import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import random
import copy
import re
from models.blip2 import Blip2Base, disabled_train
from models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
from einops import rearrange
from models.Qformer import BertConfig, BertLMHeadModel
import json

class EVCap(Blip2Base):
    
    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(
        self,
        ext_path,
        sim_scores_path,
        video_ext_path,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        num_query_token_txt=8,
        topn=10,
        video_topn=5,
        llama_model="",
        prompt_path="prompts/prompt_evcap_msrvtt.txt",
        prompt_template='###Human: {} ###Assistant: ',
        max_txt_len=128,
        end_sym='\n',
        low_resource=False,
        device_8bit=0,
        bs=1,
        frames=32,
        retriever=True,
        num_video_query_token=32, # video-llama
        freeze_video_qformer=True, # video-llama
        freeze_qformer_txt=True,
        load_video_feature=True,
        share_query_tokens=True,
        freeze_llm=True,
    ):
        super().__init__()

        self.low_resource = low_resource
        self.topn = topn
        self.video_topn = video_topn
        self.bs = bs
        self.frames = frames      
        self.retriever = retriever
        self.load_video_feature = load_video_feature
        self.share_query_tokens = share_query_tokens
          
        print('topn:', self.topn, '  video_topn:', self.video_topn)

        # ---------- 视觉编码器 ----------
        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            self.visual_encoder.eval().train = disabled_train
            for param in self.ln_vision.parameters():
                param.requires_grad = False
            self.ln_vision.eval().train = disabled_train
            print("vision encoder frozen")

        # ---------- Q-Former ----------
        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, self.visual_encoder.num_features)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)
        if freeze_qformer:
            for param in self.Qformer.parameters():
                param.requires_grad = False
            self.Qformer.eval().train = disabled_train
            print("Qformer frozen")

        # ---------- 检索数据 ----------
        if retriever: 
            print('ext_data:', ext_path)
            self.cap = json.load(open(ext_path))
            print('sim scores:', sim_scores_path)
            self.sims_scores = json.load(open(sim_scores_path))
            print('video ext data:', video_ext_path)
            self.video_ext = json.load(open(video_ext_path))

        # ---------- LLM ----------
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False, repo_type='lmsys')
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model, torch_dtype=torch.float16, load_in_8bit=True,
                device_map={'': device_8bit}, repo_type='lmsys'
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model, torch_dtype=torch.float16, repo_type='lmsys'
            )
        if freeze_llm:
            for param in self.llama_model.parameters():
                param.requires_grad = False
            print('LLAMA frozen')

        self.llama_proj = nn.Linear(self.Qformer.config.hidden_size, self.llama_model.config.hidden_size)
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        # 提示模板
        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read()
            self.prompt_list = [prompt_template.format(raw_prompts)]
            print('Load prompts, example:', random.choice(self.prompt_list))
        else:
            self.prompt_list = []

        # ---------- Video Q-Former （处理帧序列）----------
        print('Loading video_Q-Former')
        self.video_frame_position_embedding = nn.Embedding(self.frames, self.Qformer.config.hidden_size)
        self.num_video_query_token = num_video_query_token
        self.video_Qformer, self.video_query_tokens = self.init_video_Qformer(
            num_query_token=num_video_query_token,
            vision_width=self.Qformer.config.hidden_size,
            num_hidden_layers=2
        )
        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        if freeze_video_qformer:
            for param in self.video_Qformer.parameters():
                param.requires_grad = False
            print('video_Qformer frozen')

        # 加载Video-LLaMA预训练权重
        print('Loading basepretrained video_Q-Former weights')
        if '7b' in llama_model:
            self.load_from_pretrained(url_or_filename='download/videollama/baseline_2_7B_Pretrained.pth')
        elif '13b' in llama_model:
            self.load_from_pretrained(url_or_filename='download/videollama/baseline_2_13B_Pretrained.pth')

        # ---------- 文本检索融合模块（与baseline一致）----------
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.Qformer.config.hidden_size, num_heads=8, batch_first=True
        )
        self.update_gate = nn.Sequential(
            nn.Linear(self.Qformer.config.hidden_size * 2, self.Qformer.config.hidden_size),
            nn.Sigmoid()
        )
        self.reset_gate = nn.Sequential(
            nn.Linear(self.Qformer.config.hidden_size * 2, self.Qformer.config.hidden_size),
            nn.Sigmoid()
        )
        self.new_transform = nn.Sequential(
            nn.Linear(self.Qformer.config.hidden_size * 2, self.Qformer.config.hidden_size),
            nn.Tanh()
        )

        # ---------- 新增：视频检索自适应残差门控 ----------
        # 用于融合检索视频特征与当前视频特征的线性投影（生成门控值）
        self.retrieval_gate_proj = nn.Linear(self.Qformer.config.hidden_size * 2, self.Qformer.config.hidden_size)
        # 初始化偏置为负，使gate初始接近0，避免初期干扰
        nn.init.constant_(self.retrieval_gate_proj.bias, -2.0)

        # 非共享query token时复制（保持与baseline一致）
        if retriever and not self.share_query_tokens:
            self.retri_query_tokens = copy.deepcopy(self.query_tokens)
            self.retri_video_query_tokens = copy.deepcopy(self.video_query_tokens)

        self.fusion_llama_proj = copy.deepcopy(self.llama_proj)

    # ---------- 辅助方法 ----------
    def vit_to_cpu(self):
        self.ln_vision.to("cpu").float()
        self.visual_encoder.to("cpu").float()

    def prompt_wrap(self, img_embeds, atts_img, prompt_list, image_id, video_proj):
        """与baseline完全相同，不做改动"""
        if not prompt_list:
            return img_embeds, atts_img
        batch_size = img_embeds.shape[0]            
        emb_lists = []
        for i in range(batch_size):
            prompt = random.choice(prompt_list)
            p_before, p_after = prompt.split("<VideoHere>", 1)
            p_middle, p_after = p_after.split("<VideoHere>", 1)
            cap = self.cap[image_id[i]][0][:self.topn]
            cap = '\n' + '\n'.join(cap)
            p_after = p_after.format(cap)
            self.llama_tokenizer.padding_side = "right"
            p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_middle_tokens = self.llama_tokenizer(p_middle, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids)
            p_middle_embeds = self.llama_model.model.embed_tokens(p_middle_tokens.input_ids)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids)
            img_embeds_i = img_embeds[i].unsqueeze(0)
            video_proj_i = video_proj[i].unsqueeze(0)
            wrapped_embed_i = torch.cat([p_before_embeds, video_proj_i, p_middle_embeds, img_embeds_i, p_after_embeds], dim=1)
            emb_lists.append(wrapped_embed_i)  
        emb_lens = [emb.shape[1] for emb in emb_lists]
        pad_emb = self.llama_model.model.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=img_embeds.device))
        wrapped_embs = pad_emb.expand(len(emb_lens), max(emb_lens), -1).clone()
        wrapped_atts = torch.zeros([len(emb_lens), max(emb_lens)], dtype=torch.int, device=img_embeds.device)
        for i, emb in enumerate(emb_lists):
            wrapped_embs[i, :emb_lens[i]] = emb
            wrapped_atts[i, :emb_lens[i]] = 1
        return wrapped_embs, wrapped_atts

    # ---------- 文本检索融合（与baseline一致）----------
    def fusion_train(self, video_hidden, video_hidden_atts, image_id, train_dataset):
        batch_size = video_hidden.shape[0]
        device = video_hidden.device
        fused_external_list = []
        for i in range(batch_size):
            retrival_Texts = self.cap[image_id[i]][0]
            retrival_video_ids = self.cap[image_id[i]][2]
            sim_scores = self.sims_scores[image_id[i]][1]
            sim_caps = self.sims_scores[image_id[i]][0]
            cap = retrival_Texts[:self.topn]
            video_ids = retrival_video_ids[:self.topn]
            video_features = []
            sim_scores_selected = []
            for j, caption in enumerate(cap):
                corresponding_video_ids = video_ids[j]
                if not corresponding_video_ids:
                    continue
                selected_video_id = random.choice(corresponding_video_ids)
                if self.load_video_feature:
                    video_feature_path = f"ext_data/video_query/{selected_video_id[26:-4]}.pt"
                    try:
                        video_feature = torch.load(video_feature_path, map_location=device)
                    except:
                        continue
                else:
                    idx = int(selected_video_id[31:-4]) * 20
                    video_feature = train_dataset[idx]['image'].to(device)
                    video_feature, _, _ = self.encode_img(video_feature, selected_video_id, train_dataset, bs=1, fusion=False, share_query_tokens=self.share_query_tokens)
                    video_feature = video_feature.squeeze(0)
                if caption in sim_caps:
                    idx = sim_caps.index(caption)
                    sim_score_val = float(sim_scores[idx])
                    video_features.append(video_feature)
                    sim_scores_selected.append(sim_score_val)
            if not video_features:
                # fallback
                fused_external = torch.zeros(self.num_video_query_token, video_hidden.shape[-1], device=device)
            else:
                stacked_features = torch.stack(video_features, dim=0)  # [K,32,768]
                sim_tensor = torch.tensor(sim_scores_selected, device=device, dtype=torch.float32)
                weights = torch.softmax(sim_tensor, dim=0).unsqueeze(-1).unsqueeze(-1)
                fused_external = (stacked_features * weights).sum(dim=0)
            fused_external_list.append(fused_external.unsqueeze(0))
        batch_fused_external = torch.cat(fused_external_list, dim=0)
        video_fusion, _ = self.cross_attn(video_hidden, batch_fused_external, batch_fused_external)
        return video_fusion

    # ---------- 新增：视频检索特征加载（返回 [B, K, 32, 768]）----------
    def load_retrieved_video_features(self, image_id, train_dataset, device):
        """加载检索到的视频特征（K = video_topn）"""
        batch_size = len(image_id)
        batch_features = []
        for i in range(batch_size):
            video_ids = self.video_ext[image_id[i]][0][:self.video_topn]
            scores = self.video_ext[image_id[i]][1][:self.video_topn]   # 相似度分数，后续用于attention权重
            vid_feats = []
            for j, vid in enumerate(video_ids):
                if self.load_video_feature:
                    path = f"ext_data/video_query/{vid[26:-4]}.pt"
                    try:
                        feat = torch.load(path, map_location=device)   # [32,768]
                    except:
                        continue
                else:
                    idx = int(vid[31:-4]) * 20
                    frame_tensor = train_dataset[idx]['image'].to(device)
                    feat, _, _ = self.encode_img(frame_tensor, vid, train_dataset, bs=1, fusion=False, share_query_tokens=self.share_query_tokens)
                    feat = feat.squeeze(0)
                vid_feats.append(feat)
            if not vid_feats:
                # 若无有效检索，填充零
                vid_feats = [torch.zeros(self.num_video_query_token, self.Qformer.config.hidden_size, device=device)]
            batch_features.append(torch.stack(vid_feats, dim=0))   # [K,32,768]
        # 对每个样本单独stack，因为K可能不同，取最小K？实际各样本K相同（video_topn），直接stack
        return torch.stack(batch_features, dim=0)   # [B, K, 32, 768]

    # ---------- 核心编码函数（增加视频检索增强）----------
    def encode_img(self, image, image_id, train_dataset, bs=1, fusion=True, share_query_tokens=True):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            # 1. 视觉编码 + Q-Former
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
            query_tokens = (self.query_tokens if share_query_tokens else self.retri_query_tokens).expand(image_embeds.shape[0], -1, -1)
            query_outputs_img = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output_img = query_outputs_img.last_hidden_state                     # [bs*frames,32,768]
            query_output_img = rearrange(query_output_img, '(b f) hw c -> b f hw c', b=bs)  # [bs, frames,32,768]

            # 2. 帧位置编码 + Video Q-Former
            position_ids = torch.arange(self.frames, dtype=torch.long, device=device).unsqueeze(0).expand(bs, -1)
            frame_position_embeds = self.video_frame_position_embedding(position_ids).unsqueeze(-2)  # [bs, frames,1,768]
            query_output_img = query_output_img + frame_position_embeds
            query_output_img_flat = rearrange(query_output_img, 'b f hw c -> b (f hw) c')           # [bs, frames*32,768]
            query_output_img_atts = torch.ones(query_output_img_flat.size()[:-1], dtype=torch.long, device=device)

            video_query_tokens = (self.video_query_tokens if share_query_tokens else self.retri_video_query_tokens).expand(bs, -1, -1)
            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=query_output_img_flat,
                encoder_attention_mask=query_output_img_atts,
                return_dict=True,
            )
            video_hidden = video_query_output.last_hidden_state   # [bs, 32, 768]
            video_hidden_atts = torch.ones(video_hidden.size()[:-1], dtype=torch.long, device=device)

            # 3. 视频检索增强（自适应残差门控） # cider71.7
            if fusion:
                # 3.1 加载检索视频特征 [B, K, 32, 768]
                retrieved_feats = self.load_retrieved_video_features(image_id, train_dataset, device)
                # 3.2 对每个样本，计算当前视频与各检索视频的token级相似度，得到K个权重
                B, K, T, D = retrieved_feats.shape
                # 计算相似度矩阵 [B, K]：平均token余弦相似度
                sim_matrix = torch.einsum('b t d, b k t d -> b k', video_hidden, retrieved_feats) / (T * D**0.5)
                attn_weights = torch.softmax(sim_matrix / 0.1, dim=1)   # 温度0.1增强区分度
                # 加权聚合检索视频特征 -> [B, 32, 768]
                retrieved_context = torch.einsum('b k, b k t d -> b t d', attn_weights, retrieved_feats)
                # 3.3 计算残差门控
                combined = torch.cat([video_hidden, retrieved_context], dim=-1)   # [B,32,1536]
                gate = torch.sigmoid(self.retrieval_gate_proj(combined))          # [B,32,768]
                # 自适应残差：增强后的特征 = 原特征 + gate * (检索上下文 - 原特征)
                video_hidden_enhanced = video_hidden + gate * (retrieved_context - video_hidden)

                # 4. 文本检索融合（基于增强后的video_hidden）
                video_fusion = self.fusion_train(video_hidden_enhanced, video_hidden_atts, image_id, train_dataset)
                # GRU-like gate
                combined_features = torch.cat((video_hidden_enhanced, video_fusion), dim=-1)
                update_gate = self.update_gate(combined_features)
                reset_gate = self.reset_gate(combined_features)                
                reset_video = video_hidden_enhanced * reset_gate
                combined_reset = torch.cat([reset_video, video_fusion], dim=-1)                
                new_features = self.new_transform(combined_reset)                
                integrated_features = (1 - update_gate) * video_hidden_enhanced + update_gate * new_features
                qform_all_proj = self.fusion_llama_proj(integrated_features)

                #    因为文本检索分支的增强已经用于qform_all_proj，而video_proj仍然代表原始视频内容
                video_proj = self.llama_proj(video_hidden)   # 保持原语义
            else:
                qform_all_proj = video_hidden
                video_proj = self.llama_proj(video_hidden)

            atts_qform_all_proj = torch.ones(qform_all_proj.size()[:-1], dtype=torch.long, device=device)
        return qform_all_proj, atts_qform_all_proj, video_proj

    # ---------- 前向传播（与baseline相同）----------
    def forward(self, samples, train_dataset):
        image = samples["image"]                     # [bs, frames, 3, H, W]
        image_id = samples['image_id']
        image = rearrange(image, 'b f c h w -> (b f) c h w')
        qform_all_proj, atts_qform_all_proj, video_proj = self.encode_img(image, image_id, train_dataset, bs=self.bs)

        if self.prompt_list:
            prompt_embeds, atts_prompt = self.prompt_wrap(qform_all_proj, atts_qform_all_proj, self.prompt_list, image_id, video_proj)

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["text_input"]]
        text_tokens = self.llama_tokenizer(
            text, return_tensors="pt", padding="longest", truncation=True,
            max_length=self.max_txt_len, add_special_tokens=False
        ).to(image.device)

        bos = torch.ones([qform_all_proj.shape[0], 1], dtype=text_tokens.input_ids.dtype, device=image.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_qform_all_proj[:, :1]

        targets = text_tokens.input_ids.masked_fill(text_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100)
        empty_targets = torch.ones([qform_all_proj.shape[0], 1 + prompt_embeds.shape[1]], dtype=torch.long, device=image.device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)
        text_embeds = self.llama_model.model.embed_tokens(text_tokens.input_ids)

        inputs_embeds = torch.cat([bos_embeds, prompt_embeds, text_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_prompt, text_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        return {"output": outputs[0], "loss": outputs.loss}