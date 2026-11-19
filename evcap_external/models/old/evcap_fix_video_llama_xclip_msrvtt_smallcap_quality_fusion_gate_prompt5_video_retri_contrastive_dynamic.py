# evcap_fix_video_llama_xclip_msrvtt_smallcap_quality_fusion_gate_prompt5_video_retri_contrastive_dynamic.py
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
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
        contrastive_loss_weight=0.1,
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
        self.contrastive_loss_weight = contrastive_loss_weight
          
        print('topn:', self.topn, '  video_topn:', self.video_topn)

        # 视觉编码器
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

        # Q-Former
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

        # 检索数据
        if retriever: 
            self.cap = json.load(open(ext_path))
            self.sims_scores = json.load(open(sim_scores_path))
            self.video_ext = json.load(open(video_ext_path))

        # LLM
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

        self.llama_proj = nn.Linear(self.Qformer.config.hidden_size, self.llama_model.config.hidden_size)
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        # 提示模板
        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read()
            self.prompt_list = [prompt_template.format(raw_prompts)]
        else:
            self.prompt_list = []

        # Video Q-Former
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

        # 加载Video-LLaMA预训练权重
        if '7b' in llama_model:
            self.load_from_pretrained(url_or_filename='download/videollama/baseline_2_7B_Pretrained.pth')
        elif '13b' in llama_model:
            self.load_from_pretrained(url_or_filename='download/videollama/baseline_2_13B_Pretrained.pth')

        # 文本检索融合模块
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.Qformer.config.hidden_size, num_heads=8, batch_first=True)
        self.update_gate = nn.Sequential(
            nn.Linear(self.Qformer.config.hidden_size * 2, self.Qformer.config.hidden_size), nn.Sigmoid()
        )
        self.reset_gate = nn.Sequential(
            nn.Linear(self.Qformer.config.hidden_size * 2, self.Qformer.config.hidden_size), nn.Sigmoid()
        )
        self.new_transform = nn.Sequential(
            nn.Linear(self.Qformer.config.hidden_size * 2, self.Qformer.config.hidden_size), nn.Tanh()
        )

        # 视频检索融合模块
        self.video_cross_attn = nn.MultiheadAttention(embed_dim=self.Qformer.config.hidden_size, num_heads=8, batch_first=True)
        self.video_update_gate = nn.Sequential(
            nn.Linear(self.Qformer.config.hidden_size * 2, self.Qformer.config.hidden_size), nn.Sigmoid()
        )
        self.video_reset_gate = nn.Sequential(
            nn.Linear(self.Qformer.config.hidden_size * 2, self.Qformer.config.hidden_size), nn.Sigmoid()
        )
        self.video_new_transform = nn.Sequential(
            nn.Linear(self.Qformer.config.hidden_size * 2, self.Qformer.config.hidden_size), nn.Tanh()
        )

        # 动态权重预测层
        self.fusion_weight_net = nn.Sequential(
            nn.Linear(self.Qformer.config.hidden_size, self.Qformer.config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.Qformer.config.hidden_size // 2, 2)
        )

        if retriever and not self.share_query_tokens:
            self.retri_query_tokens = copy.deepcopy(self.query_tokens)
            self.retri_video_query_tokens = copy.deepcopy(self.video_query_tokens)

        self.fusion_llama_proj = copy.deepcopy(self.llama_proj)

    # ---------- 辅助方法 ----------
    def vit_to_cpu(self):
        self.ln_vision.to("cpu").float()
        self.visual_encoder.to("cpu").float()

    def prompt_wrap(self, img_embeds, atts_img, prompt_list, image_id, video_proj):
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

    def text_retrieval_fusion(self, video_hidden, image_id, train_dataset):
        """文本检索融合 (与baseline一致)"""
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
                    path = f"ext_data/video_query/{selected_video_id[26:-4]}.pt"
                    try:
                        feat = torch.load(path, map_location=device)
                    except:
                        continue
                else:
                    idx = int(selected_video_id[31:-4]) * 20
                    frame_tensor = train_dataset[idx]['image'].to(device)
                    feat, _, _, _ = self.encode_img(frame_tensor, selected_video_id, train_dataset, bs=1, fusion=False, share_query_tokens=self.share_query_tokens)
                    feat = feat.squeeze(0)
                if caption in sim_caps:
                    idx = sim_caps.index(caption)
                    sim_score_val = float(sim_scores[idx])
                    video_features.append(feat)
                    sim_scores_selected.append(sim_score_val)
            if not video_features:
                fused_external = torch.zeros(self.num_video_query_token, video_hidden.shape[-1], device=device)
            else:
                stacked = torch.stack(video_features, dim=0)
                sim_tensor = torch.tensor(sim_scores_selected, device=device, dtype=torch.float32)
                weights = torch.softmax(sim_tensor, dim=0).unsqueeze(-1).unsqueeze(-1)
                fused_external = (stacked * weights).sum(dim=0)
            fused_external_list.append(fused_external.unsqueeze(0))
        batch_fused_external = torch.cat(fused_external_list, dim=0)
        video_fusion, _ = self.cross_attn(video_hidden, batch_fused_external, batch_fused_external)
        # GRU-like gate 
        combined = torch.cat([video_hidden, video_fusion], dim=-1)
        update_gate = self.update_gate(combined)
        reset_gate = self.reset_gate(combined)
        reset_video = video_hidden * reset_gate
        combined_reset = torch.cat([reset_video, video_fusion], dim=-1)
        new_features = self.new_transform(combined_reset)
        fused = (1 - update_gate) * video_hidden + update_gate * new_features
        return fused

    def video_retrieval_fusion(self, video_hidden, image_id, train_dataset):
        """视频检索融合 (独立门控)"""
        batch_size = video_hidden.shape[0]
        device = video_hidden.device
        retrieved_feats = self.load_retrieved_video_features(image_id, train_dataset, device)
        B, K, T, D = retrieved_feats.shape
        # 基于token级相似度加权聚合
        sim = torch.einsum('b t d, b k t d -> b k', video_hidden, retrieved_feats) / (T * D**0.5)
        weights = torch.softmax(sim / 0.1, dim=1)
        aggregated = torch.einsum('b k, b k t d -> b t d', weights, retrieved_feats)
        video_fusion, _ = self.video_cross_attn(video_hidden, aggregated, aggregated)
        combined = torch.cat([video_hidden, video_fusion], dim=-1)
        update_gate = self.video_update_gate(combined)
        reset_gate = self.video_reset_gate(combined)
        reset_video = video_hidden * reset_gate
        combined_reset = torch.cat([reset_video, video_fusion], dim=-1)
        new_features = self.video_new_transform(combined_reset)
        fused = (1 - update_gate) * video_hidden + update_gate * new_features
        return fused

    def load_retrieved_video_features(self, image_id, train_dataset, device):
        """返回 [B, K, 32, 768] 检索视频特征"""
        batch_size = len(image_id)
        batch_feats = []
        for i in range(batch_size):
            video_ids = self.video_ext[image_id[i]][0][:self.video_topn]
            feats = []
            for vid in video_ids:
                if self.load_video_feature:
                    path = f"ext_data/video_query/{vid[26:-4]}.pt"
                    try:
                        feat = torch.load(path, map_location=device)
                    except:
                        continue
                else:
                    idx = int(vid[31:-4]) * 20
                    frame_tensor = train_dataset[idx]['image'].to(device)
                    feat, _, _, _ = self.encode_img(frame_tensor, vid, train_dataset, bs=1, fusion=False, share_query_tokens=self.share_query_tokens)
                    feat = feat.squeeze(0)
                feats.append(feat)
            if len(feats) == 0:
                feats = [torch.zeros(self.num_video_query_token, self.Qformer.config.hidden_size, device=device)]
            batch_feats.append(torch.stack(feats, dim=0))
        return torch.stack(batch_feats, dim=0)

    def contrastive_loss(self, video_hidden, image_id, train_dataset):
        """
        video_hidden: [B, 32, 768]
        正样本：从检索视频中选取与当前视频相似度最高的一个
        负样本：batch内其他视频（使用当前视频的全局池化表示）
        """
        B = video_hidden.shape[0]
        device = video_hidden.device
        retrieved_feats = self.load_retrieved_video_features(image_id, train_dataset, device)  # [B, K, 32, 768]
        # 计算当前视频与每个检索视频的相似度（平均token）
        sim_to_retrieved = torch.einsum('b t d, b k t d -> b k', video_hidden, retrieved_feats) / (32 * 768**0.5)
        best_idx = sim_to_retrieved.argmax(dim=1)  # [B]
        positive = retrieved_feats[torch.arange(B), best_idx]  # [B, 32, 768]
        # 全局平均池化得到视频级表示
        anchor_pool = video_hidden.mean(dim=1)   # [B,768]
        pos_pool = positive.mean(dim=1)          # [B,768]
        # 归一化
        anchor_norm = F.normalize(anchor_pool, dim=-1)
        pos_norm = F.normalize(pos_pool, dim=-1)
        # 计算相似度矩阵，对角线为正样本，其他为负样本
        logits = anchor_norm @ pos_norm.t()      # [B,B]
        labels = torch.arange(B, device=device)
        loss = F.cross_entropy(logits / 0.07, labels)
        return loss

    def encode_img(self, image, image_id, train_dataset, bs=1, fusion=True, share_query_tokens=True):
        """
        返回:
            qform_all_proj: [B, 32, LLM_dim]  用于生成caption的特征
            atts_qform_all_proj: [B, 32]      注意力掩码
            video_proj: [B, frames*32, LLM_dim]  原始视频投影，用于prompt中的<VideoHere>
            video_hidden: [B, 32, 768]        原始的video Q-Former输出（未融合前）
        """
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            # 视觉编码 + Q-Former
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
            query_tokens = (self.query_tokens if share_query_tokens else self.retri_query_tokens).expand(image_embeds.shape[0], -1, -1)
            query_outputs_img = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output_img = query_outputs_img.last_hidden_state
            query_output_img = rearrange(query_output_img, '(b f) hw c -> b f hw c', b=bs)

            # 帧位置编码 + Video Q-Former
            position_ids = torch.arange(self.frames, dtype=torch.long, device=device).unsqueeze(0).expand(bs, -1)
            frame_position_embeds = self.video_frame_position_embedding(position_ids).unsqueeze(-2)
            query_output_img = query_output_img + frame_position_embeds
            query_output_img_flat = rearrange(query_output_img, 'b f hw c -> b (f hw) c')
            query_output_img_atts = torch.ones(query_output_img_flat.size()[:-1], dtype=torch.long, device=device)

            video_query_tokens = (self.video_query_tokens if share_query_tokens else self.retri_video_query_tokens).expand(bs, -1, -1)
            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=query_output_img_flat,
                encoder_attention_mask=query_output_img_atts,
                return_dict=True,
            )
            video_hidden = video_query_output.last_hidden_state  # [B, 32, 768]

            if fusion: # cider71.3
                # 文本融合
                text_fused = self.text_retrieval_fusion(video_hidden, image_id, train_dataset)
                # 视频融合
                video_fused = self.video_retrieval_fusion(video_hidden, image_id, train_dataset)
                # 动态权重
                pool = video_hidden.mean(dim=1)  # [B,768]
                weight_logits = self.fusion_weight_net(pool)
                weights = torch.softmax(weight_logits, dim=1)  # [B,2]
                w_text = weights[:, 0].unsqueeze(-1).unsqueeze(-1)
                w_video = weights[:, 1].unsqueeze(-1).unsqueeze(-1)
                combined_fused = w_text * text_fused + w_video * video_fused
                enhanced = video_hidden + combined_fused
                qform_all_proj = self.fusion_llama_proj(enhanced)
                video_proj = self.llama_proj(video_hidden)
            else:
                qform_all_proj = video_hidden
                video_proj = self.llama_proj(video_hidden)

            atts_qform_all_proj = torch.ones(qform_all_proj.size()[:-1], dtype=torch.long, device=device)
        return qform_all_proj, atts_qform_all_proj, video_proj, video_hidden

    def forward(self, samples, train_dataset):
        image = samples["image"]                     # [bs, frames, 3, H, W]
        image_id = samples['image_id']
        image = rearrange(image, 'b f c h w -> (b f) c h w')
        qform_all_proj, atts_qform_all_proj, video_proj, video_hidden = self.encode_img(
            image, image_id, train_dataset, bs=self.bs
        )

        # 对比损失
        contrast_loss = torch.tensor(0.0, device=image.device)
        if self.training and self.contrastive_loss_weight > 0:
            contrast_loss = self.contrastive_loss(video_hidden, image_id, train_dataset)

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
        loss = outputs.loss + self.contrastive_loss_weight * contrast_loss
        return {"output": outputs[0], "loss": loss}