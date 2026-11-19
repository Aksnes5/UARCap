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
from models.Qformer_txt import BertLMHeadModel_txt, BertEmbeddings

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
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
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
        num_video_query_token=32,
        freeze_video_qformer=True,
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

        ##### Image
        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            print("vision encoder is frozen")
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            print("Qformer is frozen")
        print('Loading Q-Former Done')

        ##### Text
        if retriever:
            print('ext_data:', ext_path)
            self.cap = json.load(open(ext_path))
            print('sim scores:', sim_scores_path)
            self.sims_scores = json.load(open(sim_scores_path))
            print('video ext data:', video_ext_path)
            self.video_ext = json.load(open(video_ext_path))

        ##### Caption generation
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False, repo_type='lmsys')
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit},
                repo_type='lmsys',
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                repo_type='lmsys',
            )
        if freeze_llm:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('LLAMA is frozen')
        print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read()
            filted_prompts = [raw_prompts]

            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]

            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

        # video-llama
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
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = False
            print('video_Qformer is frozen')
        print('Loading video_Q-Former Done')

        # load video-llama pretrained weights
        print('Loading basepretrained video_Q-Former weights')
        if '7b' in llama_model:
            self.load_from_pretrained(url_or_filename='download/videollama/baseline_2_7B_Pretrained.pth')
        elif '13b' in llama_model:
            self.load_from_pretrained(url_or_filename='download/videollama/baseline_2_13B_Pretrained.pth')

        # ========== 文本检索融合模块（与baseline完全一致） ==========
        self.attn = nn.MultiheadAttention(
            embed_dim=self.Qformer.config.hidden_size,
            num_heads=8,
            batch_first=True
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

        # ========== Visual Soft Prompt: 在文本空间中注入视频检索信息 ==========
        # 核心理念：不在32-token瓶颈空间中操作，而是在LLM词嵌入空间(4096维)
        # 通过2个可学习soft prompt token注入视频检索信息
        # 这些token被放在text prompt末尾(captions之后)，LLM自主选择attend或忽略
        #
        # visual_prompt_base: 2个基础soft prompt token [2, 4096]
        # visual_prompt_proj: 将视频检索特征(768维)映射为prompt偏移(4096维)
        # 最终: visual_prompt = base + proj(retrieved_feats)
        llama_dim = self.llama_model.config.hidden_size  # 4096
        self.num_visual_prompts = 2

        self.visual_prompt_base = nn.Parameter(
            torch.randn(self.num_visual_prompts, llama_dim) * 0.02
        )
        self.visual_prompt_proj = nn.Sequential(
            nn.Linear(self.Qformer.config.hidden_size, self.Qformer.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.Qformer.config.hidden_size, llama_dim),
            nn.LayerNorm(llama_dim),
        )

        # not share_query_tokens
        if retriever and not self.share_query_tokens:
            print('copy query tokens for videok')
            self.retri_query_tokens = copy.deepcopy(self.query_tokens)
            self.retri_video_query_tokens = copy.deepcopy(self.video_query_tokens)

        self.fusion_llama_proj = copy.deepcopy(self.llama_proj)

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    # ========== 加载视频检索特征（用于生成visual prompt） ==========
    def load_retrieved_video_features(self, image_id, device):
        """
        加载检索到的视频特征，用于生成visual soft prompt。
        返回: [B, 768] 池化后的检索视频表征
        """
        batch_size = len(image_id)
        K = min(self.video_topn, 4)
        pooled_features = []

        for i in range(batch_size):
            video_ids = self.video_ext[image_id[i]][0][:K]
            vid_feats = []
            for vid in video_ids:
                if self.load_video_feature:
                    path = f"ext_data/video_query/{vid[26:-4]}.pt"
                    try:
                        feat = torch.load(path, map_location=device)  # [32, 768]
                    except Exception:
                        continue
                else:
                    raise ValueError("load_video_feature must be True")
                vid_feats.append(feat)

            if not vid_feats:
                vid_feats = [torch.zeros(self.num_video_query_token, self.Qformer.config.hidden_size, device=device)]

            # 所有检索视频的均值池化 → [768]
            pooled = torch.stack([f.mean(dim=0) for f in vid_feats], dim=0).mean(dim=0)  # [768]
            pooled_features.append(pooled)

        return torch.stack(pooled_features, dim=0)  # [B, 768]

    def prompt_wrap(self, img_embeds, atts_img, prompt_list, image_id, video_proj):
        """
        相比baseline，增加visual_prompt参数。
        visual_prompt: [B, 2, 4096] 插入到captions之后、target之前的soft prompt tokens
        """
        if prompt_list:
            batch_size = img_embeds.shape[0]
            emb_lists = []
            for i in range(batch_size):
                prompt = random.choice(prompt_list)
                p_before, p_after = prompt.split("<VideoHere>", 1)
                p_middle, p_after = p_after.split("<VideoHere>", 1)
                p_captions, p_end = p_after.split("<VideoHere>", 1)
                retrirval_Texts = self.cap[image_id[i]][0]
                cap = retrirval_Texts[:self.topn]

                cap = '\n'.join(cap)
                cap = '\n' + cap

                p_captions = p_captions.format(cap)
                self.llama_tokenizer.padding_side = "right"
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_middle_tokens = self.llama_tokenizer(
                    p_middle, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_captions_tokens = self.llama_tokenizer(
                    p_captions, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_end_tokens = self.llama_tokenizer(
                    p_end, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids)
                p_middle_embeds = self.llama_model.model.embed_tokens(p_middle_tokens.input_ids)
                p_captions_embeds = self.llama_model.model.embed_tokens(p_captions_tokens.input_ids)
                p_end_embeds = self.llama_model.model.embed_tokens(p_end_tokens.input_ids)
                img_embeds_i = img_embeds[i].unsqueeze(0)
                video_proj_i = video_proj[i].unsqueeze(0)

                # 拼接: p_before + video_proj + p_middle + img_embeds + p_captions + visual_prompt + p_end
                # ===== Visual Soft Prompt 生成 ===== # cider71.2
                # 用视频检索特征生成2个soft prompt token
                retrieved_feats = self.load_retrieved_video_features(image_id, img_embeds.device)  # [B, 768]
                visual_prompt = self.visual_prompt_base.unsqueeze(0) + self.visual_prompt_proj(retrieved_feats).unsqueeze(1)
                # visual_prompt: [B, 2, 4096]
                # 初始化: base ~ N(0, 0.02^2)，proj输出近0 → 初期几乎无影响，渐进学习
                if visual_prompt is not None:
                    visual_prompt_i = visual_prompt[i].unsqueeze(0)  # [1, 2, 4096]
                    wrapped_embed_i = torch.cat([p_before_embeds, video_proj_i, p_middle_embeds, img_embeds_i,
                                                  p_captions_embeds, visual_prompt_i, p_end_embeds], dim=1)
                else:
                    wrapped_embed_i = torch.cat([p_before_embeds, video_proj_i, p_middle_embeds, img_embeds_i,
                                                  p_captions_embeds, p_end_embeds], dim=1)

                emb_lists.append(wrapped_embed_i)

            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.llama_model.model.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=img_embeds.device))
            wrapped_embs = pad_emb.expand(len(emb_lens), max(emb_lens), -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max(emb_lens)], dtype=torch.int, device=img_embeds.device)
            for i, emb in enumerate(emb_lists):
                wrapped_embs[i, :emb_lens[i]] = emb
                wrapped_atts[i, :emb_lens[i]] = 1
            return wrapped_embs, wrapped_atts
        else:
            return img_embeds, atts_img

    def pre_name(self, caption):
        caption = re.sub(
            r"([_!,'\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")
        return caption

    # ========== 文本检索融合（与baseline完全一致，不做任何修改） ==========
    def fusion_train(self, video_hidden, video_hidden_atts, image_id, train_dataset):
        batch_size = video_hidden.shape[0]
        device = video_hidden.device
        hidden_size = video_hidden.shape[-1]
        num_tokens = self.num_video_query_token

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
                    except Exception as e:
                        print(f"Warning: failed to load {video_feature_path}, skip. Error: {e}")
                        continue
                else:
                    idx = int(selected_video_id[31:-4]) * 20
                    video_feature = train_dataset[idx]['image'].to(device)
                    video_feature, _, _, _ = self.encode_img(video_feature, selected_video_id, train_dataset, bs=1, fusion=False, share_query_tokens=self.share_query_tokens)
                    video_feature = video_feature.squeeze(0)

                if caption in sim_caps:
                    idx = sim_caps.index(caption)
                    sim_score_val = float(sim_scores[idx])
                    video_features.append(video_feature)
                    sim_scores_selected.append(sim_score_val)
                else:
                    continue

            if not video_features:
                fused_external = torch.zeros(num_tokens, hidden_size, device=device)
            else:
                stacked_features = torch.stack(video_features, dim=0)
                sim_tensor = torch.tensor(sim_scores_selected, device=device, dtype=torch.float32)

                weights = torch.softmax(sim_tensor, dim=0).unsqueeze(-1).unsqueeze(-1)

                fused_external = (stacked_features * weights).sum(dim=0)

            fused_external_list.append(fused_external.unsqueeze(0))

        batch_fused_external = torch.cat(fused_external_list, dim=0)
        return batch_fused_external

    def encode_img(self, image, image_id, train_dataset, bs=1, fusion=True, share_query_tokens=True):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            if share_query_tokens:
                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            else:
                query_tokens = self.retri_query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs_img = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output_img = query_outputs_img.last_hidden_state
            query_output_img = rearrange(query_output_img, '(b f) hw c-> b f hw c', b=bs)

            # video-llama
            position_ids = torch.arange(self.frames, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(bs, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)

            query_output_img = query_output_img + frame_position_embeddings
            query_output_img = rearrange(query_output_img, 'b f hw c-> b (f hw) c', b=bs)

            query_output_img_atts = torch.ones(query_output_img.size()[:-1], dtype=torch.long).to(device)

            # video-llama
            if share_query_tokens:
                video_query_tokens = self.video_query_tokens.expand(bs, -1, -1)
            else:
                video_query_tokens = self.retri_video_query_tokens.expand(bs, -1, -1)
            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=query_output_img,
                encoder_attention_mask=query_output_img_atts,
                return_dict=True,
            )
            video_hidden = video_query_output.last_hidden_state
            video_hidden_atts = torch.ones(video_hidden.size()[:-1], dtype=torch.long).to(device)

            if fusion:
                # 文本检索融合（与baseline完全一致）
                video_fusion = self.fusion_train(video_hidden, video_hidden_atts, image_id, train_dataset)

                video_fusion, _ = self.attn(
                    video_hidden,
                    video_fusion,
                    video_fusion
                )

                combined_features = torch.concat((video_hidden, video_fusion), dim=-1)
                update_gate = self.update_gate(combined_features)
                reset_gate = self.reset_gate(combined_features)
                reset_video = video_hidden * reset_gate
                combined_reset = torch.cat([reset_video, video_fusion], dim=-1)
                new_features = self.new_transform(combined_reset)
                query_output_all = (1 - update_gate) * video_hidden + update_gate * new_features

                qform_all_proj = self.fusion_llama_proj(query_output_all)
            else:
                qform_all_proj = video_hidden

            video_proj = self.llama_proj(video_hidden)
            atts_qform_all_proj = torch.ones(qform_all_proj.size()[:-1], dtype=torch.long).to(device)
        return qform_all_proj, atts_qform_all_proj, video_proj

    def forward(self, samples, train_dataset):
        ##### Image
        image = samples["image"]
        image_id = samples['image_id']
        image = rearrange(image, 'b f c h w -> (b f) c h w')
        qform_all_proj, atts_qform_all_proj, video_proj = self.encode_img(image, image_id, train_dataset, bs=self.bs)

        if self.prompt_list:
            prompt_embeds, atts_prompt = self.prompt_wrap(
                qform_all_proj, atts_qform_all_proj, self.prompt_list, image_id, video_proj
            )

        ##### Caption generation
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["text_input"]]
        text_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)

        bos = torch.ones([qform_all_proj.shape[0], 1],
                         dtype=text_tokens.input_ids.dtype,
                         device=text_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_qform_all_proj[:, :1]

        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones([qform_all_proj.shape[0], 1 + prompt_embeds.shape[1]],
                       dtype=torch.long).to(image.device).fill_(-100)
        )
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
        loss = outputs.loss

        return {"output": outputs[0], "loss": loss}