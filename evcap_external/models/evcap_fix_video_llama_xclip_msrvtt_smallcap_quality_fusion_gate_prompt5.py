import os
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
from models.Qformer_txt import BertLMHeadModel_txt, BertEmbeddings
# from models.Qformer_txt_global_fusion import BertAttention

# XCLIP
import json

class EVCap(Blip2Base):
    
    @classmethod # video-llama
    def init_video_Qformer(cls, num_query_token, vision_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
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
        text_topn=None,
        visual_topn=None,
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
        sim_scores_path=None,
        video_feature_dir=None,
        use_fusion=True,
        deterministic=True,
        external_features=False,
        no_retrieval=False,
    ):
        super().__init__()

        self.low_resource = low_resource
        self.topn = topn
        self.text_topn = topn if text_topn is None else text_topn
        self.visual_topn = topn if visual_topn is None else visual_topn
        self.bs = bs
        self.frames = frames      
        self.retriever = retriever
        self.load_video_feature = load_video_feature
        self.share_query_tokens = share_query_tokens
        self.video_feature_dir = video_feature_dir
        self.use_fusion = use_fusion
        self.deterministic = deterministic
        self.external_features = external_features
        self.no_retrieval = no_retrieval
          
        print('topn:', self.topn)

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
            # self.query_tokens.requires_grad = True
            print("Qformer is frozen")
            # self.query_tokens.requires_grad = False
            # print("Qformer query tokens is frozen")
        print('Loading Q-Former Done')

        ##### Text 
        if retriever: 
            # XCLIP retrieve caption
            print('ext_data:', ext_path)
            self.cap = json.load(open(ext_path))
            if sim_scores_path is None:
                sim_scores_path = 'ext_data/msrvtt_train_allGT_avg_frames_xclip_retrieve_20_captions_no_duplicate_sim_scores.json'
            print('sim scores:', sim_scores_path)
            self.sims_scores = json.load(open(sim_scores_path))

        ##### Caption generation 
        print('Loading LLAMA')
        # by lijian
        # self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False, repo_type='lmsys')
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit},
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )
        if freeze_llm:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('LLAMA is frozen')
        print('Loading LLAMA Done')

        ###
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        
        if prompt_path:
            with open(prompt_path, 'r') as f:
                # raw_prompts = f.read().splitlines()
                raw_prompts = f.read() # detail
            # filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<VideoHere>" in raw_prompt]
            filted_prompts = [raw_prompts] # detail
            
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            # no huamn_assistant
            # self.prompt_list = filted_prompts
            
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

        # video-llama
        print('Loading video_Q-Former')
        self.video_frame_position_embedding = nn.Embedding(self.frames, self.Qformer.config.hidden_size)
        self.num_video_query_token = num_video_query_token
        self.video_Qformer,self.video_query_tokens = self.init_video_Qformer(num_query_token = num_video_query_token,\
            vision_width=self.Qformer.config.hidden_size, num_hidden_layers =2)
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
        # for name, param in self.video_frame_position_embedding.named_parameters():
        #     param.requires_grad = False
        # self.video_query_tokens.requires_grad = False
        # print('video_Qformer position_embedding and video_query_tokens is frozen')
        print('Loading video_Q-Former Done')
        
        # load video-llama pretrained weights
        print('Loading basepretrained video_Q-Former weights')
        if '7b' in llama_model:
            self.load_from_pretrained(url_or_filename='download/videollama/baseline_2_7B_Pretrained.pth')
        elif '13b' in llama_model:
            self.load_from_pretrained(url_or_filename='download/videollama/baseline_2_13B_Pretrained.pth')
        
        # print('Loading video branch weights')
        # self.load_from_pretrained(url_or_filename='outputs/train_evcap_fix_video_llama_xclip_msrvtt_quality/train_evcap_fix_video_llama_xclip_20_avg_frames_allGT_msrvtt_16frames_rouge_sentence_as_inference_top10_smallcap_quality_use_basepretrained_retriever.pt')

        # fusion
        self.attn = nn.MultiheadAttention(
                                        embed_dim=self.Qformer.config.hidden_size,  # 768
                                        num_heads=8,
                                        batch_first=True
                                        )
        # gate
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
        
        # not share_query_tokens
        if retriever and not self.share_query_tokens:
            # copy framework for video retrieve
            print('copy query tokens for videok')
            self.retri_query_tokens = copy.deepcopy(self.query_tokens)
            self.retri_video_query_tokens = copy.deepcopy(self.video_query_tokens)
        
        self.fusion_llama_proj = copy.deepcopy(self.llama_proj)

    
    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def prompt_wrap(self, img_embeds, atts_img, prompt_list, image_id, video_proj):
        if prompt_list:
            batch_size = img_embeds.shape[0]            
            emb_lists = []
            for i in range(batch_size):
                prompt = random.choice(prompt_list)
                if self.no_retrieval:
                    p_before, p_after = prompt.split("<VideoHere>", 1)
                    self.llama_tokenizer.padding_side = "right"
                    pb = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                    pa = self.llama_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                    wrapped_embed_i = torch.cat([self.llama_model.model.embed_tokens(pb.input_ids),
                                                 video_proj[i].unsqueeze(0),
                                                 img_embeds[i].unsqueeze(0),
                                                 self.llama_model.model.embed_tokens(pa.input_ids)], dim=1)
                    emb_lists.append(wrapped_embed_i)
                    continue
                p_before, p_after = prompt.split("<VideoHere>", 1)
                p_middle, p_after = p_after.split("<VideoHere>", 1)
                retrirval_Texts = self.cap[image_id[i]][0]
                cap = retrirval_Texts[:self.text_topn]
                
                # smallcap
                cap = '\n'.join(cap)
                cap = '\n' + cap 
                
                p_after = p_after.format(cap)
                self.llama_tokenizer.padding_side = "right"
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_middle_tokens = self.llama_tokenizer(
                    p_middle, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)        
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids)
                p_middle_embeds = self.llama_model.model.embed_tokens(p_middle_tokens.input_ids)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids)
                img_embeds_i = img_embeds[i].unsqueeze(0)
                video_proj_i = video_proj[i].unsqueeze(0)
                wrapped_embed_i = torch.cat([p_before_embeds, video_proj_i, p_middle_embeds, img_embeds_i, p_after_embeds], dim=1) # no text Q-Former
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
    
    def fusion_train(self, video_hidden, video_hidden_atts, image_id, train_dataset):
        batch_size = video_hidden.shape[0]
        device = video_hidden.device
        hidden_size = video_hidden.shape[-1]  # 768
        num_tokens = self.num_video_query_token  # 32

        fused_external_list = []

        for i in range(batch_size):
            retrival_Texts = self.cap[image_id[i]][0]
            retrival_video_ids = self.cap[image_id[i]][2]
            sim_scores = self.sims_scores[image_id[i]][1]
            sim_caps = self.sims_scores[image_id[i]][0]

            cap = retrival_Texts[:self.visual_topn]
            video_ids = retrival_video_ids[:self.visual_topn]

            video_features = []
            sim_scores_selected = []

            for j, caption in enumerate(cap):
                corresponding_video_ids = video_ids[j]
                if not corresponding_video_ids:
                    continue
                
                selected_video_id = (corresponding_video_ids[0] if self.deterministic
                                     else random.choice(corresponding_video_ids))
                if self.load_video_feature:
                    # load video feature
                    if self.video_feature_dir:
                        video_feature_path = os.path.join(self.video_feature_dir, f"{selected_video_id}.pt")
                    else:
                        video_feature_path = f"ext_data/video_query/{selected_video_id[26:-4]}.pt"
                    try:
                        video_feature = torch.load(video_feature_path, map_location=device)  # [32, 768]
                    except Exception as e:
                        print(f"Warning: failed to load {video_feature_path}, skip. Error: {e}")
                        continue
                else:
                    # train video feature
                    idx = int(selected_video_id[31:-4]) * 20
                    video_feature = train_dataset[idx]['image'].to(device)
                    video_feature, _, _ = self.encode_img(video_feature, selected_video_id, train_dataset, bs=1, fusion=False, share_query_tokens=self.share_query_tokens)
                    video_feature = video_feature.squeeze(0)

                # 获取对应的 sim_score
                if caption in sim_caps:
                    idx = sim_caps.index(caption)
                    sim_score_val = float(sim_scores[idx])
                    video_features.append(video_feature)
                    sim_scores_selected.append(sim_score_val)
                else:
                    # 如果找不到，跳过该 caption（不使用）
                    continue

            if not video_features:
                # fallback: zero vector
                fused_external = torch.zeros(num_tokens, hidden_size, device=device)
            else:
                # return torch.concat(video_features, dim=0).unsqueeze(0)
                # Stack features: [K, 32, 768]
                stacked_features = torch.stack(video_features, dim=0)  # [K, 32, 768]
                sim_tensor = torch.tensor(sim_scores_selected, device=device, dtype=torch.float32)  # [K]
                # sim_tensor = torch.tensor(quality, device=device, dtype=torch.float32)  # [K]

                # 使用 softmax(sim_score) 作为权重
                weights = torch.softmax(sim_tensor, dim=0).unsqueeze(-1).unsqueeze(-1)  # [K, 1, 1]

                # 加权平均融合 -> [32, 768]
                fused_external = (stacked_features * weights).sum(dim=0)

            fused_external_list.append(fused_external.unsqueeze(0))  # [1, 32, 768]

        # Combine batch: [B, 32, 768]
        batch_fused_external = torch.cat(fused_external_list, dim=0)
        return batch_fused_external
    
    def encode_img(self, image, image_id, train_dataset, bs=1, fusion=True, share_query_tokens=True, raw_video_feature=False): # image: [bs*frames, 3, 224, 224]
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():              
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device) # [bs*frames, 257, 1408]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device) # [bs*frames, 257]

            if share_query_tokens:
                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            else:
                query_tokens = self.retri_query_tokens.expand(image_embeds.shape[0], -1, -1)
            # self.query_tokens: [1, 32, 768]  query_tokens: [bs*frames, 32, 768]
            query_outputs_img = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output_img = query_outputs_img.last_hidden_state # [bs*frames, 32, 768]
            query_output_img = rearrange(query_output_img, '(b f) hw c-> b f hw c', b=bs) # [bs, frames, 32, 768] video-llama
            
            # video-llama
            position_ids = torch.arange(self.frames, dtype=torch.long, device=query_tokens.device) # [32]
            position_ids = position_ids.unsqueeze(0).expand(bs, -1) # [bs, 32]
            frame_position_embeddings = self.video_frame_position_embedding(position_ids) # [bs, 32, 768]
            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2) # [bs, 32, 1, 768]
                        
            query_output_img = query_output_img + frame_position_embeddings # [bs, frames, 32, 1408] video-llama
            query_output_img = rearrange(query_output_img, 'b f hw c-> b (f hw) c', b=bs) # [bs, frames*32, 768] video-llama
            
            query_output_img_atts = torch.ones(query_output_img.size()[:-1], dtype=torch.long).to(device) # [bs, frames*32]
            
            # video-llama
            if share_query_tokens:
                video_query_tokens = self.video_query_tokens.expand(bs, -1, -1) # [bs, 32, 768]
            else:
                video_query_tokens = self.retri_video_query_tokens.expand(bs, -1, -1)
            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=query_output_img,
                encoder_attention_mask=query_output_img_atts,
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state
            video_hidden_atts = torch.ones(video_hidden.size()[:-1], dtype=torch.long).to(device) # [bs, 32]
            
            if fusion and self.use_fusion:
                video_fusion = self.fusion_train(video_hidden, video_hidden_atts, image_id, train_dataset)
                
                # concat
                # query_output_all = torch.concat((video_hidden, video_fusion), dim=1)
                
                # Self-attention
                # video_fusion = torch.concat((video_hidden, video_fusion), dim=1)
                # query_output_all, _ = self.attn(
                #     video_fusion,       # [B, 32+32, 768]
                #     video_fusion,       # [B, 32+32, 768]
                #     video_fusion        # [B, 32+32, 768]
                # )
                # query_output_all = query_output_all[:, :self.num_video_query_token, :] # # [B, 32, 768]
                
                # Cross-attention: query = current video, key/value = fused external
                video_fusion, _ = self.attn(
                    video_hidden,       # [B, 32, 768]
                    video_fusion,       # [B, 32, 768]
                    video_fusion        # [B, 32, 768]
                )
                
                # Cross-attention video as kv # cider71.5
                # video_fusion, _ = self.attn(
                #     video_fusion,       # [B, 32, 768]
                #     video_hidden,       # [B, 32, 768]
                #     video_hidden,       # [B, 32, 768]
                # )
                
                # gate
                combined_features = torch.concat((video_hidden, video_fusion), dim=-1)
                update_gate = self.update_gate(combined_features)
                reset_gate = self.reset_gate(combined_features)                
                reset_video = video_hidden * reset_gate
                combined_reset = torch.cat([reset_video, video_fusion], dim=-1)                
                new_features = self.new_transform(combined_reset)                
                query_output_all = (1 - update_gate) * video_hidden + update_gate * new_features
                
                qform_all_proj = self.fusion_llama_proj(query_output_all)
            else:
                qform_all_proj = video_hidden if raw_video_feature else self.llama_proj(video_hidden)
                
            video_proj = self.llama_proj(video_hidden)
            atts_qform_all_proj = torch.ones(qform_all_proj.size()[:-1], dtype=torch.long).to(device) # [bs, 32or40]
            # atts_qform_all_proj = torch.ones(video_fusion.size()[:-1], dtype=torch.long).to(device) # [bs, 32or40]
        return qform_all_proj, atts_qform_all_proj, video_proj
        # return video_fusion, atts_qform_all_proj, video_proj


    def forward(self, samples, train_dataset):
        ##### Image
        image = samples["image"] # [bs, frames, 3, 224, 224]
        image_id = samples['image_id'] # XCLIP
        image = rearrange(image, 'b f c h w -> (b f) c h w')
        qform_all_proj, atts_qform_all_proj, video_proj = self.encode_img(image, image_id, train_dataset, bs=self.bs) # XCLIP
        
        if self.prompt_list:
            prompt_embeds, atts_prompt = self.prompt_wrap(qform_all_proj, atts_qform_all_proj, self.prompt_list, image_id, video_proj) #(self, img_embeds, batch_names, atts_img, prompt_list):      
        
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
                         dtype=text_tokens.input_ids.dtype, # input_ids [bs, 13]
                         device=text_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) # [bs, 1, 4096]
        atts_bos = atts_qform_all_proj[:, :1]


        targets = text_tokens.input_ids.masked_fill( # [bs, 13]
            text_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = ( # [bs, 63]
            torch.ones([qform_all_proj.shape[0], 1 + prompt_embeds.shape[1]], 
                       dtype=torch.long).to(image.device).fill_(-100)  
        )
        targets = torch.cat([empty_targets, targets], dim=1) # [bs, 76]
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
