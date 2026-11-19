import os
import sys
import torch
import torch.nn as nn
import argparse
from collections import OrderedDict

# 从训练脚本中导入模型类
from models.evcap_fix_video_llama_xclip_msrvtt_smallcap_quality_fusion_gate_prompt8_cc_film3 import EVCap

# 尝试导入 thop（用于 FLOPs 统计）
try:
    from thop import profile, clever_format
except ImportError:
    profile = None
    print("未安装 thop，请运行: pip install thop")
    print("FLOPs 计算将被跳过。")

def count_parameters(model, trainable_only=False):
    """统计模型参数量（总数或可训练数）"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def print_parameter_summary(model):
    """打印参数统计，包含顶层参数（如 query_tokens）"""
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)

    print("\n========== 参数统计 ==========")
    print(f"模型总参数量: {total / 1e9:.3f} B ({total / 1e6:.2f} M)")
    print(f"可训练参数量: {trainable / 1e6:.2f} M ({trainable / total * 100:.2f}%)")
    print("-" * 50)

    # 1. 各子模块（只统计可训练参数）
    print("各子模块（可训练参数）:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if params > 0:
            print(f"  {name:30s}: {params / 1e6:8.2f} M")
        else:
            print(f"  {name:30s}: (全部冻结)")

    # 2. 顶层可训练参数（直接属于 model 的 Parameter，不在任何子模块内）
    top_level_params = {}
    for name, param in model.named_parameters():
        if '.' not in name and param.requires_grad:
            top_level_params[name] = param.numel()
    if top_level_params:
        print("\n顶层可训练参数（如 query_tokens 等）:")
        for name, numel in top_level_params.items():
            print(f"  {name:30s}: {numel / 1e3:8.2f} K ({numel})")

def compute_flops(model, args):
    """分别计算每个子模块的 FLOPs，统一以 GFLOPs 显示"""
    if profile is None:
        print("thop 未安装，跳过 FLOPs 计算。")
        return

    model.eval().cpu().float()

    def profile_module(module, inputs, name, verbose=True):
        """安全地计算单个模块的 FLOPs，输出单位 G"""
        try:
            flops_val, params_val = profile(module, inputs=inputs, verbose=False)
            flops_g = flops_val / 1e9            # 转换为 G
            # 根据数值大小决定显示 G 还是 M
            if flops_g < 0.01:
                flops_str = f"{flops_g*1000:.3f} M"
            else:
                flops_str = f"{flops_g:.3f} G"
            # 参数仍用 clever_format 自动单位
            _, params_str = clever_format([flops_val, params_val], "%.3f")
            if verbose:
                print(f"  [✓] {name:30s} FLOPs: {flops_str}FLOPs, Params: {params_str}")
            return flops_g
        except Exception as e:
            print(f"  [✗] {name:30s} 计算失败 ({e}), 使用 0 FLOPs 代替")
            return 0

    B = args.bs
    N = args.num_query_token
    F = args.max_num_frames
    hidden = model.Qformer.config.hidden_size           # 768
    vision_width = model.visual_encoder.num_features   # 1408

    print("\n========== 模块级 FLOPs 统计 (单位: GFLOPs) ==========")

    # --- 1. ViT ---
    vit_input = torch.randn(B*F, 3, args.img_res, args.img_res).cpu().float()
    class VitWrapper(nn.Module):
        def __init__(self, enc, ln):
            super().__init__()
            self.enc = enc
            self.ln = ln
        def forward(self, x):
            return self.ln(self.enc(x))
    vit = VitWrapper(model.visual_encoder, model.ln_vision)
    profile_module(vit, (vit_input,), "Visual Encoder (ViT)")

    # --- 2. Image Q-Former (修复 vision_width) ---
    q_former_input = (
        model.query_tokens.expand(B*F, -1, -1).cpu(),
        torch.randn(B*F, 257, vision_width).cpu().float(),   # 正确维度
        torch.ones(B*F, 257).long().cpu()
    )
    class QformerWrapper(nn.Module):
        def __init__(self, qformer):
            super().__init__()
            self.qformer = qformer
        def forward(self, query, encoder_hidden, encoder_mask):
            return self.qformer.bert(
                query_embeds=query,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=encoder_mask,
                return_dict=True
            ).last_hidden_state
    img_qformer = QformerWrapper(model.Qformer)
    profile_module(img_qformer, q_former_input, "Image Q-Former")

    # --- 3. Video Q-Former ---
    video_qformer_input = (
        model.video_query_tokens.expand(B, -1, -1).cpu(),
        torch.randn(B, F*N, hidden).cpu().float(),
        torch.ones(B, F*N).long().cpu()
    )
    class VideoQformerWrapper(nn.Module):
        def __init__(self, vqformer):
            super().__init__()
            self.vqformer = vqformer
        def forward(self, query, encoder_hidden, encoder_mask):
            return self.vqformer.bert(
                query_embeds=query,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=encoder_mask,
                return_dict=True
            ).last_hidden_state
    video_qformer = VideoQformerWrapper(model.video_Qformer)
    profile_module(video_qformer, video_qformer_input, "Video Q-Former")

    # --- 4. LLAMA Projection ---
    input_embeds = torch.randn(B, N, hidden).cpu().float()
    profile_module(model.llama_proj, (input_embeds,), "LLAMA Projection")

    # --- 5. Cross-Attention ---
    cross_attn_input = (
        torch.randn(B, N, hidden).cpu().float(),
        torch.randn(B, N, hidden).cpu().float(),
        torch.randn(B, N, hidden).cpu().float()
    )
    profile_module(model.attn, cross_attn_input, "Cross-Attention (attn)")

    # --- 6. Gates & Transform ---
    gate_input = torch.randn(B, N, hidden*2).cpu().float()
    profile_module(model.update_gate, (gate_input,), "Update Gate")
    profile_module(model.reset_gate, (gate_input,), "Reset Gate")
    profile_module(model.new_transform, (gate_input,), "New Transform")

    # --- 7. Fusion LLAMA proj ---
    fusion_hidden = torch.randn(B, N, hidden).cpu().float()
    profile_module(model.fusion_llama_proj, (fusion_hidden,), "Fusion LLAMA Proj")

    # --- 8. CC-FiLM 模块 ---
    profile_module(model.video_retri_proj, (fusion_hidden,), "Video Retri Proj")
    profile_module(model.film_gamma_net, (fusion_hidden,), "Film Gamma Net")
    profile_module(model.film_beta_net, (fusion_hidden,), "Film Beta Net")
    profile_module(model.token_conf_net, (fusion_hidden,), "Token Conf Net")

    # --- 9. LLAMA 包装器适配 thop ---
    class LlamaWrapper(nn.Module):
        def __init__(self, llama):
            super().__init__()
            self.llama = llama
        def forward(self, input_ids, attention_mask):
            return self.llama(input_ids=input_ids, attention_mask=attention_mask).logits

    llama_wrapper = LlamaWrapper(model.llama_model)
    llama_input_ids = torch.randint(0, 32000, (B, args.max_txt_len)).long().cpu()
    llama_attn_mask = torch.ones_like(llama_input_ids)
    try:
        flops_val, params_val = profile(llama_wrapper, inputs=(llama_input_ids, llama_attn_mask), verbose=False)
        flops_g = flops_val / 1e9
        if flops_g < 0.01:
            flops_str = f"{flops_g*1000:.3f} M"
        else:
            flops_str = f"{flops_g:.3f} G"
        _, params_str = clever_format([flops_val, params_val], "%.3f")
        print(f"  [✓] {'LLAMA (7B)':30s} FLOPs: {flops_str}FLOPs, Params: {params_str}")
    except Exception as e:
        print(f"  [✗] LLAMA FLOPs 计算失败 ({e})，可单独计算")

    print("\n注：以上为各模块单次前向 FLOPs，总 FLOPs 可将各 G 值直接相加。生成时 LLM 部分需乘以生成 token 数。")

# 自定义 Tee 类，同时输出到多个流
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def main():
    parser = argparse.ArgumentParser(description="统计 EVCap 模型的参数和 FLOPs，并保存到文件")
    # ================== 修改默认参数以匹配您的训练命令 ==================
    parser.add_argument('--ext_path', 
                        default='ext_data/rerank_sentence_by_sentence_as_inference_rouge_quality_no_duplicate_with_video_ids.json',
                        type=str)
    parser.add_argument('--sim_scores_path', 
                        default='ext_data/msrvtt_train_allGT_avg_frames_xclip_retrieve_20_captions_no_duplicate_sim_scores.json',
                        type=str)
    parser.add_argument('--video_ext_path', 
                        default='ext_data/msrvtt_video_similarity_top20.json',
                        type=str)
    parser.add_argument('--verb_noun_ext_path', 
                        default='ext_data/retrieved_video_verbs_nouns_per_video.json',
                        type=str)
    parser.add_argument('--prompt_path', 
                        default='prompts/prompt_evcap_msrvtt_detail_smallcap_prompt8.txt',
                        type=str)
    parser.add_argument('--vit_precision', default='fp16')
    parser.add_argument('--llm_model', default='vicuna-7b-v1.3')
    parser.add_argument('--num_query_token_txt', type=int, default=8)
    parser.add_argument('--topn', type=int, default=3)
    parser.add_argument('--video_topn', type=int, default=3)
    parser.add_argument('--verb_noun_topn', type=int, default=3)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--max_num_frames', type=int, default=16)
    parser.add_argument('--img_res', type=int, default=224)
    parser.add_argument('--retriever', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--freeze_vit', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--freeze_qformer', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--freeze_video_qformer', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--freeze_qformer_txt', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--freeze_llm', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--load_video_feature', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--share_query_tokens', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--num_video_query_token', type=int, default=32)
    parser.add_argument('--num_query_token', type=int, default=32)
    parser.add_argument('--vit_model', default='eva_clip_g')
    parser.add_argument('--q_former_model', 
                        default='https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--drop_path_rate', type=float, default=0)
    parser.add_argument('--use_grad_checkpoint', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--low_resource', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--device_8bit', type=int, default=0)
    parser.add_argument('--prompt_template', default='###Human: {} ###Assistant: ')
    parser.add_argument('--max_txt_len', type=int, default=128)
    parser.add_argument('--end_sym', default='\n')
    # 输出文件参数
    parser.add_argument('--output', default='model_stats.txt', help='统计结果输出文件路径')
    parser.add_argument('--skip_flops', action='store_true', help='跳过 FLOPs 计算（若模型过大耗时）')

    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 打开输出文件，并重定向 stdout 到 Tee（同时打印到屏幕和文件）
    f_out = open(args.output, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, f_out)

    try:
        print("========== 开始构建模型 ==========")
        model = EVCap(
            ext_path=args.ext_path,
            sim_scores_path=args.sim_scores_path,
            video_ext_path=args.video_ext_path,
            verb_noun_ext_path=args.verb_noun_ext_path,
            vit_model=args.vit_model,
            q_former_model=args.q_former_model,
            img_size=args.img_size,
            drop_path_rate=args.drop_path_rate,
            use_grad_checkpoint=args.use_grad_checkpoint,
            vit_precision=args.vit_precision,
            freeze_vit=args.freeze_vit,
            freeze_qformer=args.freeze_qformer,
            num_query_token=32,
            num_query_token_txt=args.num_query_token_txt,
            topn=args.topn,
            video_topn=args.video_topn,
            verb_noun_topn=args.verb_noun_topn,
            llama_model=args.llm_model,
            prompt_path=args.prompt_path,
            prompt_template=args.prompt_template,
            max_txt_len=args.max_txt_len,
            end_sym=args.end_sym,
            low_resource=args.low_resource,
            device_8bit=args.device_8bit,
            bs=args.bs,
            frames=args.max_num_frames,
            retriever=args.retriever,
            num_video_query_token=args.num_video_query_token,
            freeze_video_qformer=args.freeze_video_qformer,
            freeze_qformer_txt=args.freeze_qformer_txt,
            load_video_feature=args.load_video_feature,
            share_query_tokens=args.share_query_tokens,
            freeze_llm=args.freeze_llm,
            stats_only=True,
        )

        # 1. 参数统计
        print_parameter_summary(model)

        # 2. FLOPs 统计（除非跳过）
        if not args.skip_flops:
            compute_flops(model, args)
        else:
            print("\n已跳过 FLOPs 计算（--skip_flops 指定）。")

        print("\n统计完成，结果已保存至:", args.output)
    except Exception as e:
        print(f"程序执行出错: {e}")
        raise
    finally:
        # 恢复 stdout 并关闭文件
        sys.stdout = sys.stdout.files[0] if hasattr(sys.stdout, 'files') else sys.__stdout__
        f_out.close()

if __name__ == '__main__':
    main()