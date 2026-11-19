import os
import torch
# from torch.utils.data import DataLoader, DistributedSampler
import random
import sys
import argparse
import numpy as np
import utils
from optims import LinearWarmupCosineLRScheduler, set_optimizer, cc_film_set_optimizer
from einops import rearrange

# from dataset.msrvtt_dataset import MSRVTTDataset
# from dataset.msrvtt_dataloader.msrvtt_dataloader import make_data_loader
from dataset.msrvtt.vl_dataloader import make_data_loader
from dataset.msrvtt.data_utils.miscellaneous import str_to_bool

from models.evcap_fix_video_llama_xclip_msrvtt_smallcap_quality_fusion_gate_prompt8_cc_film5 import EVCap 
from common.dist_utils import (
    get_rank,
    init_distributed_mode,
    get_world_size,
    dist_init, # by lijian
)

# SCST
from scst_trainer import CiderReward, SCSTTrainer, get_scst_ground_truths

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model,optimizer, cur_epoch, output_dir):
    """
    Save the checkpoint at the current epoch.
    """
    model_no_ddp = model
    param_grad_dic = {
        k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
    }
    state_dict = model_no_ddp.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            del state_dict[k]
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": cur_epoch,
    }
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, output_dir))
    torch.save(save_obj, output_dir)

def train_scst(model, args, ckpt_path=None):
    """
    SCST 训练阶段：用 CIDEr 奖励微调模型
    通常在 CE 训练收敛后使用
    """
    device = torch.device(f"cuda:{get_rank()}")
    output_dir = args.out_dir
    ckpt_name = args.ckpt_name
    
    # 加载 CE 预训练权重
    if ckpt_path and os.path.exists(ckpt_path):
        print(f">>> Loading CE checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
    
    if args.distributed:
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[get_rank()])
    else:
        model.to(device)
    
    train_dataloader, train_dataset = make_data_loader(args, args.train_yaml, None, is_distributed=args.distributed, is_train=True)
    
    model.train()
    
    # SCST 使用更小的学习率
    optimizer = cc_film_set_optimizer(model, init_lr=5e-6, weight_decay=0.05)
    
    total_steps = len(train_dataloader) * args.scst_epochs
    warmup_steps = min(3000, max(1000, total_steps // 20))
    
    scheduler = LinearWarmupCosineLRScheduler(
        optimizer=optimizer,
        max_epoch=args.scst_epochs,
        iters_per_epoch=len(train_dataloader),
        min_lr=1e-6,
        init_lr=5e-6,
        decay_rate=None,
        warmup_start_lr=1e-6,
        warmup_steps=warmup_steps,
    )
    
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    use_amp = scaler is not None
    
    # 预计算语料库级别的 IDF 统计（解决 batch=1 时 CIDEr 退化为 0 的问题）
    print(">>> Precomputing corpus-level IDF for CIDEr...")
    corpus_gts = []
    for img_key in train_dataset.image_keys:
        caps = train_dataset.get_captions_by_key(img_key)
        corpus_gts.append(caps)
    
    cider_reward = CiderReward(corpus_gts=corpus_gts)
    del corpus_gts
    
    scst_trainer = SCSTTrainer(
        model=model,
        tokenizer=model.module.llama_tokenizer if args.distributed else model.llama_tokenizer,
        cider_reward=cider_reward,
        temperature=args.scst_temperature,
        scst_weight=args.scst_weight,
        use_baseline=True
    )
    
    print(
        f">>> SCST Training: temperature={args.scst_temperature}, "
        f"scst_weight={args.scst_weight}, max_new_tokens={args.scst_max_new_tokens}"
    )
    print(f">>> Total SCST steps: {len(train_dataloader) * args.scst_epochs}")
    sys.stdout.flush()
    
    for epoch in range(args.scst_epochs):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        print(f">>> SCST epoch {epoch}")
        sys.stdout.flush()
        
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=200, fmt='{value:.6f}'))
        metric_logger.add_meter('loss', utils.SmoothedValue(window_size=200, fmt='{value:.6f}'))
        metric_logger.add_meter('cider_s', utils.SmoothedValue(window_size=200, fmt='{value:.2f}'))
        metric_logger.add_meter('cider_g', utils.SmoothedValue(window_size=200, fmt='{value:.2f}'))
        metric_logger.update(loss=1000.0)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(cider_s=0.0)
        metric_logger.update(cider_g=0.0)
        
        print_freq = 10
        header = f'SCST Epoch: [{epoch}]'
        
        for idx, samples in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)):
            samples['image'] = samples['image'].to(device)
            scheduler.step(cur_epoch=epoch, cur_step=idx)
            
            # 1. 先计算 CE loss（保持语言质量）
            with torch.cuda.amp.autocast(enabled=use_amp):
                ce_output = model(samples, train_dataset)
                ce_loss = ce_output["loss"]
                
                # 获取模型输入用于 SCST
                image = samples["image"]
                image_id = samples['image_id']
                image = rearrange(image, 'b f c h w -> (b f) c h w')
                
                # 获取 qform_all_proj 和 video_proj
                qform_all_proj, atts_qform_all_proj, video_proj = model.module.encode_img(
                    image, image_id, train_dataset, bs=args.bs
                ) if args.distributed else model.encode_img(
                    image, image_id, train_dataset, bs=args.bs
                )
                
                # 构造 prompt
                prompt_list = model.module.prompt_list if args.distributed else model.prompt_list
                if prompt_list:
                    prompt_embeds, atts_prompt = model.module.prompt_wrap(
                        qform_all_proj, atts_qform_all_proj, prompt_list, image_id, video_proj
                    ) if args.distributed else model.prompt_wrap(
                        qform_all_proj, atts_qform_all_proj, prompt_list, image_id, video_proj
                    )
                
                # 构造 inputs_embeds 和 input_ids
                llama_tokenizer = model.module.llama_tokenizer if args.distributed else model.llama_tokenizer
                llama_model = model.module.llama_model if args.distributed else model.llama_model
                
                bos = torch.ones([qform_all_proj.shape[0], 1],
                                 dtype=torch.long, device=image.device) * llama_tokenizer.bos_token_id
                bos_embeds = llama_model.model.embed_tokens(bos)
                atts_bos = atts_qform_all_proj[:, :1]
                
                inputs_embeds = torch.cat([bos_embeds, prompt_embeds], dim=1)
                attention_mask = torch.cat([atts_bos, atts_prompt], dim=1)
                
                # 构造 input_ids（用于 KV cache 生成）
                # prompt 部分用 0 填充（第一步用 inputs_embeds，后续用 input_ids）
                prompt_ids = torch.zeros(qform_all_proj.shape[0], prompt_embeds.shape[1], 
                                         dtype=torch.long, device=image.device)
                input_ids = torch.cat([bos, prompt_ids], dim=1)
                
                # 使用该视频全部 GT captions 计算 CIDEr（与评估一致）
                ground_truths = get_scst_ground_truths(train_dataset, image_id)

                scst_loss, scst_metrics = scst_trainer.compute_scst_loss(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    ground_truths=ground_truths,
                    max_new_tokens=args.scst_max_new_tokens,
                    ce_loss=ce_loss,
                )
            
            # 3. 反向传播
            if use_amp:
                scaler.scale(scst_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                scst_loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            
            metric_logger.update(loss=scst_loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(cider_s=scst_metrics['cider_sampled'])
            metric_logger.update(cider_g=scst_metrics['cider_greedy'])
        
        metric_logger.synchronize_between_processes()
        print("SCST Averaged stats:", metric_logger.global_avg())
        
        # 保存 checkpoint
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # output_dir_model = os.path.join(output_dir, ckpt_name.replace('.pt', f"_epoch2.pt"))
        output_dir_model = os.path.join(output_dir, ckpt_name)
        if os.path.exists(output_dir_model):
            output_dir_model = output_dir_model.replace('.pt', '_1.pt')
        save_checkpoint(model, optimizer, epoch, output_dir_model)
    
    return model


def train(model, args):
    device = torch.device(f"cuda:{get_rank()}")
    batch_size = args.bs
    epochs = args.epochs
    accum_grad_iters = 1
    output_dir = args.out_dir
    ckpt_name = args.ckpt_name
    if args.distributed:
        # sampler = DistributedSampler(
        #     dataset,
        #     shuffle=True,
        #     num_replicas=get_world_size(),
        #     rank=get_rank(),
        # )
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[get_rank()])
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[get_rank()], find_unused_parameters=True) 
    else: # 调试时使用
        sampler = None
        model.to(device) # by lijian

    # train_dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, sampler=sampler, shuffle=False, drop_last=True)
    # train_dataloader = make_data_loader(args, dataset, is_distributed=args.distributed, is_train=True)    
    train_dataloader, train_dataset = make_data_loader(args, args.train_yaml, None, is_distributed=args.distributed, is_train=True)    
    # if args.load_video_feature:
    #      train_dataset = None
         
    model.train()
    # optimizer = set_optimizer(model, init_lr=1e-4, weight_decay=0.05)
    # cc_film_group_learn
    optimizer = cc_film_set_optimizer(model, init_lr=args.init_lr, weight_decay=0.05)
    
    scheduler = LinearWarmupCosineLRScheduler(optimizer= optimizer,
                max_epoch=epochs,
                iters_per_epoch=len(train_dataloader),
                min_lr=args.min_lr,
                init_lr=args.init_lr,
                decay_rate=None,
                warmup_start_lr=args.warmup_start_lr,
                # warmup_steps=5000,
                warmup_steps=args.warmup_steps, # cc_film_group_learn
                )
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    use_amp = scaler is not None
    print('use_amp', use_amp)

    for epoch in range(epochs):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.update(loss=1000.0)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        print_freq = 50
        header = 'Train Epoch: [{}]'.format(epoch)
        for idx, samples in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)): # image size: [B, 3, 224, 224]
            samples['image'] = samples['image'].to(device)
            
            scheduler.step(cur_epoch=epoch, cur_step=idx)    
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = model(samples, train_dataset)["loss"]
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (idx + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr = optimizer.param_groups[0]["lr"])
            
            # # cpu温度、内存以及gpu内存使用情况
            # stats = utils.get_system_stats()
            # print(f"\n[System Status] Batch {idx}")
            # print(f"CPU Temp: {stats.get('cpu_temp', 'N/A')}°C")
            # print(f"Memory Usage: {stats['mem_used']} ({stats['mem_percent']})")
            # print(f"stats: {stats['gpu']}")
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger.global_avg())
 
        # by lijian
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        if epochs != 1:
            output_dir_model = os.path.join(output_dir, ckpt_name.replace('.pt', f"_epoch{epoch}.pt"))
        else:
            output_dir_model = os.path.join(output_dir, ckpt_name)
        if os.path.exists(output_dir_model):
            output_dir_model = output_dir_model.replace('.pt', '_1.pt')
        save_checkpoint(model, optimizer, epoch, output_dir_model)
    return model


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print('Starts ...')
    print(" # PID :", os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--bs', type=int, default=6)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--device', default = 'cuda', help = 'gpu for training')
    parser.add_argument('--distributed', type=str_to_bool, default = True)
    parser.add_argument('--amp', type=str_to_bool, default = True)
    parser.add_argument('--dist_url', default = "env://")
    parser.add_argument('--world_size', type = int, default = 1)
    parser.add_argument('--num_query_token_txt', type = int, default = 8)
    parser.add_argument('--topn', type = int, default = 5)
    parser.add_argument('--disable_random_seed', action = 'store_true', default = False, help = 'set random seed for reproducing')
    parser.add_argument('--random_seed', type = int, default = 42, help = 'set random seed for reproducing')
    
    # by lijian
    # parser.add_argument('--datasets_dir', default='datasets/COCO2014/')
    parser.add_argument('--ckpt_name', default='000.pt')
    # msrvtt
    parser.add_argument("--train_yaml", default='coco_caption/train.yaml', type=str, help="Yaml file with all data for training.")
    parser.add_argument("--num_workers", default=4, type=int, help="Workers in dataloader.")
    parser.add_argument('--max_num_frames', type=int, default=32)
    parser.add_argument('--img_res', type=int, default=224)
    parser.add_argument("--shuffle", type=str_to_bool, default=True)
    parser.add_argument("--on_memory", type=str_to_bool, nargs='?', const=True, default=False,
                            help="Option to load labels/caption to memory before training.")
    parser.add_argument("--img_feature_dim", default=512, type=int,
                            help="Update model config if given.The Image Feature Dimension.")
    parser.add_argument("--debug_speed", type=str_to_bool, default=False)
    parser.add_argument('--retriever', type=str_to_bool, nargs='?', const=True, default=True, help = '是否需要检索模块和Customized Q-Former')
    # video-llama
    parser.add_argument('--num_video_query_token', type = int, default = 32)
    parser.add_argument("--freeze_video_qformer", type=str_to_bool, default=True)
    parser.add_argument("--freeze_qformer", type=str_to_bool, default=True)
    # XCLIP
    parser.add_argument("--ext_path", default='ext_data/msrvtt_traindata_one_caption_xclip_retrieve_5_captions.json', type=str)
    parser.add_argument("--sim_scores_path", default='ext_data/msrvtt_train_allGT_avg_frames_xclip_retrieve_20_captions_no_duplicate_sim_scores.json', type=str)
    # video_retri
    parser.add_argument("--video_ext_path", default='ext_data/msrvtt_video_similarity_top20.json', type=str)
    parser.add_argument("--video_topn", type=int, default=4)
    parser.add_argument("--verb_noun_topn", type=int, default=3)    
    parser.add_argument("--verb_noun_ext_path", default='ext_data/retrieved_video_verbs_nouns_per_video.json', type=str)
    
    # prompt
    parser.add_argument("--prompt_path", default='prompts/prompt_evcap_msrvtt_detail_smallcap_prompt8.txt', type=str)
    
    parser.add_argument("--freeze_qformer_txt", type=str_to_bool, default=True)
    parser.add_argument("--freeze_vit", type=str_to_bool, default=True)
    parser.add_argument("--freeze_llm", type=str_to_bool, default=True)
    parser.add_argument('--load_video_feature', type=str_to_bool, default=True)
    parser.add_argument("--share_query_tokens", type=str_to_bool, default=True)
    parser.add_argument('--vit_precision', default="fp16")
    parser.add_argument('--llm_model', default='vicuna-7b-v1.3')
    
    # SCST 参数
    parser.add_argument('--use_scst', type=str_to_bool, default=False, help='是否使用 SCST 训练')
    parser.add_argument('--scst_epochs', type=int, default=1, help='SCST 训练轮数')
    parser.add_argument('--scst_temperature', type=float, default=0.8, help='采样温度')
    parser.add_argument('--scst_weight', type=float, default=0.5, help='SCST 损失权重 (0=纯CE, 1=纯RL)')
    parser.add_argument('--scst_ckpt', type=str, default='', help='SCST 加载的 CE checkpoint 路径')
    parser.add_argument('--scst_max_new_tokens', type=int, default=32, help='SCST 生成最大 token 数（与 beam_search 默认一致）')
    
    # scheduler
    parser.add_argument('--min_lr', type=float, default=8e-5, help='最小 learning rate')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='初始 learning rate')
    parser.add_argument('--warmup_start_lr', type=float, default=1e-6, help='warmup start learning rate')
    parser.add_argument('--warmup_steps', type=int, default=8000, help='warmup steps')
    
    args = parser.parse_args()
    dist_init(args) # by lijian
    print(f'args: {vars(args)}')
    if not args.disable_random_seed:
        set_seed(args.random_seed)
    init_distributed_mode(args)
    print(f'args: {vars(args)}')
    
    # by lijian  
    # data_root = 'data/coco/coco2014'
    # data_root = args.datasets_dir
    # dataset = MSRVTTDataset(args, args.train_yaml, is_train=True)

    # ViT、Qformer预训练文件路径/home/cn/.cache/torch/hub/checkpoints/
    model = EVCap(
            ext_path = args.ext_path,
            sim_scores_path=args.sim_scores_path,            
            video_ext_path=args.video_ext_path,
            verb_noun_ext_path=args.verb_noun_ext_path,
            vit_model="eva_clip_g", 
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            img_size=args.img_res,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision=args.vit_precision,
            freeze_vit=args.freeze_vit,
            freeze_qformer=args.freeze_qformer,
            # freeze_qformer_txt=args.freeze_qformer_txt, # train txtQFormer
            num_query_token=32,
            num_query_token_txt=args.num_query_token_txt,
            topn=args.topn,
            video_topn=args.video_topn,
            verb_noun_topn=args.verb_noun_topn,
            llama_model=args.llm_model, #"vicuna-13b-v1.3" 内存不够，跑不了
            prompt_path=args.prompt_path,
            prompt_template='###Human: {} ###Assistant: ',
            max_txt_len=128,
            end_sym='\n',
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
            bs=args.bs,
            frames=args.max_num_frames,
            retriever=args.retriever,
            num_video_query_token=args.num_video_query_token, # video-llama
            freeze_video_qformer=args.freeze_video_qformer, # video-llama
            freeze_qformer_txt=args.freeze_qformer_txt, # video-llama
            load_video_feature=args.load_video_feature,
            share_query_tokens=args.share_query_tokens,
            freeze_llm=args.freeze_llm,
            )

    if args.use_scst:
        # SCST 训练模式
        ckpt_path = args.scst_ckpt if args.scst_ckpt else None
        train_scst(model, args, ckpt_path=ckpt_path)
    elif args.epochs == 1:
        train(model, args)


if __name__ == '__main__':
    main()
