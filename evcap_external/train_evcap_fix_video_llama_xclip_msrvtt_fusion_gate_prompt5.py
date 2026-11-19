import os
import torch
# from torch.utils.data import DataLoader, DistributedSampler
import random
import sys
import argparse
import numpy as np
import utils
from optims import LinearWarmupCosineLRScheduler, set_optimizer

# from dataset.msrvtt_dataset import MSRVTTDataset
# from dataset.msrvtt_dataloader.msrvtt_dataloader import make_data_loader
from dataset.msrvtt.vl_dataloader import make_data_loader
from dataset.msrvtt.data_utils.miscellaneous import str_to_bool

from models.evcap_fix_video_llama_xclip_msrvtt_smallcap_quality_fusion_gate_prompt5 import EVCap 
from common.dist_utils import (
    get_rank,
    init_distributed_mode,
    get_world_size,
    dist_init, # by lijian
)


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

def train_epochs(model, args):
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
    if args.load_video_feature:
         train_dataset = None
    
    total_steps = len(train_dataloader)
    steps_per_epoch = total_steps // args.epochs
    
    model.train()
    optimizer = set_optimizer(model, init_lr=1e-4, weight_decay=0.05)
    
    # 修改调度器：总步数=总迭代次数
    scheduler = LinearWarmupCosineLRScheduler(
        optimizer=optimizer,
        max_epoch=1,  # 整个训练视为1个epoch
        iters_per_epoch=total_steps,  # 总步数=dataloader长度
        min_lr=8e-5,
        init_lr=1e-4,
        decay_rate=None,
        warmup_start_lr=1e-6,
        warmup_steps=5000,
    )
    
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    use_amp = scaler is not None
    print('use_amp', use_amp)

    print(f">>> Total training steps: {total_steps}")
    sys.stdout.flush()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.update(loss=1000.0)
    metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    print_freq = 50
    header = 'Training Progress:'
    
    # 直接遍历整个dataloader（包含所有epoch）
    for step_idx, samples in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)):
        samples['image'] = samples['image'].to(device)
        
        # 按全局步数更新学习率
        scheduler.step(cur_epoch=0, cur_step=step_idx)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = model(samples, train_dataset)["loss"]
            
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # 按原epoch间隔保存检查点
        if (step_idx + 1) % steps_per_epoch == 0:
            epoch_num = (step_idx + 1) // steps_per_epoch
            checkpoint_path = os.path.join(
                output_dir, 
                ckpt_name.replace('.pt', f'_epoch{epoch_num}.pt')
            )
            save_checkpoint(model, optimizer, epoch_num, checkpoint_path)
            print(f"Saved checkpoint at step {step_idx+1} (epoch {epoch_num})")
    
    # 训练结束保存最终模型
    # final_path = os.path.join(output_dir, ckpt_name.replace('.pt', '_final.pt'))
    # save_checkpoint(model, optimizer, args.epochs, final_path)
    # print(f"Training completed. Final model saved to {final_path}")
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
    if args.load_video_feature:
         train_dataset = None
         
    model.train()
    optimizer = set_optimizer(model, init_lr=1e-4, weight_decay=0.05)
    scheduler = LinearWarmupCosineLRScheduler(optimizer= optimizer,
                max_epoch=epochs,
                iters_per_epoch=len(train_dataloader),
                min_lr=8e-5,
                init_lr=1e-4,
                decay_rate=None,
                warmup_start_lr=1e-6,
                warmup_steps=5000,)
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
        # if os.path.exists(output_dir_model):
        #     output_dir_model = output_dir_model.replace('.pt', '_1.pt')
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
    
    # prompt
    parser.add_argument("--prompt_path", default='prompts/prompt_evcap_msrvtt.txt', type=str)
    
    parser.add_argument("--freeze_qformer_txt", type=str_to_bool, default=True)
    parser.add_argument("--freeze_vit", type=str_to_bool, default=True)
    parser.add_argument("--freeze_llm", type=str_to_bool, default=True)
    parser.add_argument('--load_video_feature', type=str_to_bool, default=True)
    parser.add_argument("--share_query_tokens", type=str_to_bool, default=True)
    parser.add_argument('--vit_precision', default="fp16")
    parser.add_argument('--llm_model', default='vicuna-7b-v1.3')
    
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
            vit_model="eva_clip_g", 
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision=args.vit_precision,
            freeze_vit=args.freeze_vit,
            freeze_qformer=args.freeze_qformer,
            # freeze_qformer_txt=args.freeze_qformer_txt, # train txtQFormer
            num_query_token=32,
            num_query_token_txt=args.num_query_token_txt,
            topn = args.topn,
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

    if args.epochs == 1:
        train(model, args)
    else:
        train_epochs(model, args)

if __name__ == '__main__':
    main()
