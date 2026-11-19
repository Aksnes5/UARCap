"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import torch
import math

def set_optimizer(model, init_lr, weight_decay):
    # TODO make optimizer class and configurations
    num_parameters = 0
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        print(n)
        if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)
        num_parameters += p.data.nelement()
    print("number of trainable parameters: %d" % num_parameters)
    optim_params = [
        {
            "params": p_wd,
            "weight_decay": float(weight_decay),
        },
        {"params": p_non_wd, "weight_decay": 0},
    ]
    beta2 = 0.999
    _optimizer = torch.optim.AdamW(
        optim_params,
        lr=float(init_lr),
        weight_decay=float(weight_decay),
        betas=(0.9, beta2),
    )
    return _optimizer

def new_set_optimizer(model, init_lr, muti, weight_decay):
    # TODO make optimizer class and configurations
    num_parameters = 0
    p_wd, p_non_wd = [], []
    qformer_txt_wd = []
    qformer_txt_non_wd = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        else:
            print(n)
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                if "Qformer_txt" in n or "query_tokens_txt" in n:
                    qformer_txt_non_wd.append(p)
                else:
                    p_non_wd.append(p)
            else:
                if "Qformer_txt" in n or "query_tokens_txt" in n:
                    qformer_txt_wd.append(p)
                else:
                    p_wd.append(p)
            num_parameters += p.data.nelement()
    print("number of trainable parameters: %d" % num_parameters)
    optim_params = [
        {
            "params": p_wd,
            "weight_decay": float(weight_decay),
        },
        {"params": p_non_wd, "weight_decay": 0},
        {"params": qformer_txt_wd, "lr": init_lr*muti, "weight_decay": float(weight_decay)},
        {"params": qformer_txt_non_wd, "lr": init_lr*muti, "weight_decay": 0}
    ]
    
    beta2 = 0.999
    _optimizer = torch.optim.AdamW(
        optim_params,
        lr=float(init_lr),
        weight_decay=float(weight_decay),
        betas=(0.9, beta2),
    )
    return _optimizer


class LinearWarmupStepLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        min_lr,
        init_lr,
        decay_rate=1,
        warmup_start_lr=-1,
        warmup_steps=0,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.min_lr = min_lr

        self.decay_rate = decay_rate

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_epoch, cur_step):
        if cur_epoch == 0:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            step_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
                decay_rate=self.decay_rate,
            )


def cc_film_set_optimizer(model, init_lr, weight_decay):
    """
    专为 CC-FiLM 方案设计的优化器分组
    - film_*/video_retri_proj: 低学习率 (5e-5)，防震荡
    - token_conf_net: 中学习率 (1e-4)，加速置信度校准
    - 其他可训练参数: 默认学习率 (1e-4)
    """
    p_wd, p_non_wd = [], []
    film_wd, film_non_wd = [], []
    conf_wd, conf_non_wd = [], []
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 按模块名称分类
        if "film_" in n or "video_retri_proj" in n:
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                film_non_wd.append(p)
            else:
                film_wd.append(p)
        elif "token_conf_net" in n:
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                conf_non_wd.append(p)
            else:
                conf_wd.append(p)
        else:
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
    
    optim_params = [
        {"params": p_wd, "lr": init_lr, "weight_decay": weight_decay},
        {"params": p_non_wd, "lr": init_lr, "weight_decay": 0},
        {"params": film_wd, "lr": init_lr * 0.5, "weight_decay": weight_decay},  # 5e-5
        {"params": film_non_wd, "lr": init_lr * 0.5, "weight_decay": 0},
        {"params": conf_wd, "lr": init_lr, "weight_decay": weight_decay * 0.4},  # 0.02
        {"params": conf_non_wd, "lr": init_lr, "weight_decay": 0},
        
        # fix_lr # cider 70.9
        # {"params": film_wd, "lr": init_lr * 0.2, "weight_decay": weight_decay},  # 2e-5
        # {"params": film_non_wd, "lr": init_lr * 0.2, "weight_decay": 0},
        # {"params": conf_wd, "lr": init_lr * 0.5, "weight_decay": weight_decay * 0.4},
        # {"params": conf_non_wd, "lr": init_lr * 0.5, "weight_decay": 0},
    ]
    
    return torch.optim.AdamW(optim_params, lr=init_lr, weight_decay=weight_decay, betas=(0.9, 0.999))

def lr_groups_set_optimizer(model, init_lr, weight_decay, lr_groups=None):
    """
    支持分组学习率的优化器配置函数
    
    Args:
        model: PyTorch 模型
        init_lr: 基础学习率 (默认组)
        weight_decay: 权重衰减系数
        lr_groups: 可选，字典格式 {lr_multiplier: [keyword_list]}
                   示例: {0.5: ["video_retri_", "film_"], 1.0: ["sparse_router"]}
    
    Returns:
        torch.optim.AdamW 优化器实例
    """
    num_parameters = 0
    
    # 初始化分组容器: {lr_multiplier: {"wd": [], "non_wd": []}}
    if lr_groups is None:
        # 默认行为：所有参数使用相同学习率
        lr_groups = {1.0: ["__default__"]}
    
    group_params = {mult: {"wd": [], "non_wd": []} for mult in lr_groups.keys()}
    default_group = {"wd": [], "non_wd": []}  # 未匹配任何关键词的参数归入默认组
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # 跳过冻结参数
        
        print(f"{n}: requires_grad={p.requires_grad}")
        
        # 判断是否需要 weight_decay
        no_wd = p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n
        
        # 按关键词匹配分组
        matched = False
        for mult, keywords in lr_groups.items():
            if any(kw in n for kw in keywords):
                if no_wd:
                    group_params[mult]["non_wd"].append(p)
                else:
                    group_params[mult]["wd"].append(p)
                matched = True
                break
        
        # 未匹配任何关键词的参数归入默认组
        if not matched:
            if no_wd:
                default_group["non_wd"].append(p)
            else:
                default_group["wd"].append(p)
        
        num_parameters += p.data.nelement()
    
    print("number of trainable parameters: %d" % num_parameters)
    
    # 构建 optimizer param_groups
    optim_params = []
    
    # 添加默认组（基础学习率）
    if default_group["wd"]:
        optim_params.append({
            "params": default_group["wd"],
            "lr": init_lr,
            "weight_decay": float(weight_decay)
        })
    if default_group["non_wd"]:
        optim_params.append({
            "params": default_group["non_wd"],
            "lr": init_lr,
            "weight_decay": 0
        })
    
    # 添加自定义分组
    for mult, params in group_params.items():
        group_lr = init_lr * mult
        if params["wd"]:
            optim_params.append({
                "params": params["wd"],
                "lr": group_lr,
                "weight_decay": float(weight_decay)
            })
        if params["non_wd"]:
            optim_params.append({
                "params": params["non_wd"],
                "lr": group_lr,
                "weight_decay": 0
            })
    
    beta2 = 0.999
    _optimizer = torch.optim.AdamW(
        optim_params,
        lr=float(init_lr),  # 全局默认值，实际以 param_groups 为准
        weight_decay=float(weight_decay),
        betas=(0.9, beta2),
    )
    
    # 打印分组信息（调试用）
    print("\n=== Optimizer Param Groups ===")
    for i, group in enumerate(_optimizer.param_groups):
        print(f"Group {i}: lr={group['lr']:.2e}, wd={group['weight_decay']:.4f}, "
              f"num_params={sum(p.numel() for p in group['params'])}")
    print("==============================\n")
    
    return _optimizer

class LinearWarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        iters_per_epoch,
        min_lr,
        init_lr,
        warmup_steps=0,
        warmup_start_lr=-1,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.iters_per_epoch = iters_per_epoch
        self.min_lr = min_lr

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_epoch, cur_step):
        total_cur_step = cur_epoch * self.iters_per_epoch + cur_step
        if total_cur_step < self.warmup_steps:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            cosine_lr_schedule(
                epoch=total_cur_step,
                optimizer=self.optimizer,
                max_epoch=self.max_epoch * self.iters_per_epoch,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * epoch / max_epoch)
    ) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
