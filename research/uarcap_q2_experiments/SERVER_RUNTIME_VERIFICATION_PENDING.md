# 待 BN 服务器恢复后现场验证清单（SERVER RUNTIME VERIFICATION PENDING — 整改更新版）

本轮已离线补齐真实 VideoUFO X-CLIP 检索器、冻结 Video-LLaMA 18 列 TSV/检查点槽位/前向 hook 批量适配器、现代 VLM 实际帧与 processor 观测 trace 链路、统一 7 态授权状态机及 A0/A1/A2/A3/Formal B/E1 全阶段分发代码。恢复 BN 服务器访问后，**不再需要重写或补齐核心适配代码**，仅需对以下依赖真实 BN 硬件、索引、权重与人工的事项完成现场实测验证：

| 类别 | 待 BN 现场实测事项 | 所属阶段 | 执行命令 / 工具 |
|---|---|---|---|
| **主机/磁盘/锁实时探测** | BN 在线连通性、NVMe `/` 实际可用空间（$\ge 150\text{ GB}$）、`/home/bn/videoufo_research/gpu.lock` 与 `nvidia-smi` 实时空闲态 | Pilot A0 | `bash scripts/run_preflight.sh --verify-ckpt-sha` |
| **历史资产实时路径核验** | `VideoUFO.csv`、`clips_full.tsv`、`videoufo_brief_text_fp16.npy`、`k600_16_16.pth`、`vicuna-7b-v1.3`、冻结检查点（`a8b5af79...`）及 `videollama_faithful_v2` 目录实盘可读 | Pilot A0 | `bash scripts/run_preflight.sh --verify-ckpt-sha` |
| **真实百万级 VideoUFO 检索** | `VideoUFOXCLIPRetriever` 在真实 `k600_16_16.pth` 与百万级 `fp16.npy` 索引上的实际显存、检索耗时、30 条 DREAM-1K 近重复高相似尾部占比与代表 `.pt` 覆盖率 | Pilot A1 | `bash scripts/run_pilot_a01.sh` |
| **冻结 Video-LLaMA 真实批量推理** | 18 列 DREAM-1K TSV 与绑定检查点槽位在只读旧 venv（`Transformers 4.28.0`）上的真实 3 臂批量运行、前向 hook trace、吞吐与峰值显存 | Pilot A1 | `bash scripts/run_pilot_a01.sh` |
| **隔离 GPU 环境构建与跨家族裁判实测** | `env_text_evaluator.yaml` 在 BN CUDA 驱动 `580.178.04` 下的安装、候选 24B/27B/70B-AWQ 裁判 48 GB 显存/JSON 合规率，以及 30 视频双人静音人工标注后的 `A7` 差分偏置校准裁决 | Pilot A2 | `bash scripts/run_pilot_a2.sh` + `scripts/launch_annotation_ui.py` |
| **现代 VLM RTX A6000 Timed Smoke 与 VidHalluc 15.60 GiB 筛选** | `env_modern_vlm.yaml` 安装、`Qwen2.5-VL-7B-Instruct` 16 帧 48 GB 显存与 8–18 词长度稳定性、VidHalluc `ACH_videos.zip` 实盘校验与 30 候选片人工 5 项适用性通过率 $r_{\text{pass}}$ | Pilot A3 | `bash scripts/run_pilot_a3.sh` |
| **试点后科学参数冻结与正式授权** | 依据 A2/A3 真实方差与通过率锁定正式样本量、唯一裁判、唯一现代 VLM、`E3-Bridge` 启停与两阶段校正定位，并由用户签发 `formal_b_auth.json` | Formal B 前置 | 人工科学裁决 |
