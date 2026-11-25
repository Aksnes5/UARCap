# 待 BN 服务器恢复后现场验证清单（SERVER RUNTIME VERIFICATION PENDING）

以下事项在 Windows 轻薄本离线阶段**物理上无法验证**，已全部如实列出，必须在恢复 BN Ubuntu RTX A6000 服务器访问后按 `BN_SERVER_QUICKSTART.md` 逐项现场实测：

| 类别 | 待 BN 现场验证事项 | 计划验证阶段 | 对应脚本 / 工具 |
|---|---|---|---|
| **主机与资源状态** | BN 当前在线连通性、NVMe `/` 实时剩余磁盘（须 $\ge 150\text{ GB}$）、`/home/bn/videoufo_research/gpu.lock` 实时占用与 `nvidia-smi` 空闲态 | Pilot A0 | `bash scripts/run_preflight.sh` |
| **历史资产实测校验** | `/media/bn/...` 只读挂载路径、冻结 Video-LLaMA 检查点全量 SHA-256（`a8b5af79...`）、VideoUFO 百万级 X-CLIP 索引与 `videollama_faithful_v2` 目录实时可读性 | Pilot A0 | `bash scripts/run_preflight.sh --verify-ckpt-sha` |
| **数据下载与合规通路** | 用户 HF 账号对 `omni-research/DREAM-1K` gated 协议的实时认证、BN 直连 Hugging Face 速度（或启用 Route 2 本地分卷中转） | Pilot A0 / A1 | `deployment/stage_assets.py` |
| **真实百万级 VideoUFO 检索** | 30 条 DREAM-1K 真实视频经 X-CLIP 特征提取后在完整百万级 VideoUFO 索引上的真实 top-20 检索、D 重排与近重复高相似尾部占比 | Pilot A1 | `bash scripts/run_pilot_a01.sh` |
| **冻结 Video-LLaMA 真实 GPU 推理** | 打包的 DREAM-1K 16 帧 `img.tsv/yaml` 在只读旧 venv（`Transformers 4.28.0`）上的真实 90 次推理吞吐、冷启动时间与显存峰值 | Pilot A1 | `bash scripts/run_pilot_a01.sh` |
| **隔离新 Conda 环境构建** | `configs/envs/env_text_evaluator.yaml` 与 `configs/envs/env_modern_vlm.yaml` 在 BN CUDA 驱动 `580.178.04` 下的实际安装与 PyTorch CUDA 兼容性 | Pilot A2 / A3 | `conda env create -f ...` |
| **跨家族文本裁判 RTX A6000 实测** | 候选 24B/27B-AWQ 或 70B-AWQ 裁判在单卡 48 GB VRAM 上的实际显存占用、JSON 合规率、吞吐及 30 视频双人校准后的 `A7` 差分偏置裁决 | Pilot A2 | `bash scripts/run_pilot_a2.sh` |
| **现代 VLM RTX A6000 Timed Smoke** | `Qwen2.5-VL-7B-Instruct`（16 帧、256–384 visual tokens/frame、8–18 词短字幕提示）在 RTX A6000 48GB 上的真实显存、生成词长稳定性与跨模型裁判捕获率差 | Pilot A3 | `bash scripts/run_pilot_a3.sh` |
| **VidHalluc `ACH_videos.zip`（15.60 GiB）真实筛选** | 真实下载/校验 15.60 GiB 压缩包、精确排除 VideoUFO YouTube ID 重叠后的可用源数量，以及 30 候选片人工 5 项适用性实测通过率 $r_{\text{pass}}$ | Pilot A3 | `bash scripts/run_pilot_a3.sh` |
