# BN 服务器恢复后一键操作指南（BN SERVER QUICKSTART）

> 本指南覆盖从恢复 BN 服务器访问到完成 **Pilot A0 + A1** 并导出人工标注包的全部核对命令。  
> **安全红线**：绝不升级旧 Video-LLaMA 环境（`/media/bn/.../repro_evcap/venv`），绝不杀死其他项目 GPU 进程，绝不向 GitHub 提交 HF Token、原始视频或模型权重。

---

## 第 1 步：进入工作目录并拉取最新离线工程分支

```bash
cd /home/bn
git clone https://github.com/Aksnes5/UARCap.git uarcap_q2_repo   # 若目录已存在则 cd uarcap_q2_repo && git fetch origin
cd /home/bn/uarcap_q2_repo
git checkout antigravity/uarcap-q2-complete-offline-implementation
git pull origin antigravity/uarcap-q2-complete-offline-implementation
cd /home/bn/uarcap_q2_repo/research/uarcap_q2_experiments
```

## 第 2 步：一键初始化环境、目录与历史资产配置

```bash
bash scripts/bootstrap_bn.sh
```

- 自动在 `/home/bn/uarcap_q2_runtime`（NVMe 可写分区）创建全部数据与输出目录；
- 自动生成 `configs/bn_paths.local.yaml`（已预填全部历史回执路径；仅当服务器路径发生变动时编辑此唯一文件）；
- 自动创建隔离的 CPU 分析环境 `uarcap_q2_cpu`（旧 Video-LLaMA 环境 100% 保持只读不动）。

## 第 3 步：执行非破坏性 CPU/路径/GPU 锁预检（Pilot A0）

```bash
conda activate uarcap_q2_cpu
export HF_TOKEN="<your_huggingface_token_after_accepting_DREAM1K_agreement>"   # 严禁写入任何文件或提交 Git
bash scripts/run_preflight.sh --verify-ckpt-sha --check-hf-network
```

查看输出与 `/home/bn/uarcap_q2_runtime/outputs/pilot_a0/preflight_receipt.json`，确认：
- `historical_assets` 全部 `exists: true` 且 `canonical_checkpoint` SHA-256 匹配；
- `disk.ok: true`（剩余空间 $\ge 150\text{ GB}$）；
- `gpu_non_destructive_probe.active_compute_apps` 为空且 `shared_gpu_lock.exists: true`；
- `dream1k_hf_access.accessible: true`（若受阻，先在浏览器打开 `https://huggingface.co/datasets/omni-research/DREAM-1K` 点击同意协议，或改用下方路线 2 离线中转）。

## 第 4 步：准备 DREAM-1K 数据（仅 3.59 GiB；暂不下载 VidHalluc 15.60 GiB）

- **路线 1（BN 直连下载，优先）**：
```bash
python deployment/stage_assets.py download --data-root /home/bn/uarcap_q2_runtime/data --asset dream1k_metadata --confirm-download
python deployment/stage_assets.py download --data-root /home/bn/uarcap_q2_runtime/data --asset dream1k_videos --confirm-download
```

- **路线 2（若 BN 无法连 HF：Windows 本地下载分卷后上传至 BN 导入）**：
```bash
# 在 Windows 本地执行分卷：
# python deployment/stage_assets.py split --src <local_video.zip> --out-dir D:\transfer_chunks --chunk-mb 512
# 将 D:\transfer_chunks 传至 BN /home/bn/uarcap_q2_runtime/staging/ 后在 BN 执行校验重组：
python deployment/stage_assets.py import \
  --manifest /home/bn/uarcap_q2_runtime/staging/video.zip.transfer_manifest.json \
  --dest /home/bn/uarcap_q2_runtime/data/dream1k/video/video.zip
```

验证完整性：
```bash
python deployment/stage_assets.py verify --data-root /home/bn/uarcap_q2_runtime/data --asset dream1k_videos
```

## 第 5 步：写入 Pilot A0/A1 授权收据并启动 Pilot A0 + A1

```bash
cat > /home/bn/uarcap_q2_runtime/authorizations/pilot_a01_auth.json <<'EOF'
{
  "authorized_stages": ["PILOT_A0", "PILOT_A1"],
  "approved": true,
  "approval_id": "USER_APPROVED_PILOT_A01"
}
EOF

# 先跑 dry-run 核对命令与路径（0 GPU）：
bash scripts/run_pilot_a01.sh --dry-run

# 确认无误后正式启动 Pilot A0 + A1（30 视频 × 3 臂 = 90 生成）：
bash scripts/run_pilot_a01.sh
```

## 第 6 步：查看实时状态与安全断点恢复

```bash
# 随时非破坏性查看进度、GPU 锁与输出行数：
bash scripts/status.sh

# 若意外中断，安全跳过已完成项续跑：
bash scripts/resume.sh pilot_a01
```

## 第 7 步：停在 A2/A3 之前，审阅 Pilot A1 报告并导出人工标注包

Pilot A1 结束后会自动停止于 A2 之前（未写入 `pilot_a2_auth.json` 时 `run_pilot_a2.sh` 会自动拦截）。先审阅自动生成的三份报告：
- `/home/bn/uarcap_q2_runtime/outputs/pilot_a1/pilot_a1_dream1k_v1/PILOT_A1_EXECUTION_REPORT.md`
- `/home/bn/uarcap_q2_runtime/outputs/pilot_a1/pilot_a1_dream1k_v1/PILOT_A1_INTEGRITY_REPORT.md`
- `/home/bn/uarcap_q2_runtime/outputs/pilot_a1/pilot_a1_dream1k_v1/PILOT_A1_GO_NO_GO.md`

仅当 `PILOT_A1_GO_NO_GO.md` 为 `PASS` 后，再决定是否创建隔离环境 `env_text_evaluator.yaml` 与 `pilot_a2_auth.json` 并启动 `bash scripts/run_pilot_a2.sh`。

### 启动本地网页人工核验工具（Windows 或 Ubuntu 通用）

```bash
python scripts/launch_annotation_ui.py \
  --bundle /path/to/blind_tasks_e2_A1.json \
  --save-dir /path/to/annotation_saves \
  --port 8765
```

浏览器自动打开 `http://127.0.0.1:8765/`：先看静音视频填写 Phase 1 核心事件，再点击解锁 Phase 2 匿名字幕，逐条一键点击四类视频真值；每次点击原子化自动保存，随时退出续标，结束后点击右上角导出 JSONL/CSV。

## 第 8 步：仅导出小型可公开审计报告并提交 GitHub

仅将 `PILOT_A1_*.md`、`run_receipt.json` 等小型非受限文本报告复制回仓库对应记录目录并推送（**严禁提交视频、权重、Token 或 VidHalluc 原始 JSON**）。
