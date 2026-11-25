# BN 服务器一键部署与分阶段运行指南（BN SERVER QUICKSTART — CaReBench 最终离线交付版）

> **适用分支**：`antigravity/uarcap-q2-carebench-final-handoff`  
> **最终科研组合**：历史资产（MSR-VTT + MSVD + 百万级 VideoUFO） + **E2 主数据集：`CaReBench`**（`MCG-NJU/CaReBench`，`gated=False`，`MIT`，`3.65 GiB`，`913` 个 FineAction 父视频去重；`DREAM-1K` 保留为 `--e2-dataset dream1k` 备用回退） + **E3 受控干预：`VidHalluc (ACH)`**（`15.60 GiB`，已知 YouTube ID 层面无交集过滤）。  
> **Python 编排环境**：直接复用 BN 既有 `/home/bn/anaconda3/envs/videollama2/bin/python`（通过 `PYTHON_BIN` 指定，**不强制新建 Conda 环境，不访问 `conda.anaconda.org`**）。  
> **模型复用原则**：默认零新增模型下载。冻结历史 **Video-LLaMA (`Transformers 4.28.0`)** 100% 只读不动；现代视频生成器首选既有 **`Qwen/Qwen3.8-27B-FP8`**（备用 `Qwen/Qwen3.5-27B`）；跨家族文本裁判优先复用既有非 Qwen 模型（首选 `Mistral Small 3.2 24B`，备用 `Gemma 3 27B`、`Llama 3.3 70B W4A16`、`gpt-oss-20b`、`Phi-4 14B`、`Llama 3.1 8B`）；**取消默认下载 `Qwen2.5-VL`**。

---

## 1. 拉取最新 GitHub 分支

```bash
cd /home/bn
git clone https://github.com/Aksnes5/UARCap.git uarcap_q2_repo   # 若已存在：cd /home/bn/uarcap_q2_repo && git fetch origin
cd /home/bn/uarcap_q2_repo
git checkout antigravity/uarcap-q2-carebench-final-handoff
git pull origin antigravity/uarcap-q2-carebench-final-handoff
cd /home/bn/uarcap_q2_repo/research/uarcap_q2_experiments
```

---

## 2. 初始化目录与本地路径绑定（零 Conda 创建）

```bash
export PYTHON_BIN=/home/bn/anaconda3/envs/videollama2/bin/python
bash scripts/bootstrap_bn.sh
```

检查自动生成的 `configs/bn_paths.local.yaml`：
- `q2_writable_workspace.carebench_dir` = `/home/bn/uarcap_q2_runtime/data/carebench`（已自动建目录，备选 `dream1k_dir` 与 `vidhalluc_dir` 亦一并保留）；
- `historical_frozen_assets`：指向只读 VideoUFO、X-CLIP、冻结检查点 SHA-256 与 `/home/bn/videoufo_research/gpu.lock`；
- `existing_bn_modern_models`：确认既有 `Qwen3.8-27B-FP8` 与非 Qwen 裁判路径。

---

## 3. 获取并一键准备 CaReBench（免登录公开下载，支持直连 / 镜像 / 离线分卷）

CaReBench 官方地址：`https://huggingface.co/datasets/MCG-NJU/CaReBench`（`gated=False`，`MIT`，**无需 HF 登录或 Token**）：
- 元数据：`json/metadata.json`（~3.70 MB，1,000 片段 / 913 个 FineAction 父视频 ID）
- 视频压缩包：`videos/videos.zip`（**`3,918,994,506 bytes` = `3.65 GiB`**）

### 3.1 执行 Pilot A0 非破坏性预检并签发 Pilot A0/A1 授权收据
```bash
export PYTHON_BIN=/home/bn/anaconda3/envs/videollama2/bin/python
bash scripts/run_preflight.sh --e2-dataset carebench --verify-ckpt-sha

cat > /home/bn/uarcap_q2_runtime/authorizations/pilot_a01_auth.json <<'EOF'
{
  "authorized_stages": ["PILOT_A0", "PILOT_A1"],
  "approved": true,
  "approval_id": "USER_APPROVED_PILOT_A01_CAREBENCH"
}
EOF
```

### 3.2 路线 A（推荐，BN 服务器直接免登录公开下载、安全解压、规范化与校验）
```bash
# 直连 Hugging Face（若国内直连慢，可追加 --hf-endpoint https://hf-mirror.com 无需 Token！）
/home/bn/anaconda3/envs/videollama2/bin/python deployment/stage_assets.py prepare \
  --data-root /home/bn/uarcap_q2_runtime/data \
  --dataset carebench \
  --allow-download \
  --hf-endpoint https://hf-mirror.com \
  --auth-receipt /home/bn/uarcap_q2_runtime/authorizations/pilot_a01_auth.json
```

### 3.3 路线 B（离线中转：若 BN 无外网，先在 Windows 本地分卷再传 BN 导入）
1. 在 Windows 轻薄本执行（`--dry-run` 仅预览命令；确认下载时再加 `--confirm-download`）：
```powershell
python scripts/download_carebench_stage.py --stage-dir D:\UARCap_LocalData\carebench_staging --hf-endpoint https://hf-mirror.com --confirm-download
```
2. 将 `json/metadata.json` 与 `bn_transfer_chunks/` 上传至 BN `/home/bn/uarcap_q2_runtime/staging/carebench/` 后导入：
```bash
cp /home/bn/uarcap_q2_runtime/staging/carebench/metadata.json /home/bn/uarcap_q2_runtime/data/carebench/json/metadata.json
/home/bn/anaconda3/envs/videollama2/bin/python deployment/stage_assets.py prepare \
  --data-root /home/bn/uarcap_q2_runtime/data \
  --dataset carebench \
  --transfer-manifest /home/bn/uarcap_q2_runtime/staging/carebench/videos.zip.transfer_manifest.json
```

### 3.4 独立验证 CaReBench 就绪状态
```bash
/home/bn/anaconda3/envs/videollama2/bin/python deployment/stage_assets.py check-ready \
  --data-root /home/bn/uarcap_q2_runtime/data --dataset carebench
```
确认输出 `/home/bn/uarcap_q2_runtime/data/carebench_readiness.json` 中 `"ready": true`、`"n_metadata_records": 1000`、`"n_unique_fineaction_parents": 913`、`"n_missing_or_empty_videos": 0`。

---

## 4. 启动 Pilot A0 + A1（30 个 CaReBench 独立父视频 × 3 臂 A0/A1/A7 = 90 次真实生成）

```bash
export PYTHON_BIN=/home/bn/anaconda3/envs/videollama2/bin/python

# 1) 零 GPU 干跑核对路径与配置（确认 e2_dataset == carebench）：
bash scripts/run_pilot_a01.sh --dry-run

# 2) 正式执行 Pilot A0 + A1（父视频去重 + 按类别比例分层抽样 30 视频 + X-CLIP 子进程检索 + 冻结 Video-LLaMA 批量 90 次生成）：
bash scripts/run_pilot_a01.sh
```
- 输出目录：`/home/bn/uarcap_q2_runtime/outputs/pilot_a1/pilot_a1_carebench_v1/`
- 关键产物：`run_receipt.json`（核验 `status == "REAL_RUNTIME_PASS"`）、`predictions.jsonl`（90 条真实生成）、`pilot_a_excluded_carebench_vids.json`（同时永久锁定 30 条 `query_id` 与 30 个 FineAction `parent_video_id`）、`pilot_a1_carebench_sampling_audit.json` 与 `PILOT_A1_GO_NO_GO.md`。
- 如未来需回退至备用数据集 DREAM-1K，仅需传 `E2_DATASET=dream1k bash scripts/run_pilot_a01.sh`。

---

## 5. 后续 Pilot A2 / A3 / Formal B（默认贯穿 CaReBench + VidHalluc ACH）

- **Pilot A2**（读取 CaReBench A1 输出 → 跨家族非 Qwen 三值参考事实评价 → 盲法人工标注包 → 双人仲裁与校准）：
  ```bash
  export PYTHON_BIN=/home/bn/anaconda3/envs/videollama2/bin/python
  bash scripts/run_pilot_a2.sh \
    --judge-model-id mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
    --judge-family mistral \
    --judge-backend-mode vllm_subprocess \
    --judge-model-path /media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/models/Mistral-Small-3.2-24B
  # 完成双人盲标与仲裁后追加 --human-annotations <adjudicated_e2.json> 重跑（自动复用 auto-eval 与盲法映射）
  ```
- **Pilot A3**（`Qwen3.8-27B-FP8` 在相同 30 条 CaReBench 上运行 + `VidHalluc ACH` 已知 YouTube ID 无交集过滤与双模型 4 臂 `Base/Sup/Con/Neu` 受控干预）：
  ```bash
  export PYTHON_BIN=/home/bn/anaconda3/envs/videollama2/bin/python
  bash scripts/run_pilot_a3.sh \
    --modern-vlm-id Qwen/Qwen3.8-27B-FP8 \
    --modern-vlm-path /media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/models/Qwen3.8-27B-FP8 \
    --modern-vlm-backend vllm_subprocess \
    --judge-backend-mode vllm_subprocess
  ```
- **Formal Study B**（默认硬阻断；Pilot A1/A2/A3 全部 `REAL_RUNTIME_PASS` 后，依据实测父视频可用池、近重复筛查、解码结果、功效分析与人工校准精度冻结 `600–750` 规划范围内的正式样本量与 `50–80` 独立验证集，再授权运行 `bash scripts/run_formal_b.sh`）：
  ```bash
  FORMAL_CFG=/home/bn/uarcap_q2_repo/research/uarcap_q2_experiments/configs/formal_b_frozen.yaml \
  PYTHON_BIN=/home/bn/anaconda3/envs/videollama2/bin/python \
  bash scripts/run_formal_b.sh \
    --approved-e3-candidates /path/to/formal_approved_e3.json \
    --pilot-vidhalluc-exclusion /home/bn/uarcap_q2_runtime/outputs/pilot_a3/pilot_a3_carebench_vlm_vidhalluc_v1/pilot_a3_excluded_vidhalluc.json \
    --formal-e2-n <冻结正式样本量> \
    --formal-e2-val-n <冻结独立验证集样本量> \
    --formal-vidhalluc-n <冻结E3样本量>
  ```
- **进度与安全续跑**：`bash scripts/status.sh` 与 `bash scripts/resume.sh <pilot_a01|pilot_a2|pilot_a3|formal_b>`。
- **人工标注与双人仲裁**：`python scripts/launch_annotation_ui.py --bundle <blind_tasks.json> --save-dir <dir>` 与 `python scripts/adjudicate_annotations.py --task <e2|e3> ...`。
