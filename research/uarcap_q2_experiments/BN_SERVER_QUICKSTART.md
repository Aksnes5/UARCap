# BN 服务器恢复后一键操作指南（BN SERVER QUICKSTART — 真实执行链整改版）

> 适用分支：`antigravity/uarcap-q2-real-execution-integration`  
> 本指南所有命令均对应已修复的真实执行入口（零 `READY_FOR_BN_REAL_ADAPTER` 占位；严格执行 `StageRunStatus` 状态机与 `stage_unlock` 授权门控）。

---

## 第 1 步：拉取最新整改分支并进入工程目录

```bash
cd /home/bn
git clone https://github.com/Aksnes5/UARCap.git uarcap_q2_repo   # 若已存在：cd /home/bn/uarcap_q2_repo && git fetch origin
cd /home/bn/uarcap_q2_repo
git checkout antigravity/uarcap-q2-real-execution-integration
git pull origin antigravity/uarcap-q2-real-execution-integration
cd /home/bn/uarcap_q2_repo/research/uarcap_q2_experiments
```

## 第 2 步：一键初始化目录、历史资产配置与 CPU 环境

```bash
bash scripts/bootstrap_bn.sh
conda activate uarcap_q2_cpu
```

确认自动生成的 `configs/bn_paths.local.yaml` 中历史路径（已预填 `VideoUFO.csv`、`clips_full.tsv`、`videoufo_brief_text_fp16.npy`、`X-CLIP` 路径、冻结 Video-LLaMA 检查点与 `/home/bn/videoufo_research/gpu.lock`）与现场一致；**旧 Video-LLaMA 环境（`Transformers 4.28.0`）保持 100% 只读不动**。

## 第 3 步：执行 Pilot A0 非破坏性预检

```bash
export HF_TOKEN="<接受 DREAM-1K gated 协议后的 HF Token，严禁写入文件或提交 Git>"
bash scripts/run_preflight.sh --verify-ckpt-sha --check-hf-network
```

检查 `/home/bn/uarcap_q2_runtime/outputs/pilot_a0/preflight_receipt.json`，确认全部历史资产存在、检查点 SHA-256 匹配、磁盘剩余 $\ge 150\text{ GB}$、GPU 空闲且 `dream1k_hf_access.accessible == true`。

## 第 4 步：签发 Pilot A0/A1 授权收据并下载/导入 DREAM-1K（仅 3.59 GiB）

```bash
cat > /home/bn/uarcap_q2_runtime/authorizations/pilot_a01_auth.json <<'EOF'
{
  "authorized_stages": ["PILOT_A0", "PILOT_A1"],
  "approved": true,
  "approval_id": "USER_APPROVED_PILOT_A01"
}
EOF
```

- **路线 1（BN 直连，须带 `--auth-receipt` 通过 `stage_unlock` 校验）**：
```bash
python deployment/stage_assets.py download --data-root /home/bn/uarcap_q2_runtime/data --asset dream1k_metadata --confirm-download
python deployment/stage_assets.py download --data-root /home/bn/uarcap_q2_runtime/data --asset dream1k_videos --confirm-download \
  --auth-receipt /home/bn/uarcap_q2_runtime/authorizations/pilot_a01_auth.json
```
- **路线 2（Windows 分卷中转后在 BN 校验重组）**：
```bash
python deployment/stage_assets.py import \
  --manifest /home/bn/uarcap_q2_runtime/staging/video.zip.transfer_manifest.json \
  --dest /home/bn/uarcap_q2_runtime/data/dream1k/video/video.zip
```
解压 `video.zip` 至 `/home/bn/uarcap_q2_runtime/data/dream1k/video/` 后验证：
```bash
python deployment/stage_assets.py verify --data-root /home/bn/uarcap_q2_runtime/data --asset dream1k_videos
```

## 第 5 步：启动真实 Pilot A0 + A1（真实 VideoUFO X-CLIP + 冻结 Video-LLaMA 批量执行）

```bash
# 先 dry-run 核对配置（0 GPU）：
bash scripts/run_pilot_a01.sh --dry-run

# 正式运行 Pilot A0 + A1（每臂仅加载一次 7B 模型；共 30×3=90 次生成 + 真实前向 hook trace）：
bash scripts/run_pilot_a01.sh
```

监控与安全断点续跑：
```bash
bash scripts/status.sh
bash scripts/resume.sh pilot_a01
```

完成后审阅 `/home/bn/uarcap_q2_runtime/outputs/pilot_a1/pilot_a1_dream1k_v1/` 下的 `run_receipt.json`（确认 `status == "REAL_RUNTIME_PASS"`）及三份报告 `PILOT_A1_EXECUTION_REPORT.md`、`PILOT_A1_INTEGRITY_REPORT.md`、`PILOT_A1_GO_NO_GO.md`。

## 第 6 步：A1 通过后方可执行 Pilot A2（自动停于人工核验节点，退出码 10）

1. 在隔离环境 `configs/envs/env_text_evaluator.yaml` 部署 1 款跨家族文本裁判并签发 `pilot_a2_auth.json`；
2. 首次执行 `bash scripts/run_pilot_a2.sh --judge-model-path <local_judge_dir>`（或 `--judge-api-url`）：系统自动读取 A1 `predictions.jsonl`、运行三值代理评测、生成盲法标注包并安全停于 **`HUMAN_VALIDATION_PENDING`（退出码 10）**；
3. 两位标注员运行本地网页工具完成双盲静音标注与仲裁：
```bash
python scripts/launch_annotation_ui.py \
  --bundle /home/bn/uarcap_q2_runtime/outputs/pilot_a2/pilot_a2_eval_v1/blind_annotation_package/blind_tasks_e2_A1.json \
  --save-dir /home/bn/uarcap_q2_runtime/annotation_saves \
  --port 8765
```
4. 传入仲裁真值完成校准收据：`bash scripts/run_pilot_a2.sh --judge-model-path <local_judge_dir> --human-annotations <adjudicated.json>`（确认 `status == "REAL_RUNTIME_PASS"` 且 `human_calibration_completed == true`）。

## 第 7 步：A2 通过后方可执行 Pilot A3 与（经冻结授权后的）Formal B

- **Pilot A3**：构建隔离环境 `configs/envs/env_modern_vlm.yaml`、签发 `pilot_a3_auth.json`、带 `--auth-receipt` 与 `--prior-a2` 下载 VidHalluc `15.60 GiB`、运行 `bash scripts/run_pilot_a3.sh --modern-vlm-path <local_qwen25vl_dir> --judge-model-path <local_judge_dir>`（未提供人工审批候选时自动停在退出码 10 等待 5 项审核，审批后传入 `--approved-e3-candidates` 完成 A3）；
- **Formal B**：默认硬阻断，必须由用户在 Pilot A1/A2/A3 全部取得 `REAL_RUNTIME_PASS` 后冻结配置与全部 SHA-256 绑定并签发 `formal_b_auth.json` 才可运行 `bash scripts/run_formal_b.sh`。
