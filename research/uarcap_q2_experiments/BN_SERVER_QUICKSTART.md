# BN 服务器一键部署与分阶段运行指南（BN SERVER QUICKSTART — 最终离线交付版）

> 适用分支：`antigravity/uarcap-q2-final-offline-handoff`  
> 模型复用原则：**默认零新增模型下载**。现代视频生成器首选既有 **`Qwen/Qwen3.8-27B-FP8`**（`/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/models/Qwen3.8-27B-FP8`，66/66 shards，30.89 GB，备用既有 `Qwen/Qwen3.5-27B`），跨家族文本裁判首选既有 **`Mistral Small 3.2 24B`**（备用既有 `Gemma 3 27B`、`Llama 3.3 70B W4A16`、`gpt-oss-20b`、`Phi-4 14B`、`Llama 3.1 8B`），依托既有 **vLLM 0.29.0 / PyTorch 2.13.0+cu130 / CUDA 13.0** 环境运行；**取消默认下载 `Qwen2.5-VL-7B-Instruct`**；原冻结 **Video-LLaMA (`Transformers 4.28.0`)** 保持 100% 只读不动。

---

## 1. 拉取固定 GitHub 分支与 Commit

```bash
cd /home/bn
git clone https://github.com/Aksnes5/UARCap.git uarcap_q2_repo   # 若已存在：cd /home/bn/uarcap_q2_repo && git fetch origin
cd /home/bn/uarcap_q2_repo
git checkout antigravity/uarcap-q2-final-offline-handoff
git pull origin antigravity/uarcap-q2-final-offline-handoff
# 建议用固定 commit 锁死运行版本：git log -n 1 --oneline
cd /home/bn/uarcap_q2_repo/research/uarcap_q2_experiments
```

---

## 2. 自动部署及绑定已有资产（含既有 Qwen3.8-27B-FP8 与裁判模型）

```bash
bash scripts/bootstrap_bn.sh
conda activate uarcap_q2_cpu
```

检查自动生成的 `configs/bn_paths.local.yaml`：
- `historical_frozen_assets`：已预填 VideoUFO (`VideoUFO.csv`、`clips_full.tsv`、`videoufo_brief_text_fp16.npy`、`X-CLIP`、冻结检查点 SHA-256 与 `/home/bn/videoufo_research/gpu.lock`)；
- `existing_bn_modern_models`：确认既有 `existing_vllm_python`（或 `existing_vllm_conda_env`）、`primary_modern_vlm_dir`（`/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/models/Qwen3.8-27B-FP8`）与 `primary_judge_dir`（既有 `Mistral Small 3.2 24B` 目录）指向现场真实路径。

---

## 3. 一键准备 DREAM-1K（支持 BN 直连与 Windows 分卷中转）

1. 先执行 Pilot A0 非破坏性预检：
```bash
export HF_TOKEN="<接受 DREAM-1K gated 协议后的 HF Token，严禁写入文件或提交 Git>"
bash scripts/run_preflight.sh --verify-ckpt-sha --check-hf-network
```

2. 签发 Pilot A0/A1 授权收据：
```bash
cat > /home/bn/uarcap_q2_runtime/authorizations/pilot_a01_auth.json <<'EOF'
{
  "authorized_stages": ["PILOT_A0", "PILOT_A1"],
  "approved": true,
  "approval_id": "USER_APPROVED_PILOT_A01"
}
EOF
```

3. 一键自动化准备（自动串联：已有文件识别/受控下载或分卷导入 → 完整性校验 → 防路径穿越安全解压 → 目录规范化 → 元数据与视频对应校验 → 落盘 `dream1k_readiness.json`）：
- **路线 1（BN 直连下载）**：
```bash
python deployment/stage_assets.py prepare \
  --data-root /home/bn/uarcap_q2_runtime/data \
  --dataset dream1k \
  --allow-download \
  --auth-receipt /home/bn/uarcap_q2_runtime/authorizations/pilot_a01_auth.json
```
- **路线 2（Windows 下载并分卷中转至 BN）**：
  - 在 Windows 本地拆分：`python deployment/stage_assets.py split --src video.zip --out-dir ./dream1k_chunks --chunk-mb 512`
  - 上传 `dream1k_chunks/` 与 `metadata.json` 后在 BN 一键导入解压规范化：
```bash
python deployment/stage_assets.py prepare \
  --data-root /home/bn/uarcap_q2_runtime/data \
  --dataset dream1k \
  --transfer-manifest /home/bn/uarcap_q2_runtime/staging/video.zip.transfer_manifest.json
```
- 随时独立核验数据就绪状态：
```bash
python deployment/stage_assets.py check-ready --data-root /home/bn/uarcap_q2_runtime/data --dataset dream1k
```

---

## 4. 启动 Pilot A0 + A1（X-CLIP 独立子进程 + 冻结 Video-LLaMA 批量执行）

```bash
# 先 dry-run 核对配置（0 GPU）：
bash scripts/run_pilot_a01.sh --dry-run

# 正式运行 Pilot A0 + A1（X-CLIP 在既有冻结环境子进程运行并释放显存；每臂仅加载一次 7B Video-LLaMA）：
bash scripts/run_pilot_a01.sh
```

完成后审阅 `/home/bn/uarcap_q2_runtime/outputs/pilot_a1/pilot_a1_dream1k_v1/` 下的 `run_receipt.json`（确认 `status == "REAL_RUNTIME_PASS"`）及 `PILOT_A1_GO_NO_GO.md`。

---

## 5. 查看进度与安全断点续跑

```bash
bash scripts/status.sh
bash scripts/resume.sh pilot_a01   # 或 pilot_a2 / pilot_a3 / formal_b
```

续跑严格校验配对 `trace_sha256`、JSONL 完整性及 `run_config_fingerprint`（配置/权重漂移或截断文件自动拒绝），并基于全量记录重新计算质量指标。

---

## 6. 后续 Pilot A2 / A3 启动入口（复用既有 BN 模型，零新增下载）

### Pilot A2（跨家族文本裁判校准 — 首选既有 Mistral Small 3.2 24B）
1. 签发 `/home/bn/uarcap_q2_runtime/authorizations/pilot_a2_auth.json`（`authorized_stages: ["PILOT_A2"]`, `approved: true`, `approval_id`）。
2. 第 1 步自动评测与盲法打包（自动停在退出码 `10` = `HUMAN_VALIDATION_PENDING`）：
```bash
bash scripts/run_pilot_a2.sh \
  --judge-model-id mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --judge-family mistral \
  --judge-backend-mode vllm_subprocess \
  --judge-model-path /media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/models/Mistral-Small-3.2-24B
# 若现场已启动本地 vLLM OpenAI 服务，可改传：--judge-backend-mode openai_compat --judge-api-url http://127.0.0.1:8000/v1
```
3. 完成双人盲标与仲裁（见第 7 节）后，重入 A2（自动复用既有 `a2_ref_fact_eval.jsonl` 与盲法映射，不重跑裁判，按稳定 `claim_id` 严格对账）：
```bash
bash scripts/run_pilot_a2.sh \
  --judge-model-id mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --judge-family mistral \
  --judge-backend-mode vllm_subprocess \
  --judge-model-path /media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/models/Mistral-Small-3.2-24B \
  --human-annotations /home/bn/uarcap_q2_runtime/outputs/pilot_a2/adjudicated/adjudicated_e2.json
```

### Pilot A3（既有 Qwen3.8-27B-FP8 多模态视频推理 + VidHalluc 4 臂受控干预 + 人工核验闭环）
1. A2 取得 `REAL_RUNTIME_PASS` 且 `human_calibration_completed == true` 后，签发 `pilot_a3_auth.json` 并一键准备 VidHalluc：
```bash
python deployment/stage_assets.py prepare \
  --data-root /home/bn/uarcap_q2_runtime/data \
  --dataset vidhalluc \
  --allow-download \
  --auth-receipt /home/bn/uarcap_q2_runtime/authorizations/pilot_a3_auth.json \
  --prior-a2 /home/bn/uarcap_q2_runtime/outputs/pilot_a2/pilot_a2_eval_v1/calibration_receipt.json
```
2. 第 1 次运行 A3（Qwen3.8-27B-FP8 在 DREAM-1K 30 样本运行，关闭 thinking，输出 VidHalluc 候选并停于退出码 `10` 等待 5 项人工适用性审核）：
```bash
bash scripts/run_pilot_a3.sh \
  --modern-vlm-id Qwen/Qwen3.8-27B-FP8 \
  --modern-vlm-path /media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/models/Qwen3.8-27B-FP8 \
  --modern-vlm-backend vllm_subprocess \
  --judge-backend-mode vllm_subprocess
```
3. 传入人工批准的 E3 候选 `--approved-e3-candidates <approved.json>` 运行双模型四臂与盲法评测；系统导出 E3 与跨模型 DREAM-1K 匿名标注包并**再次停于退出码 `10`（`HUMAN_VALIDATION_PENDING`）**等待人工核验。
4. 完成 E3 人工核验与仲裁后，传入 `--e3-human-annotations <adjudicated_e3.json>`（及可选 `--dream1k-vlm-human-annotations` 与 `--a2-aligned-records`），自动复用既有生成/评测结果，完成独立 `Y_true`/`Y_conflict` 一致性、`Con vs Neu` 差分偏置与跨模型偏置审计，产出最终 `run_receipt.json`！

---

## 7. 人工核验工具使用方法（A2 与 A3 全流程，零手工改 JSON）

1. **启动本地网页标注器**（支持 Windows 轻薄本与 Ubuntu BN）：
```bash
# A2 标注示例（A1、A2 两位标注员分别使用各自盲法任务包）
python scripts/launch_annotation_ui.py \
  --bundle /home/bn/uarcap_q2_runtime/outputs/pilot_a2/pilot_a2_eval_v1/blind_annotation_package/blind_tasks_e2_A1.json \
  --save-dir /home/bn/uarcap_q2_runtime/annotation_saves \
  --port 8765

# A3 E3 双目标标注示例
python scripts/launch_annotation_ui.py \
  --bundle /home/bn/uarcap_q2_runtime/outputs/pilot_a3/pilot_a3_vlm_vidhalluc_v1/e3_blind_annotation_package/blind_tasks_e3_A1.json \
  --save-dir /home/bn/uarcap_q2_runtime/annotation_saves \
  --port 8766
```
网页内操作：静音播放视频 → （E2 先锁定 Phase-1 核心事件）→ 一键点击四分类标签 / 补录遗漏命题 / 拆分或合并命题 / 勾选 E3 Target A & B → 每次点击自动原子保存、随时续标 → 完成后点击顶部 **Export JSON** 导出个人标注文件。

2. **一键双人分歧提取与仲裁**：
```bash
# E2 仲裁（若存在分歧则返回退出码 10 并写出 e2_disagreements_to_resolve.json；填入 resolved_label 后带 --resolutions 重跑即可产出 adjudicated_e2.json）
python scripts/adjudicate_annotations.py \
  --task e2 \
  --ann1 /path/to/export_A1.json \
  --ann2 /path/to/export_A2.json \
  --blind-key /home/bn/uarcap_q2_runtime/outputs/pilot_a2/pilot_a2_eval_v1/blind_annotation_package/blind_key_e2_DO_NOT_OPEN.json \
  --out-dir /home/bn/uarcap_q2_runtime/outputs/pilot_a2/adjudicated

# E3 仲裁（产出 adjudicated_e3.json，直接传入 run_pilot_a3.sh --e3-human-annotations）
python scripts/adjudicate_annotations.py \
  --task e3 \
  --ann1 /path/to/e3_export_A1.json \
  --ann2 /path/to/e3_export_A2.json \
  --blind-key /home/bn/uarcap_q2_runtime/outputs/pilot_a3/pilot_a3_vlm_vidhalluc_v1/e3_blind_annotation_package/blind_key_e3_DO_NOT_OPEN.json \
  --out-dir /home/bn/uarcap_q2_runtime/outputs/pilot_a3/adjudicated
```

---

## 8. 正式实验（Formal Study B）授权与配置冻结入口

Formal B 默认硬阻断，**必须**在真实 Pilot A1、A2（含人工校准）、A3（含 E3 人工核验）全部取得 `REAL_RUNTIME_PASS` 后，由用户根据试点结果冻结参数并签发授权：

1. 根据 Pilot A2/A3 实测结果填写冻结配置文件（如 `configs/formal_b_frozen.yaml`），并计算 3 个文件的 SHA-256（`sha256sum`）：
   - `frozen_formal_config_sha256`
   - `asset_manifest_sha256`（`DATA_AND_MODEL_ASSET_MANIFEST.json`）
   - `pilot_exclusion_manifest_sha256`（`pilot_a_excluded_dream1k_vids.json`）
2. 写入 `/home/bn/uarcap_q2_runtime/authorizations/formal_b_auth.json`（含 `stage: "FORMAL_B"`, `approved: true`, `approval_id`, `frozen_pilot_params_confirmed: true`, `evaluator_version` 及上述 3 个 SHA-256）。
3. 启动正式实验（自动执行分层概率抽样、近重复/Pilot/同来源隔离、E2 + E3 + 可选真实 E3-Bridge、两阶段误差校正、MSR-VTT 历史桥接与全量统计）：
```bash
FORMAL_CFG=/home/bn/uarcap_q2_repo/research/uarcap_q2_experiments/configs/formal_b_frozen.yaml \
bash scripts/run_formal_b.sh \
  --approved-e3-candidates /path/to/formal_approved_e3.json \
  --pilot-vidhalluc-exclusion /home/bn/uarcap_q2_runtime/outputs/pilot_a3/pilot_a3_vlm_vidhalluc_v1/pilot_a3_excluded_vidhalluc.json \
  --formal-dream1k-n <冻结样本量> \
  --formal-e2-val-n <冻结验证集样本量> \
  --formal-vidhalluc-n <冻结E3样本量> \
  --modern-vlm-id Qwen/Qwen3.8-27B-FP8 \
  --modern-vlm-backend vllm_subprocess \
  --judge-backend-mode vllm_subprocess
```
