#!/usr/bin/env bash
# Pilot A2 automatic stage (real cross-family judge + blind annotation package + diagnostics).
#
# Ends at PILOT_A2_HUMAN_VALIDATION_PENDING. Does NOT rerun A1 and does NOT download data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
Q2_ROOT="${Q2_ROOT:-/home/bn/uarcap_q2_runtime}"
A1_DIR="${A1_DIR:-${Q2_ROOT}/outputs/pilot_a1/pilot_a1_carebench_v1}"
OUT_DIR="${OUT_DIR:-${Q2_ROOT}/outputs/pilot_a2/pilot_a2_carebench_eval_v1}"
AUTH_RECEIPT="${AUTH_RECEIPT:-${Q2_ROOT}/authorizations/pilot_a2_auth.json}"
PATHS_CFG="${PATHS_CFG:-${EXP_ROOT}/configs/bn_paths.local.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/home/bn/anaconda3/envs/videollama2/bin/python}"
VLLM_PY="${VLLM_PY:-/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/.venv_rag_vllm/bin/python}"

# Existing BN judge (priority-2 Gemma 3 27B): Mistral Small 3.2 24B FP8 cannot be loaded by
# the installed vLLM 0.29 + transformers 5.17 (removed Pixtral symbols), and new downloads are
# prohibited. See PILOT_A2_RUNBOOK.md for the full compatibility note.
JUDGE_MODEL_ID="${JUDGE_MODEL_ID:-RedHatAI/gemma-3-27b-it-FP8-dynamic}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/models_campaign/M04_gemma3_27b}"
JUDGE_FAMILY="${JUDGE_FAMILY:-gemma}"

METADATA="${Q2_ROOT}/data/carebench/json/metadata.json"
VIDEO_ROOT="${Q2_ROOT}/data/carebench/videos"

mkdir -p "${OUT_DIR}"

# 1. Pin A1 artifacts + freeze core-event candidates BEFORE any judge call.
PYTHONPATH="${EXP_ROOT}" "${PYTHON_BIN}" "${EXP_ROOT}/scripts/freeze_pilot_a2_inputs.py" \
  --a1-dir "${A1_DIR}" \
  --metadata "${METADATA}" \
  --video-root "${VIDEO_ROOT}" \
  --out-dir "${OUT_DIR}"

# 2. Real batched cross-family fact evaluation (single model load; GPU lock acquired inside).
PYTHONPATH="${EXP_ROOT}" VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "${VLLM_PY}" -u "${EXP_ROOT}/scripts/run_pilot_a2_auto.py" \
  --a1-dir "${A1_DIR}" \
  --out-dir "${OUT_DIR}" \
  --paths-config "${PATHS_CFG}" \
  --auth-receipt "${AUTH_RECEIPT}" \
  --prior-a1 "${A1_DIR}/run_receipt.json" \
  --judge-model-id "${JUDGE_MODEL_ID}" \
  --judge-model-path "${JUDGE_MODEL_PATH}" \
  --judge-family "${JUDGE_FAMILY}" \
  --max-model-len 4096 \
  --max-new-tokens 768 \
  --gpu-memory-utilization 0.85 \
  --chunk-size 16 \
  --max-retries 2

# 3. Preliminary uncalibrated diagnostics + human workload estimate.
PYTHONPATH="${EXP_ROOT}" "${PYTHON_BIN}" "${EXP_ROOT}/scripts/pilot_a2_diagnostics.py" \
  --a2-dir "${OUT_DIR}" --a1-dir "${A1_DIR}"

echo "[run_pilot_a2_auto] DONE: PILOT_A2_HUMAN_VALIDATION_PENDING"
echo "[run_pilot_a2_auto] Blind package: ${OUT_DIR}/blind_annotation_package"
echo "[run_pilot_a2_auto] Diagnostics:  ${OUT_DIR}/PILOT_A2_AUTO_DIAGNOSTICS.md"
