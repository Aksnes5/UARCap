#!/usr/bin/env bash
# Formal Study B Entrypoint — HARD-BLOCKED BY DEFAULT!
# Forwards ALL real prior receipts & frozen bindings into BOTH outer authorization_gate.py AND inner run_stage_cli.py!
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATHS_CFG="${PATHS_CFG:-${EXP_ROOT}/configs/bn_paths.local.yaml}"
FORMAL_CFG="${FORMAL_CFG:-${EXP_ROOT}/configs/formal_b_candidate.yaml}"
ASSET_MANIFEST="${ASSET_MANIFEST:-${EXP_ROOT}/DATA_AND_MODEL_ASSET_MANIFEST.json}"
Q2_ROOT="${Q2_ROOT:-/home/bn/uarcap_q2_runtime}"
AUTH_RECEIPT="${AUTH_RECEIPT:-${Q2_ROOT}/authorizations/formal_b_auth.json}"
PRIOR_A1="${PRIOR_A1:-${Q2_ROOT}/outputs/pilot_a1/pilot_a1_dream1k_v1/run_receipt.json}"
PRIOR_A2="${PRIOR_A2:-${Q2_ROOT}/outputs/pilot_a2/pilot_a2_eval_v1/calibration_receipt.json}"
PRIOR_A3="${PRIOR_A3:-${Q2_ROOT}/outputs/pilot_a3/pilot_a3_vlm_vidhalluc_v1/run_receipt.json}"
PILOT_EXCL="${PILOT_EXCL:-${Q2_ROOT}/outputs/pilot_a1/pilot_a1_dream1k_v1/pilot_a_excluded_dream1k_vids.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${RUN_ID:-formal_b_v1}"
OUT_DIR="${Q2_ROOT}/outputs/formal_b/${RUN_ID}"

# 1. Outer hard authorization gate (rejects MOCK_PASS, unconfirmed params, or hash mismatch)
"${PYTHON_BIN}" "${EXP_ROOT}/deployment/authorization_gate.py" \
  --stage FORMAL_B \
  --auth-receipt "${AUTH_RECEIPT}" \
  --prior-a1 "${PRIOR_A1}" \
  --prior-a2 "${PRIOR_A2}" \
  --prior-a3 "${PRIOR_A3}" \
  --formal-config "${FORMAL_CFG}" \
  --asset-manifest "${ASSET_MANIFEST}" \
  --pilot-exclusion "${PILOT_EXCL}"

# 2. Inner CLI receives IDENTICAL full prerequisite & frozen binding arguments!
"${PYTHON_BIN}" "${EXP_ROOT}/deployment/run_stage_cli.py" \
  --stage FORMAL_B \
  --paths-config "${PATHS_CFG}" \
  --stage-config "${FORMAL_CFG}" \
  --out-dir "${OUT_DIR}" \
  --run-id "${RUN_ID}" \
  --auth-receipt "${AUTH_RECEIPT}" \
  --prior-a1 "${PRIOR_A1}" \
  --prior-a2 "${PRIOR_A2}" \
  --prior-a3 "${PRIOR_A3}" \
  --asset-manifest "${ASSET_MANIFEST}" \
  --pilot-exclusion "${PILOT_EXCL}" \
  "$@"

echo "[run_formal_b] Formal B receipt written: ${OUT_DIR}/formal_b_receipt.json"
