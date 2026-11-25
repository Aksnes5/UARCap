#!/usr/bin/env bash
# Formal Study B Entrypoint — HARD-BLOCKED BY DEFAULT!
# Requires signed FORMAL_B authorization receipt + PASS receipts from PILOT_A1, PILOT_A2, and PILOT_A3.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATHS_CFG="${PATHS_CFG:-${EXP_ROOT}/configs/bn_paths.local.yaml}"
Q2_ROOT="${Q2_ROOT:-/home/bn/uarcap_q2_runtime}"
AUTH_RECEIPT="${AUTH_RECEIPT:-${Q2_ROOT}/authorizations/formal_b_auth.json}"
PRIOR_A1="${PRIOR_A1:-${Q2_ROOT}/outputs/pilot_a1/pilot_a1_dream1k_v1/run_receipt.json}"
PRIOR_A2="${PRIOR_A2:-${Q2_ROOT}/outputs/pilot_a2/pilot_a2_eval_v1/calibration_receipt.json}"
PRIOR_A3="${PRIOR_A3:-${Q2_ROOT}/outputs/pilot_a3/pilot_a3_vlm_vidhalluc_v1/run_receipt.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${RUN_ID:-formal_b_v1}"

# Mandatory hard authorization + prerequisite gate check
"${PYTHON_BIN}" "${EXP_ROOT}/deployment/authorization_gate.py" \
  --stage FORMAL_B \
  --auth-receipt "${AUTH_RECEIPT}" \
  --prior-a1 "${PRIOR_A1}" \
  --prior-a2 "${PRIOR_A2}" \
  --prior-a3 "${PRIOR_A3}"

"${PYTHON_BIN}" "${EXP_ROOT}/deployment/run_stage_cli.py" \
  --stage FORMAL_B \
  --paths-config "${PATHS_CFG}" \
  --stage-config "${EXP_ROOT}/configs/formal_b_candidate.yaml" \
  --out-dir "${Q2_ROOT}/outputs/formal_b/${RUN_ID}" \
  --run-id "${RUN_ID}" \
  --auth-receipt "${AUTH_RECEIPT}" \
  "$@"
