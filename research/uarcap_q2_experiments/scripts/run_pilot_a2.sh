#!/usr/bin/env bash
# Execute Authorized Pilot A2 (Isolated Cross-Family 3-Way Evaluator + Human Calibration Prep)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATHS_CFG="${PATHS_CFG:-${EXP_ROOT}/configs/bn_paths.local.yaml}"
Q2_ROOT="${Q2_ROOT:-/home/bn/uarcap_q2_runtime}"
AUTH_RECEIPT="${AUTH_RECEIPT:-${Q2_ROOT}/authorizations/pilot_a2_auth.json}"
PRIOR_A1="${PRIOR_A1:-${Q2_ROOT}/outputs/pilot_a1/pilot_a1_dream1k_v1/run_receipt.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${RUN_ID:-pilot_a2_eval_v1}"

if [[ ! -f "${PRIOR_A1}" ]]; then
  echo "[run_pilot_a2] BLOCKED: Prerequisite Pilot A1 receipt missing: ${PRIOR_A1}" >&2
  exit 3
fi

"${PYTHON_BIN}" "${EXP_ROOT}/deployment/authorization_gate.py" \
  --stage PILOT_A2 \
  --auth-receipt "${AUTH_RECEIPT}"

"${PYTHON_BIN}" "${EXP_ROOT}/deployment/run_stage_cli.py" \
  --stage PILOT_A2 \
  --paths-config "${PATHS_CFG}" \
  --stage-config "${EXP_ROOT}/configs/pilot_a23_config.yaml" \
  --out-dir "${Q2_ROOT}/outputs/pilot_a2/${RUN_ID}" \
  --run-id "${RUN_ID}" \
  --auth-receipt "${AUTH_RECEIPT}" \
  "$@"

echo "[run_pilot_a2] Completed. Export blind annotation package & verify A7 differential bias before A3."
