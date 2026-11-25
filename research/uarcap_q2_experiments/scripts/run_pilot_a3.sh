#!/usr/bin/env bash
# Execute Authorized Pilot A3 (Isolated Modern VLM + VidHalluc 20-Clip Controlled E3 Pilot)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATHS_CFG="${PATHS_CFG:-${EXP_ROOT}/configs/bn_paths.local.yaml}"
Q2_ROOT="${Q2_ROOT:-/home/bn/uarcap_q2_runtime}"
AUTH_RECEIPT="${AUTH_RECEIPT:-${Q2_ROOT}/authorizations/pilot_a3_auth.json}"
PRIOR_A2="${PRIOR_A2:-${Q2_ROOT}/outputs/pilot_a2/pilot_a2_eval_v1/calibration_receipt.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${RUN_ID:-pilot_a3_vlm_vidhalluc_v1}"

if [[ ! -f "${PRIOR_A2}" ]]; then
  echo "[run_pilot_a3] BLOCKED: Prerequisite Pilot A2 calibration receipt missing: ${PRIOR_A2}" >&2
  exit 3
fi

"${PYTHON_BIN}" "${EXP_ROOT}/deployment/authorization_gate.py" \
  --stage PILOT_A3 \
  --auth-receipt "${AUTH_RECEIPT}"

# Verify VidHalluc ACH_videos.zip (15.60 GiB) is staged ONLY at A3
"${PYTHON_BIN}" "${EXP_ROOT}/deployment/stage_assets.py" \
  verify --data-root "${Q2_ROOT}/data" --asset vidhalluc_ach_videos

"${PYTHON_BIN}" "${EXP_ROOT}/deployment/run_stage_cli.py" \
  --stage PILOT_A3 \
  --paths-config "${PATHS_CFG}" \
  --stage-config "${EXP_ROOT}/configs/pilot_a23_config.yaml" \
  --out-dir "${Q2_ROOT}/outputs/pilot_a3/${RUN_ID}" \
  --run-id "${RUN_ID}" \
  --auth-receipt "${AUTH_RECEIPT}" \
  "$@"

echo "[run_pilot_a3] Completed. Review all 6 formal Go/No-Go gates before authorizing Formal B."
