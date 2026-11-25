#!/usr/bin/env bash
# Execute Authorized Pilot A3 (Modern VLM on DREAM-1K 30 + VidHalluc 4-arm E3 after A2 REAL_RUNTIME_PASS)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATHS_CFG="${PATHS_CFG:-${EXP_ROOT}/configs/bn_paths.local.yaml}"
Q2_ROOT="${Q2_ROOT:-/home/bn/uarcap_q2_runtime}"
AUTH_RECEIPT="${AUTH_RECEIPT:-${Q2_ROOT}/authorizations/pilot_a3_auth.json}"
PRIOR_A2="${PRIOR_A2:-${Q2_ROOT}/outputs/pilot_a2/pilot_a2_eval_v1/calibration_receipt.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${RUN_ID:-pilot_a3_vlm_vidhalluc_v1}"
OUT_DIR="${Q2_ROOT}/outputs/pilot_a3/${RUN_ID}"

# Strictly verify PILOT_A2 status==REAL_RUNTIME_PASS & human_calibration_completed==True
"${PYTHON_BIN}" "${EXP_ROOT}/deployment/authorization_gate.py" \
  --stage PILOT_A3 \
  --auth-receipt "${AUTH_RECEIPT}" \
  --prior-a2 "${PRIOR_A2}"

# Verify VidHalluc ACH_videos.zip (15.60 GiB) is staged ONLY after A2 pass
"${PYTHON_BIN}" "${EXP_ROOT}/deployment/stage_assets.py" \
  verify --data-root "${Q2_ROOT}/data" --asset vidhalluc_ach_videos

set +e
"${PYTHON_BIN}" "${EXP_ROOT}/deployment/run_stage_cli.py" \
  --stage PILOT_A3 \
  --paths-config "${PATHS_CFG}" \
  --stage-config "${EXP_ROOT}/configs/pilot_a23_config.yaml" \
  --out-dir "${OUT_DIR}" \
  --run-id "${RUN_ID}" \
  --auth-receipt "${AUTH_RECEIPT}" \
  --prior-a2 "${PRIOR_A2}" \
  "$@"
RC=$?
set -e

if [[ "${RC}" -eq 10 ]]; then
  echo "[run_pilot_a3] PAUSED (HUMAN_VALIDATION_PENDING): Check ${OUT_DIR}/run_receipt.json for pending human step (--approved-e3-candidates or --e3-human-annotations)."
  exit 10
elif [[ "${RC}" -ne 0 ]]; then
  echo "[run_pilot_a3] FAILED or BLOCKED (exit code ${RC})." >&2
  exit "${RC}"
fi

echo "[run_pilot_a3] Receipt written: ${OUT_DIR}/run_receipt.json"
