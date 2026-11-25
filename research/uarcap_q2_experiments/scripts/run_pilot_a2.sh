#!/usr/bin/env bash
# Execute Authorized Pilot A2 (Reads A1 outputs -> 3-way evaluator -> blind annotation package -> human calibration)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATHS_CFG="${PATHS_CFG:-${EXP_ROOT}/configs/bn_paths.local.yaml}"
Q2_ROOT="${Q2_ROOT:-/home/bn/uarcap_q2_runtime}"
AUTH_RECEIPT="${AUTH_RECEIPT:-${Q2_ROOT}/authorizations/pilot_a2_auth.json}"
E2_DATASET="${E2_DATASET:-carebench}"
A1_DIR="${A1_DIR:-${Q2_ROOT}/outputs/pilot_a1/pilot_a1_${E2_DATASET}_v1}"
PRIOR_A1="${PRIOR_A1:-${A1_DIR}/run_receipt.json}"
A1_PREDS="${A1_PREDS:-${A1_DIR}/predictions.jsonl}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/home/bn/anaconda3/envs/videollama2/bin/python" ]]; then
    PYTHON_BIN="/home/bn/anaconda3/envs/videollama2/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
RUN_ID="${RUN_ID:-pilot_a2_${E2_DATASET}_eval_v1}"
OUT_DIR="${Q2_ROOT}/outputs/pilot_a2/${RUN_ID}"

# Strictly verify REAL_RUNTIME_PASS on PILOT_A1 (rejects missing file AND rejects MOCK_PASS!)
"${PYTHON_BIN}" "${EXP_ROOT}/deployment/authorization_gate.py" \
  --stage PILOT_A2 \
  --auth-receipt "${AUTH_RECEIPT}" \
  --prior-a1 "${PRIOR_A1}"

set +e
"${PYTHON_BIN}" "${EXP_ROOT}/deployment/run_stage_cli.py" \
  --stage PILOT_A2 \
  --e2-dataset "${E2_DATASET}" \
  --paths-config "${PATHS_CFG}" \
  --stage-config "${EXP_ROOT}/configs/pilot_a23_config.yaml" \
  --out-dir "${OUT_DIR}" \
  --run-id "${RUN_ID}" \
  --auth-receipt "${AUTH_RECEIPT}" \
  --prior-a1 "${PRIOR_A1}" \
  --a1-predictions "${A1_PREDS}" \
  "$@"
RC=$?
set -e

if [[ "${RC}" -eq 10 ]]; then
  echo "[run_pilot_a2] PAUSED (HUMAN_VALIDATION_PENDING): Blind annotation package ready under ${OUT_DIR}/blind_annotation_package"
  echo "[run_pilot_a2] Complete human annotation, then re-run with --human-annotations <adjudicated.json>."
  exit 10
elif [[ "${RC}" -ne 0 ]]; then
  echo "[run_pilot_a2] FAILED or BLOCKED (exit code ${RC})." >&2
  exit "${RC}"
fi

echo "[run_pilot_a2] Calibration receipt written: ${OUT_DIR}/calibration_receipt.json"
