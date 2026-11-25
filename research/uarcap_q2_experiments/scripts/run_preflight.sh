#!/usr/bin/env bash
# Non-destructive BN & CPU Preflight Check (Pilot A0)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATHS_CFG="${PATHS_CFG:-${EXP_ROOT}/configs/bn_paths.local.yaml}"
if [[ ! -f "${PATHS_CFG}" ]]; then
  PATHS_CFG="${EXP_ROOT}/configs/bn_paths.example.yaml"
fi
Q2_ROOT="${Q2_ROOT:-/home/bn/uarcap_q2_runtime}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/home/bn/anaconda3/envs/videollama2/bin/python" ]]; then
    PYTHON_BIN="/home/bn/anaconda3/envs/videollama2/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

mkdir -p "${Q2_ROOT}/outputs/pilot_a0"
echo "[run_preflight] Running non-destructive A0 preflight check..."
"${PYTHON_BIN}" "${EXP_ROOT}/deployment/bn_preflight_check.py" \
  --paths-config "${PATHS_CFG}" \
  --out-receipt "${Q2_ROOT}/outputs/pilot_a0/preflight_receipt.json" \
  "$@"

echo "[run_preflight] Receipt written to ${Q2_ROOT}/outputs/pilot_a0/preflight_receipt.json"
