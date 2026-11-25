#!/usr/bin/env bash
# Execute Authorized Pilot A0 + A1 (30 DREAM-1K videos x 3 arms A0/A1/A7 = 90 generations on frozen Video-LLaMA)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATHS_CFG="${PATHS_CFG:-${EXP_ROOT}/configs/bn_paths.local.yaml}"
Q2_ROOT="${Q2_ROOT:-/home/bn/uarcap_q2_runtime}"
AUTH_RECEIPT="${AUTH_RECEIPT:-${Q2_ROOT}/authorizations/pilot_a01_auth.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${RUN_ID:-pilot_a1_dream1k_v1}"
OUT_DIR="${Q2_ROOT}/outputs/pilot_a1/${RUN_ID}"

# 1. Verify PILOT_A1 authorization receipt BEFORE touching GPU
"${PYTHON_BIN}" "${EXP_ROOT}/deployment/authorization_gate.py" \
  --stage PILOT_A1 \
  --auth-receipt "${AUTH_RECEIPT}"

# 2. Run A0 preflight first
bash "${SCRIPT_DIR}/run_preflight.sh" --verify-ckpt-sha

# 3. Verify DREAM-1K normalized dataset readiness (metadata + videos correspondence)
"${PYTHON_BIN}" "${EXP_ROOT}/deployment/stage_assets.py" \
  check-ready --data-root "${Q2_ROOT}/data" --dataset dream1k

# 4. Run real Pilot A1 pipeline (X-CLIP in frozen venv subprocess + frozen Video-LLaMA batch adapter + gpu.lock)
"${PYTHON_BIN}" "${EXP_ROOT}/deployment/run_stage_cli.py" \
  --stage PILOT_A1 \
  --paths-config "${PATHS_CFG}" \
  --stage-config "${EXP_ROOT}/configs/pilot_a01_config.yaml" \
  --out-dir "${OUT_DIR}" \
  --run-id "${RUN_ID}" \
  --auth-receipt "${AUTH_RECEIPT}" \
  --xclip-subprocess \
  "$@"

if [[ -f "${OUT_DIR}/run_receipt.json" ]]; then
  echo "[run_pilot_a01] Receipt written: ${OUT_DIR}/run_receipt.json"
  echo "[run_pilot_a01] Review ${OUT_DIR}/PILOT_A1_GO_NO_GO.md before authorizing Pilot A2."
fi
