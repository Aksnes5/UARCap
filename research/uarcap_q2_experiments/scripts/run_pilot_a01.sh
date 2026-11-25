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

# 1. Verify authorization receipt before touching GPU
"${PYTHON_BIN}" "${EXP_ROOT}/deployment/authorization_gate.py" \
  --stage PILOT_A1 \
  --auth-receipt "${AUTH_RECEIPT}"

# 2. Run A0 preflight first
bash "${SCRIPT_DIR}/run_preflight.sh" --verify-ckpt-sha

# 3. Verify DREAM-1K video.zip is staged (Route 1 or Route 2)
"${PYTHON_BIN}" "${EXP_ROOT}/deployment/stage_assets.py" \
  verify --data-root "${Q2_ROOT}/data" --asset dream1k_videos

# 4. Run Pilot A1 stage CLI (cooperative gpu.lock enforced inside runner)
"${PYTHON_BIN}" "${EXP_ROOT}/deployment/run_stage_cli.py" \
  --stage PILOT_A1 \
  --paths-config "${PATHS_CFG}" \
  --stage-config "${EXP_ROOT}/configs/pilot_a01_config.yaml" \
  --out-dir "${Q2_ROOT}/outputs/pilot_a1/${RUN_ID}" \
  --run-id "${RUN_ID}" \
  --auth-receipt "${AUTH_RECEIPT}" \
  "$@"

echo "[run_pilot_a01] Completed. Review PILOT_A1_GO_NO_GO.md before authorizing Pilot A2."
