#!/usr/bin/env bash
# One-click BN server bootstrap (non-destructive; NEVER touches frozen /media/bn or upgrades legacy Video-LLaMA venv!)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATHS_CFG="${PATHS_CFG:-${EXP_ROOT}/configs/bn_paths.local.yaml}"
Q2_ROOT="${Q2_ROOT:-/home/bn/uarcap_q2_runtime}"

echo "[bootstrap_bn] EXP_ROOT=${EXP_ROOT}"
echo "[bootstrap_bn] Q2_ROOT=${Q2_ROOT}"

# 1. Initialize editable local paths config if missing
if [[ ! -f "${PATHS_CFG}" ]]; then
  cp "${EXP_ROOT}/configs/bn_paths.example.yaml" "${PATHS_CFG}"
  echo "[bootstrap_bn] Created ${PATHS_CFG} from template. Review paths if BN layout changed."
fi

# 2. Prepare writable Q2 runtime directories on '/' NVMe (NEVER write to read-only /media/bn)
mkdir -p \
  "${Q2_ROOT}/data/dream1k/json" \
  "${Q2_ROOT}/data/dream1k/video" \
  "${Q2_ROOT}/data/vidhalluc/data" \
  "${Q2_ROOT}/models" \
  "${Q2_ROOT}/outputs/pilot_a0" \
  "${Q2_ROOT}/outputs/pilot_a1" \
  "${Q2_ROOT}/outputs/pilot_a2" \
  "${Q2_ROOT}/outputs/pilot_a3" \
  "${Q2_ROOT}/outputs/formal_b" \
  "${Q2_ROOT}/authorizations"

# 3. Check shared GPU lock readability (NEVER create/overwrite/delete foreign lock!)
LOCK_PATH="/home/bn/videoufo_research/gpu.lock"
if [[ -f "${LOCK_PATH}" ]]; then
  echo "[bootstrap_bn] Verified shared GPU lock exists: ${LOCK_PATH}"
else
  echo "[bootstrap_bn] WARNING: Shared GPU lock not found at ${LOCK_PATH}; verify historical path in ${PATHS_CFG}."
fi

# 4. Prepare isolated CPU analysis Conda env if conda is available (opt-in for GPU envs at A2/A3!)
if command -v conda >/dev/null 2>&1; then
  if ! conda env list | awk '{print $1}' | grep -qx "uarcap_q2_cpu"; then
    echo "[bootstrap_bn] Creating isolated CPU analysis env: uarcap_q2_cpu"
    conda env create -f "${EXP_ROOT}/configs/envs/env_cpu_analysis.yaml"
  else
    echo "[bootstrap_bn] Isolated env uarcap_q2_cpu already exists."
  fi
  echo "[bootstrap_bn] NOTE: GPU envs (uarcap_q2_evaluator, uarcap_q2_modern_vlm) are installed only when A2/A3 unlock."
  echo "[bootstrap_bn] NOTE: Legacy Video-LLaMA venv (/media/bn/.../repro_evcap/venv) is kept 100% READ-ONLY."
else
  echo "[bootstrap_bn] conda not in PATH; ensure Python 3.10+ with packages from configs/envs/env_cpu_analysis.yaml."
fi

echo "[bootstrap_bn] Done. Next step: bash scripts/run_preflight.sh"
