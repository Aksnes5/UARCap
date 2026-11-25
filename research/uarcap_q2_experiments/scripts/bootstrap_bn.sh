#!/usr/bin/env bash
# One-click BN server bootstrap (non-destructive; NEVER touches frozen /media/bn or upgrades existing envs!)
# Supports existing BN Python via PYTHON_BIN (e.g. /home/bn/anaconda3/envs/videollama2/bin/python)
# without forcing `conda env create` or network access to conda.anaconda.org.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATHS_CFG="${PATHS_CFG:-${EXP_ROOT}/configs/bn_paths.local.yaml}"
Q2_ROOT="${Q2_ROOT:-/home/bn/uarcap_q2_runtime}"
CREATE_CPU_CONDA_ENV="${CREATE_CPU_CONDA_ENV:-0}"

# Resolve Python interpreter: explicit PYTHON_BIN -> existing BN videollama2 env -> python3
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/home/bn/anaconda3/envs/videollama2/bin/python" ]]; then
    PYTHON_BIN="/home/bn/anaconda3/envs/videollama2/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

echo "[bootstrap_bn] EXP_ROOT=${EXP_ROOT}"
echo "[bootstrap_bn] Q2_ROOT=${Q2_ROOT}"
echo "[bootstrap_bn] PYTHON_BIN=${PYTHON_BIN}"

# 1. Initialize editable local paths config if missing
if [[ ! -f "${PATHS_CFG}" ]]; then
  cp "${EXP_ROOT}/configs/bn_paths.example.yaml" "${PATHS_CFG}"
  echo "[bootstrap_bn] Created ${PATHS_CFG} from template. Review paths if BN layout changed."
fi

# 2. Prepare writable Q2 runtime directories on '/' NVMe (NEVER write to read-only /media/bn)
mkdir -p \
  "${Q2_ROOT}/data/carebench/json" \
  "${Q2_ROOT}/data/carebench/videos" \
  "${Q2_ROOT}/data/dream1k/json" \
  "${Q2_ROOT}/data/dream1k/video" \
  "${Q2_ROOT}/data/vidhalluc/data" \
  "${Q2_ROOT}/staging" \
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

# 4. Verify existing Python interpreter satisfies CPU orchestration dependencies (READ-ONLY check; no pip/conda upgrades!)
if "${PYTHON_BIN}" -c "import numpy, scipy, yaml, PIL, av, pytest" >/dev/null 2>&1; then
  echo "[bootstrap_bn] Verified existing Python environment satisfies CPU orchestration dependencies (numpy, scipy, yaml, PIL, av, pytest): ${PYTHON_BIN}"
  echo "[bootstrap_bn] Skipping conda env creation (no conda.anaconda.org access needed)."
elif [[ "${CREATE_CPU_CONDA_ENV}" == "1" ]] && command -v conda >/dev/null 2>&1; then
  if ! conda env list | awk '{print $1}' | grep -qx "uarcap_q2_cpu"; then
    echo "[bootstrap_bn] Creating isolated CPU analysis env (opt-in CREATE_CPU_CONDA_ENV=1): uarcap_q2_cpu"
    conda env create -f "${EXP_ROOT}/configs/envs/env_cpu_analysis.yaml"
  else
    echo "[bootstrap_bn] Isolated env uarcap_q2_cpu already exists."
  fi
else
  echo "[bootstrap_bn] WARNING: ${PYTHON_BIN} is missing one or more CPU deps (numpy, scipy, yaml, PIL, av, pytest)." >&2
  echo "[bootstrap_bn] Pass an existing interpreter via PYTHON_BIN=/home/bn/anaconda3/envs/videollama2/bin/python or set CREATE_CPU_CONDA_ENV=1." >&2
fi

echo "[bootstrap_bn] NOTE: Existing model environments (legacy Video-LLaMA venv & vLLM 0.29.0) are kept 100% READ-ONLY."
echo "[bootstrap_bn] Done. Next step: PYTHON_BIN=${PYTHON_BIN} bash scripts/run_preflight.sh --verify-ckpt-sha"
