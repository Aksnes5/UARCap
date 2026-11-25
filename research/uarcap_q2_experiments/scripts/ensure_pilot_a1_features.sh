#!/usr/bin/env bash
# Ensure every Pilot A1 retrieval candidate has a cached faithful_v2 visual feature.
#
# Retrieval over the full VideoUFO corpus can return clips whose Video-LLaMA visual
# features were never extracted (the shared cache covers only previously retrieved
# clips). This script discovers the exact top-20 candidates for the frozen 30-video
# sample, extracts only the missing features with the same faithful_v2 pipeline, and
# then re-verifies. Retrieval semantics are NOT changed.
#
# Usage:
#   bash scripts/ensure_pilot_a1_features.sh \
#     [--manifest bn_carebench_pilot_a01/PILOT_A1_FROZEN_SAMPLING_MANIFEST.json]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
Q2_ROOT="${Q2_ROOT:-/home/bn/uarcap_q2_runtime}"
RUN_ID="${RUN_ID:-pilot_a1_carebench_v1}"
OUT_DIR="${OUT_DIR:-${Q2_ROOT}/outputs/pilot_a1/${RUN_ID}}"
MANIFEST="${MANIFEST:-${EXP_ROOT}/bn_carebench_pilot_a01/PILOT_A1_FROZEN_SAMPLING_MANIFEST.json}"
PATHS_CFG="${PATHS_CFG:-${EXP_ROOT}/configs/bn_paths.local.yaml}"
METADATA="${METADATA:-${Q2_ROOT}/data/carebench/json/metadata.json}"
VIDEO_ROOT="${VIDEO_ROOT:-${Q2_ROOT}/data/carebench/videos}"
FROZEN_PY="${FROZEN_PY:-/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python}"
HF_CACHE="${HF_CACHE:-/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/hf_cache}"
TORCH_CACHE="${TORCH_CACHE:-/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/torch_cache}"
GPU_LOCK="${GPU_LOCK:-/home/bn/videoufo_research/gpu.lock}"
MISSING="${OUT_DIR}/pilot_a1_missing_features.txt"
DISCOVERY="${OUT_DIR}/pilot_a1_retrieval_discovery.json"

mkdir -p "${OUT_DIR}"
discover() {
  PYTHONPATH="${EXP_ROOT}" "${FROZEN_PY}" -u "${EXP_ROOT}/scripts/discover_pilot_a1_candidates.py" \
    --manifest "${MANIFEST}" \
    --metadata "${METADATA}" \
    --video-root "${VIDEO_ROOT}" \
    --paths-config "${PATHS_CFG}" \
    --out-json "${DISCOVERY}" \
    --missing-out "${MISSING}"
}

discover
N_MISSING=$(grep -c . "${MISSING}" || true)
echo "[ensure_features] missing cached visual features: ${N_MISSING}"
if [[ "${N_MISSING}" -gt 0 ]]; then
  flock "${GPU_LOCK}" -c "HUGGINGFACE_HUB_CACHE='${HF_CACHE}' HF_HOME='${HF_CACHE}' TORCH_HOME='${TORCH_CACHE}' \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '${FROZEN_PY}' -u /home/bn/videoufo_research/scripts/extract_external_features_v2.py '${MISSING}'"
  discover
  N_MISSING=$(grep -c . "${MISSING}" || true)
  echo "[ensure_features] after extraction missing: ${N_MISSING}"
fi
if [[ "${N_MISSING}" -gt 0 ]]; then
  echo "[ensure_features] ERROR: ${N_MISSING} features still missing" >&2
  exit 2
fi
echo "[ensure_features] OK: all Pilot A1 candidate features are cached"
