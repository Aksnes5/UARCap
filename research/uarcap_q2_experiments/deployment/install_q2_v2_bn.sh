#!/usr/bin/env bash
# Run on BN from the pinned V2 checkout. Dry-run and approval are explicit modes.
set -euo pipefail

commit="${1:?pass the complete accepted V2 commit SHA}"
mode="${2:---dry-run}"
approval="${3:-}"
case "$mode" in
  --dry-run|--install) ;;
  *) echo "Use --dry-run or --install" >&2; exit 2 ;;
esac
if [[ ! "$commit" =~ ^[a-f0-9]{40}$ ]]; then
  echo "Expected a complete commit SHA" >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test "$(git rev-parse HEAD)" = "$commit"
cpu=/home/bn/anaconda3/envs/videollama2/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUTF8=1
"$cpu" deployment/q2_v2_bn_preflight.py
"$cpu" -m pytest -q -rs
"$cpu" deployment/q2_laptop_queue.py --commit "$commit" --dry-run
if [[ "$mode" == --install ]]; then
  "$cpu" deployment/q2_laptop_queue.py --commit "$commit" --install --gpu-approval "${approval:?provide the signed BN execution approval file}"
fi
