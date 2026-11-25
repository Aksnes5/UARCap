#!/usr/bin/env bash
# Non-destructive status dashboard for UARCap Q2 stages, GPU lock, prediction counts, and logs
set -euo pipefail

Q2_ROOT="${Q2_ROOT:-/home/bn/uarcap_q2_runtime}"
LOCK_PATH="${LOCK_PATH:-/home/bn/videoufo_research/gpu.lock}"

echo "=== UARCap Q2 Runtime Status ==="
echo "Q2_ROOT: ${Q2_ROOT}"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "--- GPU Compute Clients (Read-Only) ---"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
fi

if [[ -f "${LOCK_PATH}" ]]; then
  echo "--- Shared GPU Lock: ${LOCK_PATH} (present) ---"
  ls -l "${LOCK_PATH}"
else
  echo "--- Shared GPU Lock: ${LOCK_PATH} (NOT FOUND) ---"
fi

echo "--- Stage Receipts & Output Counts ---"
for stg in pilot_a0 pilot_a1 pilot_a2 pilot_a3 formal_b; do
  dir="${Q2_ROOT}/outputs/${stg}"
  if [[ -d "${dir}" ]]; then
    preds=$(find "${dir}" -name "predictions.jsonl" -exec wc -l {} + 2>/dev/null | tail -n 1 || echo "0")
    fails=$(find "${dir}" -name "failures.jsonl" -exec wc -l {} + 2>/dev/null | tail -n 1 || echo "0")
    receipts=$(find "${dir}" -name "*receipt*.json" 2>/dev/null | wc -l || echo "0")
    echo "  ${stg}: receipts=${receipts} | predictions_lines=[${preds}] | failure_lines=[${fails}]"
  else
    echo "  ${stg}: not initialized"
  fi
done
