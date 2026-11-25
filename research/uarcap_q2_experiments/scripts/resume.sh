#!/usr/bin/env bash
# Safe checkpointed resume entrypoint (re-runs target stage script; completed (query, arm, model) rows are skipped atomically)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:-}"
if [[ -z "${STAGE}" ]]; then
  echo "Usage: bash scripts/resume.sh <pilot_a01|pilot_a2|pilot_a3|formal_b> [extra args...]" >&2
  exit 2
fi
shift

case "${STAGE}" in
  pilot_a01) exec bash "${SCRIPT_DIR}/run_pilot_a01.sh" "$@" ;;
  pilot_a2)  exec bash "${SCRIPT_DIR}/run_pilot_a2.sh" "$@" ;;
  pilot_a3)  exec bash "${SCRIPT_DIR}/run_pilot_a3.sh" "$@" ;;
  formal_b)  exec bash "${SCRIPT_DIR}/run_formal_b.sh" "$@" ;;
  *)
    echo "Unknown stage '${STAGE}'. Expected one of: pilot_a01, pilot_a2, pilot_a3, formal_b" >&2
    exit 2
    ;;
esac
