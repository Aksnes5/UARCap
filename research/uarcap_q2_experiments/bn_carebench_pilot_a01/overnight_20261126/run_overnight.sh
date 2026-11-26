#!/usr/bin/env bash
# Detached entrypoint for the persistent overnight queue (survives SSH/desktop/agent loss).
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${BASE}/logs" "${BASE}/progress"
cd "${BASE}"
exec /usr/bin/python3 -u run_overnight.py --base "${BASE}" >> "${BASE}/logs/orchestrator.log" 2>&1
