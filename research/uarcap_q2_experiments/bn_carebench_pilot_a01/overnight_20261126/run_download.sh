#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${BASE}/logs"
cd "${BASE}"
export OVERNIGHT_BASE="${BASE}"
exec /usr/bin/python3 -u tasks/download_vidhalluc.py >> "${BASE}/logs/download_stdout.log" 2>&1
