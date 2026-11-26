#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${BASE}/logs"
cd "${BASE}"
exec /usr/bin/python3 -u overnight_supervisor.py --base "${BASE}" --check-interval 1800 --heartbeat 60 --duration-hours 14 >> "${BASE}/logs/supervisor_stdout.log" 2>&1
