#!/usr/bin/env bash
# Launch the two blind Pilot A2 annotator servers (A1 on :8765, A2 on :8766), localhost-only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
Q2_ROOT="${Q2_ROOT:-/home/bn/uarcap_q2_runtime}"
A2_DIR="${A2_DIR:-${Q2_ROOT}/outputs/pilot_a2/pilot_a2_carebench_eval_v1}"
PKG="${A2_DIR}/blind_annotation_package"
SAVE_ROOT="${SAVE_ROOT:-${A2_DIR}/annotation_saves}"
PY="${PYTHON_BIN:-/home/bn/anaconda3/envs/videollama2/bin/python}"

mkdir -p "${SAVE_ROOT}"
for pair in "A1:8765" "A2:8766"; do
  ann="${pair%%:*}"; port="${pair##*:}"
  bundle="${PKG}/blind_tasks_e2_${ann}.json"
  [[ -f "${bundle}" ]] || { echo "Missing bundle: ${bundle}" >&2; exit 2; }
  nohup "${PY}" "${EXP_ROOT}/scripts/launch_annotation_ui.py" \
    --bundle "${bundle}" --save-dir "${SAVE_ROOT}" --host 127.0.0.1 --port "${port}" --no-browser \
    > "${A2_DIR}/ui_${ann}.log" 2>&1 &
  echo $! > "${A2_DIR}/ui_${ann}.pid"
  echo "[annotation] ${ann} -> http://127.0.0.1:${port}/  (pid $(cat "${A2_DIR}/ui_${ann}.pid"))"
done

cat <<EOF

Both servers bind 127.0.0.1 only. From your laptop, open an SSH tunnel:

  ssh -N -L 8765:127.0.0.1:8765 -L 8766:127.0.0.1:8766 bn@172.20.72.8

Then browse:
  Annotator A1: http://127.0.0.1:8765/
  Annotator A2: http://127.0.0.1:8766/

Progress auto-saves to ${SAVE_ROOT}/annotator_A1 and .../annotator_A2.
Do not open ${PKG}/blind_key_e2_DO_NOT_OPEN.json.
EOF
