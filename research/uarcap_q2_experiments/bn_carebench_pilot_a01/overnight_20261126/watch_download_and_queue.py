#!/usr/bin/env python3
"""Local watcher: when the VidHalluc download completes, atomically append the full E3 task
batch to task_queue.json. No Claude session or network required after this starts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

BASE = Path(os.environ.get("OVERNIGHT_BASE", "/home/bn/uarcap_q2_runtime/overnight/20261126"))
STATE = BASE / "download_state.json"
QUEUE = BASE / "task_queue.json"
TEMPLATE = BASE / "vidhalluc_task_template.json"
LOG = BASE / "logs" / "download_trigger.log"
RECEIPT = BASE / "vidhalluc_trigger_receipt.json"
STOP = BASE / "STOP_REQUESTED"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    line = f"{utcnow()} {msg}\n"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def read_json(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def main() -> int:
    log(f"watcher started pid={os.getpid()}")
    appended = False
    while True:
        if STOP.exists():
            log("STOP_REQUESTED; exiting watcher")
            return 0
        st = read_json(STATE, {})
        status = st.get("status")
        if status == "COMPLETED" and not appended:
            q = read_json(QUEUE, {"tasks": []})
            ids = {t["id"] for t in q.get("tasks", [])}
            if "vidhalluc_extract" not in ids:
                tpl = read_json(TEMPLATE, {"tasks": []})
                q["tasks"].extend(tpl.get("tasks", []))
                q["updated_utc"] = utcnow()
                tmp = QUEUE.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(q, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(tmp, QUEUE)
                log(f"appended {len(tpl.get('tasks', []))} VidHalluc tasks")
            else:
                log("VidHalluc tasks already present")
            with RECEIPT.open("w", encoding="utf-8") as f:
                json.dump({"status": "VIDHALLUC_TASKS_QUEUED", "utc": utcnow(),
                           "download_sha256": st.get("expected_sha256"), "download_bytes": st.get("bytes_on_disk"),
                           "downloader_pid": st.get("pid")}, f, indent=2)
            appended = True
        if status == "FAILED":
            log(f"download FAILED: {st.get('last_error')}")
        time.sleep(20)


if __name__ == "__main__":
    raise SystemExit(main())
