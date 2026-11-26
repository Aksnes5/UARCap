#!/usr/bin/env python3
"""Detached VidHalluc ACH downloader: resumable parallel ranges, fresh signed URL per cycle,
full SHA-256 verification. Network-only; never blocks the GPU queue."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
import threading
import time
import urllib.request

BASE = Path(os.environ.get("OVERNIGHT_BASE", "/home/bn/uarcap_q2_runtime/overnight/20261126"))
DATA = Path("/home/bn/uarcap_q2_runtime/data/vidhalluc")
DEST = DATA / "data" / "ACH_videos.zip"
STATE = BASE / "download_state.json"
LOG = BASE / "logs" / "download_vidhalluc.log"
EXPECTED_BYTES = 16754119398
EXPECTED_SHA = "8f163f29a72f5bc8d15720847b1c0b125c8a593d3bdaedfabbeb4f83e3c25af5"
HF = "https://hf-mirror.com/datasets/chaoyuli/VidHalluc/resolve/main"
API = "https://hf-mirror.com/api/datasets/chaoyuli/VidHalluc/tree/main/data"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    line = f"{utcnow()} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def atomic_state(payload: dict) -> None:
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, STATE)


def ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def resolve_url(path: str) -> str:
    req = urllib.request.Request(f"{HF}/{path}", method="HEAD", headers={"User-Agent": "UARCap-Overnight/1.0"})
    with urllib.request.urlopen(req, timeout=30, context=ctx()) as resp:
        loc = resp.headers.get("Location")
        if loc:
            return loc
    return f"{HF}/{path}"


def bytes_on_disk() -> int:
    total = 0
    if DEST.is_file():
        total += DEST.stat().st_size
    part = DEST.parent / f".{DEST.name}.chunks"
    if part.is_dir():
        for p in part.glob("chunk_*"):
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def state_writer(stop: threading.Event, info: dict) -> None:
    last = 0
    while not stop.wait(20):
        now = time.time()
        cur = bytes_on_disk()
        speed = (cur - last) / 20.0 if last else 0.0
        last = cur
        info.update({"bytes_on_disk": cur, "percent": round(100.0 * cur / EXPECTED_BYTES, 3),
                     "speed_kbps": round(speed / 1024.0, 1), "updated_utc": utcnow()})
        atomic_state(dict(info))


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "data").mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, "/home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments")
    from uarcap_q2.datasets.asset_transfer import download_parallel_ranges, download_with_resume

    info = {"status": "STARTING", "started_utc": utcnow(), "expected_bytes": EXPECTED_BYTES,
            "expected_sha256": EXPECTED_SHA, "dest": str(DEST), "pid": os.getpid(), "attempts": 0}
    atomic_state(info)
    # small metadata files first (non-fatal if they fail)
    for name in ("ach_mcq.json", "ach_binaryqa.json"):
        try:
            out = DATA / name
            r = download_with_resume(f"{HF}/{name}", out)
            log(f"metadata {name}: {r.get('status')} {out.stat().st_size if out.is_file() else 'missing'}")
        except Exception as exc:
            log(f"metadata {name} failed: {exc}")

    if DEST.is_file() and DEST.stat().st_size == EXPECTED_BYTES:
        import hashlib
        h = hashlib.sha256()
        with DEST.open("rb") as f:
            for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() == EXPECTED_SHA:
            info.update({"status": "COMPLETED", "bytes_on_disk": EXPECTED_BYTES, "percent": 100.0,
                         "sha256": h.hexdigest(), "completed_utc": utcnow()})
            atomic_state(info)
            log("already complete and verified")
            return 0

    stop = threading.Event()
    writer = threading.Thread(target=state_writer, args=(stop, info), daemon=True)
    writer.start()
    try:
        for cycle in range(1, 60):
            info["attempts"] = cycle
            try:
                url = resolve_url("data/ACH_videos.zip")
                log(f"cycle {cycle}: resolved host={url.split('/')[2] if '://' in url else '?'}")
                info["status"] = "RUNNING"
                atomic_state(dict(info))
                res = download_parallel_ranges(
                    [url], DEST, expected_bytes=EXPECTED_BYTES, expected_sha256=EXPECTED_SHA,
                    workers=32, chunk_bytes=128 * 1024 * 1024, max_retries=5, timeout_sec=45,
                )
                info.update({"status": "COMPLETED", "result": res, "completed_utc": utcnow(),
                             "bytes_on_disk": DEST.stat().st_size, "percent": 100.0})
                atomic_state(dict(info))
                log(f"COMPLETED sha={res.get('sha256')}")
                return 0
            except Exception as exc:
                info.update({"status": "RETRYING", "last_error": str(exc)[:500], "updated_utc": utcnow()})
                atomic_state(dict(info))
                log(f"cycle {cycle} failed: {exc}")
                time.sleep(min(30 * cycle, 300))
        info.update({"status": "FAILED", "last_error": "cycles_exhausted", "updated_utc": utcnow()})
        atomic_state(dict(info))
        return 2
    finally:
        stop.set()
        writer.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
