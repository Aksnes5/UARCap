#!/usr/bin/env python3
"""Persistent overnight task orchestrator for UARCap Q2 (no Claude session dependency).

- Reads task_queue.json (reload each loop), writes atomic task_state.json.
- Runs one task at a time; each task has its own log and optional progress JSON.
- Heartbeat every 60 s (task, GPU, progress, completed/remaining, download state).
- STOP_REQUESTED file => finish current task, mark the rest SKIPPED_STOPPED, exit.
- Never kills unrelated processes; tasks themselves honour the shared GPU lock.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

PENDING = "PENDING"
RUNNING = "RUNNING"
COMPLETED = "COMPLETED"
RETRYABLE = "RETRYABLE_FAILURE"
BLOCKED = "BLOCKED"
SKIPPED = "SKIPPED"
AWAITING_HUMAN = "AWAITING_HUMAN"
SKIPPED_STOP = "SKIPPED_STOPPED"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


class Runner:
    def __init__(self, base: Path) -> None:
        self.base = base
        self.queue_path = base / "task_queue.json"
        self.state_path = base / "task_state.json"
        self.heartbeat_path = base / "heartbeat.json"
        self.stop_path = base / "STOP_REQUESTED"
        self.logs = base / "logs"
        self.progress = base / "progress"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.progress.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.current_task_id: str | None = None
        self.current_proc: subprocess.Popen | None = None
        self.stop_requested = False
        self.started = utcnow()
        self.state = read_json(self.state_path, {"schema": "uarcap_overnight_v1", "started_utc": self.started, "tasks": {}})
        self.state.setdefault("tasks", {})
        self.save_state()
        self.hb_stop = threading.Event()
        self.hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)

    # ---------- state helpers ----------
    def save_state(self) -> None:
        with self.lock:
            self.state["updated_utc"] = utcnow()
            atomic_write_json(self.state_path, self.state)

    def task_state(self, task_id: str) -> dict:
        return self.state["tasks"].setdefault(task_id, {"status": PENDING, "attempts": 0})

    def set_status(self, task_id: str, status: str, **extra) -> None:
        with self.lock:
            st = self.state["tasks"].setdefault(task_id, {})
            st["status"] = status
            st.update(extra)
            st["updated_utc"] = utcnow()
        self.save_state()

    # ---------- heartbeat ----------
    def _gpu_snapshot(self) -> dict:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                text=True, timeout=5,
            ).strip().splitlines()[0]
            u, used, total = [x.strip() for x in out.split(",")]
            return {"utilization_pct": int(u), "memory_used_mib": int(used), "memory_total_mib": int(total)}
        except Exception:
            return {"utilization_pct": None, "memory_used_mib": None, "memory_total_mib": None}

    def _heartbeat_loop(self) -> None:
        while not self.hb_stop.wait(60):
            try:
                self._write_heartbeat()
            except Exception:
                pass

    def _write_heartbeat(self) -> None:
        tasks = self.state.get("tasks", {})
        counts: dict[str, int] = {}
        gpu_seconds = 0.0
        for st in tasks.values():
            counts[st.get("status", PENDING)] = counts.get(st.get("status", PENDING), 0) + 1
            if st.get("requires_gpu") and st.get("elapsed_sec"):
                gpu_seconds += float(st["elapsed_sec"])
        prog = None
        if self.current_task_id:
            prog = read_json(self.progress / f"{self.current_task_id}.json", None)
        download = read_json(self.base / "download_state.json", None)
        payload = {
            "updated_utc": utcnow(),
            "started_utc": self.started,
            "current_task": self.current_task_id,
            "current_task_pid": self.current_proc.pid if self.current_proc else None,
            "progress": prog,
            "gpu": self._gpu_snapshot(),
            "cumulative_task_elapsed_sec": round(gpu_seconds, 1),
            "task_counts": counts,
            "n_tasks": len(tasks),
            "stop_requested": self.stop_path.exists(),
            "network_download": download,
        }
        atomic_write_json(self.heartbeat_path, payload)

    # ---------- task loop ----------
    def _load_queue(self) -> list[dict]:
        q = read_json(self.queue_path, {"tasks": []})
        return q.get("tasks", []) if isinstance(q, dict) else list(q)

    def _ready(self, task: dict, by_id: dict[str, dict]) -> bool:
        for dep in task.get("depends_on", []):
            st = self.state["tasks"].get(dep, {}).get("status")
            if st not in (COMPLETED, SKIPPED):
                return False
        return True

    def _run_task(self, task: dict) -> None:
        tid = task["id"]
        cmd = task["cmd"]
        cwd = task.get("cwd") or str(self.base)
        timeout = int(task.get("timeout_sec", 3600))
        log_path = self.logs / f"{tid}.log"
        st = self.task_state(tid)
        attempt = int(st.get("attempts", 0)) + 1
        self.set_status(tid, RUNNING, attempts=attempt, started_utc=utcnow(), log=str(log_path),
                        requires_gpu=bool(task.get("requires_gpu")), network=bool(task.get("network")))
        self.current_task_id = tid
        with log_path.open("ab") as logf:
            logf.write(f"\n===== {utcnow()} attempt {attempt} =====\n".encode())
            logf.flush()
            self.current_proc = subprocess.Popen(
                ["/bin/bash", "-lc", cmd], cwd=cwd, stdout=logf, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            start = time.time()
            timed_out = False
            try:
                rc = self.current_proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(self.current_proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    rc = self.current_proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.current_proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    rc = -9
            elapsed = round(time.time() - start, 1)
        self.current_proc = None
        self.current_task_id = None
        extra = {"ended_utc": utcnow(), "elapsed_sec": elapsed, "exit_code": rc, "timed_out": timed_out}
        if rc == 0 and not timed_out:
            self.set_status(tid, COMPLETED, **extra)
        else:
            # Detect explicit hard-block markers in the task log tail.
            hard = ""
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                if "HARD_BLOCK" in tail or "AWAITING_HUMAN" in tail:
                    hard = "hard_block_marker"
            except Exception:
                pass
            max_retries = int(task.get("max_retries", 1))
            if hard:
                self.set_status(tid, BLOCKED, **extra, note=hard)
            elif attempt < max_retries:
                self.set_status(tid, RETRYABLE, **extra)
                time.sleep(min(60 * attempt, 300))
            else:
                self.set_status(tid, BLOCKED, **extra, note="max_retries_exhausted")

    def loop(self) -> int:
        self.hb_thread.start()
        self.set_status("__orchestrator__", RUNNING, pid=os.getpid(), started_utc=self.started)
        while True:
            if self.stop_path.exists():
                self.stop_requested = True
                with self.lock:
                    for tid, st in self.state["tasks"].items():
                        if st.get("status") in (PENDING, RETRYABLE):
                            st["status"] = SKIPPED_STOP
                            st["updated_utc"] = utcnow()
                self.save_state()
                self.set_status("__orchestrator__", SKIPPED_STOP, ended_utc=utcnow(), note="STOP_REQUESTED")
                self.hb_stop.set()
                self._write_heartbeat()
                return 0
            queue = self._load_queue()
            by_id = {t["id"]: t for t in queue}
            # register new tasks
            for t in queue:
                self.task_state(t["id"])
            self.save_state()
            candidates = [
                t for t in queue
                if self.state["tasks"][t["id"]].get("status") in (PENDING, RETRYABLE)
                and self._ready(t, by_id)
            ]
            if not candidates:
                if all(self.state["tasks"][t["id"]].get("status") in (COMPLETED, SKIPPED, BLOCKED, SKIPPED_STOP, AWAITING_HUMAN)
                       for t in queue):
                    self.set_status("__orchestrator__", COMPLETED, ended_utc=utcnow())
                    self.hb_stop.set()
                    self._write_heartbeat()
                    return 0
                time.sleep(10)
                continue
            candidates.sort(key=lambda t: (int(t.get("priority", 100)), t["id"]))
            self._run_task(candidates[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    return Runner(args.base).loop()


if __name__ == "__main__":
    raise SystemExit(main())
