#!/usr/bin/env python3
"""Independent overnight supervisor (local only; no Claude API / network dependency).

- Lightweight heartbeat every 60 s.
- Full health check every 30 min for the whole 12-14 h window.
- Detects: dead orchestrator, stalled current task, missing results, OOM/parse errors,
  stagnant downloads, blocked dependency chains, GPU idle with ready work.
- Auto-recovery is deliberately conservative: restart a dead orchestrator; terminate a
  task only after >=2 consecutive checks with no log growth, no progress, and idle GPU.
  Never touches foreign processes, locks, network, remote-desktop or watchdog services.
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
import time

PENDING = "PENDING"
RUNNING = "RUNNING"
COMPLETED = "COMPLETED"
RETRYABLE = "RETRYABLE_FAILURE"
BLOCKED = "BLOCKED"
SKIPPED = "SKIPPED_STOPPED"

STALL_CHECKS = 2          # ~60 min with zero log growth / progress / GPU work
ORCH_RESTART_COOLDOWN = 600


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def atomic_write(p: Path, payload) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def append_jsonl(p: Path, payload) -> None:
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def file_state(p: Path) -> dict:
    try:
        st = p.stat()
        return {"exists": True, "size": st.st_size, "mtime": st.st_mtime}
    except Exception:
        return {"exists": False, "size": 0, "mtime": 0}


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def cmdline(pid: int | None) -> str:
    if not pid:
        return ""
    try:
        return Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
    except Exception:
        return ""


def gpu() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip().splitlines()[0]
        u, used, total = [int(x.strip()) for x in out.split(",")]
        return {"utilization_pct": u, "memory_used_mib": used, "memory_total_mib": total}
    except Exception:
        return {"utilization_pct": None, "memory_used_mib": None, "memory_total_mib": None}


def proc_table() -> list[dict]:
    rows = []
    try:
        for d in Path("/proc").iterdir():
            if not d.name.isdigit():
                continue
            try:
                stat = (d / "stat").read_text().split()
                state = stat[2]
                rows.append({"pid": int(d.name), "state": state, "cmd": cmdline(int(d.name))[:400]})
            except Exception:
                continue
    except Exception:
        pass
    return rows


class Supervisor:
    def __init__(self, base: Path, check_interval: int = 1800, heartbeat: int = 60) -> None:
        self.base = base
        self.check_interval = check_interval
        self.heartbeat_interval = heartbeat
        self.state_path = base / "task_state.json"
        self.heartbeat_path = base / "heartbeat.json"
        self.download_state = base / "download_state.json"
        self.stop_path = base / "STOP_REQUESTED"
        self.orch_pid_path = base / "orchestrator.pid"
        self.supervisor_state_path = base / "supervisor_state.json"
        self.log_path = base / "supervisor.log"
        self.health_path = base / "health_checks.jsonl"
        self.recovery_path = base / "recovery_actions.jsonl"
        self.pending_path = base / "pending_repairs.json"
        self.shb_path = base / "supervisor_heartbeat.json"
        self.state = read_json(self.supervisor_state_path, {"last_check_utc": None, "stall": {}, "restarts": [], "kills": {}})
        self._log(f"supervisor started pid={os.getpid()} interval={check_interval}s base={base}")

    def _log(self, msg: str) -> None:
        line = f"{utcnow()} {msg}\n"
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def _recover(self, action: str, detail: dict) -> None:
        rec = {"utc": utcnow(), "action": action, **detail}
        append_jsonl(self.recovery_path, rec)
        self._log(f"RECOVERY {action} {json.dumps(detail, ensure_ascii=False)[:500]}")

    def _pending(self, items: list[dict]) -> None:
        atomic_write(self.pending_path, {"updated_utc": utcnow(), "items": items})

    def heartbeat(self) -> None:
        state = read_json(self.state_path, {"tasks": {}})
        counts: dict[str, int] = {}
        for st in state.get("tasks", {}).values():
            counts[st.get("status", PENDING)] = counts.get(st.get("status", PENDING), 0) + 1
        hb = read_json(self.heartbeat_path, {})
        atomic_write(self.shb_path, {
            "updated_utc": utcnow(), "supervisor_pid": os.getpid(),
            "orchestrator_alive": pid_alive(int(read_json(self.orch_pid_path, {}).get("pid", 0) or 0)),
            "heartbeat": hb, "task_counts": counts, "gpu": gpu(),
            "stop_requested": self.stop_path.exists(),
            "network_download": read_json(self.download_state, None),
        })

    def orchestrator_pid(self) -> int:
        data = read_json(self.orch_pid_path, {})
        return int(data.get("pid", 0) or 0)

    def start_orchestrator(self) -> None:
        log = self.base / "logs" / "orchestrator_restart.log"
        p = subprocess.Popen(
            ["/bin/bash", "-lc", f"cd {self.base} && exec /usr/bin/python3 run_overnight.py --base {self.base}"],
            stdout=log.open("ab"), stderr=subprocess.STDOUT, start_new_session=True,
        )
        atomic_write(self.orch_pid_path, {"pid": p.pid, "started_utc": utcnow()})
        self._recover("restart_orchestrator", {"new_pid": p.pid})

    def check(self) -> None:
        now = time.time()
        state = read_json(self.state_path, {"tasks": {}})
        hb = read_json(self.heartbeat_path, {})
        dstate = read_json(self.download_state, None)
        tasks = state.get("tasks", {})
        health = {"utc": utcnow(), "gpu": gpu(), "orchestrator_pid": self.orchestrator_pid(),
                  "orchestrator_alive": pid_alive(self.orchestrator_pid()),
                  "current_task": hb.get("current_task"), "task_counts": {}}
        anomalies: list[str] = []
        repairs: list[dict] = []
        for st in tasks.values():
            health["task_counts"][st.get("status", PENDING)] = health["task_counts"].get(st.get("status", PENDING), 0) + 1

        # 1) Orchestrator dead while work remains
        if not health["orchestrator_alive"] and not self.stop_path.exists():
            ready_or_pending = [t for t, st in tasks.items() if st.get("status") in (PENDING, RETRYABLE, RUNNING)]
            if ready_or_pending:
                last = self.state.get("last_restart_ts", 0)
                if now - last >= ORCH_RESTART_COOLDOWN:
                    self.state["last_restart_ts"] = now
                    self.start_orchestrator()
                    repairs.append({"issue": "orchestrator_dead", "action": "restart_orchestrator"})
                else:
                    anomalies.append("orchestrator_dead_cooldown")

        # 2) Current task stall detection
        tid = hb.get("current_task")
        pid = hb.get("current_task_pid")
        if tid and pid and pid_alive(pid):
            progress = read_json(self.base / "progress" / f"{tid}.json", {})
            logf = self.base / "logs" / f"{tid}.log"
            fs = file_state(logf)
            prev = self.state.setdefault("stall", {}).get(tid, {})
            unchanged = prev.get("log_size") == fs["size"] and prev.get("progress_sig") == self._sig(progress)
            stalled_checks = int(prev.get("stalled_checks", 0)) + (1 if unchanged else 0)
            self.state["stall"][tid] = {"log_size": fs["size"], "progress_sig": self._sig(progress),
                                        "stalled_checks": stalled_checks, "pid": pid, "ts": now,
                                        "progress": progress}
            g = health["gpu"]
            idle = (g.get("utilization_pct") is not None and g["utilization_pct"] < 5
                    and (g.get("memory_used_mib") or 0) < 2048)
            if stalled_checks >= STALL_CHECKS and idle:
                c = cmdline(pid)
                if "/overnight/" in c or "uarcap_q2_carebench" in c or str(self.base) in c:
                    kills = int(self.state.setdefault("kills", {}).get(tid, 0))
                    if kills < 1:
                        try:
                            os.killpg(pid, signal.SIGTERM)
                            self.state["kills"][tid] = kills + 1
                            self._recover("terminate_stalled_task", {"task": tid, "pid": pid,
                                                                     "stalled_checks": stalled_checks, "cmd": c[:200]})
                            repairs.append({"issue": "task_stalled", "task": tid, "action": "terminated_for_retry"})
                        except Exception as exc:
                            anomalies.append(f"terminate_failed:{exc}")
                    else:
                        anomalies.append(f"task_stalled_no_more_auto_kill:{tid}")
                else:
                    anomalies.append(f"task_stalled_foreign_cmd_not_killed:{tid}")
            elif stalled_checks >= STALL_CHECKS:
                anomalies.append(f"task_no_log_growth_but_gpu_busy:{tid}")
            health["task_progress"] = progress
        elif tid and pid and not pid_alive(pid):
            anomalies.append(f"current_task_pid_dead:{tid}")

        # 3) OOM / repeated parse failures in current log tail
        if tid:
            try:
                txt = (self.base / "logs" / f"{tid}.log").read_text(encoding="utf-8", errors="replace")[-8000:]
                for pat, name in (("CUDA out of memory", "cuda_oom"), ("OutOfMemoryError", "cuda_oom"),
                                  ("PARSE_FAILED", "parse_failed_present"), ("HARD_BLOCK", "hard_block_marker")):
                    if pat in txt:
                        anomalies.append(f"{name}:{tid}")
            except Exception:
                pass

        # 4) Download stagnation
        if dstate:
            f = dstate.get("file_state") or {}
            prev_dl = self.state.setdefault("download_prev", {})
            unchanged = prev_dl.get("size") == f.get("size")
            self.state["download_prev"] = {"size": f.get("size"), "ts": now}
            if unchanged and dstate.get("status") in ("RUNNING", "RETRYING"):
                anomalies.append("download_no_growth")
            if dstate.get("status") == "FAILED":
                anomalies.append("download_failed")

        # 5) Blocked dependency chains
        for t in tasks:
            st = tasks[t]
            if st.get("status") == BLOCKED:
                anomalies.append(f"task_blocked:{t}")
        for t, st in tasks.items():
            if st.get("status") in (PENDING, RETRYABLE) and st.get("depends_on_blocked"):
                anomalies.append(f"dependency_blocked:{t}")

        # 6) GPU idle with ready work
        g = health["gpu"]
        if (g.get("utilization_pct") is not None and g["utilization_pct"] < 5
                and not (tid and pid_alive(pid)) and not self.stop_path.exists()):
            ready = [t for t, st in tasks.items() if st.get("status") in (PENDING, RETRYABLE)]
            if ready:
                anomalies.append(f"gpu_idle_with_ready_tasks:{len(ready)}")

        health["anomalies"] = anomalies
        health["recovery_actions"] = repairs
        append_jsonl(self.health_path, health)
        self._pending([{"utc": utcnow(), "issue": a, "requires_code_fix": True} for a in anomalies if "oom" in a or "parse" in a or "blocked" in a or "dead" in a or "stalled" in a])
        self.state["last_check_utc"] = utcnow()
        atomic_write(self.supervisor_state_path, self.state)
        self._log(f"check anomalies={anomalies[:8]} repairs={repairs}")

    @staticmethod
    def _sig(obj) -> str:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False)[:2000] if obj else ""

    def run(self, duration_hours: float = 14.0) -> int:
        end = time.time() + duration_hours * 3600
        next_check = 0.0
        while time.time() < end:
            if self.stop_path.exists():
                self._log("STOP_REQUESTED observed; supervisor exiting")
                return 0
            try:
                self.heartbeat()
            except Exception as exc:
                self._log(f"heartbeat_error {exc}")
            if time.time() >= next_check:
                try:
                    self.check()
                except Exception as exc:
                    self._log(f"check_error {exc}")
                next_check = time.time() + self.check_interval
            time.sleep(self.heartbeat_interval)
        self._log("supervisor duration reached; exiting")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--check-interval", type=int, default=1800)
    ap.add_argument("--heartbeat", type=int, default=60)
    ap.add_argument("--duration-hours", type=float, default=14.0)
    args = ap.parse_args()
    return Supervisor(args.base, args.check_interval, args.heartbeat).run(args.duration_hours)


if __name__ == "__main__":
    raise SystemExit(main())
