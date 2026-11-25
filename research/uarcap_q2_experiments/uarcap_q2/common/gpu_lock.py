"""Cooperative shared GPU lock compatible with BN server (/home/bn/videoufo_research/gpu.lock).

Safety guarantees:
  - Uses non-blocking OS file locking (fcntl.flock on Linux, msvcrt on Windows for CPU testing).
  - Queries nvidia-smi on Linux real-GPU mode to refuse acquisition if any foreign GPU process is running.
  - NEVER kills foreign GPU processes and NEVER deletes or overwrites foreign lock files.
  - Writes own run metadata to an isolated sidecar file only while holding the lock, and removes only its own sidecar.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
from typing import Iterator

from .io_utils import atomic_write_json

# `fcntl.flock` is per open-file-description, so a second open() + LOCK_EX in the
# same process conflicts with itself. Nested GPU work (e.g. the orchestrator holding
# the lock while an isolated X-CLIP subprocess wrapper acquires it again) must be
# re-entrant; this registry tracks locks held by *this* process.
_HELD_LOCKS: dict[str, dict[str, object]] = {}
_HELD_LOCKS_GUARD = threading.Lock()


class GPULockBusyError(RuntimeError):
    """Raised when the shared GPU lock is already held or foreign GPU processes are active."""


@dataclass
class GPULockConfig:
    lock_path: Path
    run_id: str
    gpu_uuid: str | None = None
    require_nvidia_smi_idle: bool = True
    sidecar_dir: Path | None = None


def query_nvidia_smi_compute_pids(gpu_uuid: str | None = None) -> tuple[bool, list[str], str]:
    """Non-destructive read-only nvidia-smi check. Never kills processes."""
    cmd = ["nvidia-smi"]
    if gpu_uuid:
        cmd.extend(["-i", gpu_uuid])
    cmd.extend(["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=10)
    except FileNotFoundError:
        return False, [], "nvidia-smi not found"
    except subprocess.CalledProcessError as exc:
        return False, [], f"nvidia-smi error (rc={exc.returncode}): {exc.output.strip()}"
    except Exception as exc:
        return False, [], f"nvidia-smi failed: {exc}"
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return True, lines, "ok"


def _acquire_os_lock(file_obj) -> None:
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        import fcntl

        try:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GPULockBusyError("Shared GPU lock is currently held by another process (fcntl.flock).") from exc
    elif os.name == "nt":
        import msvcrt

        # Ensure at least 1 byte exists for Windows byte-range lock without truncating foreign data
        file_obj.seek(0, os.SEEK_END)
        if file_obj.tell() == 0:
            file_obj.write(b" ")
            file_obj.flush()
        file_obj.seek(0)
        try:
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise GPULockBusyError("Shared GPU lock is currently held by another process (msvcrt).") from exc
    else:
        raise RuntimeError(f"Unsupported platform for file lock: {sys.platform}")


def _release_os_lock(file_obj) -> None:
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        import fcntl

        fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
    elif os.name == "nt":
        import msvcrt

        file_obj.seek(0)
        try:
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass


@contextmanager
def acquire_cooperative_gpu_lock(cfg: GPULockConfig) -> Iterator[dict[str, object]]:
    """Acquire shared BN GPU lock non-destructively. Never deletes cfg.lock_path."""
    lock_path = Path(cfg.lock_path).resolve()
    key = str(lock_path)

    # Re-entrant acquisition: this process already holds the OS lock.
    with _HELD_LOCKS_GUARD:
        held = _HELD_LOCKS.get(key)
        if held is not None:
            held["count"] = int(held["count"]) + 1
            nested_meta = dict(held["meta"])
            nested_meta["reentrant"] = True
            nested_meta["nesting"] = held["count"]
            nested_meta["nested_run_id"] = cfg.run_id
            nesting = int(held["count"])
        else:
            nested_meta = None
            nesting = 0
    if nested_meta is not None:
        try:
            yield nested_meta
        finally:
            with _HELD_LOCKS_GUARD:
                cur = _HELD_LOCKS.get(key)
                if cur is not None:
                    cur["count"] = max(0, int(cur["count"]) - 1)
                    if int(cur["count"]) == 0:
                        _HELD_LOCKS.pop(key, None)
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # Open in a+b so we never truncate or recreate an existing lock inode!
    mode = "r+b" if lock_path.exists() else "a+b"
    fh = lock_path.open(mode)
    sidecar: Path | None = None
    locked = False
    try:
        _acquire_os_lock(fh)
        locked = True

        if cfg.require_nvidia_smi_idle:
            ok, active_apps, msg = query_nvidia_smi_compute_pids(cfg.gpu_uuid)
            if not ok:
                raise GPULockBusyError(f"Cannot confirm GPU idle state via nvidia-smi: {msg}")
            if active_apps:
                raise GPULockBusyError(
                    "Refusing to run: active GPU compute process(es) detected (NEVER auto-killed): "
                    + "; ".join(active_apps)
                )

        meta = {
            "run_id": cfg.run_id,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "gpu_uuid": cfg.gpu_uuid,
            "lock_path": str(lock_path),
            "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if cfg.sidecar_dir is not None:
            cfg.sidecar_dir.mkdir(parents=True, exist_ok=True)
            sidecar = cfg.sidecar_dir / f"gpu_lock_holder_{cfg.run_id}.json"
            atomic_write_json(sidecar, meta)

        with _HELD_LOCKS_GUARD:
            _HELD_LOCKS[key] = {"count": 1, "fh": fh, "sidecar": sidecar, "meta": meta}
        try:
            yield meta
        finally:
            with _HELD_LOCKS_GUARD:
                _HELD_LOCKS.pop(key, None)
    finally:
        if sidecar is not None and sidecar.exists():
            try:
                sidecar.unlink()
            except OSError:
                pass
        if locked:
            try:
                _release_os_lock(fh)
            except OSError:
                pass
        fh.close()
