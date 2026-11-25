"""Cross-environment subprocess launcher, shared GPU lock helper, and VRAM cleanup utilities.

Allows the lightweight CPU orchestration environment (`uarcap_q2_cpu`) to invoke GPU tasks inside
strictly isolated model environments (`frozen_videollama_python`, `uarcap_q2_modern_vlm`,
`uarcap_q2_evaluator`) without installing CUDA in `uarcap_q2_cpu` and without touching the frozen
Transformers 4.28.0 Video-LLaMA environment.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
import subprocess
from typing import Any, Callable

from .gpu_lock import GPULockConfig, acquire_cooperative_gpu_lock

SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]


def release_gpu_memory() -> dict[str, Any]:
    """Explicitly release Python/PyTorch CUDA cache between model tasks (safe no-op on CPU)."""
    gc.collect()
    info: dict[str, Any] = {"gc_collected": True, "cuda_available": False}
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["allocated_before_bytes"] = int(torch.cuda.memory_allocated())
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            info["allocated_after_bytes"] = int(torch.cuda.memory_allocated())
    except ImportError:
        pass
    return info


def build_isolated_python_argv(
    script_or_module: str,
    script_args: list[str],
    python_executable: str | None = None,
    conda_env: str | None = None,
    as_module: bool = False,
) -> list[str]:
    """Build argv targeting either an explicit venv/conda python binary or `conda run -n <env>`."""
    flag = ["-m", script_or_module] if as_module else [script_or_module]
    if python_executable:
        return [python_executable, *flag, *script_args]
    if conda_env:
        return ["conda", "run", "--no-capture-output", "-n", conda_env, "python", *flag, *script_args]
    raise ValueError("Isolated invocation requires either python_executable or conda_env")


def run_isolated_gpu_subprocess(
    script_or_module: str,
    script_args: list[str],
    *,
    python_executable: str | None = None,
    conda_env: str | None = None,
    as_module: bool = False,
    cwd: Path | str | None = None,
    env_overrides: dict[str, str] | None = None,
    gpu_lock_cfg: GPULockConfig | None = None,
    timeout_sec: int = 3600,
    subprocess_runner: SubprocessRunner | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a model worker in an isolated environment under optional cooperative BN GPU lock + VRAM release."""
    argv = build_isolated_python_argv(
        script_or_module=script_or_module,
        script_args=script_args,
        python_executable=python_executable,
        conda_env=conda_env,
        as_module=as_module,
    )
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    runner = subprocess_runner or subprocess.run

    def _exec() -> subprocess.CompletedProcess[str]:
        try:
            return runner(
                argv,
                cwd=str(cwd) if cwd else None,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        finally:
            release_gpu_memory()

    if gpu_lock_cfg is not None:
        with acquire_cooperative_gpu_lock(gpu_lock_cfg):
            return _exec()
    return _exec()
