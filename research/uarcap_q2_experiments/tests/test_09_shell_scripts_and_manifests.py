"""Tests 09: Linux bash script syntax check (bash -n), YAML config validation, and CLI preflight/gate smoke."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

EXP_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = EXP_ROOT / "scripts"
CONFIGS_DIR = EXP_ROOT / "configs"


def _find_bash() -> str | None:
    cand = Path(r"C:\Program Files\Git\bin\bash.exe")
    if cand.is_file():
        return str(cand)
    return shutil.which("bash")


def test_all_linux_bash_scripts_pass_syntax_check() -> None:
    bash_exe = _find_bash()
    if not bash_exe:
        pytest.skip("bash not available on this host")
    sh_files = sorted(SCRIPTS_DIR.glob("*.sh"))
    assert len(sh_files) >= 8
    for sf in sh_files:
        # Ensure Unix LF line endings (no CRLF breaking Ubuntu #!/usr/bin/env bash!)
        raw = sf.read_bytes()
        assert b"\r\n" not in raw, f"CRLF detected in Linux script {sf.name}"
        proc = subprocess.run([bash_exe, "-n", str(sf)], capture_output=True, text=True)
        assert proc.returncode == 0, f"bash -n failed for {sf.name}: {proc.stderr}"


def test_all_yaml_configs_parse_cleanly() -> None:
    yaml_files = sorted(CONFIGS_DIR.rglob("*.yaml"))
    assert len(yaml_files) >= 7
    for yf in yaml_files:
        parsed = yaml.safe_load(yf.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict) and len(parsed) > 0


def test_cli_authorization_gate_blocks_formal_b_and_preflight_runs(tmp_path: Path) -> None:
    gate_cli = EXP_ROOT / "deployment" / "authorization_gate.py"
    p_block = subprocess.run([sys.executable, str(gate_cli), "--stage", "FORMAL_B"], capture_output=True, text=True)
    assert p_block.returncode == 3 and "AUTHORIZATION_BLOCKED" in p_block.stderr

    pre_cli = EXP_ROOT / "deployment" / "bn_preflight_check.py"
    out_rec = tmp_path / "preflight.json"
    p_pre = subprocess.run(
        [sys.executable, str(pre_cli), "--paths-config", str(CONFIGS_DIR / "bn_paths.example.yaml"), "--out-receipt", str(out_rec)],
        capture_output=True,
        text=True,
    )
    assert p_pre.returncode == 0 and out_rec.is_file()
