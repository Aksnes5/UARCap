"""Non-destructive BN Server & Local CPU Preflight Checker (Pilot A0).

NEVER kills GPU processes, NEVER modifies /media/bn read-only assets, NEVER logs secrets/HF tokens.
Checks:
  1. Configurable BN paths (VideoUFO, evcap_external, canonical checkpoint SHA-256, Vicuna, feature dir, gpu.lock)
  2. Writable NVMe disk space (>= min_free_disk_gb)
  3. Non-destructive nvidia-smi GPU idle & driver query (on Linux)
  4. Optional DREAM-1K HF gated access check (token read from env only)
  5. Emits preflight_receipt.json for auditability
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uarcap_q2.common.gpu_lock import query_nvidia_smi_compute_pids
from uarcap_q2.common.io_utils import atomic_write_json, sha256_file
from uarcap_q2.datasets.carebench import check_carebench_hf_access
from uarcap_q2.datasets.dream1k import check_dream1k_hf_access


def run_preflight(
    paths_yaml: Path,
    out_receipt: Path | None = None,
    check_network_hf: bool = False,
    verify_large_checkpoint_sha: bool = False,
    e2_dataset: str = "carebench",
) -> dict[str, Any]:
    cfg = yaml.safe_load(paths_yaml.read_text(encoding="utf-8"))
    hist = cfg.get("historical_frozen_assets", {})
    host = cfg.get("host", {})
    q2w = cfg.get("q2_writable_workspace", {})

    path_checks: dict[str, Any] = {}
    for key in (
        "videoufo_dataset_root",
        "videoufo_research_root",
        "original_evcap_root",
        "frozen_videollama_python",
        "canonical_checkpoint",
        "vicuna_7b_dir",
        "video_feature_dir",
        "torch_home",
        "hf_hub_cache",
    ):
        raw = hist.get(key, "")
        p = Path(raw) if raw else None
        exists = bool(p and p.exists())
        entry: dict[str, Any] = {"path": raw, "exists": exists}
        if key == "canonical_checkpoint" and exists and verify_large_checkpoint_sha:
            actual_sha = sha256_file(p)  # type: ignore[arg-type]
            expected = hist.get("canonical_checkpoint_sha256", "")
            entry["sha256"] = actual_sha
            entry["sha256_match"] = actual_sha == expected
        path_checks[key] = entry

    lock_path = Path(host.get("shared_gpu_lock", ""))
    lock_check = {"path": str(lock_path), "exists": lock_path.exists()}

    # Disk space check on q2_root (or current filesystem fallback on Windows dry preflight)
    q2_root = Path(q2w.get("q2_root", "."))
    probe_dir = q2_root if q2_root.exists() else Path.cwd()
    du = shutil.disk_usage(probe_dir)
    free_gb = round(du.free / (1024**3), 2)
    min_gb = float(host.get("min_free_disk_gb", 150))
    disk_check = {"probed_path": str(probe_dir), "free_gb": free_gb, "min_required_gb": min_gb, "ok": free_gb >= min_gb}

    # Non-destructive nvidia-smi query
    if sys.platform.startswith("linux"):
        smi_ok, active_apps, smi_msg = query_nvidia_smi_compute_pids(host.get("gpu_uuid"))
    else:
        smi_ok, active_apps, smi_msg = False, [], "non-linux local workstation (GPU check deferred to BN)"

    e2_key = (e2_dataset or "carebench").strip().lower().replace("-", "")
    if check_network_hf:
        hf_check = check_carebench_hf_access() if e2_key == "carebench" else check_dream1k_hf_access()
    else:
        hf_check = {"skipped": True}

    all_hist_present = all(v["exists"] for v in path_checks.values())
    bn_ready = bool(sys.platform.startswith("linux") and all_hist_present and lock_check["exists"] and disk_check["ok"] and smi_ok and not active_apps)

    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "paths_yaml": str(paths_yaml),
        "e2_dataset_selected": "CaReBench" if e2_key == "carebench" else "DREAM-1K",
        "historical_assets": path_checks,
        "shared_gpu_lock": lock_check,
        "disk": disk_check,
        "gpu_non_destructive_probe": {"ok": smi_ok, "active_compute_apps": active_apps, "message": smi_msg},
        "e2_hf_access": hf_check,
        "carebench_hf_access": hf_check if e2_key == "carebench" else {"skipped": True},
        "dream1k_hf_access": hf_check if e2_key == "dream1k" else {"skipped": True},
        "bn_live_ready": bn_ready,
    }
    if out_receipt is not None:
        atomic_write_json(out_receipt, report)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Non-destructive BN / CPU Preflight Check (Pilot A0)")
    ap.add_argument("--paths-config", type=Path, required=True)
    ap.add_argument("--out-receipt", type=Path, default=None)
    ap.add_argument("--e2-dataset", choices=["carebench", "dream1k"], default="carebench")
    ap.add_argument("--check-hf-network", action="store_true")
    ap.add_argument("--verify-ckpt-sha", action="store_true")
    args = ap.parse_args()

    rep = run_preflight(
        paths_yaml=args.paths_config,
        out_receipt=args.out_receipt,
        check_network_hf=args.check_hf_network,
        verify_large_checkpoint_sha=args.verify_ckpt_sha,
        e2_dataset=args.e2_dataset,
    )
    print(yaml.safe_dump(rep, sort_keys=False, allow_unicode=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
