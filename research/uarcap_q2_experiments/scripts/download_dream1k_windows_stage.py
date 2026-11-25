r"""Windows Local DREAM-1K Downloader, Verifier, and BN Offline Staging Packager.

Usage (after accepting gated access at https://huggingface.co/datasets/omni-research/DREAM-1K):
  PowerShell:
    $env:HF_TOKEN = "<your_hf_read_token>"
    python scripts/download_dream1k_windows_stage.py --stage-dir D:\UARCap_LocalData\dream1k_staging

What it does (strictly outside Git repo; NEVER logs or writes tokens):
  1. Verifies HF login (from HF_TOKEN env var or ~/.cache/huggingface/token) and gated DREAM-1K access.
  2. Pins official HF commit revision (default: 99da78dc8df846a14e731748b8e4edac4ef9f184).
  3. Downloads `json/metadata.json` and `video/video.zip` (~3.59 GiB) with HTTP Range resume.
  4. Verifies exact byte size (3,849,677,473) & computes SHA-256 checksums.
  5. Safely extracts into local verification workspace and validates metadata-to-video correspondence.
  6. Splits `video.zip` into verified 512 MiB chunks + writes `video.zip.transfer_manifest.json` and
     `dream1k_bn_offline_import_manifest.json` ready for BN `stage_assets.py prepare --transfer-manifest`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uarcap_q2.common.io_utils import atomic_write_json, sha256_file
from uarcap_q2.datasets.asset_transfer import download_with_resume, split_file_into_chunks
from uarcap_q2.datasets.data_preparer import (
    normalize_dataset_layout,
    safe_extract_zip,
    validate_dataset_readiness,
)
from uarcap_q2.datasets.dream1k import DREAM1K_EXPECTED_VIDEO_ZIP_BYTES, DREAM1K_HF_REPO

PINNED_DREAM1K_REVISION = "99da78dc8df846a14e731748b8e4edac4ef9f184"


def _resolve_hf_token() -> str:
    tok = os.environ.get("HF_TOKEN", "").strip()
    if tok:
        return tok
    for cand in (
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ):
        if cand.is_file():
            val = cand.read_text(encoding="utf-8").strip()
            if val:
                return val
    return ""


def _check_gated_access(token: str, revision: str) -> tuple[bool, str]:
    if not token:
        return False, "No HF token found in HF_TOKEN env var or ~/.cache/huggingface/token."
    url = f"https://huggingface.co/datasets/{DREAM1K_HF_REPO}/resolve/{revision}/json/metadata.json"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return (200 <= resp.status < 400), f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return False, "HTTP 401 Unauthorized: HF token is invalid or expired."
        if exc.code == 403:
            return (
                False,
                "HTTP 403 Forbidden: Your HF account has not yet been granted access to gated dataset "
                f"https://huggingface.co/datasets/{DREAM1K_HF_REPO}. Open that URL in browser and click 'Agree and access repository'.",
            )
        return False, f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:
        return False, f"Network error checking HF gated access: {exc}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download, verify, and split DREAM-1K locally on Windows for BN offline import")
    ap.add_argument("--stage-dir", type=Path, default=Path(r"D:\UARCap_LocalData\dream1k_staging"))
    ap.add_argument("--revision", default=PINNED_DREAM1K_REVISION)
    ap.add_argument("--chunk-mb", type=int, default=512)
    ap.add_argument("--check-only", action="store_true", help="Only verify HF login & gated access without downloading")
    ap.add_argument("--skip-extract-verify", action="store_true", help="Skip local zip extraction & correspondence check")
    args = ap.parse_args(argv)

    token = _resolve_hf_token()
    ok, msg = _check_gated_access(token, args.revision)
    if not ok:
        print(f"[DREAM-1K ACCESS CHECK FAILED] {msg}", file=sys.stderr)
        print("Next steps:", file=sys.stderr)
        print(f"  1. Visit https://huggingface.co/datasets/{DREAM1K_HF_REPO} while signed into Hugging Face and accept terms.", file=sys.stderr)
        print("  2. Create a Read token at https://huggingface.co/settings/tokens", file=sys.stderr)
        print('  3. Run in PowerShell: $env:HF_TOKEN="<your_token>"; python scripts/download_dream1k_windows_stage.py', file=sys.stderr)
        return 3

    print(f"[DREAM-1K] Gated access verified ({msg}) at pinned revision {args.revision}.")
    if args.check_only:
        return 0

    # Ensure download_with_resume sees token via env (never logged or written to disk)
    os.environ["HF_TOKEN"] = token
    stage = args.stage_dir
    raw_root = stage / "raw_data"
    chunks_dir = stage / "bn_transfer_chunks"
    meta_dest = raw_root / "dream1k" / "json" / "metadata.json"
    zip_dest = raw_root / "dream1k" / "video" / "video.zip"

    meta_url = f"https://huggingface.co/datasets/{DREAM1K_HF_REPO}/resolve/{args.revision}/json/metadata.json"
    zip_url = f"https://huggingface.co/datasets/{DREAM1K_HF_REPO}/resolve/{args.revision}/video/video.zip"

    print(f"[DREAM-1K] Downloading metadata.json -> {meta_dest}")
    meta_res = download_with_resume(meta_url, meta_dest, timeout_sec=60)
    print(f"[DREAM-1K] Downloading video.zip (3.59 GiB, resumable) -> {zip_dest}")
    zip_res = download_with_resume(
        zip_url, zip_dest, expected_bytes=DREAM1K_EXPECTED_VIDEO_ZIP_BYTES, timeout_sec=120
    )

    readiness = None
    if not args.skip_extract_verify:
        print("[DREAM-1K] Safely extracting and validating metadata-to-video correspondence locally...")
        safe_extract_zip(zip_dest, raw_root / "dream1k" / "video")
        normalize_dataset_layout("dream1k", raw_root)
        readiness = validate_dataset_readiness("dream1k", raw_root)
        if not readiness.get("ready"):
            print(f"[DREAM-1K] Readiness check FAILED: {json.dumps(readiness, indent=2)}", file=sys.stderr)
            return 2

    print(f"[DREAM-1K] Splitting video.zip into {args.chunk_mb} MiB SHA-256 verified transfer chunks -> {chunks_dir}")
    transfer_manifest = split_file_into_chunks(zip_dest, chunks_dir, chunk_bytes=args.chunk_mb * 1024 * 1024)
    # Copy metadata.json alongside chunks for convenient single-folder transfer to BN
    meta_copy = chunks_dir / "metadata.json"
    meta_copy.write_bytes(meta_dest.read_bytes())

    import_manifest = {
        "schema_version": "uarcap-q2-dream1k-bn-offline-import-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hf_repo": DREAM1K_HF_REPO,
        "pinned_revision": args.revision,
        "metadata_file": {"name": "metadata.json", "bytes": meta_res["bytes"], "sha256": sha256_file(meta_copy)},
        "video_zip": {"name": "video.zip", "bytes": zip_res["bytes"], "sha256": zip_res["sha256"]},
        "transfer_manifest_file": transfer_manifest.name,
        "local_readiness_verified": readiness,
        "bn_import_commands": [
            "mkdir -p /home/bn/uarcap_q2_runtime/data/dream1k/json /home/bn/uarcap_q2_runtime/staging/dream1k",
            "cp /home/bn/uarcap_q2_runtime/staging/dream1k/metadata.json /home/bn/uarcap_q2_runtime/data/dream1k/json/metadata.json",
            "PYTHON_BIN=/home/bn/anaconda3/envs/videollama2/bin/python "
            "/home/bn/anaconda3/envs/videollama2/bin/python deployment/stage_assets.py prepare "
            "--data-root /home/bn/uarcap_q2_runtime/data --dataset dream1k "
            "--transfer-manifest /home/bn/uarcap_q2_runtime/staging/dream1k/video.zip.transfer_manifest.json",
        ],
    }
    imp_path = atomic_write_json(chunks_dir / "dream1k_bn_offline_import_manifest.json", import_manifest)
    print(f"[DREAM-1K] READY FOR BN TRANSFER! Upload directory: {chunks_dir}")
    print(f"[DREAM-1K] Offline import manifest: {imp_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
