"""Public (ungated, MIT) CaReBench download, SHA-256 verification, and optional Route-2 chunk staging helper.

Supports BOTH direct BN server public download (with optional HF mirror via --hf-endpoint / HF_ENDPOINT)
AND Windows local staging + SHA-256 chunk splitting for offline transfer to BN.

Safety guarantees:
  - NO Hugging Face login / HF_TOKEN required (`MCG-NJU/CaReBench` is `gated=False`, `MIT` license).
  - Refuses full video download (3.65 GiB) unless `--confirm-download` is explicitly supplied!
  - NEVER commits raw videos to Git; keep `--stage-dir` outside the repository.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from uarcap_q2.common.io_utils import atomic_write_json, sha256_file
from uarcap_q2.datasets.asset_transfer import download_with_resume, split_file_into_chunks
from uarcap_q2.datasets.carebench import (
    CAREBENCH_EXPECTED_VIDEO_ZIP_BYTES,
    CAREBENCH_HF_REPO,
    check_carebench_hf_access,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage public CaReBench metadata (~3.7 MB) and videos.zip (3.65 GiB)")
    ap.add_argument(
        "--stage-dir",
        type=Path,
        required=True,
        help="Target staging directory OUTSIDE the Git repo (e.g., D:/UARCap_LocalData/carebench_staging or /home/bn/uarcap_q2_runtime/staging/carebench)",
    )
    ap.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        help="HF base URL or public mirror (e.g., https://hf-mirror.com; no token needed for CaReBench)",
    )
    ap.add_argument("--metadata-only", action="store_true", help="Download only json/metadata.json (~3.7 MB)")
    ap.add_argument("--confirm-download", action="store_true", help="Required safety flag before downloading videos.zip (3.65 GiB)")
    ap.add_argument("--skip-chunk-split", action="store_true", help="Skip 512 MiB Route-2 chunk splitting after downloading videos.zip")
    ap.add_argument("--chunk-mb", type=int, default=512)
    ap.add_argument("--dry-run", action="store_true", help="Print exact URLs, target paths, and BN import commands without network I/O")
    args = ap.parse_args(argv)

    base = args.hf_endpoint.rstrip("/")
    meta_url = f"{base}/datasets/{CAREBENCH_HF_REPO}/resolve/main/json/metadata.json"
    zip_url = f"{base}/datasets/{CAREBENCH_HF_REPO}/resolve/main/videos/videos.zip"

    stage_dir = args.stage_dir.resolve()
    meta_dst = stage_dir / "json" / "metadata.json"
    zip_dst = stage_dir / "videos" / "videos.zip"
    chunks_dir = stage_dir / "bn_transfer_chunks"

    plan: dict[str, Any] = {
        "dataset": "CaReBench",
        "repo": CAREBENCH_HF_REPO,
        "gated": False,
        "token_required": False,
        "hf_endpoint": base,
        "metadata_url": meta_url,
        "videos_zip_url": zip_url,
        "expected_videos_zip_bytes": CAREBENCH_EXPECTED_VIDEO_ZIP_BYTES,
        "stage_dir": str(stage_dir),
        "metadata_dest": str(meta_dst),
        "videos_zip_dest": str(zip_dst),
        "bn_prepare_direct_cmd": (
            "/home/bn/anaconda3/envs/videollama2/bin/python deployment/stage_assets.py prepare "
            "--data-root /home/bn/uarcap_q2_runtime/data --dataset carebench "
            "--allow-download --auth-receipt /home/bn/uarcap_q2_runtime/authorizations/pilot_a01_auth.json"
        ),
        "bn_import_route2_cmd": (
            "/home/bn/anaconda3/envs/videollama2/bin/python deployment/stage_assets.py prepare "
            "--data-root /home/bn/uarcap_q2_runtime/data --dataset carebench "
            "--transfer-manifest /home/bn/uarcap_q2_runtime/staging/carebench/videos.zip.transfer_manifest.json"
        ),
    }

    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    if not args.metadata_only and not args.confirm_download:
        print(
            "Refusing to download 3.65 GiB videos.zip without --confirm-download "
            "(use --dry-run or --metadata-only if you do not intend to fetch videos yet).",
            file=sys.stderr,
        )
        return 2

    probe = check_carebench_hf_access(hf_endpoint=base)
    if not probe.get("accessible"):
        print(json.dumps({"error": "CaReBench endpoint unreachable", "probe": probe}, indent=2), file=sys.stderr)
        return 2

    stage_dir.mkdir(parents=True, exist_ok=True)
    meta_res = download_with_resume(meta_url, meta_dst)
    receipt: dict[str, Any] = {**plan, "metadata_download": meta_res, "metadata_sha256": sha256_file(meta_dst)}

    if not args.metadata_only:
        zip_res = download_with_resume(
            zip_url,
            zip_dst,
            expected_bytes=CAREBENCH_EXPECTED_VIDEO_ZIP_BYTES,
        )
        receipt["videos_zip_download"] = zip_res
        receipt["videos_zip_sha256"] = sha256_file(zip_dst)
        if not args.skip_chunk_split:
            manifest_path = split_file_into_chunks(zip_dst, chunks_dir, chunk_bytes=args.chunk_mb * 1024 * 1024)
            receipt["transfer_manifest"] = str(manifest_path)

    atomic_write_json(stage_dir / "carebench_staging_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
