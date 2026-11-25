"""CLI for Route 1 BN resumable downloads and Route 2 Windows-to-BN chunked offline transfer.

Strictly enforces stage-gated unlocking:
  - PILOT_A1: DREAM-1K video.zip (3.59 GiB) + metadata only.
  - PILOT_A3: VidHalluc ACH_videos.zip (15.60 GiB) + JSONs only after A2 approval.
NEVER logs or persists HF_TOKEN!
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uarcap_q2.datasets.asset_transfer import (
    AssetSpec,
    download_with_resume,
    reassemble_and_verify_chunks,
    split_file_into_chunks,
    verify_local_asset,
)
from uarcap_q2.datasets.dream1k import DREAM1K_EXPECTED_VIDEO_ZIP_BYTES, DREAM1K_HF_REPO
from uarcap_q2.datasets.vidhalluc import VIDHALLUC_ACH_ZIP_BYTES, VIDHALLUC_HF_REPO

KNOWN_ASSETS: dict[str, AssetSpec] = {
    "dream1k_metadata": AssetSpec(
        asset_id="dream1k_metadata",
        stage_unlock="PILOT_A0",
        source_url=f"https://huggingface.co/datasets/{DREAM1K_HF_REPO}/resolve/main/json/metadata.json",
        relative_path="dream1k/json/metadata.json",
        gated=True,
        license_name="apache-2.0 (academic video copyright notice)",
        redistribution_allowed=False,
    ),
    "dream1k_videos": AssetSpec(
        asset_id="dream1k_videos",
        stage_unlock="PILOT_A1",
        source_url=f"https://huggingface.co/datasets/{DREAM1K_HF_REPO}/resolve/main/video/video.zip",
        relative_path="dream1k/video/video.zip",
        expected_bytes=DREAM1K_EXPECTED_VIDEO_ZIP_BYTES,
        gated=True,
        license_name="apache-2.0 metadata / original video copyright academic-only",
        redistribution_allowed=False,
    ),
    "vidhalluc_ach_mcq": AssetSpec(
        asset_id="vidhalluc_ach_mcq",
        stage_unlock="PILOT_A3",
        source_url=f"https://huggingface.co/datasets/{VIDHALLUC_HF_REPO}/resolve/main/ach_mcq.json",
        relative_path="vidhalluc/ach_mcq.json",
        gated=False,
        license_name="cc-by-nc-nd-4.0",
        redistribution_allowed=False,
    ),
    "vidhalluc_ach_binaryqa": AssetSpec(
        asset_id="vidhalluc_ach_binaryqa",
        stage_unlock="PILOT_A3",
        source_url=f"https://huggingface.co/datasets/{VIDHALLUC_HF_REPO}/resolve/main/ach_binaryqa.json",
        relative_path="vidhalluc/ach_binaryqa.json",
        gated=False,
        license_name="cc-by-nc-nd-4.0",
        redistribution_allowed=False,
    ),
    "vidhalluc_ach_videos": AssetSpec(
        asset_id="vidhalluc_ach_videos",
        stage_unlock="PILOT_A3",
        source_url=f"https://huggingface.co/datasets/{VIDHALLUC_HF_REPO}/resolve/main/data/ACH_videos.zip",
        relative_path="vidhalluc/data/ACH_videos.zip",
        expected_bytes=VIDHALLUC_ACH_ZIP_BYTES,
        gated=False,
        license_name="cc-by-nc-nd-4.0",
        redistribution_allowed=False,
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage-gated dataset downloader & Route 2 offline chunk transfer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ver = sub.add_parser("verify", help="Verify local staged asset")
    p_ver.add_argument("--data-root", type=Path, required=True)
    p_ver.add_argument("--asset", choices=sorted(KNOWN_ASSETS), required=True)

    p_dl = sub.add_parser("download", help="Download a single stage-approved asset with HTTP resume")
    p_dl.add_argument("--data-root", type=Path, required=True)
    p_dl.add_argument("--asset", choices=sorted(KNOWN_ASSETS), required=True)
    p_dl.add_argument("--confirm-download", action="store_true", help="Required explicit flag to prevent accidental downloads")

    p_sp = sub.add_parser("split", help="Route 2: Split local archive into SHA-256 verified transfer chunks")
    p_sp.add_argument("--src", type=Path, required=True)
    p_sp.add_argument("--out-dir", type=Path, required=True)
    p_sp.add_argument("--chunk-mb", type=int, default=512)

    p_im = sub.add_parser("import", help="Route 2: Reassemble & verify chunks on BN server")
    p_im.add_argument("--manifest", type=Path, required=True)
    p_im.add_argument("--dest", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "verify":
        res = verify_local_asset(args.data_root, KNOWN_ASSETS[args.asset])
        print(json.dumps(res, indent=2))
        return 0 if res["ok"] else 2
    if args.cmd == "download":
        if not args.confirm_download:
            print("Refusing download without --confirm-download (offline design safety guard).", file=sys.stderr)
            return 2
        spec = KNOWN_ASSETS[args.asset]
        res = download_with_resume(
            url=spec.source_url,
            dest_path=args.data_root / spec.relative_path,
            expected_bytes=spec.expected_bytes,
            expected_sha256=spec.expected_sha256,
        )
        print(json.dumps(res, indent=2))
        return 0
    if args.cmd == "split":
        m = split_file_into_chunks(args.src, args.out_dir, chunk_bytes=args.chunk_mb * 1024 * 1024)
        print(json.dumps({"manifest": str(m)}, indent=2))
        return 0
    if args.cmd == "import":
        res = reassemble_and_verify_chunks(args.manifest, args.dest)
        print(json.dumps(res, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
