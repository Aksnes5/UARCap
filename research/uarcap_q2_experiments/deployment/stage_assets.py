"""Stage-gated dataset downloader & Route 2 offline chunk transfer (P0-5 remediated: actively enforces stage_unlock!).

Rules:
  - carebench_metadata (PILOT_A0, ~3.70 MB, ungated MIT): permitted during A0 preflight with --confirm-download.
  - carebench_videos (PILOT_A1, 3.65 GiB, ungated MIT): REQUIRES valid PILOT_A1 --auth-receipt!
  - dream1k_metadata / dream1k_videos (backup E2, gated: auto, 3.59 GiB): retained as full compatible fallback.
  - vidhalluc_ach_* (PILOT_A3, 15.60 GiB, ungated CC-BY-NC-ND-4.0): REQUIRES PILOT_A3 --auth-receipt + PILOT_A2 REAL_RUNTIME_PASS!
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uarcap_q2.common.gates import AuthorizationError, verify_asset_download_authorization
from uarcap_q2.common.schemas import PilotStage
from uarcap_q2.datasets.asset_transfer import (
    AssetSpec,
    download_with_resume,
    reassemble_and_verify_chunks,
    split_file_into_chunks,
    verify_local_asset,
)
from uarcap_q2.datasets.carebench import CAREBENCH_EXPECTED_VIDEO_ZIP_BYTES, CAREBENCH_HF_REPO
from uarcap_q2.datasets.data_preparer import prepare_dataset_pipeline, validate_dataset_readiness
from uarcap_q2.datasets.dream1k import DREAM1K_EXPECTED_VIDEO_ZIP_BYTES, DREAM1K_HF_REPO
from uarcap_q2.datasets.vidhalluc import VIDHALLUC_ACH_ZIP_BYTES, VIDHALLUC_HF_REPO


def build_known_assets(hf_endpoint: str | None = None) -> dict[str, AssetSpec]:
    base = (hf_endpoint or os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    return {
        "carebench_metadata": AssetSpec(
            asset_id="carebench_metadata",
            stage_unlock="PILOT_A0",
            source_url=f"{base}/datasets/{CAREBENCH_HF_REPO}/resolve/main/json/metadata.json",
            relative_path="carebench/json/metadata.json",
            gated=False,
            license_name="MIT (metadata/code)",
            redistribution_allowed=False,
        ),
        "carebench_videos": AssetSpec(
            asset_id="carebench_videos",
            stage_unlock="PILOT_A1",
            source_url=f"{base}/datasets/{CAREBENCH_HF_REPO}/resolve/main/videos/videos.zip",
            relative_path="carebench/videos/videos.zip",
            expected_bytes=CAREBENCH_EXPECTED_VIDEO_ZIP_BYTES,
            gated=False,
            license_name="MIT code/metadata; upstream FineAction academic video copyrights apply",
            redistribution_allowed=False,
        ),
        "dream1k_metadata": AssetSpec(
            asset_id="dream1k_metadata",
            stage_unlock="PILOT_A0",
            source_url=f"{base}/datasets/{DREAM1K_HF_REPO}/resolve/main/json/metadata.json",
            relative_path="dream1k/json/metadata.json",
            gated=True,
            license_name="apache-2.0 (academic video copyright notice)",
            redistribution_allowed=False,
        ),
        "dream1k_videos": AssetSpec(
            asset_id="dream1k_videos",
            stage_unlock="PILOT_A1",
            source_url=f"{base}/datasets/{DREAM1K_HF_REPO}/resolve/main/video/video.zip",
            relative_path="dream1k/video/video.zip",
            expected_bytes=DREAM1K_EXPECTED_VIDEO_ZIP_BYTES,
            gated=True,
            license_name="apache-2.0 metadata / original video copyright academic-only",
            redistribution_allowed=False,
        ),
        "vidhalluc_ach_mcq": AssetSpec(
            asset_id="vidhalluc_ach_mcq",
            stage_unlock="PILOT_A3",
            source_url=f"{base}/datasets/{VIDHALLUC_HF_REPO}/resolve/main/ach_mcq.json",
            relative_path="vidhalluc/ach_mcq.json",
            gated=False,
            license_name="cc-by-nc-nd-4.0",
            redistribution_allowed=False,
        ),
        "vidhalluc_ach_binaryqa": AssetSpec(
            asset_id="vidhalluc_ach_binaryqa",
            stage_unlock="PILOT_A3",
            source_url=f"{base}/datasets/{VIDHALLUC_HF_REPO}/resolve/main/ach_binaryqa.json",
            relative_path="vidhalluc/ach_binaryqa.json",
            gated=False,
            license_name="cc-by-nc-nd-4.0",
            redistribution_allowed=False,
        ),
        "vidhalluc_ach_videos": AssetSpec(
            asset_id="vidhalluc_ach_videos",
            stage_unlock="PILOT_A3",
            source_url=f"{base}/datasets/{VIDHALLUC_HF_REPO}/resolve/main/data/ACH_videos.zip",
            relative_path="vidhalluc/data/ACH_videos.zip",
            expected_bytes=VIDHALLUC_ACH_ZIP_BYTES,
            gated=False,
            license_name="cc-by-nc-nd-4.0",
            redistribution_allowed=False,
        ),
    }


KNOWN_ASSETS: dict[str, AssetSpec] = build_known_assets("https://huggingface.co")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage-gated dataset downloader & Route 2 offline chunk transfer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ver = sub.add_parser("verify", help="Verify local staged asset")
    p_ver.add_argument("--data-root", type=Path, required=True)
    p_ver.add_argument("--asset", choices=sorted(KNOWN_ASSETS), required=True)

    p_dl = sub.add_parser("download", help="Download a single stage-authorized asset with HTTP resume")
    p_dl.add_argument("--data-root", type=Path, required=True)
    p_dl.add_argument("--asset", choices=sorted(KNOWN_ASSETS), required=True)
    p_dl.add_argument("--confirm-download", action="store_true")
    p_dl.add_argument("--hf-endpoint", default=None, help="Optional HF base URL or mirror (e.g. https://hf-mirror.com for ungated assets)")
    p_dl.add_argument("--auth-receipt", type=Path, default=None, help="Required for PILOT_A1/A3 gated assets")
    p_dl.add_argument("--prior-a1", type=Path, default=None)
    p_dl.add_argument("--prior-a2", type=Path, default=None)

    p_sp = sub.add_parser("split", help="Route 2: Split local archive into SHA-256 verified transfer chunks")
    p_sp.add_argument("--src", type=Path, required=True)
    p_sp.add_argument("--out-dir", type=Path, required=True)
    p_sp.add_argument("--chunk-mb", type=int, default=512)

    p_im = sub.add_parser("import", help="Route 2: Reassemble & verify chunks on BN server")
    p_im.add_argument("--manifest", type=Path, required=True)
    p_im.add_argument("--dest", type=Path, required=True)

    p_prep = sub.add_parser("prepare", help="One-click dataset prepare: detect/import/download -> verify -> safe extract -> normalize -> validate")
    p_prep.add_argument("--data-root", type=Path, required=True)
    p_prep.add_argument("--dataset", choices=["carebench", "dream1k", "vidhalluc"], required=True)
    p_prep.add_argument("--allow-download", action="store_true", help="Permit stage-gated HTTP download if missing")
    p_prep.add_argument("--hf-endpoint", default=None, help="Optional HF base URL or mirror for ungated assets")
    p_prep.add_argument("--transfer-manifest", type=Path, default=None, help="Route 2 chunk manifest to import before extracting")
    p_prep.add_argument("--auth-receipt", type=Path, default=None)
    p_prep.add_argument("--prior-a1", type=Path, default=None)
    p_prep.add_argument("--prior-a2", type=Path, default=None)
    p_prep.add_argument("--skip-archive-size-check", action="store_true")

    p_rdy = sub.add_parser("check-ready", help="Check normalized dataset schema & metadata-to-video readiness")
    p_rdy.add_argument("--data-root", type=Path, required=True)
    p_rdy.add_argument("--dataset", choices=["carebench", "dream1k", "vidhalluc"], required=True)

    args = ap.parse_args(argv)
    assets = build_known_assets(getattr(args, "hf_endpoint", None))
    if args.cmd == "verify":
        res = verify_local_asset(args.data_root, assets[args.asset])
        print(json.dumps(res, indent=2))
        return 0 if res["ok"] else 2
    if args.cmd == "check-ready":
        res = validate_dataset_readiness(args.dataset, args.data_root)
        print(json.dumps(res, indent=2))
        return 0 if res.get("ready") else 2
    if args.cmd == "prepare":
        priors = {}
        if args.prior_a1:
            priors[PilotStage.PILOT_A1.value] = args.prior_a1
        if args.prior_a2:
            priors[PilotStage.PILOT_A2.value] = args.prior_a2
        try:
            res = prepare_dataset_pipeline(
                dataset_name=args.dataset,
                data_root=args.data_root,
                known_assets=assets,
                auth_receipt_path=args.auth_receipt,
                prior_receipts=priors,
                allow_download=args.allow_download,
                transfer_manifest=args.transfer_manifest,
                skip_size_check_for_local_archive=args.skip_archive_size_check,
            )
        except AuthorizationError as exc:
            print(f"[STAGE_UNLOCK_BLOCKED] Dataset prepare '{args.dataset}' blocked: {exc}", file=sys.stderr)
            return 3
        print(json.dumps(res, indent=2))
        return 0 if res.get("status") in ("READY", "ALREADY_READY") else 2
    if args.cmd == "download":
        if not args.confirm_download:
            print("Refusing download without --confirm-download (offline safety guard).", file=sys.stderr)
            return 2
        spec = assets[args.asset]
        priors = {}
        if args.prior_a1:
            priors[PilotStage.PILOT_A1.value] = args.prior_a1
        if args.prior_a2:
            priors[PilotStage.PILOT_A2.value] = args.prior_a2
        try:
            verify_asset_download_authorization(spec.asset_id, spec.stage_unlock, args.auth_receipt, priors)
        except AuthorizationError as exc:
            print(f"[STAGE_UNLOCK_BLOCKED] Asset '{spec.asset_id}' requires stage {spec.stage_unlock}: {exc}", file=sys.stderr)
            return 3
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
