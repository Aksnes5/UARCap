"""One-click automated dataset preparation pipeline (DREAM-1K & VidHalluc-ACH).

Chains:
  1. Existing-file detection (already normalized, local archive present, or Route-2 chunk manifest).
  2. Optional stage-gated HTTP Range download (preserving `stage_unlock` enforcement).
  3. Size / SHA-256 integrity verification.
  4. Zip-slip-safe extraction.
  5. Canonical directory normalization (`dream1k/json/metadata.json` + `dream1k/video/*.mp4`,
     `vidhalluc/ach_mcq.json` + `vidhalluc/data/*.mp4`).
  6. Metadata-to-video correspondence check & schema validation.
  7. Pre-stage readiness receipt (`<dataset>_readiness.json`).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any
import zipfile

from ..common.gates import verify_asset_download_authorization
from ..common.io_utils import atomic_write_json, sha256_file
from .asset_transfer import (
    AssetSpec,
    download_with_resume,
    reassemble_and_verify_chunks,
    verify_local_asset,
)
from .dream1k import load_dream1k_records
from .vidhalluc import load_vidhalluc_ach_mcq


class UnsafeArchiveError(ValueError):
    """Raised when a zip archive attempts directory traversal (zip-slip) or symlink escape."""


def safe_extract_zip(zip_path: Path | str, dest_dir: Path | str) -> dict[str, Any]:
    """Safely extract a zip archive rejecting absolute paths, `..` traversal, and symlinks."""
    zp = Path(zip_path)
    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    if not zp.is_file():
        raise FileNotFoundError(f"Archive not found: {zp}")

    extracted = 0
    with zipfile.ZipFile(zp, "r") as zf:
        for info in zf.infolist():
            name = info.filename
            if name.startswith("/") or name.startswith("\\") or ".." in Path(name).parts:
                raise UnsafeArchiveError(f"Blocked unsafe zip member path: {name!r}")
            # Reject Unix symlinks stored in external_attr (0o120000)
            unix_mode = (info.external_attr >> 16) & 0o170000
            if unix_mode == 0o120000:
                raise UnsafeArchiveError(f"Blocked symlink member in zip archive: {name!r}")
            target = (dest / name).resolve()
            if dest != target and dest not in target.parents:
                raise UnsafeArchiveError(f"Zip-slip escape blocked for member: {name!r}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
    return {"zip_path": str(zp), "dest_dir": str(dest), "files_extracted": extracted}


def normalize_dataset_layout(dataset_name: str, data_root: Path | str) -> dict[str, Any]:
    """Flatten nested zip folder layouts (e.g., `video/video/*.mp4` or `data/ACH_videos/*.mp4`) safely."""
    root = Path(data_root)
    moved = 0
    if dataset_name == "dream1k":
        vid_dir = root / "dream1k" / "video"
        vid_dir.mkdir(parents=True, exist_ok=True)
        for mp4 in list(vid_dir.rglob("*.mp4")):
            if mp4.parent != vid_dir:
                dst = vid_dir / mp4.name
                if not dst.exists():
                    shutil.move(str(mp4), str(dst))
                    moved += 1
        return {"dataset": "dream1k", "canonical_video_dir": str(vid_dir), "files_normalized": moved}

    if dataset_name == "vidhalluc":
        vid_dir = root / "vidhalluc" / "data"
        vid_dir.mkdir(parents=True, exist_ok=True)
        for mp4 in list(vid_dir.rglob("*.mp4")):
            if mp4.parent != vid_dir:
                dst = vid_dir / mp4.name
                if not dst.exists():
                    shutil.move(str(mp4), str(dst))
                    moved += 1
        return {"dataset": "vidhalluc", "canonical_video_dir": str(vid_dir), "files_normalized": moved}

    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def validate_dataset_readiness(dataset_name: str, data_root: Path | str) -> dict[str, Any]:
    """Run metadata schema validation + metadata-to-video correspondence audit & write readiness receipt."""
    root = Path(data_root)
    if dataset_name == "dream1k":
        meta_path = root / "dream1k" / "json" / "metadata.json"
        vid_dir = root / "dream1k" / "video"
        if not meta_path.is_file():
            res = {"dataset": "dream1k", "ready": False, "error": f"Missing metadata: {meta_path}"}
            atomic_write_json(root / "dream1k_readiness.json", res)
            return res
        recs = load_dream1k_records(meta_path, vid_dir, require_video_exists=False)
        existing_vids = {p.name for p in vid_dir.glob("*.mp4")} if vid_dir.is_dir() else set()
        expected_vids = {Path(r.video_path).name for r in recs}
        missing_vids = sorted(
            r.query_id for r in recs if not (Path(r.video_path).is_file() and Path(r.video_path).stat().st_size > 0)
        )
        orphan_vids = sorted(existing_vids - expected_vids)
        empty_refs = [r.query_id for r in recs if not r.reference_description.strip()]
        ready = bool(recs) and not missing_vids and not empty_refs
        res = {
            "dataset": "dream1k",
            "ready": ready,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "metadata_path": str(meta_path),
            "metadata_sha256": sha256_file(meta_path),
            "video_dir": str(vid_dir),
            "n_metadata_records": len(recs),
            "n_matched_videos": len(recs) - len(missing_vids),
            "n_missing_or_empty_videos": len(missing_vids),
            "missing_video_sample": missing_vids[:10],
            "n_orphan_videos": len(orphan_vids),
            "n_empty_reference_descriptions": len(empty_refs),
        }
        atomic_write_json(root / "dream1k_readiness.json", res)
        return res

    if dataset_name == "vidhalluc":
        meta_path = root / "vidhalluc" / "ach_mcq.json"
        vid_dir = root / "vidhalluc" / "data"
        if not meta_path.is_file():
            res = {"dataset": "vidhalluc", "ready": False, "error": f"Missing ach_mcq.json: {meta_path}"}
            atomic_write_json(root / "vidhalluc_readiness.json", res)
            return res
        recs = load_vidhalluc_ach_mcq(meta_path, vid_dir, require_video_exists=False)
        existing_vids = {p.name for p in vid_dir.glob("*.mp4")} if vid_dir.is_dir() else set()
        expected_vids = {Path(r.video_path).name for r in recs}
        missing_vids = sorted(
            r.query_id for r in recs if not (Path(r.video_path).is_file() and Path(r.video_path).stat().st_size > 0)
        )
        orphan_vids = sorted(existing_vids - expected_vids)
        ready = bool(recs) and not missing_vids
        res = {
            "dataset": "vidhalluc",
            "ready": ready,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "metadata_path": str(meta_path),
            "metadata_sha256": sha256_file(meta_path),
            "video_dir": str(vid_dir),
            "n_metadata_records": len(recs),
            "n_matched_videos": len(recs) - len(missing_vids),
            "n_missing_or_empty_videos": len(missing_vids),
            "missing_video_sample": missing_vids[:10],
            "n_orphan_videos": len(orphan_vids),
        }
        atomic_write_json(root / "vidhalluc_readiness.json", res)
        return res

    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def prepare_dataset_pipeline(
    dataset_name: str,
    data_root: Path | str,
    known_assets: dict[str, AssetSpec],
    *,
    auth_receipt_path: Path | str | None = None,
    prior_receipts: dict[str, Path | str] | None = None,
    allow_download: bool = False,
    transfer_manifest: Path | str | None = None,
    skip_size_check_for_local_archive: bool = False,
) -> dict[str, Any]:
    """One-click automated preparation: detect/import/download -> verify -> safe extract -> normalize -> validate."""
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []

    if dataset_name == "dream1k":
        meta_spec = known_assets["dream1k_metadata"]
        vid_spec = known_assets["dream1k_videos"]
        extract_dest = root / "dream1k" / "video"
    elif dataset_name == "vidhalluc":
        meta_spec = known_assets["vidhalluc_ach_mcq"]
        vid_spec = known_assets["vidhalluc_ach_videos"]
        extract_dest = root / "vidhalluc" / "data"
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    # 1. Check if already fully normalized & ready (idempotent!)
    pre_check = validate_dataset_readiness(dataset_name, root)
    if pre_check.get("ready"):
        return {"dataset": dataset_name, "status": "ALREADY_READY", "steps": steps, "readiness": pre_check}

    # 2. Metadata check or gated download
    meta_file = root / meta_spec.relative_path
    if not meta_file.is_file() and allow_download:
        verify_asset_download_authorization(meta_spec.asset_id, meta_spec.stage_unlock, auth_receipt_path, prior_receipts)
        steps.append({"step": "download_metadata", **download_with_resume(meta_spec.source_url, meta_file)})
    else:
        steps.append({"step": "detect_metadata", "exists": meta_file.is_file(), "path": str(meta_file)})

    # 3. Video archive: Route-2 chunk import OR existing file OR stage-gated Route-1 download
    archive_file = root / vid_spec.relative_path
    if transfer_manifest and Path(transfer_manifest).is_file():
        steps.append({"step": "import_chunks", **reassemble_and_verify_chunks(transfer_manifest, archive_file)})
    elif not archive_file.is_file() and allow_download:
        verify_asset_download_authorization(vid_spec.asset_id, vid_spec.stage_unlock, auth_receipt_path, prior_receipts)
        steps.append(
            {
                "step": "download_videos",
                **download_with_resume(
                    vid_spec.source_url,
                    archive_file,
                    expected_bytes=vid_spec.expected_bytes,
                    expected_sha256=vid_spec.expected_sha256,
                ),
            }
        )
    else:
        steps.append({"step": "detect_archive", "exists": archive_file.is_file(), "path": str(archive_file)})

    # 4. Integrity verify & safe extract if archive exists
    if archive_file.is_file():
        if not skip_size_check_for_local_archive:
            vrep = verify_local_asset(root, vid_spec)
            steps.append({"step": "verify_archive", **vrep})
            if not vrep["ok"]:
                return {"dataset": dataset_name, "status": "ARCHIVE_VERIFICATION_FAILED", "steps": steps, "readiness": pre_check}
        steps.append({"step": "safe_extract", **safe_extract_zip(archive_file, extract_dest)})

    # 5. Canonical layout normalization & final schema/correspondence check
    steps.append({"step": "normalize_layout", **normalize_dataset_layout(dataset_name, root)})
    readiness = validate_dataset_readiness(dataset_name, root)
    return {
        "dataset": dataset_name,
        "status": "READY" if readiness.get("ready") else "INCOMPLETE",
        "steps": steps,
        "readiness": readiness,
    }
