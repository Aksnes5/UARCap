"""Offline dataset & model asset transfer manager (Route 1 BN direct + Route 2 Windows-to-BN staging).

Guarantees:
  - Resumable HTTP Range downloads with atomic completion and SHA-256 verification.
  - Skips already-verified local assets without re-downloading.
  - Optional chunked file split & verified reassembly for constrained transfer links.
  - Reads HF_TOKEN strictly from environment; NEVER writes tokens to logs, manifests, or command history.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any
import urllib.request

from ..common.io_utils import atomic_write_json, read_json, sha256_file


@dataclass
class AssetSpec:
    asset_id: str
    stage_unlock: str  # PILOT_A1 | PILOT_A2 | PILOT_A3
    source_url: str
    relative_path: str
    expected_bytes: int | None = None
    expected_sha256: str | None = None
    gated: bool = False
    license_name: str = ""
    redistribution_allowed: bool = False
    revision: str = "main"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_local_asset(root_dir: Path | str, spec: AssetSpec) -> dict[str, Any]:
    target = Path(root_dir) / spec.relative_path
    if not target.is_file():
        return {"asset_id": spec.asset_id, "path": str(target), "status": "MISSING", "ok": False}
    size = target.stat().st_size
    if spec.expected_bytes is not None and size != spec.expected_bytes:
        return {
            "asset_id": spec.asset_id,
            "path": str(target),
            "status": "SIZE_MISMATCH",
            "expected_bytes": spec.expected_bytes,
            "actual_bytes": size,
            "ok": False,
        }
    actual_sha = sha256_file(target)
    if spec.expected_sha256 is not None and actual_sha != spec.expected_sha256:
        return {
            "asset_id": spec.asset_id,
            "path": str(target),
            "status": "SHA256_MISMATCH",
            "expected_sha256": spec.expected_sha256,
            "actual_sha256": actual_sha,
            "ok": False,
        }
    return {
        "asset_id": spec.asset_id,
        "path": str(target),
        "status": "VERIFIED",
        "bytes": size,
        "sha256": actual_sha,
        "ok": True,
    }


def download_with_resume(
    url: str,
    dest_path: Path | str,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    token_env_var: str = "HF_TOKEN",
    timeout_sec: int = 30,
    chunk_size: int = 1024 * 1024,
) -> dict[str, Any]:
    """Resumable HTTP Range downloader. Never logs or stores token values."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        cur_size = dest.stat().st_size
        if expected_bytes is None or cur_size == expected_bytes:
            sha = sha256_file(dest)
            if expected_sha256 is None or sha == expected_sha256:
                return {"status": "SKIPPED_VERIFIED", "path": str(dest), "bytes": cur_size, "sha256": sha}

    part = dest.with_suffix(dest.suffix + ".part")
    existing = part.stat().st_size if part.is_file() else 0
    headers = {"User-Agent": "UARCap-Q2-AssetTransfer/0.1"}
    token = os.environ.get(token_env_var, "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        status_code = getattr(resp, "status", 200)
        mode = "ab" if (existing > 0 and status_code == 206) else "wb"
        with part.open(mode) as out_f:
            while True:
                buf = resp.read(chunk_size)
                if not buf:
                    break
                out_f.write(buf)

    final_size = part.stat().st_size
    if expected_bytes is not None and final_size != expected_bytes:
        raise ValueError(f"Download size mismatch for {dest.name}: expected {expected_bytes}, got {final_size}")
    final_sha = sha256_file(part)
    if expected_sha256 is not None and final_sha != expected_sha256:
        raise ValueError(f"Download SHA-256 mismatch for {dest.name}: expected {expected_sha256}, got {final_sha}")
    os.replace(str(part), str(dest))
    return {"status": "DOWNLOADED_VERIFIED", "path": str(dest), "bytes": final_size, "sha256": final_sha}


def split_file_into_chunks(src_file: Path | str, out_dir: Path | str, chunk_bytes: int = 512 * 1024 * 1024) -> Path:
    """Split a large archive into verified transfer chunks for Route 2 Windows -> BN staging."""
    src = Path(src_file)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    full_sha = sha256_file(src)
    parts: list[dict[str, Any]] = []
    idx = 0
    with src.open("rb") as fin:
        while True:
            buf = fin.read(chunk_bytes)
            if not buf:
                break
            part_name = f"{src.name}.part{idx:03d}"
            part_path = out / part_name
            part_path.write_bytes(buf)
            parts.append({"name": part_name, "bytes": len(buf), "sha256": sha256_file(part_path)})
            idx += 1
    manifest = {
        "original_name": src.name,
        "original_bytes": src.stat().st_size,
        "original_sha256": full_sha,
        "chunk_bytes": chunk_bytes,
        "parts": parts,
    }
    return atomic_write_json(out / f"{src.name}.transfer_manifest.json", manifest)


def reassemble_and_verify_chunks(manifest_path: Path | str, dest_file: Path | str) -> dict[str, Any]:
    """Reassemble Route 2 staged chunks on Ubuntu BN and verify every chunk + full SHA-256."""
    mpath = Path(manifest_path)
    manifest = read_json(mpath)
    src_dir = mpath.parent
    dest = Path(dest_file)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".assembling")
    with tmp.open("wb") as fout:
        for part in manifest["parts"]:
            pp = src_dir / part["name"]
            if not pp.is_file():
                raise FileNotFoundError(f"Missing transfer chunk: {pp}")
            actual_sha = sha256_file(pp)
            if actual_sha != part["sha256"]:
                raise ValueError(f"Corrupt transfer chunk {pp.name}: SHA-256 mismatch")
            fout.write(pp.read_bytes())
    assembled_sha = sha256_file(tmp)
    if assembled_sha != manifest["original_sha256"]:
        tmp.unlink(missing_ok=True)
        raise ValueError("Reassembled archive SHA-256 mismatch")
    os.replace(str(tmp), str(dest))
    return {"path": str(dest), "bytes": dest.stat().st_size, "sha256": assembled_sha, "status": "IMPORTED_VERIFIED"}
