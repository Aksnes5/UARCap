"""Offline dataset & model asset transfer manager (Route 1 BN direct + Route 2 Windows-to-BN staging).

Guarantees:
  - Resumable HTTP Range downloads with atomic completion and SHA-256 verification.
  - Skips already-verified local assets without re-downloading.
  - Optional chunked file split & verified reassembly for constrained transfer links.
  - Reads HF_TOKEN strictly from environment; NEVER writes tokens to logs, manifests, or command history.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import shutil
import ssl
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from ..common.io_utils import atomic_write_json, read_json, sha256_file


def _build_ssl_context() -> ssl.SSLContext:
    """Use certifi's CA bundle when the active env's OpenSSL paths are stale (common in conda envs)."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001 - fall back to the platform default
        return ssl.create_default_context()


_SSL_CONTEXT = _build_ssl_context()


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
    mirror_urls: list[str] = field(default_factory=list)

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
    with urllib.request.urlopen(req, timeout=timeout_sec, context=_SSL_CONTEXT) as resp:
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


def _download_range_to_file(
    urls: list[str],
    start: int,
    end: int,
    part_path: Path,
    *,
    timeout_sec: int = 30,
    max_retries: int = 10,
    user_agent: str = "UARCap-Q2-AssetTransfer/0.2",
) -> str:
    """Download byte range [start, end] into `part_path`, resuming at byte granularity.

    Rotates through `urls` (primary + mirrors) and retries with exponential backoff.
    Returns the URL that completed the range.
    """
    want = end - start + 1
    last_err: Exception | None = None
    for attempt in range(max_retries):
        have = part_path.stat().st_size if part_path.is_file() else 0
        if have >= want:
            return "already-complete"
        url = urls[attempt % len(urls)]
        req = urllib.request.Request(
            url,
            headers={"Range": f"bytes={start + have}-{end}", "User-Agent": user_agent},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec, context=_SSL_CONTEXT) as resp:
                if getattr(resp, "status", 206) == 200 and have > 0:
                    # Server ignored Range: restart this chunk from zero.
                    have = 0
                    part_path.unlink(missing_ok=True)
                mode = "ab" if have > 0 else "wb"
                with part_path.open(mode) as f:
                    while True:
                        remaining = want - (part_path.stat().st_size if part_path.is_file() else 0)
                        if remaining <= 0:
                            break
                        buf = resp.read(min(1024 * 1024, remaining))
                        if not buf:
                            break
                        f.write(buf)
            if part_path.stat().st_size >= want:
                return url
            last_err = IOError(f"short range read: {part_path.stat().st_size}/{want}")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_err = exc
        time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"Range download failed [{start}-{end}] after {max_retries} attempts: {last_err}")


def download_parallel_ranges(
    urls: str | list[str],
    dest_path: Path | str,
    expected_bytes: int,
    expected_sha256: str | None = None,
    *,
    workers: int = 16,
    chunk_bytes: int = 64 * 1024 * 1024,
    max_retries: int = 10,
    timeout_sec: int = 30,
) -> dict[str, Any]:
    """Parallel resumable HTTP Range downloader with full-file SHA-256 verification.

    Primary source + trusted mirrors are tried per chunk (rotation), so a single flaky
    endpoint cannot block the transfer. Chunk parts are kept on disk between runs.
    """
    url_list = [urls] if isinstance(urls, str) else list(urls)
    if not url_list:
        raise ValueError("download_parallel_ranges requires at least one URL")
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == expected_bytes:
        sha = sha256_file(dest)
        if expected_sha256 is None or sha == expected_sha256:
            return {"status": "SKIPPED_VERIFIED", "path": str(dest), "bytes": expected_bytes, "sha256": sha}

    part_dir = dest.parent / f".{dest.name}.chunks"
    part_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (expected_bytes + chunk_bytes - 1) // chunk_bytes
    started = time.time()

    def _fetch(idx: int) -> dict[str, Any]:
        start = idx * chunk_bytes
        end = min(expected_bytes, start + chunk_bytes) - 1
        part_path = part_dir / f"chunk_{idx:05d}"
        used = _download_range_to_file(
            url_list, start, end, part_path, timeout_sec=timeout_sec, max_retries=max_retries
        )
        return {"chunk": idx, "bytes": part_path.stat().st_size, "source": used}

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch, i) for i in range(n_chunks)]
        try:
            for fut in as_completed(futures):
                results.append(fut.result())
        except Exception:
            for f in futures:
                f.cancel()
            raise

    # Assemble in deterministic order.
    tmp = dest.with_suffix(dest.suffix + ".assembling")
    with tmp.open("wb") as out:
        for idx in range(n_chunks):
            pp = part_dir / f"chunk_{idx:05d}"
            with pp.open("rb") as fin:
                shutil.copyfileobj(fin, out, 8 * 1024 * 1024)
    final_size = tmp.stat().st_size
    if final_size != expected_bytes:
        raise ValueError(f"Assembled size mismatch for {dest.name}: expected {expected_bytes}, got {final_size}")
    final_sha = sha256_file(tmp)
    if expected_sha256 is not None and final_sha != expected_sha256:
        raise ValueError(f"Assembled SHA-256 mismatch for {dest.name}: expected {expected_sha256}, got {final_sha}")
    os.replace(str(tmp), str(dest))
    shutil.rmtree(part_dir, ignore_errors=True)
    return {
        "status": "DOWNLOADED_VERIFIED",
        "path": str(dest),
        "bytes": final_size,
        "sha256": final_sha,
        "chunks": n_chunks,
        "workers": workers,
        "elapsed_sec": round(time.time() - started, 1),
        "sources": sorted({r["source"] for r in results}),
    }


def download_asset(spec: AssetSpec, dest_path: Path | str) -> dict[str, Any]:
    """Download an asset using its primary URL plus trusted mirrors (parallel ranges when size is known)."""
    urls = [spec.source_url, *spec.mirror_urls]
    if spec.expected_bytes and spec.expected_bytes > 32 * 1024 * 1024:
        try:
            return download_parallel_ranges(
                urls,
                dest_path,
                expected_bytes=spec.expected_bytes,
                expected_sha256=spec.expected_sha256,
            )
        except RuntimeError:
            # Fall through to the single-stream resumer (which also tries mirrors).
            pass
    last_err: Exception | None = None
    for url in urls:
        try:
            return download_with_resume(
                url, dest_path, expected_bytes=spec.expected_bytes, expected_sha256=spec.expected_sha256
            )
        except Exception as exc:  # noqa: BLE001 - try next mirror
            last_err = exc
    raise RuntimeError(f"All sources failed for {spec.asset_id}: {last_err}")


def probe_modelscope_dataset(repo_id: str, *, root: str = "/", timeout_sec: int = 20) -> dict[str, Any]:
    """Check whether a trusted ModelScope mirror exists for a HuggingFace dataset repo.

    Returns the ModelScope API tree (paths, sizes, sha256) so callers can prove that the
    mirror copy is bit-identical to the HF LFS object (compare against HF `lfs.oid`).
    """
    api = (
        "https://modelscope.cn/api/v1/datasets/"
        f"{repo_id}/repo/tree?Revision=master&Root={urllib.parse.quote(root, safe='/')}"
        "&PageSize=1000&PageNumber=1"
    )
    req = urllib.request.Request(api, headers={"User-Agent": "UARCap-Q2-AssetTransfer/0.2"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec, context=_SSL_CONTEXT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"repo_id": repo_id, "available": False, "error": str(exc)}
    if payload.get("Code") != 200:
        return {"repo_id": repo_id, "available": False, "error": payload.get("Message", "unknown")}
    files = payload.get("Data", {}).get("Files", [])
    return {
        "repo_id": repo_id,
        "available": True,
        "resolve_base": f"https://modelscope.cn/datasets/{repo_id}/resolve/master",
        "files": [
            {"path": f.get("Path"), "size": f.get("Size"), "sha256": f.get("Sha256"), "type": f.get("Type")}
            for f in files
        ],
    }


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
