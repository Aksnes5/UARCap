"""Atomic file I/O, streaming SHA-256 digests, canonical hashing, and safe ZIP extraction."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable
import zipfile


def primitive_hash(value: Any) -> str:
    """Canonical SHA-256 over JSON-serializable primitives (compatible with v0.6 frozen builder)."""
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return sha256(raw).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    p = Path(path)
    h = sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(path: Path | str, data: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(target))
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return target


def atomic_write_text(path: Path | str, text: str, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: Path | str, obj: Any, indent: int = 2) -> Path:
    text = json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=False) + "\n"
    return atomic_write_text(path, text)


def atomic_write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> Path:
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    body = ("\n".join(lines) + "\n") if lines else ""
    return atomic_write_text(path, body)


def append_jsonl_durable(path: Path | str, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    with target.open("ab") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        s = raw.strip()
        if not s:
            continue
        try:
            out.append(json.loads(s))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSONL at {p}:{idx}: {exc}") from exc
    return out


def safe_extract_zip(zip_path: Path | str, dest_dir: Path | str, members: list[str] | None = None) -> list[Path]:
    """Zip-Slip protected extraction returning extracted file paths."""
    zpath = Path(zip_path)
    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zpath, "r") as zf:
        infos = [zf.getinfo(m) for m in members] if members is not None else zf.infolist()
        for info in infos:
            member_name = info.filename
            if member_name.startswith("/") or member_name.startswith("\\") or ".." in Path(member_name).parts:
                raise ValueError(f"Unsafe ZIP member path blocked (Zip-Slip): {member_name}")
            target = (dest / member_name).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"ZIP extraction target escapes destination: {member_name}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target.open("wb") as dst:
                while True:
                    buf = src.read(1024 * 1024)
                    if not buf:
                        break
                    dst.write(buf)
            extracted.append(target)
    return extracted
