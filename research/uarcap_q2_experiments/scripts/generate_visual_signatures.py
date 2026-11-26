"""Decode only frozen query videos and unique clips referenced by the D Top-7."""

import argparse
from contextlib import contextmanager
import csv
from functools import lru_cache
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uarcap_q2.common.io_utils import atomic_write_json, atomic_write_jsonl, primitive_hash, read_json, read_jsonl, sha256_file


@contextmanager
def video_file(path, member=None):
    """Read one declared regular archive member without writing archive paths."""
    if not member:
        yield path
        return
    member_path = PurePosixPath(member)
    if member_path.is_absolute() or ".." in member_path.parts or "\\" in member:
        raise ValueError("Unsafe archive member")
    with TemporaryDirectory() as tmp:
        dest = Path(tmp, "video" + member_path.suffix)
        if path.suffix.lower() == ".zip":
            with ZipFile(path) as archive:
                item = archive.getinfo(member)
                if item.is_dir() or item.file_size > 512 * 1024**2:
                    raise ValueError("Invalid archive video size/type")
                with archive.open(item) as source, dest.open("wb") as target:
                    shutil.copyfileobj(source, target)
        else:
            with tarfile.open(path) as archive:
                item = archive.getmember(member)
                if not item.isfile() or item.size > 512 * 1024**2:
                    raise ValueError("Invalid archive video size/type")
                with archive.extractfile(item) as source, dest.open("wb") as target:
                    shutil.copyfileobj(source, target)
        yield dest


@lru_cache(maxsize=1)
def cosine_basis():
    import numpy as np
    n = np.arange(32)
    return np.cos(np.pi * np.arange(8)[:, None] * (n + .5) / 32)


def phash(image):
    import numpy as np
    from PIL import Image
    arr = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    basis = cosine_basis()
    coeff = (basis @ arr @ basis.T).ravel()
    median = np.median(coeff[1:])
    return f"{int(''.join('1' if value > median else '0' for value in coeff), 2):016x}"


def cache_signature(path, member, cache, expected_sha=None):
    from uarcap_q2.datasets.video_decoder import decode_and_sample_video
    stat = path.stat()
    binding = primitive_hash(["dhash-phash-crop-16-v1", str(path.resolve()), member, stat.st_size,
                              stat.st_mtime_ns, stat.st_ctime_ns, expected_sha])
    target = cache / f"{binding}.json"
    if target.exists():
        row = read_json(target)
        if row["binding"] != binding:
            raise ValueError("Signature cache binding changed")
        return row
    with video_file(path, member) as video:
        frames, rep = decode_and_sample_video(video, num_frames=16)
    if not rep.ok or len(set(rep.sampled_indices)) != 16 or not rep.duration_sec:
        raise ValueError(f"Insufficient usable video frames: {rep.status}")
    if expected_sha and expected_sha != rep.video_sha256:
        raise ValueError("Video checksum mismatch")
    row = {"binding": binding, "asset_path": str(path), "member": member,
           "video_sha256": rep.video_sha256, "duration_seconds": rep.duration_sec,
           "ordered_dhash": rep.frame_dhashes, "ordered_phash": [phash(f) for f in frames],
           "ordered_center_phash": [phash(f.crop((f.width//10, f.height//10, f.width*9//10, f.height*9//10))) for f in frames],
           "frame_indices": rep.sampled_indices, "timestamps_seconds": rep.sampled_pts_sec,
           "warnings": rep.warnings}
    atomic_write_json(target, row)
    return row


def top7_ids(retrieval):
    ids = set()
    for row in retrieval:
        ranked = row["reranked_d"][:7]
        if len(ranked) != 7 or any(not item["clip_ids"] for item in ranked):
            raise ValueError("Incomplete frozen D Top-7")
        ids.update(item["clip_ids"][0] for item in ranked)
    return ids


def clip_locations(clips, location, ids):
    with clips.open(encoding="utf-8-sig", newline="") as stream:
        known = {r["clip_id"]: r["source_id"] for r in csv.DictReader(stream, delimiter="\t") if r["clip_id"] in ids}
    loc = {}
    if location is not None and location.is_file():
        for row in read_jsonl(location):
            cid = row["clip_id"]
            if cid not in ids:
                continue
            if cid in loc or known.get(cid) != row.get("source_id") or not known.get(cid):
                raise ValueError(f"Ambiguous or unbound official clip location: {cid}")
            loc[cid] = row
    return known, loc


def index_locations(clips, retrieval, inventory, out):
    """Resolve frozen representatives using a declared list of video roots or shards."""
    ids = top7_ids(read_jsonl(retrieval))
    known, _ = clip_locations(clips, None, ids)
    aliases = {cid if cid.endswith(".mp4") else f"{cid}.mp4": cid for cid in known}
    loc = {}
    for asset in read_json(inventory)["assets"]:
        path = Path(asset["path"])
        candidates = {}
        if path.is_dir():
            for p in path.rglob("*.mp4"):
                if p.is_file() and p.name in aliases:
                    cid = aliases[p.name]
                    candidates.setdefault(cid, []).append({"path":str(p.resolve())})
        elif path.suffix.lower()==".zip":
            with ZipFile(path) as z:
                for member in z.infolist():
                    cid=aliases.get(PurePosixPath(member.filename).name)
                    if cid and not member.is_dir():
                        candidates.setdefault(cid,[]).append({"path":str(path.resolve()),"member":member.filename})
        else:
            with tarfile.open(path) as archive:
                for member in archive:
                    cid = aliases.get(PurePosixPath(member.name).name)
                    if member.isfile() and cid:
                        candidates.setdefault(cid, []).append({"path":str(path.resolve()),"member":member.name})
        for cid, matches in candidates.items():
            loc.setdefault(cid, []).extend(matches)
    rows = [{"clip_id":cid,"source_id":known[cid],**matches[0]} for cid,matches in sorted(loc.items()) if len(matches)==1]
    atomic_write_jsonl(out, rows)
    atomic_write_json(out.with_suffix(".receipt.json"), {"schema":"uarcap_q2.clip_locator_receipt.v2",
        "input_sha256":{str(p):sha256_file(p) for p in (clips,retrieval,inventory)},
        "n_retrieved_unique":len(ids),"n_official_ids":len(known),"n_resolved":len(rows),
        "ambiguous_clip_ids":sorted(cid for cid,matches in loc.items() if len(matches)>1),
        "missing_clip_ids":sorted(ids-loc.keys()),"locations_sha256":sha256_file(out)})


def factorial(plan, retrieval, clips, location, out):
    from vlm_statement_verify import dense_frames
    ids = read_json(plan)["query_ids"]
    if len(ids) != 40 or len(set(ids)) != 40:
        raise ValueError("Need the fixed 40-query factorial subset")
    ranks = {r["query_id"]: r["reranked_d"][0]["clip_ids"][0] for r in read_jsonl(retrieval)}
    _, locations = clip_locations(clips, location, {ranks[q] for q in ids})
    cache = {}
    n = 8
    for cid in sorted({ranks[q] for q in ids}):
        row = locations.get(cid)
        if not row:
            raise ValueError(f"Cannot run factorial without verified Top-1 clip: {cid}")
        with video_file(Path(row["path"]), row.get("member")) as path:
            info = dense_frames(path, out.parent / "frames" / primitive_hash(cid), n)
        if row.get("sha256") and info["source_video_sha256"] != row["sha256"]:
            raise ValueError(f"Top-1 byte mismatch: {cid}")
        cache[cid] = {"clip_id": cid, **info}
    atomic_write_json(out, {"schema": "uarcap_q2.factorial_frames.v1", "num_frames": n,
                            "queries": {q: cache[ranks[q]] for q in ids},
                            "input_sha256": {str(p): sha256_file(p) for p in (plan, retrieval, clips, location)}})


def generate(candidates, retrieval, clips, location, out):
    cands = read_jsonl(candidates)
    ranks = read_jsonl(retrieval)
    if len({r["query_id"] for r in cands}) != len(cands) or {r["query_id"] for r in cands} != {r["query_id"] for r in ranks} or len(ranks) != len(cands):
        raise ValueError("Query and retrieval manifests differ")
    ids = top7_ids(ranks)
    known, loc = clip_locations(clips, location, ids)

    def sign(key, identifier, path, member=None, checksum=None):
        try:
            if not path:
                raise FileNotFoundError("No asset binding for frozen identity")
            return {key: identifier, **cache_signature(Path(path), member, out / "cache", checksum)}
        except (ValueError, OSError, KeyError, tarfile.TarError, BadZipFile) as exc:
            return {key: identifier, "status": "UNVERIFIED", "reason": str(exc)}

    query_rows = [sign("query_id", r["query_id"], r["video_path"], checksum=r.get("video_sha256")) for r in cands]
    clip_rows = []
    for cid in sorted(ids):
        row = loc.get(cid, {})
        clip_rows.append(sign("clip_id", cid, row.get("path"), row.get("member"), row.get("sha256")))
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(out / "query_signatures.jsonl", query_rows)
    atomic_write_jsonl(out / "clip_signatures.jsonl", clip_rows)
    result = {"schema": "uarcap_q2.visual_signature_inventory.v1", "queries": len(query_rows),
              "unique_top7_clips": len(clip_rows), "official_clip_ids_resolved": len(known),
              "unverified_queries": sum("video_sha256" not in r for r in query_rows),
              "unverified_clips": sum("video_sha256" not in r for r in clip_rows),
              "inputs_sha256": {str(p): sha256_file(p) if p.is_file() else None for p in (candidates, retrieval, clips, location)},
              "outputs_sha256": {name: sha256_file(out / name) for name in ("query_signatures.jsonl", "clip_signatures.jsonl")}}
    atomic_write_json(out / "SIGNATURE_INVENTORY.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command, input_flag in (("sign", "candidates"), ("factorial", "plan"), ("index", "inventory")):
        item = sub.add_parser(command)
        flags = (input_flag, "retrieval", "clips", "out") + (() if command=="index" else ("location",))
        for name in flags:
            item.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "sign":
        generate(args.candidates, args.retrieval, args.clips, args.location, args.out)
    elif args.command == "factorial":
        factorial(args.plan, args.retrieval, args.clips, args.location, args.out)
    else:
        index_locations(args.clips, args.retrieval, args.inventory, args.out)
