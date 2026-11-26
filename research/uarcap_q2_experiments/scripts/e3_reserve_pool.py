"""Prepare an outcome-blind reserve and combine only identically screened candidates."""

import argparse
import json
from pathlib import Path
import random
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e3_repair import YT, youtube_sources
from uarcap_q2.common.io_utils import atomic_write_json, primitive_hash, read_json, sha256_file
from uarcap_q2.datasets.vidhalluc import load_vidhalluc_ach_mcq
from uarcap_q2.interventions.neutral_repair import V2

SEED = 20261126
MINIMUM = 150
POLICY = "e3-reserve-v2-original-criteria-20261126"


def source_ids(path):
    ids = {s.strip() for s in path.read_text(encoding="utf-8").splitlines() if s.strip()}
    if any(not YT.fullmatch(s) for s in ids):
        raise ValueError(f"Malformed source ID in {path}")
    return ids


def make_pool(ach, video_root, clips, pilot, historical, formal, debug):
    p = read_json(pilot)
    f = read_json(formal)["candidates"]
    if len(p["excluded_source_ids"]) != 30 or len(set(p["excluded_source_ids"])) != 30 or len(f) != 180:
        raise ValueError("Require complete original 30 pilot and 180 formal source rosters")
    blocked = {"PILOT_SOURCE": set(p["excluded_source_ids"]), "VIDEOUFO_SOURCE": youtube_sources(clips),
               "HISTORICAL_SOURCE": source_ids(historical), "FORMAL_SOURCE": {r["source_id"] for r in f},
               "DEBUG_SOURCE": source_ids(debug)}
    if not blocked["HISTORICAL_SOURCE"] or any(not YT.fullmatch(s or "") for values in blocked.values() for s in values):
        raise ValueError("Source exclusions are not authoritative")
    records = load_vidhalluc_ach_mcq(ach, video_root)
    parsed = {r.metadata["clip_key"] for r in records}
    rejected = []
    observed = set()
    for group, clips_in_group in json.loads(ach.read_text(encoding="utf-8")).items():
        for key in clips_in_group:
            if key not in parsed or key in observed:
                rejected.append({"group": group, "clip_key": key, "reasons": ["DUPLICATE_RAW_CLIP" if key in observed else "UNPARSEABLE_MCQ"]})
            observed.add(key)
    seen = set()
    kept = []
    for row in records:
        sid = row.source_id
        reasons = [reason for reason, ids in blocked.items() if sid in ids]
        if not sid:
            reasons.append("UNKNOWN_SOURCE")
        if sid in seen:
            reasons.append("DUPLICATE_SOURCE")
        seen.add(sid)
        if reasons:
            rejected.append({"query_id": row.query_id, "source_id": sid, "reasons": reasons})
        else:
            kept.append({"query_id": row.query_id, "source_id": sid, "video_path": row.video_path,
                         "true_action": row.target_true_fact, "conflict_action": row.candidate_conflict_fact,
                         "question": row.metadata["question"]})
    random.Random(SEED).shuffle(kept)
    return {"schema": "uarcap_q2.e3_reserve_pool.v1", "policy": POLICY, "seed": SEED, "candidates": kept,
            "excluded": rejected, "input_sha256": {str(p): sha256_file(p) for p in (ach, clips, pilot, historical, formal, debug)}}


def require_approval(authorization, pool, path):
    auth = read_json(authorization)
    cap = auth.get("max_additional_candidates")
    if (auth.get("approved") is not True or not auth.get("approved_by") or not auth.get("protocol_revision")
            or auth.get("policy") != pool["policy"] or auth.get("pool_sha256") != sha256_file(path)
            or type(cap) is not int or cap < 1):
        raise ValueError("Approval must explicitly bind the revised design, candidate cap and frozen reserve")
    return cap


def combine(base, reserve, pool, cap):
    if base["schema"] != V2 or any(r.get("human_verification") != "PENDING" for r in base["approved"]):
        raise ValueError("Require original V2 machine-only screen")
    if base["n_approved"] >= MINIMUM:
        return base
    approved = list(base["approved"])
    rejected = list(base["rejected"])
    if reserve is not None:
        if reserve["schema"] != V2 or reserve["config_sha256"] != base["config_sha256"]:
            raise ValueError("Reserve must use identical screen configuration")
        allow = {r["query_id"]: r for r in pool["candidates"][:cap]}
        screened = [*reserve["approved"], *reserve["rejected"]]
        if len({r["query_id"] for r in screened}) != len(screened) or allow.keys() != {r["query_id"] for r in screened}:
            raise ValueError("Screen coverage does not equal the authorized reserve prefix")
        by_id = {r["query_id"]: r for r in reserve["approved"]}
        available = MINIMUM - len(approved)
        for q, candidate in allow.items():
            row = by_id.get(q)
            if row:
                if row["original_candidate_sha256"] != primitive_hash(candidate) or row.get("human_verification") != "PENDING":
                    raise ValueError("Reserve candidate or verification scope changed")
                if available:
                    approved.append(row)
                    available -= 1
                else:
                    rejected.append({"query_id": q, "reason": "QUALIFIED_AFTER_COHORT_FILLED"})
        rejected.extend(reserve["rejected"])
    for key in ("query_id", "source_id"):
        if not all(r.get(key) for r in approved) or len({r[key] for r in approved}) != len(approved):
            raise ValueError("Duplicate or unidentified original/reserve candidate")
    eligible = len(approved) >= MINIMUM
    return {**base, "status": "MACHINE_ASSISTED_UNVALIDATED" if eligible else "E3_INSUFFICIENT_VALID_CANDIDATES",
            "eligible": eligible, "n_approved": len(approved), "approved": approved, "rejected": rejected,
            "original_freeze_sha256": primitive_hash(base), "reserve_freeze_sha256": primitive_hash(reserve) if reserve else None,
            "reserve_policy": POLICY, "reserve_cap": cap, "reserve_pool_sha256": primitive_hash(pool) if pool else None}


def continue_screen(args):
    from vlm_statement_verify import dense_frames
    base = read_json(args.base)
    path = args.pool
    if base["n_approved"] >= MINIMUM or not path.is_file() or not args.approval.is_file():
        result = combine(base, None, None, 0)
        if not result["eligible"]:
            result["reserve_state"] = "BLOCKED_MISSING_RESERVE_OR_APPROVAL"
        atomic_write_json(args.out, result)
        if not result["eligible"]:
            print(f"HARD_BLOCK E3_INSUFFICIENT_VALID_CANDIDATES {result['n_approved']}/{MINIMUM}")
        return 0 if result["eligible"] else 3
    pool = read_json(path)
    cap = require_approval(args.approval, pool, path)
    work = args.out.parent / "reserve"
    work.mkdir(parents=True, exist_ok=True)
    selected = pool["candidates"][:cap]
    # The prepared pool binds original bytes and candidate order. Missing media
    # remains a rejected candidate under the same screen rather than substituted.
    frames = work / "frames"
    for c in selected:
        q = c["query_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", q):
            raise ValueError("Unsafe VidHalluc query binding")
        try:
            dense_frames(c["video_path"], frames / q, 16)
        except (ValueError, OSError):
            continue
    inputs = work / "selection.json"
    proposals = work / "proposals.json"
    screened = work / "screened.json"
    jobs = work / "jobs.json"
    decoded = work / "decoded.json"
    verdicts = work / "verdicts.jsonl"
    reserve = work / "RESERVE_SCREEN_FREEZE.json"
    atomic_write_json(inputs, {"approved": selected, "reserve_pool_sha256": sha256_file(path)})
    root = Path(__file__).resolve().parent
    def launch(python, script, command, options):
        argv = [str(python), "-u", str(root / script), command]
        for key,value in options.items():
            argv.extend(("--" + key.replace("_","-"), str(value)))
        subprocess.run(argv, check=True)
    launch(args.vllm, "e3_repair.py", "prepare", dict(candidates=inputs, frames=frames,
           vicuna=args.vicuna, qwen=args.qwen, gemma_id=args.gemma_id, out=proposals,
           videoufo_clips=args.clips, historical_sources=args.historical, pilot_exclusions=args.pilot))
    if base["config_sha256"] != read_json(proposals)["config_sha256"]:
        raise ValueError("Reserve and original proposal screening configurations differ")
    lock = dict(lock=args.lock, gpu_uuid=args.gpu_uuid)
    launch(args.vllm, "e3_repair.py", "screen",
           dict(proposals=proposals, gemma=args.gemma, gemma_id=args.gemma_id, out=screened, jobs=jobs, **lock))
    launch(sys.executable, "vlm_statement_verify.py", "decode", dict(jobs=jobs, work=work/"dense", out=decoded))
    launch(args.vllm, "vlm_statement_verify.py", "verify", dict(jobs=jobs, frames=decoded, qwen=args.qwen, out=verdicts, **lock))
    from uarcap_q2.common.io_utils import read_jsonl
    from uarcap_q2.interventions.neutral_repair import freeze
    frozen = freeze(read_json(inputs), read_json(screened), read_jsonl(verdicts))
    atomic_write_json(reserve, frozen)
    result = combine(base, frozen, pool, cap)
    reserve_ids = {c["query_id"] for c in selected}
    for c in result["approved"]:
        if c["query_id"] in reserve_ids:
            c["verified_reserve_frames_dir"] = str(frames / c["query_id"])
    result["reserve_authorization_sha256"] = sha256_file(args.approval)
    if args.out.exists() and read_json(args.out).get("eligible") is True and read_json(args.out) != result:
        raise ValueError("Accepted combined freeze is immutable")
    atomic_write_json(args.out, result)
    if not result["eligible"]:
        print(f"HARD_BLOCK E3_INSUFFICIENT_VALID_CANDIDATES {result['n_approved']}/{MINIMUM}")
    return 0 if result["eligible"] else 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command, names, optional in (
        ("pool", ("ach", "video-root", "clips", "pilot", "historical", "formal", "debug", "out"), ()),
        ("select", ("base", "pool", "approval", "out"), ()),
        ("combine", ("base", "out"), ("pool", "approval", "reserve")),
        ("continue", ("base", "pool", "approval", "out", "vicuna","qwen","clips","historical","pilot","gemma","lock","vllm"), ()),
    ):
        item = sub.add_parser(command)
        for name in names + optional:
            item.add_argument("--" + name, type=Path, required=name in names)
        if command == "continue":
            item.add_argument("--gemma-id", required=True)
            item.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    if args.command == "continue":
        return continue_screen(args)
    if args.command == "pool":
        result = make_pool(args.ach, args.video_root, args.clips, args.pilot, args.historical, args.formal, args.debug)
    else:
        base = read_json(args.base)
        path = args.pool
        pool = read_json(path) if path else None
        cap = require_approval(args.approval, pool, path) if pool and args.approval else 0
        if args.command == "select":
            candidates = pool["candidates"][:cap] if base["n_approved"] < MINIMUM else []
            result = {"schema": "uarcap_q2.e3_reserve_selected.v1", "approved": candidates,
                      "selected_before_generation": True, "pool_sha256": sha256_file(path), "cap": cap}
        else:
            if args.reserve and not cap:
                raise ValueError("Unauthorized reserve supplementation")
            result = combine(base, read_json(args.reserve) if args.reserve else None, pool, cap)
    if args.out.exists() and read_json(args.out) != result:
        raise ValueError("Output version is immutable")
    atomic_write_json(args.out, result)


if __name__ == "__main__":
    raise SystemExit(main())
