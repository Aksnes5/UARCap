#!/usr/bin/env python3
"""Prepare a 150-180 candidate formal E3 pool from VidHalluc ACH after VideoUFO/source
filtering and pilot-source exclusion. Machine-assisted, human-unvalidated."""

from __future__ import annotations
from datetime import datetime, timezone
import json, os
from pathlib import Path
import random, sys

BASE = Path(os.environ.get("OVERNIGHT_BASE", "/home/bn/uarcap_q2_runtime/overnight/20261126"))
REPO = "/home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments"
sys.path.insert(0, REPO)
WORK = Path("/home/bn/uarcap_q2_runtime/outputs/pilot_e3_exploratory")
VH = Path("/home/bn/uarcap_q2_runtime/data/vidhalluc")


def utcnow(): return datetime.now(timezone.utc).isoformat()


def main() -> int:
    import yaml
    from uarcap_q2.datasets.vidhalluc import load_vidhalluc_ach_mcq, deduplicate_and_filter_vidhalluc
    cfg = yaml.safe_load(Path(f"{REPO}/configs/bn_paths.local.yaml").read_text(encoding="utf-8"))
    clips = Path(cfg["historical_frozen_assets"]["clips_full_tsv_path"])
    vufo = {ln.split("\t")[0].strip().rsplit(".", 1)[0] for ln in clips.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("clip_id")}
    raw = json.loads((VH/"ach_mcq.json").read_text(encoding="utf-8"))
    probe = next(iter(next(iter(raw.values())).keys()))
    root = next((p.parent for p in (VH/"videos").rglob(f"{probe}.mp4")), VH/"videos")
    recs = load_vidhalluc_ach_mcq(VH/"ach_mcq.json", root)
    kept, audit = deduplicate_and_filter_vidhalluc(recs, vufo)
    pilot = json.loads((WORK/"E3_PILOT_EXCLUSION.json").read_text())
    pilot_sources = set(pilot.get("excluded_source_ids", [])); pilot_qids = set(pilot.get("excluded_query_ids", []))
    pool = [r for r in kept if r.source_id not in pilot_sources and r.query_id not in pilot_qids]
    rng = random.Random(20261125)
    chosen = sorted(rng.sample(pool, min(180, len(pool))), key=lambda r: r.query_id)
    rows = [{"query_id": r.query_id, "source_id": r.source_id, "video_path": r.video_path,
             "true_action": r.target_true_fact, "conflict_action": r.candidate_conflict_fact,
             "question": (r.metadata or {}).get("question")} for r in chosen]
    payload = {"schema": "uarcap_q2.e3_formal_candidates.v1", "status": "MACHINE_ASSISTED_UNVALIDATED",
               "utc": utcnow(), "n_requested": 180, "n_prepared": len(rows), "n_pool_after_filters": len(pool),
               "source_audit": audit, "pilot_excluded_sources": len(pilot_sources), "candidates": rows}
    (WORK/"E3_FORMAL_CANDIDATES_180.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n_prepared": len(rows), "n_pool_after_filters": len(pool), "source_audit": audit}, indent=2)[:1500])
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
