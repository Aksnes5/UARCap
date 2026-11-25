#!/usr/bin/env python3
"""Verify Pilot A1 inputs and freeze Pilot A2 core-event candidates BEFORE any judge call.

Writes:
  - a2_input_verification.json : SHA-256 of every reused A1 artifact + metadata
  - a2_core_event_lock.json    : per-query human opening-overview core-event candidate,
                                 reference-field texts, and a single lock SHA-256

The core-event lock is derived only from the human CaReBench metadata and the frozen
sampling manifest; it never depends on any generated caption or evaluator output.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uarcap_q2.common.io_utils import atomic_write_json, sha256_file  # noqa: E402
from uarcap_q2.datasets.carebench import load_carebench_records  # noqa: E402

EXPECTED_A1_SHA256 = {
    "predictions.jsonl": "f958c87c83d31b7f28468e6637c22a3f45df338f36fc94f36c317f034a119428",
    "input_traces.jsonl": "cbf789fe4261f0431d4949c398eba424036c8c2d01400210112ef8dca2bbd133",
    "retrieval_candidates.jsonl": "14f29dde5ae9f5484b5ae404946b42a695e468eaf24821df34d32aa0dc86c2dd",
    "run_receipt.json": "e792d35ba5b9d3e8a45d5b7278a6d3542961426797a8be906f3e19b0b5703ae3",
}
EXPECTED_METADATA_SHA256 = "740f43b71a1f8131521e2a6cde95a59c3db10a5c4bdd554dc64959b48b0fc951"
# Internal manifest canonical digest (NOT the file's SHA-256).
EXPECTED_FROZEN_MANIFEST_SHA256 = "26fbf6abaf7ed12309ca5be40ccc583c068e5bd2c9b427f0753c4600d5b5bd21"


def _canonical_sha(payload: dict) -> str:
    return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1-dir", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, dict] = {}
    ok = True
    for name, expected in EXPECTED_A1_SHA256.items():
        p = args.a1_dir / name
        actual = sha256_file(p) if p.is_file() else None
        match = actual == expected
        ok = ok and match
        files[name] = {"path": str(p), "expected_sha256": expected, "actual_sha256": actual, "match": match}

    meta_sha = sha256_file(args.metadata)
    meta_match = meta_sha == EXPECTED_METADATA_SHA256
    ok = ok and meta_match

    receipt = json.loads((args.a1_dir / "run_receipt.json").read_text(encoding="utf-8"))
    a1_status = receipt.get("status")
    ok = ok and a1_status == "REAL_RUNTIME_PASS"

    manifest_path = args.a1_dir / "FROZEN_SAMPLING_MANIFEST.json"
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_internal_sha = frozen.get("manifest_sha256")
    manifest_match = manifest_internal_sha == EXPECTED_FROZEN_MANIFEST_SHA256
    ok = ok and manifest_match
    selected_ids = [e["query_id"] for e in frozen["selected"]]
    records = load_carebench_records(args.metadata, args.video_root, require_video_exists=False)
    by_id = {r.query_id: r for r in records}
    missing = [q for q in selected_ids if q not in by_id]
    ok = ok and not missing

    verification = {
        "schema_version": "uarcap_q2.pilot_a2_input_verification.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "a1_dir": str(args.a1_dir),
        "a1_run_status": a1_status,
        "a1_files": files,
        "metadata": {"path": str(args.metadata), "expected_sha256": EXPECTED_METADATA_SHA256, "actual_sha256": meta_sha, "match": meta_match},
        "frozen_manifest": {
            "path": str(manifest_path),
            "file_sha256": sha256_file(manifest_path),
            "internal_manifest_sha256": manifest_internal_sha,
            "expected_internal_sha256": EXPECTED_FROZEN_MANIFEST_SHA256,
            "match": manifest_match,
        },
        "n_frozen_queries": len(selected_ids),
        "missing_metadata_queries": missing,
        "all_verified": ok,
    }
    atomic_write_json(args.out_dir / "a2_input_verification.json", verification)
    if not ok:
        print(json.dumps(verification, indent=2))
        return 2

    core_entries = []
    for qid in selected_ids:
        rec = by_id[qid]
        core_entries.append(
            {
                "query_id": qid,
                "core_event_candidate": (rec.core_events[0] if rec.core_events else None),
                "core_event_source": "human_caption_opening_overview_sentence_candidate",
                "core_event_status": "UNVERIFIED_CANDIDATE_REQUIRES_PILOT_A2_HUMAN_AUDIT",
                "reference_caption": rec.reference_description,
                "human_reference_fields": (rec.metadata or {}).get("human_reference_fields", []),
                "spatial_caption": (rec.metadata or {}).get("spatial_caption"),
                "temporal_caption": (rec.metadata or {}).get("temporal_caption"),
                "llm_extracted_events_NOT_gold": (rec.metadata or {}).get("llm_extracted_events", []),
                "llm_extracted_objects_NOT_gold": (rec.metadata or {}).get("llm_extracted_objects", []),
            }
        )
    lock_body = {
        "schema_version": "uarcap_q2.pilot_a2_core_event_lock.v1",
        "dataset": "CaReBench",
        "created_before_any_judge_call": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_manifest_sha256": frozen.get("manifest_sha256"),
        "metadata_sha256": meta_sha,
        "n_queries": len(core_entries),
        "entries": core_entries,
    }
    lock_body["lock_sha256"] = _canonical_sha(lock_body)
    atomic_write_json(args.out_dir / "a2_core_event_lock.json", lock_body)
    print(json.dumps({"all_verified": True, "n_queries": len(core_entries), "lock_sha256": lock_body["lock_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
