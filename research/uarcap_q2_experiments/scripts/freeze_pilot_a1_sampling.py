#!/usr/bin/env python3
"""Freeze the Pilot A1 CaReBench 30-video stratified sampling manifest.

The pilot's A0 preview is deterministic (seed=20261125), but a preview is not a
frozen artifact. This script re-derives the sample from the verified metadata,
checks it against the preview when supplied, and writes a SHA-256 pinned
manifest that downstream stages can audit against.

Usage:
  python scripts/freeze_pilot_a1_sampling.py \
    --metadata /home/bn/uarcap_q2_runtime/data/carebench/json/metadata.json \
    --video-root /home/bn/uarcap_q2_runtime/data/carebench/videos \
    --out /home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments/bn_carebench_pilot_a01/PILOT_A1_FROZEN_SAMPLING_MANIFEST.json \
    --preview /home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments/bn_carebench_pilot_a01/sampling_preview.json
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
from uarcap_q2.datasets.carebench import (  # noqa: E402
    load_carebench_records,
    sample_carebench_stratified_pilot,
)


def _manifest_sha(payload: dict) -> str:
    canonical = json.dumps({k: v for k, v in payload.items() if k != "manifest_sha256"}, sort_keys=True)
    return sha256(canonical.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Freeze Pilot A1 CaReBench sampling manifest")
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--preview", type=Path, default=None)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261125)
    args = ap.parse_args(argv)

    records = load_carebench_records(args.metadata, args.video_root, require_video_exists=False)
    selected, remaining, audit = sample_carebench_stratified_pilot(records, total_queries=args.n, seed=args.seed)

    selected_entries = []
    for rec in selected:
        meta = rec.metadata or {}
        selected_entries.append(
            {
                "query_id": rec.query_id,
                "video_filename": meta.get("video_filename"),
                "parent_video_id": meta.get("parent_video_id"),
                "clip_index": meta.get("clip_index"),
                "category": rec.domain,
                "subcategory": meta.get("subcategory"),
                "duration": rec.duration,
                "reference_caption_sha256": sha256((rec.reference_description or "").encode("utf-8")).hexdigest(),
                "video_path_expected": str(args.video_root / str(meta.get("video_filename"))),
            }
        )

    preview_check: dict = {"checked": False}
    if args.preview and args.preview.is_file():
        preview = json.loads(args.preview.read_text(encoding="utf-8"))
        preview_ids = [e.get("query_id") for e in preview.get("selected", [])]
        derived_ids = [e["query_id"] for e in selected_entries]
        preview_check = {
            "checked": True,
            "preview_path": str(args.preview),
            "preview_n": len(preview_ids),
            "derived_n": len(derived_ids),
            "identical": preview_ids == derived_ids,
            "preview_only": sorted(set(preview_ids) - set(derived_ids)),
            "derived_only": sorted(set(derived_ids) - set(preview_ids)),
            "preview_sha256": sha256_file(args.preview),
        }

    payload = {
        "schema_version": "uarcap_q2.pilot_a1_sampling_manifest.v1",
        "dataset": "CaReBench",
        "metadata_path": str(args.metadata),
        "metadata_sha256": sha256_file(args.metadata),
        "video_root": str(args.video_root),
        "seed": args.seed,
        "strategy": audit["strategy"],
        "n_selected": len(selected_entries),
        "n_unique_parents": audit["n_unique_pilot_parents"],
        "n_unique_subcategories": audit["n_unique_subcategories_in_pilot"],
        "category_pool_counts": audit["category_pool_counts"],
        "category_pilot_allocation": audit["category_pilot_allocation"],
        "remaining_formal_parent_pool_n": audit["remaining_formal_parent_pool_n"],
        "parent_video_dedup_audit": audit["parent_video_dedup_audit"],
        "selected": selected_entries,
        "preview_check": preview_check,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload["manifest_sha256"] = _manifest_sha(payload)
    atomic_write_json(args.out, payload)
    print(json.dumps({k: payload[k] for k in ("manifest_sha256", "n_selected", "n_unique_parents", "preview_check")}, indent=2))
    if preview_check.get("checked") and not preview_check.get("identical"):
        print("[ERROR] Frozen sample differs from the A0 preview!", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
