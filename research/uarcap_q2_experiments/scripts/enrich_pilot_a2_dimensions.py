#!/usr/bin/env python3
"""Add the fine-grained object/attribute/action/relation/mixed/unclassified dimension
to an existing Pilot A2 auto-eval file WITHOUT changing any claim text or label.

Deterministic post-processing only; used when the judge run predates the field.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uarcap_q2.common.io_utils import atomic_write_json, atomic_write_jsonl, read_jsonl  # noqa: E402
from uarcap_q2.evaluators.reference_fact_eval import (  # noqa: E402
    classify_claim_fact_dimension,
    classify_claim_fact_dimension_detail,
)

DETAILS = ("object_entity", "attribute", "action_event", "relation_spatial", "mixed", "unclassified")


def _semantic_sha(rows: list[dict]) -> str:
    payload = [
        {
            "query_id": r["query_id"],
            "arm": r["arm"],
            "model_id": r["model_id"],
            "caption": r["caption"],
            "claims": [{"claim": c["claim"], "label": c["label"], "evidence": c.get("evidence", "")} for c in r.get("claims", [])],
            "core": [{"claim": c["claim"], "label": c["label"]} for c in r.get("core_event_verdicts", [])],
            "parse_status": r.get("parse_status"),
        }
        for r in rows
    ]
    return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", type=Path, required=True)
    args = ap.parse_args(argv)
    src = args.eval_jsonl
    rows = read_jsonl(src)
    before_sha = _semantic_sha(rows)

    counts = {d: 0 for d in DETAILS}
    changed = 0
    for r in rows:
        for c in r.get("claims", []):
            if not c.get("fact_dimension_aux"):
                c["fact_dimension_aux"] = classify_claim_fact_dimension(c["claim"])
            if not c.get("fact_dimension_detail"):
                c["fact_dimension_detail"] = classify_claim_fact_dimension_detail(c["claim"])
                changed += 1
            counts[c["fact_dimension_detail"]] = counts.get(c["fact_dimension_detail"], 0) + 1
        breakdown = {d: {"total": 0, "entailed": 0, "contradicted": 0, "unmentioned": 0} for d in DETAILS}
        for c in r.get("claims", []):
            b = breakdown[c["fact_dimension_detail"]]
            b["total"] += 1
            if c.get("label") in b:
                b[c["label"]] += 1
        r["auxiliary_dimension_detail_breakdown"] = breakdown

    after_sha = _semantic_sha(rows)
    if before_sha != after_sha:
        raise SystemExit("Semantic hash changed during enrichment; refusing to write")

    backup = src.with_suffix(".pre_detail_enrich.jsonl")
    if not backup.exists():
        shutil.copy2(src, backup)
    atomic_write_jsonl(src, rows)
    receipt = {
        "schema_version": "uarcap_q2.pilot_a2_dimension_enrichment.v1",
        "eval_jsonl": str(src),
        "backup_jsonl": str(backup),
        "n_rows": len(rows),
        "n_claims": sum(len(r.get("claims", [])) for r in rows),
        "n_new_detail_labels": changed,
        "detail_counts": counts,
        "semantic_sha256_before": before_sha,
        "semantic_sha256_after": after_sha,
        "claim_text_or_label_changed": False,
    }
    atomic_write_json(src.parent / "a2_fact_dimension_enrichment_receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("n_rows", "n_claims", "n_new_detail_labels", "detail_counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
