#!/usr/bin/env python3
"""Pilot A2 preliminary auto-evaluation diagnostics (UNCALIBRATED — not human Gold).

Reads A1 predictions, the frozen core-event lock, and A2 auto-eval rows. Produces
per-arm fact counts, core-event coverage, arm-to-arm fact transitions, static/dynamic
breakdowns, failure rates, representative change cases, and a human-annotation workload
estimate after within-video duplicate reuse.

This is descriptive only: no significance tests, no sample selection.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uarcap_q2.common.io_utils import atomic_write_json, atomic_write_text, read_json, read_jsonl  # noqa: E402

ARMS = ["ARM_A0", "ARM_A1", "ARM_A7"]
TRANSITIONS = [("ARM_A0", "ARM_A1"), ("ARM_A1", "ARM_A7"), ("ARM_A0", "ARM_A7")]
FOCUS = ["carebench_v_00002093_8", "carebench_v_00008320_0", "carebench_v_00010336_1", "carebench_v_00013412_0", "carebench_v_00015614_0"]


def _norm(text: str) -> str:
    s = re.sub(r"[^a-z0-9'\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", s).strip()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", (text or "").lower()))


def _jaccard(a: str, b: str) -> float:
    sa, sb = _tokens(a), _tokens(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2-dir", type=Path, required=True)
    ap.add_argument("--a1-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    eval_rows = read_jsonl(args.a2_dir / "a2_ref_fact_eval.jsonl")
    preds = read_jsonl(args.a1_dir / "predictions.jsonl")
    lock = read_json(args.a2_dir / "a2_core_event_lock.json")
    core_by_qid = {e["query_id"]: e for e in lock["entries"]}
    qids = [e["query_id"] for e in lock["entries"]]

    by_q_arm = {(r["query_id"], r["arm"]): r for r in eval_rows}
    ok_rows = [r for r in eval_rows if r.get("parse_status") == "OK"]
    fail_rows = [r for r in eval_rows if r.get("parse_status") != "OK"]

    arm_stats = {}
    for arm in ARMS:
        rows = [r for r in ok_rows if r["arm"] == arm]
        n_claims = sum(r["n_claims"] for r in rows)
        arm_stats[arm] = {
            "n_captions": sum(1 for r in eval_rows if r["arm"] == arm),
            "n_captions_parsed_ok": len(rows),
            "n_claims": n_claims,
            "entailed": sum(r["n_entail_ref"] for r in rows),
            "contradicted": sum(r["n_contra_ref"] for r in rows),
            "unmentioned": sum(r["n_unment_ref"] for r in rows),
            "mean_claims_per_caption": round(n_claims / len(rows), 3) if rows else None,
            "support_rate": round(sum(r["n_entail_ref"] for r in rows) / n_claims, 4) if n_claims else None,
            "contradiction_rate": round(sum(r["n_contra_ref"] for r in rows) / n_claims, 4) if n_claims else None,
            "unmentioned_rate": round(sum(r["n_unment_ref"] for r in rows) / n_claims, 4) if n_claims else None,
            "mean_core_event_coverage": round(
                sum(r["r_core_ref"] for r in rows if r.get("r_core_ref") is not None)
                / max(1, sum(1 for r in rows if r.get("r_core_ref") is not None)),
                4,
            )
            if rows
            else None,
        }

    dims = ("static_entity_attribute", "dynamic_action_temporal", "mixed", "unclassified")
    dim_stats = {arm: {d: Counter() for d in dims} for arm in ARMS}
    detail_dims = ("object_entity", "attribute", "action_event", "relation_spatial", "mixed", "unclassified")
    detail_stats = {arm: {d: Counter() for d in detail_dims} for arm in ARMS}
    for r in ok_rows:
        for d, counts in (r.get("auxiliary_dimension_breakdown") or {}).items():
            if d in dims:
                dim_stats[r["arm"]][d].update(counts)
        for d, counts in (r.get("auxiliary_dimension_detail_breakdown") or {}).items():
            if d in detail_dims:
                detail_stats[r["arm"]][d].update(counts)
    dim_report = {
        arm: {d: dict(dim_stats[arm][d]) for d in dims}
        for arm in ARMS
    }
    detail_report = {
        arm: {d: dict(detail_stats[arm][d]) for d in detail_dims}
        for arm in ARMS
    }

    transitions = {}
    for src, dst in TRANSITIONS:
        name = f"{src}->{dst}"
        added = removed = flips = replacements = 0
        examples_added, examples_removed, examples_replaced = [], [], []
        for qid in qids:
            a = by_q_arm.get((qid, src))
            b = by_q_arm.get((qid, dst))
            if not a or not b or a.get("parse_status") != "OK" or b.get("parse_status") != "OK":
                continue
            ac = {_norm(c["claim"]): c for c in a.get("claims", [])}
            bc = {_norm(c["claim"]): c for c in b.get("claims", [])}
            add_keys = [k for k in bc if k not in ac]
            rem_keys = [k for k in ac if k not in bc]
            added += len(add_keys)
            removed += len(rem_keys)
            for k in set(ac) & set(bc):
                if ac[k]["label"] != bc[k]["label"]:
                    flips += 1
            used_rem = set()
            for ak in add_keys:
                best = None
                for rk in rem_keys:
                    if rk in used_rem:
                        continue
                    j = _jaccard(bc[ak]["claim"], ac[rk]["claim"])
                    if j >= 0.6 and (best is None or j > best[1]):
                        best = (rk, j)
                if best:
                    used_rem.add(best[0])
                    replacements += 1
                    if len(examples_replaced) < 6:
                        examples_replaced.append(
                            {"query_id": qid, "from": ac[best[0]]["claim"], "to": bc[ak]["claim"], "jaccard": round(best[1], 3),
                             "from_label": ac[best[0]]["label"], "to_label": bc[ak]["label"]}
                        )
                elif len(examples_added) < 8:
                    examples_added.append({"query_id": qid, "claim": bc[ak]["claim"], "label": bc[ak]["label"]})
            for rk in rem_keys:
                if rk not in used_rem and len(examples_removed) < 8:
                    examples_removed.append({"query_id": qid, "claim": ac[rk]["claim"], "label": ac[rk]["label"]})
        transitions[name] = {
            "added_facts": added,
            "removed_facts": removed,
            "label_flips": flips,
            "replacement_pairs_jaccard_ge_0.6": replacements,
            "examples_added": examples_added,
            "examples_removed": examples_removed,
            "examples_replaced": examples_replaced,
        }

    focus_cases = {}
    for qid in FOCUS:
        focus_cases[qid] = {
            "core_event_candidate": core_by_qid[qid]["core_event_candidate"],
            "reference_caption": core_by_qid[qid]["reference_caption"],
            "arms": {
                arm: (
                    None
                    if not by_q_arm.get((qid, arm))
                    else {
                        "caption": by_q_arm[(qid, arm)]["caption"],
                        "claims": [
                            {"claim": c["claim"], "label": c["label"], "dimension": c.get("fact_dimension_aux")}
                            for c in by_q_arm[(qid, arm)].get("claims", [])
                        ],
                        "core_coverage": by_q_arm[(qid, arm)].get("r_core_ref"),
                        "parse_status": by_q_arm[(qid, arm)].get("parse_status"),
                    }
                )
                for arm in ARMS
            },
        }

    # Within-video duplicate reuse + human workload estimate
    per_video = defaultdict(list)
    for p in preds:
        per_video[p["query_id"]].append(p)
    unique_captions_total = 0
    unique_claims_total = 0
    total_claims = 0
    for qid, ps in per_video.items():
        unique_captions_total += len({p["caption"] for p in ps})
        norms = set()
        for p in ps:
            rows = [r for r in ok_rows if r["query_id"] == qid and r["arm"] == p["arm"]]
            for r in rows:
                for c in r.get("claims", []):
                    norms.add(_norm(c["claim"]))
                    total_claims += 1
        unique_claims_total += len(norms)

    n_videos = len(per_video)
    reused_caption_cards = len(preds) - unique_captions_total
    # Conservative human estimate: 90s video+core per video; 8s per unique claim; 6s per unique caption read.
    est_min_low = (n_videos * (90 + 30) + unique_claims_total * 8 + unique_captions_total * 6) / 60
    est_min_high = (n_videos * (150 + 60) + unique_claims_total * 15 + unique_captions_total * 10) / 60

    workload = {
        "n_videos": n_videos,
        "n_caption_cards_in_blind_bundle": len(preds),
        "n_unique_captions_within_video": unique_captions_total,
        "n_reusable_duplicate_caption_cards": reused_caption_cards,
        "n_total_atomic_claims": total_claims,
        "n_unique_normalized_claims_within_video": unique_claims_total,
        "n_claim_labels_avoidable_via_within_video_reuse": total_claims - unique_claims_total,
        "estimated_minutes_range": [round(est_min_low, 1), round(est_min_high, 1)],
        "estimate_basis": "90-150s video watch + 30-60s core events per video; 8-15s per unique claim; 6-10s per unique caption; duplicates within a video reuse labels (blind).",
    }

    report = {
        "schema_version": "uarcap_q2.pilot_a2_auto_diagnostics.v1",
        "status": "AUTO_EVAL_UNCALIBRATED",
        "warning": "All fact labels are automatic judge outputs and are NOT yet validated by human annotators. Do not interpret as accuracy, hallucination rate, or significance.",
        "n_eval_rows": len(eval_rows),
        "n_parse_failures": len(fail_rows),
        "parse_failure_rate": round(len(fail_rows) / len(eval_rows), 4) if eval_rows else None,
        "parse_failure_rows": [
            {"query_id": r["query_id"], "arm": r["arm"], "error": r.get("parse_error", "")[:300]} for r in fail_rows[:20]
        ],
        "arm_stats": arm_stats,
        "fact_dimension_breakdown": dim_report,
        "fact_dimension_detail_breakdown": detail_report,
        "transitions": transitions,
        "focus_videos": focus_cases,
        "workload_estimate": workload,
        "core_event_lock_sha256": lock.get("lock_sha256"),
    }
    atomic_write_json(args.a2_dir / "a2_auto_diagnostics.json", report)

    lines = [
        "# Pilot A2 — Preliminary Automatic Diagnostics",
        "",
        "> **STATUS: AUTO_EVAL_UNCALIBRATED.** These labels come from the automatic judge and have **not** been validated by the two human annotators. They must not be reported as accuracy, hallucination rate, or significance.",
        "",
        "## Per-arm automatic fact counts",
        "",
        "| Arm | Captions | Parsed OK | Atomic facts | Entailed | Contradicted | Unmentioned | Support rate | Contradiction rate | Unmentioned rate | Mean core-event coverage |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in ARMS:
        s = arm_stats[arm]
        lines.append(
            f"| {arm} | {s['n_captions']} | {s['n_captions_parsed_ok']} | {s['n_claims']} | {s['entailed']} | {s['contradicted']} | {s['unmentioned']} | "
            f"{s['support_rate']} | {s['contradiction_rate']} | {s['unmentioned_rate']} | {s['mean_core_event_coverage']} |"
        )
    lines += [
        "",
        f"- Automatic evaluation failures: **{len(fail_rows)} / {len(eval_rows)}** (rate {report['parse_failure_rate']}).",
        "- `UNMENTIONED` is not counted as a factual error.",
        "",
        "## Fact transitions (descriptive only)",
        "",
        "| Transition | Added | Removed | Label flips | Replacement pairs (Jaccard>=0.6) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, t in transitions.items():
        lines.append(f"| {name} | {t['added_facts']} | {t['removed_facts']} | {t['label_flips']} | {t['replacement_pairs_jaccard_ge_0.6']} |")
    lines += ["", "## Static vs dynamic auxiliary breakdown", ""]
    for arm in ARMS:
        parts = []
        for d in dims:
            c = dim_report[arm][d]
            parts.append(f"{d}: total={c.get('total',0)}, ent={c.get('entailed',0)}, con={c.get('contradicted',0)}, unm={c.get('unmentioned',0)}")
        lines.append(f"- **{arm}** — " + "; ".join(parts))
    lines += ["", "## Fine-grained object / attribute / action / relation breakdown", ""]
    for arm in ARMS:
        parts = []
        for d in detail_dims:
            c = detail_report[arm][d]
            parts.append(f"{d}: total={c.get('total',0)}, ent={c.get('entailed',0)}, con={c.get('contradicted',0)}, unm={c.get('unmentioned',0)}")
        lines.append(f"- **{arm}** — " + "; ".join(parts))
    lines += ["", "## Representative automatic change cases", ""]
    for name, t in transitions.items():
        lines.append(f"### {name}")
        for ex in t["examples_added"][:3]:
            lines.append(f"- added: `{ex['query_id']}` — {ex['claim']} [{ex['label']}]")
        for ex in t["examples_replaced"][:3]:
            lines.append(f"- replaced: `{ex['query_id']}` — “{ex['from']}” → “{ex['to']}” [{ex['from_label']}→{ex['to_label']}]")
        for ex in t["examples_removed"][:2]:
            lines.append(f"- removed: `{ex['query_id']}` — {ex['claim']} [{ex['label']}]")
        lines.append("")
    lines += ["## Focus videos", ""]
    for qid, fc in focus_cases.items():
        lines.append(f"### `{qid}`")
        lines.append(f"- Core-event candidate (locked pre-eval): {fc['core_event_candidate']}")
        for arm in ARMS:
            a = fc["arms"][arm]
            if not a:
                lines.append(f"- **{arm}**: missing")
                continue
            lines.append(
                f"- **{arm}** ({a['parse_status']}, core={a['core_coverage']}): {a['caption']}  \n"
                + "  "
                + "; ".join(f"{c['claim']} [{c['label']}/{c['dimension']}]" for c in a["claims"])
            )
        lines.append("")
    w = workload
    lines += [
        "## Human annotation workload (after within-video duplicate reuse)",
        "",
        f"- Videos: **{w['n_videos']}**; caption cards in blind bundle: **{w['n_caption_cards_in_blind_bundle']}**.",
        f"- Unique captions within video: **{w['n_unique_captions_within_video']}** (reusable duplicate cards: {w['n_reusable_duplicate_caption_cards']}).",
        f"- Atomic claims: **{w['n_total_atomic_claims']}** total, **{w['n_unique_normalized_claims_within_video']}** unique within video; "
        f"{w['n_claim_labels_avoidable_via_within_video_reuse']} labels can be reused blindly within the same video.",
        f"- Estimated annotation effort: **{w['estimated_minutes_range'][0]}–{w['estimated_minutes_range'][1]} minutes** per annotator (basis: {w['estimate_basis']}).",
        "",
        "The blind bundle still covers all 90 captions and preserves arm identity only in the secret key. Duplicate reuse never changes experimental coupling or statistical weights.",
        "",
    ]
    atomic_write_text(args.a2_dir / "PILOT_A2_AUTO_DIAGNOSTICS.md", "\n".join(lines))
    print(json.dumps({"status": report["status"], "arm_stats": arm_stats, "transitions": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("examples")} for k, v in transitions.items()}, "workload": workload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
