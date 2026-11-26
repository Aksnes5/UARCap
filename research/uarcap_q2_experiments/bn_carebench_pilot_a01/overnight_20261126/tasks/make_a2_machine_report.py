#!/usr/bin/env python3
"""Assemble machine-assisted A2 annotations, cross-judge agreement, visual evidence and a
minimal human-review list. Never writes human Gold or changes human-calibration gates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path

BASE = Path(os.environ.get("OVERNIGHT_BASE", "/home/bn/uarcap_q2_runtime/overnight/20261126"))
PROGRESS = BASE / "progress" / "a2_machine_report.json"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []


def atomic_write(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def norm(s: str) -> str:
    import re
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    return " ".join(t for t in s.split() if t not in {"a", "an", "the"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2-dir", type=Path, default=Path("/home/bn/uarcap_q2_runtime/outputs/pilot_a2/pilot_a2_carebench_eval_v1"))
    ap.add_argument("--machine-dir", type=Path, default=Path("/home/bn/uarcap_q2_runtime/outputs/pilot_a2_machine"))
    ap.add_argument("--visual-dir", type=Path, default=Path("/home/bn/uarcap_q2_runtime/outputs/pilot_a2_machine_visual"))
    ap.add_argument("--out-dir", type=Path, default=Path("/home/bn/uarcap_q2_runtime/outputs/pilot_a2_machine"))
    ap.add_argument("--allow-missing-visual", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.a2_dir / "a2_ref_fact_eval.jsonl")
    lock = json.loads((args.a2_dir / "a2_core_event_lock.json").read_text(encoding="utf-8"))
    ref = {e["query_id"]: "\n".join(x for x in [
        ("Caption: " + e["reference_caption"]) if e.get("reference_caption") else "",
        ("Spatial: " + e["spatial_caption"]) if e.get("spatial_caption") else "",
        ("Temporal: " + e["temporal_caption"]) if e.get("temporal_caption") else "",
    ] if x) for e in lock["entries"]}
    cross = {(r["query_id"], r["norm_key"]): r for r in read_jsonl(args.machine_dir / "A2_CROSS_JUDGE_LABELS.jsonl")}
    visual = {(r["query_id"], r["norm_key"]): r for r in read_jsonl(args.visual_dir / "A2_VISUAL_VERIFICATION_RESULTS.jsonl")}
    if not visual and not args.allow_missing_visual:
        print("visual results missing; run with --allow-missing-visual for text-only report")
        return 0

    claims = {}
    for r in rows:
        if r.get("parse_status") != "OK":
            continue
        for c in r.get("claims", []):
            key = (r["query_id"], norm(c["claim"]))
            u = claims.setdefault(key, {"query_id": r["query_id"], "claim": c["claim"], "reference": ref[r["query_id"]],
                                        "gemma": c["label"], "llama": None, "visual": None, "arms": [], "claim_ids": []})
            u["arms"].append(r["arm"]); u["claim_ids"].append(c.get("claim_id"))
            if key in cross: u["llama"] = cross[key]["llama_label"]
            if key in visual:
                v = visual[key]
                u["visual"] = {"video_truth": v["video_truth"], "visible_in_16_frames": v["visible_in_16_frames"],
                               "evidence": v.get("evidence", ""), "frames_sha256": v.get("frames_sha256")}

    annotations = []
    review = []
    for key, u in sorted(claims.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        g, l, v = u["gemma"], u["llama"], u["visual"]
        reasons = []
        final = None
        if l is not None and g == l:
            final = g
            if v and v["video_truth"] != g:
                reasons.append("visual_conflicts_with_text_consensus"); final = None
        else:
            reasons.append("text_judges_disagree")
        if v:
            if final is None and l is not None and v["video_truth"] == l:
                final = l; reasons.append("visual_supports_second_judge")
            elif final is None and v["video_truth"] == g:
                final = g; reasons.append("visual_supports_first_judge")
            if v["visible_in_16_frames"] != "yes":
                reasons.append("evidence_not_fully_visible_in_16_frames")
                final = None if v["video_truth"] == "unverifiable" else final
        if g == "unmentioned":
            reasons.append("reference_unmentioned")
        rec = {
            "query_id": u["query_id"], "claim": u["claim"], "norm_key": key[1],
            "gemma": g, "llama": l, "visual_video_truth": (v or {}).get("video_truth"),
            "visual_visible_16f": (v or {}).get("visible_in_16_frames"),
            "machine_label": final, "requires_human_review": final is None,
            "reasons": sorted(set(reasons)), "arms": sorted(set(u["arms"])), "claim_ids": u["claim_ids"],
            "machine_assisted_only": True, "not_human_gold": True,
        }
        annotations.append(rec)
        if final is None or "visual_conflicts_with_text_consensus" in reasons or "evidence_not_fully_visible_in_16_frames" in reasons:
            frames_dir = args.a2_dir / "blind_annotation_package" / "frames" / u["query_id"]
            frames = sorted(str(p) for p in frames_dir.glob("frame_*.jpg"))
            review.append({
                "query_id": u["query_id"], "claim": u["claim"], "reference": u["reference"],
                "video_path": f"/home/bn/uarcap_q2_runtime/data/carebench/videos/{u['query_id'].replace('carebench_','')}.mp4",
                "sampled_frames": frames[:16], "model_opinions": {"gemma": g, "llama": l, "qwen_visual": v},
                "why_review": sorted(set(reasons)), "choices": ["entailed", "contradicted", "unmentioned", "split_error"],
            })

    atomic_write(args.out_dir / "A2_MACHINE_ANNOTATIONS.json", {
        "schema": "uarcap_q2.a2_machine_annotations.v1", "status": "MACHINE_ASSISTED_UNCALIBRATED",
        "generated_utc": utcnow(), "n_claims": len(annotations), "n_requires_human_review": len(review),
        "annotations": annotations,
    })
    atomic_write(args.out_dir / "A2_HUMAN_REVIEW_REQUIRED.json", {
        "schema": "uarcap_q2.a2_human_review_required.v1", "generated_utc": utcnow(),
        "n_items": len(review), "items": review,
    })

    by_arm = defaultdict(Counter)
    for a in annotations:
        for arm in a["arms"]:
            by_arm[arm]["total"] += 1
            by_arm[arm][a["machine_label"] or "REQUIRES_HUMAN_REVIEW"] += 1
    md = [
        "# Pilot A2 — Machine-Assisted Annotation Report",
        "",
        "> **MACHINE-ASSISTED / UNCALIBRATED.** These labels are not human Gold and do not satisfy the human validation gate.",
        f"- Generated: {utcnow()}",
        f"- Unique claims: {len(annotations)}; require human review: {len(review)}",
        "",
        "## Label sources",
        f"- Gemma 3 27B FP8 + Llama 3.3 70B W4A16 text judges; Qwen3.8 visual verification available for {len(visual)} flagged claims.",
        "",
        "## Per-arm machine labels (with 90-row experimental weights preserved)",
        "",
        "| Arm | Claims | Labels / review |",
        "| --- | --- | --- |",
    ]
    for arm in ("ARM_A0", "ARM_A1", "ARM_A7"):
        c = by_arm.get(arm, Counter())
        md.append(f"| {arm} | {c.get('total',0)} | {dict(c)} |")
    md += [
        "",
        "## Human review",
        f"See `A2_HUMAN_REVIEW_REQUIRED.json` — **{len(review)} items**, each with video path, 16 frame paths, the reference, all model opinions, and a single final-label choice.",
        "",
        "## Integrity notes",
        "- No human annotator name, signature, agreement or adjudication is generated.",
        "- `UNMENTIONED` is never counted as a factual error.",
        "- Qwen visual outputs are auxiliary and are not independent confirmation when Qwen also generates A3 text.",
    ]
    (args.out_dir / "A2_MACHINE_ASSISTED_CALIBRATION_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    atomic_write(PROGRESS, {"task": "a2_machine_report", "phase": "completed", "completed": len(annotations),
                            "total": len(annotations), "review_items": len(review), "updated_utc": utcnow()})
    print(json.dumps({"n_claims": len(annotations), "n_review": len(review), "by_arm": {k: dict(v) for k, v in by_arm.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
