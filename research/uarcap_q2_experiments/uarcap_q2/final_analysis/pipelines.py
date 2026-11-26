"""Parameterized downstream pipelines: formal E3, E3-Bridge, historical MSR-VTT/MSVD bridge,
and the 60-video independent validation set. No confirmatory analysis is run here."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from .core import ARMS, norm_claim, read_jsonl, tokens

O = Path("/home/bn/uarcap_q2_runtime/outputs")
FORMAL = O / "formal_candidate_unvalidated"
E3 = O / "pilot_e3_exploratory"


def utcnow(): return datetime.now(timezone.utc).isoformat()


def formal_e3_plan(min_approved: int = 150, candidates_path: Path | None = None) -> dict:
    """Formal E3 generation plan; refuses to run below the pre-declared candidate floor."""
    p = candidates_path or (E3 / "E3_FORMAL_APPROVED.json")
    data = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"approved": [], "n_approved": 0}
    n = int(data.get("n_approved", len(data.get("approved", []))))
    approved = data.get("approved", [])
    plan = {
        "schema": "uarcap_q2.formal_e3_plan.v1", "utc": utcnow(),
        "n_approved_candidates": n, "min_required": min_approved,
        "eligible": n >= min_approved,
        "models": ["videollama_evcap_frozen_v08", "Qwen/Qwen3.8-27B-FP8"],
        "arms": ["E3_BASE", "E3_SUP", "E3_CON", "E3_NEU"],
        "planned_generations": n * 4 * 2,
        "status": "MACHINE_ASSISTED_UNVALIDATED" if n >= min_approved else "AWAITING_SUFFICIENT_CANDIDATES",
        "note": "Formal E3 remains exploratory until real human verification; no fabricated human gate.",
    }
    (E3 / "E3_FORMAL_PLAN.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def build_e3_bridge_candidates(eval_rows: list[dict], retrieval: list[dict], max_items: int = 60) -> dict:
    """Natural-conflict bridge: pair a model claim contradicted by the human reference with
    an external retrieved caption that overlaps but disagrees on a content token.

    This is a conservative, machine-assisted candidate generator; every pair remains
    CANDIDATE_PENDING_VERIFICATION and requires visual/human checks before use.
    """
    ret = {r["query_id"]: [c.get("caption", "") for c in (r.get("reranked_d") or [])[:20]] for r in retrieval}
    neg = re.compile(r"\b(not|no|never|without|instead of|rather than)\b", re.I)
    out = []
    for r in eval_rows:
        if r.get("parse_status") != "OK": continue
        for c in r.get("claims", []):
            if c.get("label") != "contradicted": continue
            ct = tokens(c["claim"])
            for cap in ret.get(r["query_id"], []):
                kt = tokens(cap)
                if len(ct & kt) < 2: continue
                if (bool(neg.search(cap)) != bool(neg.search(c["claim"]))) or len(ct & kt) / max(1, len(ct | kt)) >= 0.4:
                    out.append({"query_id": r["query_id"], "external_caption": cap, "model_claim": c["claim"],
                                "reference_label": "contradicted", "status": "CANDIDATE_PENDING_VERIFICATION",
                                "needs_visual_check": True, "needs_human_check": True})
                    break
        if len(out) >= max_items: break
    payload = {"schema": "uarcap_q2.e3_bridge_candidates.v1", "utc": utcnow(), "n": len(out),
               "status": "MACHINE_ASSISTED_UNVALIDATED", "candidates": out}
    (O / "final_analysis").mkdir(parents=True, exist_ok=True)
    (O / "final_analysis/E3_BRIDGE_CANDIDATES.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_historical_metrics(roots: list[Path] | None = None) -> list[dict]:
    """Read already-computed historical MSR-VTT/MSVD metrics; never rerun them."""
    roots = roots or [Path("/home/bn/uarcap_q2_runtime/outputs"), Path("/home/bn/videoufo_research")]
    rows = []
    for root in roots:
        if not root.exists(): continue
        for p in root.rglob("*.json"):
            name = p.name.lower()
            if not any(k in name for k in ("metric", "result", "eval", "cider", "msrvtt", "msvd")):
                continue
            try: d = json.loads(p.read_text(encoding="utf-8"))
            except Exception: continue
            if isinstance(d, dict) and any(k in d for k in ("CIDEr", "cider", "Bleu_4", "BLEU_4", "METEOR")):
                rows.append({"source": str(p), "dataset": "msrvtt" if "msrvtt" in str(p).lower() else ("msvd" if "msvd" in str(p).lower() else "unknown"),
                             "metrics": {k: d[k] for k in ("CIDEr", "cider", "Bleu_4", "BLEU_4", "METEOR", "ROUGE_L", "SPICE") if k in d}})
    return rows


def prepare_validation_package(out_dir: Path | None = None) -> dict:
    """Build a blind 60-video E2 validation package from frozen predictions/eval/frames."""
    out = Path(out_dir or (O / "final_analysis/validation_60"))
    out.mkdir(parents=True, exist_ok=True)
    from uarcap_q2.annotation.package_builder import build_blind_e2_annotation_package
    manifest = json.loads((FORMAL / "FORMAL_CAREBENCH_ANALYSIS_MANIFEST.json").read_text(encoding="utf-8"))
    val_ids = set(manifest.get("e2_validation_set", {}).get("validation_query_ids", []))
    preds = {m: read_jsonl(FORMAL / m / "predictions.jsonl") for m in ("videollama", "qwen")}
    frames_idx = {r["query_id"]: r.get("frames", []) for r in read_jsonl(FORMAL / "formal_frame_indices.jsonl")}
    # choose the frozen modern model as the validation generator; keep claims from its eval
    ev = {r["query_id"]: r for r in read_jsonl(FORMAL / "qwen_eval/a2_ref_fact_eval.jsonl") if r["query_id"] in val_ids and r["parse_status"] == "OK"}
    videos = defaultdict(lambda: {"captions": []})
    for p in preds["qwen"]:
        if p["query_id"] in val_ids:
            v = videos[p["query_id"]]
            v.update({"query_id": p["query_id"], "dataset_id": "CaReBench", "video_path": p.get("query_id"),
                      "sampled_frame_paths": frames_idx.get(p["query_id"], [])})
            r = ev.get(p["query_id"])
            v["captions"].append({"arm": p["arm"], "model_id": p["model_id"], "caption": p["caption"],
                                  "claims": [{"claim_id": c.get("claim_id"), "claim": c["claim"]} for c in (r or {}).get("claims", [])]})
    for q, v in videos.items():
        v["video_path"] = f"/home/bn/uarcap_q2_runtime/data/carebench/videos/{q.replace('carebench_','')}.mp4"
    pkg = build_blind_e2_annotation_package(list(videos.values()), ["V1", "V2"], out_dir=out)
    receipt = {"schema": "uarcap_q2.validation60_package.v1", "utc": utcnow(), "n_videos": len(videos),
               "n_validation_ids": len(val_ids), "files": {k: str(v) for k, v in pkg.items()},
               "note": "Blind validation package; human verification pending."}
    (out / "VALIDATION60_RECEIPT.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt
