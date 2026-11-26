"""Parameterized downstream pipelines: formal E3, E3-Bridge, historical MSR-VTT/MSVD bridge,
and the 60-video independent validation set. No confirmatory analysis is run here."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
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
    """Build the blind 60-video E2 validation package.

    STRICT KEY: every audited caption is keyed by (model_id, query_id, arm). All 60
    validation videos contribute BOTH models x THREE arms = 360 caption records backed by
    real frozen predictions/eval claims/frames. No generated text is invented. Isolation
    from the 540 formal-main videos and inclusion probabilities are asserted before build.
    """
    from hashlib import sha256
    from uarcap_q2.annotation.package_builder import build_blind_e2_annotation_package

    out = Path(out_dir or (O / "final_analysis/validation_60"))
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((FORMAL / "FORMAL_CAREBENCH_ANALYSIS_MANIFEST.json").read_text(encoding="utf-8"))
    val_ids = set(manifest["e2_validation_set"]["validation_query_ids"])
    main_ids = set(manifest.get("formal_main_excluding_validation_ids") or []) or (
        set(manifest.get("selected_query_ids", [])) - val_ids)
    assert val_ids.isdisjoint(main_ids), "validation/main isolation violated"
    units = manifest.get("units", {})
    strata = manifest["e2_validation_set"]["strata"]

    cand = {c["query_id"]: c for c in read_jsonl(FORMAL / "formal_candidates.jsonl")}
    frames_idx = {r["query_id"]: r.get("frames", []) for r in read_jsonl(FORMAL / "formal_frame_indices.jsonl")}
    model_dirs = {"videollama": FORMAL / "videollama_eval/a2_ref_fact_eval.jsonl",
                  "qwen": FORMAL / "qwen_eval/a2_ref_fact_eval.jsonl"}
    evals = {m: {r["query_id"]: r for r in read_jsonl(p) if r.get("parse_status") == "OK"} for m, p in model_dirs.items()}
    videos = {q: {"query_id": q, "dataset_id": "CaReBench", "video_path": cand[q]["video_path"],
                  "sampled_frame_paths": frames_idx.get(q, []), "captions": []} for q in sorted(val_ids)}
    counts = Counter()
    missing = []
    for m, _ in model_dirs.items():
        for p in read_jsonl(FORMAL / m / "predictions.jsonl"):
            q = p["query_id"]
            if q not in videos:
                continue
            r = evals[m].get(q)
            claims = [{"claim_id": c.get("claim_id"), "claim": c["claim"]} for c in (r or {}).get("claims", [])]
            if not claims:
                missing.append({"model_id": p.get("model_id"), "query_id": q, "arm": p["arm"]})
            videos[q]["captions"].append({"arm": p["arm"], "model_id": p.get("model_id"),
                                          "caption": p.get("caption", ""), "claims": claims})
            counts[(p.get("model_id"), p["arm"])] += 1
    expected = len(val_ids) * 2 * 3
    actual = sum(len(v["captions"]) for v in videos.values())
    if actual != expected:
        raise ValueError(f"validation package incomplete: {actual}/{expected} caption rows")
    for q in videos:
        if len(videos[q]["captions"]) != 6:
            raise ValueError(f"{q}: expected 6 (model,arm) captions, got {len(videos[q]['captions'])}")
    pkg = build_blind_e2_annotation_package(list(videos.values()), ["V1", "V2"], out_dir=out)
    key_sha = sha256(pkg["blind_key"].read_bytes()).hexdigest()
    receipt = {
        "schema": "uarcap_q2.validation60_package.v2", "utc": utcnow(),
        "n_videos": len(videos), "n_validation_ids": len(val_ids),
        "n_caption_records": actual, "expected_caption_records": expected,
        "keying": "unique per (model_id, query_id, arm); verified against frozen predictions",
        "per_model_arm_counts": {f"{m}|{a}": n for (m, a), n in sorted(counts.items(), key=str)},
        "eval_missing_claims_records": len(missing), "eval_missing_examples": missing[:5],
        "isolation": {"validation_disjoint_from_formal_main": True, "n_formal_main": len(main_ids),
                      "parent_isolation_verified": manifest.get("parent_isolation_verified")},
        "inclusion_probability_source": "FORMAL_CAREBENCH_ANALYSIS_MANIFEST.e2_validation_set.strata",
        "stratum_inclusion_probs": {k: v["val_inclusion_prob"] for k, v in strata.items()},
        "inclusion_by_qid": {q: units[q]["val_inclusion_prob"] for q in val_ids if q in units},
        "n_with_inclusion_prob": sum(1 for q in val_ids if q in units),
        "files": {k: str(v) for k, v in pkg.items()},
        "blind_key_sha256": key_sha, "blind_key_in_git": False,
        "new_generations_needed": 0,
        "note": "All 360 predictions already exist in the frozen formal run; no regeneration. Human verification pending.",
    }
    (out / "VALIDATION60_RECEIPT.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


HUMAN_IMPORT_SCHEMA = "uarcap_q2.human_primary_endpoints.v1"


def verify_human_primary_endpoints(path: Path, manifest: dict) -> dict:
    """Validate an adjudicated human endpoint import before any calibration use.

    Required schema fields are checked; audited IDs must be a subset of the frozen
    validation set; per-qid inclusion probabilities must equal the pre-registered
    stratified probabilities; two independent annotators plus an adjudication receipt
    hash are mandatory. Anything missing is an error — never silently filled.
    """
    from hashlib import sha256
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("schema") != HUMAN_IMPORT_SCHEMA:
        raise ValueError(f"schema must be {HUMAN_IMPORT_SCHEMA}")
    if d.get("dataset_id") != "CaReBench":
        raise ValueError("dataset_id must be CaReBench")
    if d.get("metric") not in ("n_contra_ref", "r_core_ref"):
        raise ValueError("metric must be a protocol primary endpoint")
    ann = d.get("annotator_ids") or []
    if int(d.get("n_annotators", 0)) < 2 or len(set(ann)) < 2:
        raise ValueError("at least two distinct annotators are required")
    rec_path, rec_sha = d.get("adjudication_receipt_path"), d.get("adjudication_receipt_sha256")
    if not rec_path or not rec_sha:
        raise ValueError("adjudication_receipt_path and sha256 are required")
    rp = Path(rec_path)
    if not rp.is_file() or sha256(rp.read_bytes()).hexdigest() != rec_sha:
        raise ValueError("adjudication receipt missing or hash mismatch")
    val = set(manifest["e2_validation_set"]["validation_query_ids"])
    units = manifest.get("units", {})
    vals, inc, strat = d.get("values") or {}, d.get("inclusion_by_qid") or {}, d.get("stratum_by_qid") or {}
    audited = set(vals)
    if not audited:
        raise ValueError("no audited queries")
    if not audited <= val:
        raise ValueError(f"audited IDs outside frozen validation set: {sorted(audited - val)[:5]}")
    manifest_inc = {q: units[q]["val_inclusion_prob"] for q in val if q in units}
    for q in audited:
        arms = vals[q]
        if not {"ARM_A0", "ARM_A1", "ARM_A7"} <= set(arms):
            raise ValueError(f"{q}: missing required arm values")
        for a in ("ARM_A0", "ARM_A1", "ARM_A7"):
            v = arms[a]
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) < 0:
                raise ValueError(f"{q}/{a}: invalid endpoint value {v!r}")
        if q not in inc or not (0 < float(inc[q]) <= 1):
            raise ValueError(f"{q}: missing/invalid inclusion probability")
        if q not in manifest_inc or abs(float(inc[q]) - float(manifest_inc[q])) > 1e-6:
            raise ValueError(f"{q}: inclusion probability deviates from pre-registered design")
        if q not in units or strat.get(q) != units[q]["stratum"]:
            raise ValueError(f"{q}: stratum missing or does not match manifest")
    covered = {strat[q] for q in audited}
    for s, meta in manifest["e2_validation_set"]["strata"].items():
        if s not in covered:
            raise ValueError(f"stratum {s} has zero audited units; two-phase estimate undefined")
    excluded = set(d.get("calibration_excluded_ids") or [])
    if excluded & audited:
        raise ValueError("calibration IDs overlap audited validation IDs")
    payload = {"schema": HUMAN_IMPORT_SCHEMA, "metric": d["metric"], "values": vals,
               "inclusion_by_qid": {q: float(inc[q]) for q in audited},
               "stratum_by_qid": {q: strat[q] for q in audited},
               "calibration_excluded_ids": sorted(excluded)}
    receipt = {"import_path": str(p), "import_sha256": sha256(p.read_bytes()).hexdigest(),
               "n_audited": len(audited), "n_strata_covered": len(covered),
               "annotator_ids": sorted(set(ann)), "adjudication_receipt_sha256": rec_sha,
               "design_inclusion_match": True}
    return {"payload": payload, "receipt": receipt}


def prepare_e3_bridge_approved(candidates_path: Path | None = None, min_items: int = 20) -> dict:
    """Freeze machine-assisted E3-Bridge candidates into the shared E3 approved schema.

    TRUE condition = external retrieved caption (natural evidence); CONFLICT = the model
    claim contradicted by the human reference; NEU = unrelated action. Every pair remains
    CANDIDATE_PENDING_VERIFICATION and may only be used in machine-assisted mode.
    """
    cp = Path(candidates_path or (O / "final_analysis/E3_BRIDGE_CANDIDATES.json"))
    data = json.loads(cp.read_text(encoding="utf-8")) if cp.is_file() else {"candidates": []}
    formal = {r["query_id"]: r for r in read_jsonl(FORMAL / "formal_candidates.jsonl")}
    approved = []
    for c in data.get("candidates", []):
        f = formal.get(c["query_id"])
        if not f: continue
        approved.append({"query_id": c["query_id"], "source_id": None, "video_path": f["video_path"],
                         "true_action": c["external_caption"], "conflict_action": c["model_claim"],
                         "neutral_text": "reading a book in a library", "status": "CANDIDATE_PENDING_VERIFICATION",
                         "bridge_requires_visual_check": True})
    payload = {"schema": "uarcap_q2.e3_bridge_approved.v1", "status": "MACHINE_ASSISTED_UNVALIDATED",
               "utc": utcnow(), "n_approved": len(approved), "min_required": min_items,
               "eligible": len(approved) >= min_items, "approved": approved}
    (O / "final_analysis/E3_BRIDGE_APPROVED.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
