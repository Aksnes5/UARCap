"""Parameterized downstream pipelines: formal E3, E3-Bridge, historical MSR-VTT/MSVD bridge,
and the 60-video independent validation set. No confirmatory analysis is run here."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any

from .core import ARMS, norm_claim, read_jsonl, tokens

O = Path(os.environ.get("UARCAP_OUTPUTS","/home/bn/uarcap_q2_runtime/outputs"))
FORMAL = O / "formal_candidate_unvalidated"
E3 = O / "pilot_e3_exploratory"


def utcnow(): return datetime.now(timezone.utc).isoformat()


def formal_e3_plan(min_approved: int = 150, candidates_path: Path | None = None) -> dict:
    """Formal E3 generation plan; refuses to run below the pre-declared candidate floor."""
    p = candidates_path or (O/"laptop-20261126-v1/e3/E3_FORMAL_APPROVED_V2.json")
    data = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"approved": [], "n_approved": 0}
    from ..interventions.neutral_repair import V2
    approved = data.get("approved", [])
    n = len(approved)
    ready = data.get("schema") == V2 and data.get("eligible") is True and n >= max(150, min_approved)
    plan = {
        "schema": "uarcap_q2.formal_e3_plan.v1", "utc": utcnow(),
        "n_approved_candidates": n, "min_required": min_approved,
        "eligible": ready,
        "models": ["videollama_evcap_frozen_v08", "Qwen/Qwen3.8-27B-FP8"],
        "arms": ["E3_BASE", "E3_SUP", "E3_CON", "E3_NEU"],
        "planned_generations": n * 4 * 2,
        "status": "MACHINE_ASSISTED_UNVALIDATED" if ready else "AWAITING_SUFFICIENT_CANDIDATES",
        "note": "Formal E3 remains exploratory until real human verification; no fabricated human gate.",
    }
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

    from ..common.io_utils import primitive_hash, sha256_file
    from ..common.io_utils import read_json
    core_lock = FORMAL / "formal_core_lock.json"
    core_lock_sha = sha256_file(core_lock)
    ref_lock={r["query_id"]:r for r in read_json(core_lock)["entries"]}
    out = Path(out_dir or (O / "final_analysis/validation_60_v3"))
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((FORMAL / "FORMAL_CAREBENCH_ANALYSIS_MANIFEST.json").read_text(encoding="utf-8"))
    val_ids = set(manifest["e2_validation_set"]["validation_query_ids"])
    main_ids = set(manifest.get("formal_main_excluding_validation_ids") or []) or (
        set(manifest.get("selected_query_ids", [])) - val_ids)
    from .validation_identity import caption_key, index, require_cohort, verify_caption_pair
    require_cohort(manifest)
    units = manifest.get("units", {})
    strata = manifest["e2_validation_set"]["strata"]

    cand = {c["query_id"]: c for c in read_jsonl(FORMAL / "formal_candidates.jsonl")}
    frames_idx = {r["query_id"]: r.get("frames", []) for r in read_jsonl(FORMAL / "formal_frame_indices.jsonl")}
    model_dirs = {"videollama": FORMAL / "videollama_eval/a2_ref_fact_eval.jsonl",
                  "qwen": FORMAL / "qwen_eval/a2_ref_fact_eval.jsonl"}
    evals = {m: index(read_jsonl(p)) for m, p in model_dirs.items()}
    videos = {q: {"query_id": q, "dataset_id": "CaReBench", "video_path": cand[q]["video_path"],
                  "sampled_frame_paths": frames_idx.get(q, []), "captions": [],
                  "human_reference_spans":[ref_lock[q][field] for field in ("reference_caption","spatial_caption","temporal_caption")
                                           if ref_lock[q].get(field)],
                  "frozen_reference_core":ref_lock[q]["core_event_candidate"]} for q in sorted(val_ids)}
    counts = Counter()
    missing = []
    for m, _ in model_dirs.items():
        for p in index(read_jsonl(FORMAL / m / "predictions.jsonl")).values():
            q = p["query_id"]
            if q not in videos:
                continue
            r = evals[m][caption_key(p)]
            claims = [{"claim_id": c["claim_id"], "claim": c["claim"]} for c in verify_caption_pair(p, r)]
            if not claims:
                missing.append({"model_id": p.get("model_id"), "query_id": q, "arm": p["arm"]})
            videos[q]["captions"].append({"arm": p["arm"], "model_id": p.get("model_id"),
                                          "caption": p["caption"], "claims": claims, "caption_sha256": r["caption_sha256"],
                                          "reference_description_sha256":r["reference_description_sha256"],
                                          "evaluator_id":r["judge_model_id"], "claim_split_sha256": primitive_hash(claims),
                                          "core_lock_sha256": core_lock_sha})
            counts[(p.get("model_id"), p["arm"])] += 1
    expected = len(val_ids) * 2 * 3
    actual = sum(len(v["captions"]) for v in videos.values())
    if actual != expected:
        raise ValueError(f"validation package incomplete: {actual}/{expected} caption rows")
    for q in videos:
        if len(videos[q]["captions"]) != 6:
            raise ValueError(f"{q}: expected 6 (model,arm) captions, got {len(videos[q]['captions'])}")
    if any(len(v["sampled_frame_paths"])!=16 for v in videos.values()):
        raise ValueError("Expected the sixteen original frame paths for every validation clip")
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
    """Validate a model-specific, fully bound human primary endpoint import."""
    from ..annotation.human_endpoints import verify
    return verify(path,manifest)


def prepare_e3_bridge_approved(candidates_path: Path | None = None, min_items: int = 30) -> dict:
    """Report eligibility from the versioned, outcome-blind bridge freeze."""
    from ..interventions.neutral_repair import V2
    cp=Path(candidates_path or (O/"laptop-20261126-v1/bridge/E3_BRIDGE_APPROVED_V2.json"))
    data=json.loads(cp.read_text(encoding="utf-8")) if cp.is_file() else {}
    if (data.get("schema")==V2 and data.get("scope")=="CAREB_BRIDGE" and data.get("eligible") is True
        and len(data.get("approved",[]))>=max(30,min_items)):
        return data
    return {"eligible":False,"status":"BLOCKED_STRICT_VERIFICATION_REQUIRED","approved":[]}
