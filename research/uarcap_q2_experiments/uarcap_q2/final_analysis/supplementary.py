"""Supplementary analyses and the strict E3-Bridge pipeline.

Every function here is either descriptive or an explicitly post-hoc/exploratory analysis.
No human labels are invented. E3-Bridge approval requires machine-verified video facts and
validated neutral evidence; if verification is missing or insufficient the bridge is BLOCKED.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

import numpy as np
from scipy.stats import spearmanr

O = Path("/home/bn/uarcap_q2_runtime/outputs")
FORMAL = O / "formal_candidate_unvalidated"
SUPP = O / "supplementary"
E3 = O / "pilot_e3_exploratory"

NEUTRAL_POOL = (
    "a person is washing a car outdoors",
    "a person is reading a book in a library",
    "a person is folding laundry indoors",
    "a person is typing on a laptop at a desk",
    "a person is riding a bicycle on a road",
    "a person is painting a wall with a brush",
    "a person is playing a guitar on a stage",
    "a person is swimming in a pool",
    "a person is watering plants in a garden",
    "a person is driving a car on a highway",
    "a person is brushing their teeth in a bathroom",
    "a person is flying a kite in a park",
    "a person is baking bread in an oven",
    "a person is shoveling snow from a driveway",
    "a person is assembling a wooden shelf",
    "a person is feeding a dog in a kitchen",
    "a person is taking a photograph with a camera",
    "a person is rowing a boat on a lake",
    "a person is writing notes on a whiteboard",
    "a person is repairing a bicycle tire",
)

STOP = set("a an the this that these those of to in on at with and or is are was were be being by for from as it its his her their he she they person video showcases process".split())
BOILER = re.compile(r"^(this video (showcases|shows|depicts|captures)( the process of)?|the video (shows|depicts)|in this video,?)\s*", re.I)
NEG = re.compile(r"\b(not|no|never|without|instead of|rather than)\b", re.I)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(p: Path) -> list[dict]:
    p = Path(p)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []


def tokens(s: str) -> set[str]:
    return {t for t in re.sub(r"[^a-z0-9'\s]", " ", (s or "").lower()).split() if t not in STOP and len(t) > 1}


def jaccard(a: str, b: str) -> float:
    A, B = tokens(a), tokens(b)
    return len(A & B) / max(1, len(A | B))


def clean_fact(s: str) -> str:
    s = " ".join((s or "").split())
    s = BOILER.sub("", s).strip(" .,")
    return (s[0].lower() + s[1:]) if s else s


def atomic_write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- E3-Bridge
def build_e3_bridge_candidates_v2(max_candidates: int = 60) -> dict:
    """Build 50-60 natural-conflict candidates.

    f+  : human-written core event from the frozen reference (cleaned), pending visual verification.
    f-  : a caption retrieved from a DIFFERENT source video with lexical overlap but a
          different/negated action (natural evidence), pending visual verification.
    neu : neutral action from a fixed pool with zero content-token overlap with f+/f-.
    """
    cands = {c["query_id"]: c for c in read_jsonl(FORMAL / "formal_candidates.jsonl")}
    retr = {r["query_id"]: r for r in read_jsonl(FORMAL / "formal_retrieval.jsonl")}
    out = []
    for qid in sorted(cands):
        c = cands[qid]
        fpos = clean_fact(c.get("core_event_candidate", ""))
        if not (3 <= len(fpos.split()) <= 30):
            continue
        best = None
        for rank, rc in enumerate((retr.get(qid, {}).get("reranked_d") or [])[:20]):
            cap = " ".join(str(rc.get("caption", "")).split())
            n = len(tokens(cap))
            if not (4 <= n <= 40):
                continue
            ov = tokens(fpos) & tokens(cap)
            if len(ov) < 2:
                continue
            if fpos.lower() in cap.lower() or cap.lower() in fpos.lower():
                continue
            neg_diff = bool(NEG.search(cap)) != bool(NEG.search(fpos))
            diff = len(tokens(cap) ^ tokens(fpos))
            j = jaccard(fpos, cap)
            if not (neg_diff or j >= 0.10 or diff >= 4):
                continue
            score = (2 if neg_diff else 0) + j + 0.01 * (20 - rank)
            if best is None or score > best[0]:
                best = (score, cap, rc, rank)
        if not best:
            continue
        _, fneg, rc, rank = best
        neu = None
        for cand_neu in NEUTRAL_POOL:
            if not (tokens(cand_neu) & (tokens(fpos) | tokens(fneg))):
                h = int(sha256(f"{qid}|{cand_neu}".encode()).hexdigest()[:8], 16)
                if neu is None or h % 1000 < neu[0]:
                    neu = (h % 1000, cand_neu)
        if not neu:
            continue
        out.append({
            "bridge_id": f"bridge_{len(out)+1:03d}", "query_id": qid,
            "source_id": None, "source_id_status": "UNKNOWN_OPAQUE",
            "video_path": c["video_path"], "frames_dir": str(FORMAL / "formal_frames" / qid),
            "f_plus": fpos, "f_minus": fneg, "neutral_text": neu[1],
            "f_plus_provenance": "human_reference_core_event; pending independent visual verification",
            "f_minus_provenance": {"external_caption": fneg, "retrieved_rank": rank,
                                   "retrieval_score": rc.get("score"), "clip_ids": rc.get("clip_ids"),
                                   "retrieval_source_ids": rc.get("source_ids"),
                                   "note": "retrieved from a different video; natural evidence"},
            "neutral_provenance": "fixed pre-declared pool; zero content-token overlap (text only)",
            "text_validity": {"f_plus_minus_jaccard": round(jaccard(fpos, fneg), 4),
                              "negation_differs": bool(NEG.search(fneg)) != bool(NEG.search(fpos)),
                              "neutral_overlap": 0},
            "status": "CANDIDATE_PENDING_INDEPENDENT_VISUAL_VERIFICATION",
        })
        if len(out) >= max_candidates:
            break
    payload = {"schema": "uarcap_q2.e3_bridge_candidates.v2", "utc": utcnow(),
               "n_candidates": len(out), "min_valid_required": 30, "status": "PENDING_VERIFICATION",
               "verification_model": "Qwen3.8-27B-FP8 (independent of data source; machine, not human)",
               "candidates": out}
    atomic_write(SUPP / "E3_BRIDGE_CANDIDATES_V2.json", payload)
    return payload


def verification_jobs(candidates_payload: dict | None = None) -> dict:
    data = candidates_payload or json.loads((SUPP / "E3_BRIDGE_CANDIDATES_V2.json").read_text())
    jobs = []
    for c in data["candidates"]:
        for role, stmt in (("f_plus", c["f_plus"]), ("f_minus", c["f_minus"]), ("neutral", c["neutral_text"])):
            jobs.append({"job_id": f"{c['bridge_id']}|{role}", "bridge_id": c["bridge_id"],
                         "query_id": c["query_id"], "role": role, "statement": stmt,
                         "frames_dir": c["frames_dir"]})
    payload = {"schema": "uarcap_q2.e3_bridge_jobs.v1", "utc": utcnow(), "n_jobs": len(jobs), "jobs": jobs}
    atomic_write(SUPP / "E3_BRIDGE_VERIFICATION_JOBS.json", payload)
    return payload


def merge_e3_bridge_verification(candidates_payload: dict, verdicts: list[dict],
                                 min_valid: int = 30, expected_roles=("f_plus", "f_minus", "neutral")) -> dict:
    """Approve only candidates with verified f+, verified-absent f- and verified-absent neutral."""
    by = defaultdict(dict)
    for v in verdicts:
        by[v["bridge_id"]][v["role"]] = v
    approved, rejected = [], []
    reasons = Counter()
    for c in candidates_payload["candidates"]:
        v = by.get(c["bridge_id"], {})
        miss = [r for r in expected_roles if r not in v or not v[r].get("parse_ok")]
        if miss:
            reasons["missing_or_unparsed_verification"] += 1
            rejected.append({**c, "rejection": f"missing verification: {miss}"})
            continue
        fp, fm, nu = v["f_plus"], v["f_minus"], v["neutral"]
        ok = (fp["video_truth"] == "entailed" and fp["visible_in_16_frames"] == "yes"
              and fm["video_truth"] == "contradicted"
              and nu["video_truth"] in ("contradicted", "unverifiable"))
        if not ok:
            reasons["verification_failed"] += 1
            rejected.append({**c, "rejection": "verification criteria not met",
                             "verdicts": {k: {"video_truth": x.get("video_truth"),
                                              "visible": x.get("visible_in_16_frames")} for k, x in v.items()}})
            continue
        approved.append({**c, "status": "APPROVED_MACHINE_VERIFIED_NOT_HUMAN",
                         "verification": {k: {"video_truth": x["video_truth"],
                                              "visible_in_16_frames": x["visible_in_16_frames"],
                                              "evidence": x.get("evidence", ""),
                                              "frames_sha256": x.get("frames_sha256"),
                                              "n_frames": x.get("n_frames", 16)} for k, x in v.items()},
                         "human_verification": "PENDING"})
    eligible = len(approved) >= min_valid
    payload = {"schema": "uarcap_q2.e3_bridge_approved.v2", "utc": utcnow(),
               "n_candidates": len(candidates_payload["candidates"]), "n_approved": len(approved),
               "n_rejected": len(rejected), "min_valid_required": min_valid, "eligible": eligible,
               "status": "APPROVED_MACHINE_VERIFIED_NOT_HUMAN" if eligible else "BLOCKED_INSUFFICIENT_VERIFIED_CANDIDATES",
               "rejection_reasons": dict(reasons),
               "requirements": "f+ entailed & visible yes; f- contradicted; neutral not entailed; all parsed",
               "approved": approved, "rejected": rejected}
    atomic_write(SUPP / "E3_BRIDGE_APPROVED_V2.json", payload)
    return payload


# ---------------------------------------------------------------- descriptive supplementary analyses
def _formal_models():
    return {"formal_videollama": FORMAL / "videollama_eval/a2_ref_fact_eval.jsonl",
            "formal_qwen38": FORMAL / "qwen_eval/a2_ref_fact_eval.jsonl"}


def e2_length_analysis() -> dict:
    """Caption length vs fact outcomes, with query-fixed-effect arm contrasts and matched bins."""
    out = {}
    for name, path in _formal_models().items():
        rows = [r for r in read_jsonl(path) if r.get("parse_status") == "OK"]
        per = {a: [] for a in ("ARM_A0", "ARM_A1", "ARM_A7")}
        for r in rows:
            w = len(str(r.get("caption", "")).split())
            per.setdefault(r["arm"], []).append({"q": r["query_id"], "w": w,
                                                 "support": r.get("p_entail_ref") or 0.0,
                                                 "conflict": (r.get("n_contra_ref", 0) or 0) / max(1, r.get("n_claims", 0)),
                                                 "core": r.get("r_core_ref")})
        res = {"per_arm": {}}
        for a, rs in per.items():
            if not rs:
                continue
            ws = [x["w"] for x in rs]
            res["per_arm"][a] = {"n": len(rs), "mean_words": round(float(np.mean(ws)), 2),
                                 "median_words": float(np.median(ws))}
            for k in ("support", "conflict", "core"):
                xs = [(x["w"], x[k]) for x in rs if x[k] is not None]
                if len(xs) > 3:
                    rho = spearmanr([x[0] for x in xs], [x[1] for x in xs])
                    res["per_arm"][a][f"spearman_length_{k}"] = round(float(rho.correlation), 4)
        # within-query fixed-effect regression: y ~ query FE + arm indicators + words
        qids = sorted({r["query_id"] for r in rows})
        qidx = {q: i for i, q in enumerate(qids)}
        X, y, meta = [], [], []
        for r in rows:
            if r["arm"] not in ("ARM_A0", "ARM_A1", "ARM_A7"):
                continue
            row = [0.0] * len(qids)
            row[qidx[r["query_id"]]] = 1.0
            row += [1.0 if r["arm"] == "ARM_A1" else 0.0, 1.0 if r["arm"] == "ARM_A7" else 0.0,
                    float(len(str(r.get("caption", "")).split()))]
            X.append(row)
            y.append(float(r.get("p_entail_ref") or 0.0))
            meta.append(r)
        if len(X) > len(qids) + 4:
            beta, *_ = np.linalg.lstsq(np.asarray(X), np.asarray(y), rcond=None)
            res["query_fe_ols_support"] = {"coef_words": round(float(beta[-1]), 6),
                                           "coef_A1_vs_A0": round(float(beta[-3]), 6),
                                           "coef_A7_vs_A0": round(float(beta[-2]), 6),
                                           "n_rows": len(X), "n_queries": len(qids)}
        # length-matched A0-A7 contrast within each model (quintile bins of A0 words)
        q0 = {r["query_id"]: len(str(r["caption"]).split()) for r in rows if r["arm"] == "ARM_A0"}
        q7 = {r["query_id"]: (len(str(r["caption"]).split()), r.get("p_entail_ref") or 0.0)
              for r in rows if r["arm"] == "ARM_A7"}
        common = sorted(set(q0) & set(q7))
        if len(common) >= 10:
            w0 = np.array([q0[q] for q in common]); order = np.argsort(w0)
            bins = np.array_split(order, 5)
            diffs = []
            for b in bins:
                qs = [common[i] for i in b]
                if len(qs) < 2:
                    continue
                d0 = np.mean([r.get("p_entail_ref") or 0.0 for r in rows if r["arm"] == "ARM_A0" and r["query_id"] in qs])
                d7 = np.mean([q7[q][1] for q in qs])
                diffs.append(d7 - d0)
            res["length_stratified_A7_minus_A0_support"] = {
                "n_bins": len(diffs), "mean_within_bin_delta": round(float(np.mean(diffs)), 4),
                "bin_deltas": [round(float(x), 4) for x in diffs]}
        out[name] = res
    atomic_write(SUPP / "E2_LENGTH_ANALYSIS.json", {"utc": utcnow(), "status": "POST_HOC_EXPLORATORY", "models": out})
    return out


def length_control_plan(n_videos: int = 120, seed: int = 20261126) -> dict:
    """Deterministic 120-video subset for the Qwen short-vs-detailed length experiment."""
    cands = [c["query_id"] for c in read_jsonl(FORMAL / "formal_candidates.jsonl")]
    rng = np.random.default_rng(seed)
    picked = sorted(rng.choice(sorted(cands), size=min(n_videos, len(cands)), replace=False).tolist())
    plan = {"schema": "uarcap_q2.length_control_plan.v1", "utc": utcnow(), "n_videos": len(picked),
            "query_ids": picked, "generator": "Qwen/Qwen3.8-27B-FP8",
            "conditions": {"short_aligned": "existing A0/A1/A7 captions (8-18 words)",
                           "detailed_sensitivity": "same frames/evidence, 2-4 sentences, max_new_tokens=128"},
            "planned_new_generations": len(picked) * 3,
            "videollama_note": "frozen Video-LLaMA prompt cannot be length-manipulated without protocol violation; excluded.",
            "design": "within-video paired; length is measured post hoc; no hypothesis test until human-calibrated endpoints"}
    atomic_write(SUPP / "LENGTH_CONTROL_PLAN.json", plan)
    return plan


def visual_sampling_plan(n_videos: int = 150, seed: int = 20261127) -> dict:
    cands = sorted(c["query_id"] for c in read_jsonl(FORMAL / "formal_candidates.jsonl"))
    rng = np.random.default_rng(seed)
    picked = sorted(rng.choice(cands, size=min(n_videos, len(cands)), replace=False).tolist())
    plan = {"schema": "uarcap_q2.visual_sampling_plan.v1", "utc": utcnow(), "n_videos": len(picked),
            "query_ids": picked, "generator": "Qwen/Qwen3.8-27B-FP8", "arm": "ARM_A7",
            "frame_settings": [32, 64], "planned_new_generations": len(picked) * 2,
            "baseline": "existing 16-frame A7 captions",
            "note": "Video-LLaMA frozen 16-frame feature contract cannot be changed; excluded from this sensitivity."}
    atomic_write(SUPP / "VISUAL_SAMPLING_PLAN.json", plan)
    return plan


def retrieval_relevance() -> dict:
    retr = read_jsonl(FORMAL / "formal_retrieval.jsonl")
    cands = {c["query_id"]: c for c in read_jsonl(FORMAL / "formal_candidates.jsonl")}
    rows = []
    for r in retr:
        rr = r.get("reranked_d") or []
        ref = cands[r["query_id"]]["reference_caption"]
        core = cands[r["query_id"]]["core_event_candidate"]
        if not rr:
            continue
        scores = [float(x.get("rerank_score") if x.get("rerank_score") is not None else x.get("score", 0)) for x in rr]
        rows.append({"query_id": r["query_id"], "n_reranked": len(rr),
                     "top1_score": scores[0], "top7_mean": float(np.mean(scores[:7])),
                     "top1_minus_top7": scores[0] - float(np.mean(scores[:7])),
                     "top7_ref_jaccard": float(np.mean([jaccard(x.get("caption", ""), ref) for x in rr[:7]])),
                     "top7_core_jaccard": float(np.mean([jaccard(x.get("caption", ""), core) for x in rr[:7]]))})
    out = {"schema": "uarcap_q2.retrieval_relevance.v1", "utc": utcnow(), "n_queries": len(rows),
           "status": "DESCRIPTIVE_NO_HUMAN_RELEVANCE_LABELS",
           "top1_score_mean": round(float(np.mean([x["top1_score"] for x in rows])), 4) if rows else None,
           "top7_core_jaccard_mean": round(float(np.mean([x["top7_core_jaccard"] for x in rows])), 4) if rows else None,
           "top7_ref_jaccard_mean": round(float(np.mean([x["top7_ref_jaccard"] for x in rows])), 4) if rows else None,
           "warning": "lexical overlap is not human relevance; X-CLIP score is retrieval confidence, not correctness",
           "per_query": rows}
    atomic_write(SUPP / "RETRIEVAL_RELEVANCE.json", out)
    return {k: v for k, v in out.items() if k != "per_query"}


def natural_evidence_provenance() -> dict:
    out = {}
    for name, d in (("videollama", FORMAL / "videollama"), ("qwen", FORMAL / "qwen")):
        rows = read_jsonl(d / "predictions.jsonl")
        tr = read_jsonl(d / "input_traces.jsonl")
        by_arm = defaultdict(lambda: {"n": 0, "with_text": 0, "with_visual": 0, "text_counts": Counter()})
        for r in tr:
            a = by_arm[r["arm"]]
            a["n"] += 1
            if r.get("text_evidence"):
                a["with_text"] += 1
                a["text_counts"][len(r["text_evidence"])] += 1
            if r.get("visual_evidence_ids"):
                a["with_visual"] += 1
        out[name] = {"n_predictions": len(rows), "n_traces": len(tr),
                     "per_arm": {k: {"n": v["n"], "with_text_evidence": v["with_text"],
                                     "with_visual_evidence": v["with_visual"],
                                     "text_evidence_counts": dict(v["text_counts"])} for k, v in by_arm.items()},
                     "modality_fairness": "Video-LLaMA receives text+visual EVCap evidence; Qwen receives text evidence + raw frames only (PROMPT_TEXT_ONLY). Not an architecture-isolated comparison."}
    atomic_write(SUPP / "NATURAL_EVIDENCE_PROVENANCE.json", {"utc": utcnow(), "models": out})
    return out


def contamination_audit() -> dict:
    man = json.loads((FORMAL / "FORMAL_CAREBENCH_ANALYSIS_MANIFEST.json").read_text())
    val = set(man["e2_validation_set"]["validation_query_ids"])
    main = set(man.get("formal_main_excluding_validation_ids") or [])
    pilot = set(man.get("pilot_excluded_query_ids") or [])
    vid = json.loads((O / "pilot_e3_exploratory/E3_EXTRACT_RECEIPT.json").read_text()) if (E3 / "E3_EXTRACT_RECEIPT.json").is_file() else {}
    vh = {}
    for p in (E3 / "E3_PILOT_EXCLUSION.json", E3 / "E3_FORMAL_APPROVED.json"):
        if p.is_file():
            try:
                d = json.loads(p.read_text()); vh[p.name] = d.get("n_excluded", d.get("n_approved", d.get("n", None)))
            except Exception:
                pass
    srcs = [a.get("source_id") for a in json.loads((E3 / "E3_FORMAL_APPROVED.json").read_text()).get("approved", [])] if (E3 / "E3_FORMAL_APPROVED.json").is_file() else []
    pilot_excl = json.loads((E3 / "E3_PILOT_EXCLUSION.json").read_text()) if (E3 / "E3_PILOT_EXCLUSION.json").is_file() else {}
    out = {"schema": "uarcap_q2.contamination_audit.v1", "utc": utcnow(),
           "carebench": {"n_formal": len(main | val), "n_validation_audit": len(val), "n_main": len(main),
                         "validation_main_overlap": len(val & main),
                         "pilot_main_overlap": len(pilot & (main | val)),
                         "parent_isolation_verified": man.get("parent_isolation_verified"),
                         "near_dup_scan": man.get("n_auto_near_dup_excluded"),
                         "caveat": "CaReBench parent IDs are anonymized; YouTube-level source independence cannot be proven."},
           "vidhalluc": {"n_approved_formal": len(srcs), "n_unique_source_ids": len(set(srcs)),
                         "duplicate_source_within_formal": len(srcs) - len(set(srcs)),
                         "pilot_excluded_query_ids": pilot_excl.get("count"),
                         "pilot_excluded_source_ids": len(pilot_excl.get("excluded_source_ids") or []),
                         "video_ufo_overlap_excluded": 2,
                         "video_ufo_source": "E3_EXPLORATORY_REPORT.md (2 exact VideoUFO overlaps excluded)"},
           "cross_dataset": {"status": "NOT_JOINABLE",
                             "reason": "CaReBench query IDs/parent IDs are anonymized; no trustworthy YouTube source key exists, so a join would fabricate provenance.",
                             "n_vidhalluc_sources": len(set(s for s in srcs if s)),
                             "carebench_vs_vidhalluc_source_overlap": None}}
    atomic_write(SUPP / "CONTAMINATION_AUDIT.json", out)
    return out


def pareto_costs() -> dict:
    costs = {
        "videollama_formal_600x3": {"seconds": 6525, "generations": 1800, "model": "Video-LLaMA EVCap frozen"},
        "qwen_formal_600x3": {"seconds": 1910, "generations": 1800, "model": "Qwen3.8-27B-FP8"},
        "qwen_a3_pilot_90x3": {"seconds": 113.9, "generations": 90, "model": "Qwen3.8-27B-FP8"},
        "gemma_eval_1800": {"seconds": 287.1 * 1800 / 90, "generations": 1800, "model": "Gemma-3-27B judge (extrapolated)"},
    }
    rows = []
    for k, v in costs.items():
        rows.append({"pipeline": k, "model": v["model"], "generations": v["generations"],
                     "gpu_seconds": round(v["seconds"], 1),
                     "sec_per_generation": round(v["seconds"] / max(1, v["generations"]), 3)})
    out = {"schema": "uarcap_q2.pareto_costs.v1", "utc": utcnow(),
           "note": "measured GPU-task elapsed from real run receipts; Gemma eval projection is marked extrapolated",
           "rows": rows,
           "planned_supplementary": {"budget_extension_A3_A5": {"generations": 2400},
                                     "length_control_qwen": {"generations": 360},
                                     "visual_sampling_32_64": {"generations": 300},
                                     "e3_bridge_verification_vlm": {"calls": 180}}}
    atomic_write(SUPP / "PARETO_COSTS.json", out)
    return out


def historical_reverse_bridge() -> dict:
    """Compare direction of fact-vs-metric relations from historical MSR-VTT/MSVD outputs."""
    rows = []
    for root in (Path("/home/bn/uarcap_q2_runtime/outputs"), Path("/home/bn/videoufo_research"),
                 Path("/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display")):
        if not root.exists():
            continue
        for p in root.rglob("*.json"):
            n = p.name.lower()
            if not any(k in n for k in ("metric", "result", "eval", "cider")):
                continue
            if not any(k in str(p).lower() for k in ("msrvtt", "msvd")):
                continue
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            if isinstance(d, dict) and any(k in d for k in ("CIDEr", "cider", "Bleu_4", "BLEU_4", "METEOR")):
                rows.append({"source": str(p), "dataset": "msrvtt" if "msrvtt" in str(p).lower() else "msvd",
                             "metrics": {k: d[k] for k in ("CIDEr", "cider", "Bleu_4", "BLEU_4", "METEOR", "ROUGE_L") if k in d}})
    out = {"schema": "uarcap_q2.historical_reverse_bridge.v1", "utc": utcnow(),
           "n_sources_found": len(rows), "sources": rows,
           "status": "NO_RERUN" if rows else "PENDING_NO_HISTORICAL_METRIC_FILES",
           "caution": "MSR-VTT/MSVD are multi-reference and different datasets; only DIRECTION of effects may be compared, never absolute values."}
    atomic_write(SUPP / "HISTORICAL_REVERSE_BRIDGE.json", out)
    return {k: v for k, v in out.items() if k != "sources"}


def e3_authenticity_audit() -> dict:
    approved = json.loads((E3 / "E3_FORMAL_APPROVED.json").read_text()) if (E3 / "E3_FORMAL_APPROVED.json").is_file() else {"approved": []}
    apps = approved.get("approved", [])
    src = [a.get("source_id") for a in apps]
    bad_src = [s for s in src if not (isinstance(s, str) and len(s) == 11 and all(ch.isalnum() or ch in "-_" for ch in s))]
    noneu = [a.get("query_id") for a in apps if not str(a.get("neutral_text") or "").strip()]
    eval_rows = read_jsonl(E3 / "formal_evaluation/E3_FORMAL_EVAL.jsonl")
    parse_bad = [r for r in eval_rows if not r.get("parse_ok")]
    preds = read_jsonl(E3 / "formal_videollama/predictions.jsonl")
    traces = read_jsonl(E3 / "formal_videollama/input_traces.jsonl")
    trace_by = {(r["query_id"], r["arm"]): r for r in traces}
    mism = 0
    for p in preds:
        t = trace_by.get((p["query_id"], p["arm"]))
        if not t or t.get("visual_input_sha256") != p.get("input_trace", {}).get("visual_input_sha256"):
            mism += 1
    out = {"schema": "uarcap_q2.e3_authenticity_audit.v1", "utc": utcnow(),
           "n_approved": len(apps), "source_id_invalid": len(bad_src), "source_id_examples_bad": bad_src[:5],
           "neutral_text_missing": len(noneu),
           "eval_rows": len(eval_rows), "eval_parse_failures": len(parse_bad),
           "parse_failure_policy": "None/excluded, never 0",
           "videollama_predictions": len(preds), "visual_trace_hash_mismatch": mism,
           "frame_contract": "16 uniformly sampled frames; not full-video understanding",
           "full_video_vs_16frames": "unresolved by design; report as limitation, sensitivity queued for Qwen only",
           "neutral_visual_validation": "PENDING (VLM statement verification task queued)",
           "human_screening": "PENDING; no human sign-off claimed"}
    atomic_write(SUPP / "E3_AUTHENTICITY_AUDIT.json", out)
    return out


def fusion_fairness_audit() -> dict:
    prov = natural_evidence_provenance()
    out = {"schema": "uarcap_q2.fusion_fairness.v1", "utc": utcnow(),
           "comparison": "frozen EVCap Video-LLaMA (text + visual retrieval fusion) vs Qwen3.8 (text evidence injected into prompt; raw 16 frames as visual input)",
           "not_architecture_isolated": True,
           "reasons": ["different pretraining data and scale", "different evidence injection interfaces",
                       "EVCap fuses retrieval visual features; Qwen never receives retrieved clip features",
                       "prompts differ (frozen prompt5 vs short_aligned)"],
           "required_claim_language": "model-system contrast under different fusion interfaces; no isolated architectural causal claim",
           "per_model_evidence": prov}
    atomic_write(SUPP / "FUSION_FAIRNESS_AUDIT.json", out)
    return out


def run_all() -> dict:
    SUPP.mkdir(parents=True, exist_ok=True)
    cand = None
    if (SUPP / "E3_BRIDGE_CANDIDATES_V2.json").is_file():
        cand = json.loads((SUPP / "E3_BRIDGE_CANDIDATES_V2.json").read_text())
    else:
        cand = build_e3_bridge_candidates_v2()
    jobs = verification_jobs(cand)
    result = {
        "e2_length": e2_length_analysis(),
        "length_control_plan": length_control_plan(),
        "visual_sampling_plan": visual_sampling_plan(),
        "retrieval_relevance": retrieval_relevance(),
        "natural_evidence": natural_evidence_provenance(),
        "contamination": contamination_audit(),
        "pareto": pareto_costs(),
        "historical_reverse": historical_reverse_bridge(),
        "e3_authenticity": e3_authenticity_audit(),
        "fusion_fairness": fusion_fairness_audit(),
        "e3_bridge_candidates": {"n_candidates": cand["n_candidates"], "n_jobs": jobs["n_jobs"]},
    }
    atomic_write(SUPP / "SUPPLEMENTARY_ANALYSIS_SUMMARY.json",
                 {"schema": "uarcap_q2.supplementary_summary.v1", "utc": utcnow(), **result})
    return result
