"""Unified descriptive statistics, paired fact transitions, formal statistics, and
judge-reliability analysis for UARCap Q2 (E2 formal candidates, A3, E3).

All machine labels remain uncalibrated until human calibration is imported. Functions are
pure over JSONL inputs and can be tested with fixtures.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Callable, Iterable

import numpy as np

ARMS = ("ARM_A0", "ARM_A1", "ARM_A7")
LABELS = ("entailed", "contradicted", "unmentioned")


def read_jsonl(p: Path | str) -> list[dict]:
    q = Path(p)
    return [json.loads(x) for x in q.read_text(encoding="utf-8").splitlines() if x.strip()] if q.is_file() else []


def norm_claim(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    return " ".join(t for t in s.split() if t not in {"a", "an", "the"})


def tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", (s or "").lower()))


def jaccard(a: str, b: str) -> float:
    sa, sb = tokens(a), tokens(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0


def load_eval(eval_path: Path | str) -> list[dict]:
    return read_jsonl(eval_path)


def per_model_arm_descriptives(rows: list[dict]) -> dict[str, Any]:
    """Counts/rates/core coverage/dimensions/lengths per arm; parse failures included."""
    out: dict[str, Any] = {}
    for arm in ARMS:
        ar = [r for r in rows if r.get("arm") == arm]
        ok = [r for r in ar if r.get("parse_status") == "OK"]
        n_claims = sum(r.get("n_claims", 0) for r in ok)
        ent = sum(r.get("n_entail_ref", 0) or 0 for r in ok)
        con = sum(r.get("n_contra_ref", 0) or 0 for r in ok)
        unm = sum(r.get("n_unment_ref", 0) or 0 for r in ok)
        cores = [r["r_core_ref"] for r in ok if r.get("r_core_ref") is not None]
        dims = Counter()
        for r in ok:
            for d, c in (r.get("auxiliary_dimension_breakdown") or {}).items():
                dims[d] += c.get("total", 0)
        out[arm] = {
            "captions": len(ar), "parsed_ok": len(ok),
            "parse_failure_rate": round((len(ar) - len(ok)) / len(ar), 4) if ar else 0.0,
            "claims": n_claims, "entailed": ent, "contradicted": con, "unmentioned": unm,
            "support_rate": round(ent / n_claims, 4) if n_claims else None,
            "conflict_rate": round(con / n_claims, 4) if n_claims else None,
            "unmentioned_rate": round(unm / n_claims, 4) if n_claims else None,
            "mean_claims_per_caption": round(n_claims / len(ok), 3) if ok else None,
            "mean_caption_words": round(sum(len(str(r.get("caption", "")).split()) for r in ok) / len(ok), 2) if ok else None,
            "mean_contra_ref_per_caption": round(con / len(ok), 4) if ok else None,
            "core_coverage_mean": round(float(np.mean(cores)), 4) if cores else None,
            "dimensions": dict(dims),
        }
    return out


@dataclass
class TransitionResult:
    added_correct: int = 0
    lost_correct: int = 0
    added_error: int = 0
    removed_error: int = 0
    added_unmentioned: int = 0
    removed_unmentioned: int = 0
    label_flips: int = 0
    replacements: int = 0
    unmatched: int = 0
    core_coverage_delta: float | None = None
    match_method: str = "lexical_jaccard_candidate"
    semantic_verified: bool = False
    uncertain_matches: int = 0


def pair_transitions(rows: list[dict], src: str, dst: str, match_jaccard: float = 0.6,
                     semantic_verifier: Callable[[str, str], bool] | None = None) -> TransitionResult:
    """Match facts across arms.

    Default matching is a LEXICAL CANDIDATE match (Jaccard on word sets). It is deliberately
    labelled `lexical_jaccard_candidate` and is NOT a semantic-equivalence certificate.
    A supplied `semantic_verifier` may upgrade individual candidate pairs; unverified pairs
    are counted in `uncertain_matches`. High lexical overlap can hide negation/color/count/
    entity substitutions, so downstream reporting must keep this caveat.
    """
    by = defaultdict(dict)
    for r in rows:
        if r.get("parse_status") == "OK":
            by[r["query_id"]][r["arm"]] = r
    res = TransitionResult()
    core_deltas = []
    for qid, arms in by.items():
        a, b = arms.get(src), arms.get(dst)
        if not a or not b:
            continue
        ac = [(c["claim"], c["label"]) for c in a.get("claims", [])]
        bc = [(c["claim"], c["label"]) for c in b.get("claims", [])]
        used_b = set()
        unmatched_a = []
        for ca, la in ac:
            best, bestj = None, 0.0
            for j, (cb, lb) in enumerate(bc):
                if j in used_b:
                    continue
                jac = jaccard(ca, cb)
                if jac >= match_jaccard and jac > bestj:
                    best, bestj = j, jac
            if best is None:
                unmatched_a.append((ca, la)); continue
            used_b.add(best)
            if semantic_verifier is None or not semantic_verifier(ca, bc[best][0]):
                res.uncertain_matches += 1
            lb = bc[best][1]
            if la != lb:
                res.label_flips += 1
                res.replacements += 1
                if la == "entailed" and lb != "entailed": res.lost_correct += 1
                if la != "entailed" and lb == "entailed": res.added_correct += 1
                if la == "contradicted" and lb != "contradicted": res.removed_error += 1
                if la != "contradicted" and lb == "contradicted": res.added_error += 1
        for ca, la in unmatched_a:
            res.unmatched += 1
            if la == "entailed": res.lost_correct += 1
            elif la == "contradicted": res.removed_error += 1
            else: res.removed_unmentioned += 1
        for j, (cb, lb) in enumerate(bc):
            if j in used_b: continue
            if lb == "entailed": res.added_correct += 1
            elif lb == "contradicted": res.added_error += 1
            else: res.added_unmentioned += 1
        if a.get("r_core_ref") is not None and b.get("r_core_ref") is not None:
            core_deltas.append(float(b["r_core_ref"]) - float(a["r_core_ref"]))
    if core_deltas:
        res.core_coverage_delta = round(float(np.mean(core_deltas)), 4)
    return res


def all_transitions(rows: list[dict]) -> dict[str, Any]:
    pairs = (("ARM_A0", "ARM_A1"), ("ARM_A1", "ARM_A7"), ("ARM_A0", "ARM_A7"))
    return {f"{a}->{b}": pair_transitions(rows, a, b).__dict__ for a, b in pairs}


# ---------------- statistics ----------------
def cluster_bootstrap(diffs_by_qid: dict[str, float], n_boot: int = 2000, seed: int = 20261125) -> dict:
    """Parent-video (cluster) bootstrap of a paired mean difference."""
    qids = sorted(diffs_by_qid)
    if not qids:
        return {"n_clusters": 0, "mean": None, "ci95": None}
    rng = np.random.default_rng(seed)
    vals = np.array([diffs_by_qid[q] for q in qids], dtype=float)
    means = []
    n = len(vals)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        means.append(float(vals[idx].mean()))
    return {"n_clusters": n, "n_observations": n, "mean": round(float(vals.mean()), 4),
            "ci95": [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)],
            "bootstrap_sd": round(float(np.std(means, ddof=1)), 4)}


def paired_permutation(diffs_by_qid: dict[str, float], n_perm: int = 5000, seed: int = 20261125) -> dict:
    qids = sorted(diffs_by_qid)
    if not qids:
        return {"n": 0, "p_two_sided": None}
    rng = np.random.default_rng(seed)
    vals = np.array([diffs_by_qid[q] for q in qids], dtype=float)
    obs = abs(float(vals.mean()))
    cnt = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(vals))
        if abs(float((vals * signs).mean())) >= obs:
            cnt += 1
    return {"n": len(vals), "observed_mean": round(float(vals.mean()), 4),
            "p_two_sided": round((cnt + 1) / (n_perm + 1), 6)}


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict:
    items = sorted((v, k) for k, v in pvalues.items() if v is not None)
    m = len(items); out = {}; prev = 0.0
    for i, (p, k) in enumerate(items):
        adj = min(1.0, max(prev, p * (m - i)))
        prev = adj; out[k] = {"p": p, "p_holm": round(adj, 6), "reject": adj < alpha}
    return out


def paired_effect(diffs_by_qid: dict[str, float]) -> dict:
    vals = np.array(list(diffs_by_qid.values()), dtype=float)
    if len(vals) < 2: return {"n": len(vals), "mean_diff": None, "cohen_dz": None}
    sd = float(vals.std(ddof=1))
    return {"n": len(vals), "mean_diff": round(float(vals.mean()), 4),
            "cohen_dz": round(float(vals.mean() / sd), 4) if sd > 0 else None}


PRIMARY_METRICS = ("n_contra_ref", "r_core_ref")
AUXILIARY_METRICS = ("support_rate", "conflict_rate", "unmentioned_rate", "n_claims")


def metric_by_qid_arm(rows: list[dict], metric: str) -> dict[str, dict[str, float]]:
    """Per-query metric. Primary protocol endpoints: n_contra_ref (count per caption) and
    r_core_ref (core-event coverage). Proportions are auxiliary."""
    out = defaultdict(dict)
    for arm in ARMS:
        byq = defaultdict(list)
        for r in rows:
            if r.get("arm") == arm and r.get("parse_status") == "OK":
                byq[r["query_id"]].append(r)
        for q, rs in byq.items():
            n = sum(x.get("n_claims", 0) for x in rs)
            if metric in ("n_contra_ref", "contra_count"):
                v = float(sum(x.get("n_contra_ref", 0) or 0 for x in rs))
            elif metric in ("r_core_ref", "core_coverage"):
                z = [x["r_core_ref"] for x in rs if x.get("r_core_ref") is not None]
                v = float(np.mean(z)) if z else None
            elif metric == "support_rate": v = sum(x.get("n_entail_ref", 0) for x in rs) / n if n else None
            elif metric == "conflict_rate": v = sum(x.get("n_contra_ref", 0) for x in rs) / n if n else None
            elif metric == "unmentioned_rate": v = sum(x.get("n_unment_ref", 0) for x in rs) / n if n else None
            elif metric == "n_claims": v = float(n)
            else: raise ValueError(f"unknown metric {metric}")
            if v is not None: out[arm][q] = float(v)
    return dict(out)


def paired_arm_stats(rows: list[dict], metric: str) -> dict:
    m = metric_by_qid_arm(rows, metric)
    res = {}
    for a, b in (("ARM_A0", "ARM_A1"), ("ARM_A1", "ARM_A7"), ("ARM_A0", "ARM_A7")):
        common = sorted(set(m.get(a, {})) & set(m.get(b, {})))
        diffs = {q: m[b][q] - m[a][q] for q in common}
        res[f"{a}->{b}"] = {**cluster_bootstrap(diffs), **paired_permutation(diffs), **paired_effect(diffs)}
    return res


def cross_model_paired(rows_a: list[dict], rows_b: list[dict], metric: str) -> dict:
    """Direct same-arm cross-model differences (descriptive; NOT the protocol's budget effect)."""
    ma, mb = metric_by_qid_arm(rows_a, metric), metric_by_qid_arm(rows_b, metric)
    res = {}
    for arm in ARMS:
        common = sorted(set(ma.get(arm, {})) & set(mb.get(arm, {})))
        diffs = {q: mb[arm][q] - ma[arm][q] for q in common}
        res[arm] = {**cluster_bootstrap(diffs), **paired_permutation(diffs), **paired_effect(diffs)}
    return res


def cross_model_budget_differential(rows_a: list[dict], rows_b: list[dict], metric: str) -> dict:
    """Protocol cross-model budget effect:
        (model_b[A7] - model_b[A0]) - (model_a[A7] - model_a[A0])
    plus A1-A0 and A7-A1 contrasts. Cluster bootstrap/permutation over query_id.

    Retrieval-fusion mechanisms differ between the frozen EVCap model and Qwen prompt
    injection; this is a model-system contrast, not an isolated architectural causal effect.
    """
    ma, mb = metric_by_qid_arm(rows_a, metric), metric_by_qid_arm(rows_b, metric)
    out = {}
    for low, high in (("ARM_A0", "ARM_A1"), ("ARM_A1", "ARM_A7"), ("ARM_A0", "ARM_A7")):
        common = sorted(set(ma.get(low, {})) & set(ma.get(high, {})) & set(mb.get(low, {})) & set(mb.get(high, {})))
        diffs = {q: (mb[high][q] - mb[low][q]) - (ma[high][q] - ma[low][q]) for q in common}
        out[f"{low}->{high}"] = {**cluster_bootstrap(diffs), **paired_permutation(diffs), **paired_effect(diffs)}
    out["interpretation"] = "model-system budget contrast; fusion mechanisms differ (not isolated architecture effect)"
    return out


def missing_value_sensitivity(rows: list[dict]) -> dict:
    """Compare per-caption vs per-claim weighting and drop-parse-failure sensitivity."""
    ok = [r for r in rows if r.get("parse_status") == "OK"]
    out = {}
    for arm in ARMS:
        ar = [r for r in ok if r["arm"] == arm]
        per_claim = sum(r.get("n_entail_ref", 0) for r in ar) / max(1, sum(r.get("n_claims", 0) for r in ar))
        per_cap = float(np.mean([r.get("p_entail_ref", 0) for r in ar])) if ar else None
        out[arm] = {"support_per_claim": round(per_claim, 4), "support_per_caption": round(per_cap, 4) if per_cap is not None else None}
    return out


# ---------------- judge reliability ----------------
def judge_reliability(eval_rows: list[dict], cross_labels: list[dict]) -> dict:
    """Gemma vs Llama confusion, kappa, per-arm disagreement/directional bias, budget delta."""
    cross = {(r["query_id"], r["norm_key"]): r for r in cross_labels}
    conf = Counter(); per_arm = defaultdict(lambda: {"n": 0, "agree": 0, "dir": Counter()})
    flips = []
    for r in eval_rows:
        if r.get("parse_status") != "OK": continue
        for c in r.get("claims", []):
            k = (r["query_id"], norm_claim(c["claim"]))
            o = cross.get(k)
            if not o: continue
            a, b = c["label"], o["llama_label"]
            conf[(a, b)] += 1
            pa = per_arm[r["arm"]]; pa["n"] += 1
            if a == b: pa["agree"] += 1
            else:
                pa["dir"][f"{a}->{b}"] += 1
                if len(flips) < 100: flips.append({"query_id": r["query_id"], "claim": c["claim"], "gemma": a, "llama": b})
    n = sum(conf.values())
    po = sum(v for (a, b), v in conf.items() if a == b) / n if n else 0.0
    ca = Counter(); cb = Counter()
    for (a, b), v in conf.items(): ca[a] += v; cb[b] += v
    pe = sum((ca[l] / n) * (cb[l] / n) for l in LABELS) if n else 0.0
    kappa = (po - pe) / (1 - pe) if n and pe < 1 else None
    budget = {}
    for arm in ARMS:
        d = per_arm[arm]
        budget[arm] = {"n": d["n"], "disagreement_rate": round(1 - d["agree"] / d["n"], 4) if d["n"] else None,
                       "directions": dict(d["dir"])}
    # budget differential: A7 minus A0 disagreement/directional bias
    diff = {}
    if budget["ARM_A0"]["disagreement_rate"] is not None and budget["ARM_A7"]["disagreement_rate"] is not None:
        diff["disagreement_rate_A7_minus_A0"] = round(budget["ARM_A7"]["disagreement_rate"] - budget["ARM_A0"]["disagreement_rate"], 4)
    # Paired budget-effect difference between judges (not just overall agreement):
    #   (Llama[A7]-Llama[A0]) - (Gemma[A7]-Gemma[A0]), same query clusters.
    q_arm_gemma = defaultdict(lambda: Counter()); q_arm_llama = defaultdict(lambda: Counter())
    for r in eval_rows:
        if r.get("parse_status") != "OK": continue
        for c in r.get("claims", []):
            o = cross.get((r["query_id"], norm_claim(c["claim"])))
            if not o: continue
            q_arm_gemma[(r["query_id"], r["arm"])][c["label"]] += 1
            q_arm_llama[(r["query_id"], r["arm"])][o["llama_label"]] += 1
    def _rate(counter, label):
        t = sum(counter.values())
        return counter[label] / t if t else None
    budget_effect = {}
    for low, high in (("ARM_A0", "ARM_A1"), ("ARM_A1", "ARM_A7"), ("ARM_A0", "ARM_A7")):
        for label in ("entailed", "contradicted"):
            pairs = {}
            for q in {k[0] for k in q_arm_gemma}:
                gl, gh = q_arm_gemma.get((q, low)), q_arm_gemma.get((q, high))
                ll, lh = q_arm_llama.get((q, low)), q_arm_llama.get((q, high))
                if not all((gl, gh, ll, lh)): continue
                rg_l, rg_h = _rate(gl, label), _rate(gh, label)
                rl_l, rl_h = _rate(ll, label), _rate(lh, label)
                if None in (rg_l, rg_h, rl_l, rl_h): continue
                pairs[q] = (rl_h - rl_l) - (rg_h - rg_l)
            budget_effect[f"{low}->{high}|{label}"] = {**cluster_bootstrap(pairs), **paired_permutation(pairs), **paired_effect(pairs)} if pairs else {"n_clusters": 0}
    budget_effect["interpretation"] = "paired judge budget-effect difference (Llama minus Gemma), not overall agreement"
    return {"n_claims": n, "confusion_gemma_to_llama": {f"{a}->{b}": v for (a, b), v in conf.items()},
            "observed_agreement": round(po, 4), "cohen_kappa": round(kappa, 4) if kappa is not None else None,
            "per_arm": budget, "budget_differential": diff,
            "judge_budget_effect_difference": budget_effect, "disagreement_examples": flips[:20]}


def random_agreed_sample(eval_rows: list[dict], cross_labels: list[dict], n: int = 20, seed: int = 20261126) -> list[dict]:
    cross = {(r["query_id"], r["norm_key"]): r for r in cross_labels}
    pool = []
    for r in eval_rows:
        if r.get("parse_status") != "OK": continue
        for c in r.get("claims", []):
            o = cross.get((r["query_id"], norm_claim(c["claim"])))
            if o and o["llama_label"] == c["label"]:
                pool.append({"query_id": r["query_id"], "arm": r["arm"], "claim": c["claim"], "label": c["label"]})
    random.Random(seed).shuffle(pool)
    return pool[:n]


def core_event_audit(lock: dict) -> dict:
    ents = lock.get("entries", [])
    return {"n": len(ents), "n_with_core": sum(1 for e in ents if e.get("core_event_candidate")),
            "core_lock_sha256": lock.get("lock_sha256"), "created_before_generation": bool(lock.get("created_before_any_generation") or lock.get("created_before_any_judge_call"))}


def two_phase_primary_correction(
    auto_rows: list[dict],
    human_arm_metric: dict[str, dict[str, float]],
    inclusion_by_qid: dict[str, float],
    stratum_by_qid: dict[str, str],
    metric: str = "n_contra_ref",
    arm_low: str = "ARM_A0",
    arm_high: str = "ARM_A7",
    calibration_excluded_ids: set[str] | None = None,
) -> dict:
    """Design-based two-phase (IPW) correction for a PRIMARY endpoint paired contrast.

    Uses the repository's existing `two_phase_correction.py` (Horvitz-Thompson residual
    correction with stratified finite-population variance, inclusion probabilities, and
    calibration/validation leakage blocking). It does NOT use a global precision/recall
    correction and does NOT apply one machine-agreement rate to every arm.

    `human_arm_metric[qid][arm]` must contain adjudicated human metric values for audited
    queries; `inclusion_by_qid` must carry the recorded probability of selection into the
    human audit. Missing/empty human samples yield an explicit INCONCLUSIVE result.
    """
    from ..statistics.two_phase_correction import TwoPhaseUnit, estimate_two_phase_corrected_delta

    auto = metric_by_qid_arm(auto_rows, metric)
    low_map, high_map = auto.get(arm_low, {}), auto.get(arm_high, {})
    qids = sorted(set(low_map) & set(high_map))
    units: list[TwoPhaseUnit] = []
    for q in qids:
        unit_human = human_arm_metric.get(q)
        in_sample = bool(unit_human and arm_low in unit_human and arm_high in unit_human)
        pi = float(inclusion_by_qid.get(q, 0.0)) if in_sample else 0.0
        if in_sample and not (0.0 < pi <= 1.0):
            raise ValueError(f"Missing/invalid inclusion probability for audited query {q}: {pi}")
        units.append(TwoPhaseUnit(
            query_id=q,
            stratum=stratum_by_qid.get(q, "unknown"),
            auto_delta=float(high_map[q] - low_map[q]),
            in_human_sample=in_sample,
            inclusion_prob=pi if in_sample else 0.0,
            human_delta=float(unit_human[arm_high] - unit_human[arm_low]) if in_sample else None,
        ))
    res = estimate_two_phase_corrected_delta(units, calibration_excluded_ids=calibration_excluded_ids)
    res.update({"metric": metric, "contrast": f"{arm_high}-{arm_low}", "n_units": len(units),
                "n_human_sampled": sum(1 for u in units if u.in_human_sample),
                "estimator": "two_phase_correction.py (stratified IPW; inclusion probabilities required)"})
    return res
