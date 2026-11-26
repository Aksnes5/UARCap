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


def pair_transitions(rows: list[dict], src: str, dst: str, match_jaccard: float = 0.6) -> TransitionResult:
    """Match facts across arms semantically; rewrites are matches (label flips), not add+delete."""
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


def metric_by_qid_arm(rows: list[dict], metric: str) -> dict[str, dict[str, float]]:
    """metric: support_rate|conflict_rate|unmentioned_rate|core_coverage|claims."""
    out = defaultdict(dict)
    for arm in ARMS:
        byq = defaultdict(list)
        for r in rows:
            if r.get("arm") == arm and r.get("parse_status") == "OK":
                byq[r["query_id"]].append(r)
        for q, rs in byq.items():
            n = sum(x.get("n_claims", 0) for x in rs)
            if metric == "support_rate": v = sum(x.get("n_entail_ref", 0) for x in rs) / n if n else None
            elif metric == "conflict_rate": v = sum(x.get("n_contra_ref", 0) for x in rs) / n if n else None
            elif metric == "unmentioned_rate": v = sum(x.get("n_unment_ref", 0) for x in rs) / n if n else None
            elif metric == "core_coverage":
                z = [x["r_core_ref"] for x in rs if x.get("r_core_ref") is not None]
                v = float(np.mean(z)) if z else None
            else: v = float(n)
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
    ma, mb = metric_by_qid_arm(rows_a, metric), metric_by_qid_arm(rows_b, metric)
    res = {}
    for arm in ARMS:
        common = sorted(set(ma.get(arm, {})) & set(mb.get(arm, {})))
        diffs = {q: mb[arm][q] - ma[arm][q] for q in common}
        res[arm] = {**cluster_bootstrap(diffs), **paired_permutation(diffs), **paired_effect(diffs)}
    return res


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
    return {"n_claims": n, "confusion_gemma_to_llama": {f"{a}->{b}": v for (a, b), v in conf.items()},
            "observed_agreement": round(po, 4), "cohen_kappa": round(kappa, 4) if kappa is not None else None,
            "per_arm": budget, "budget_differential": diff, "disagreement_examples": flips[:20]}


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
