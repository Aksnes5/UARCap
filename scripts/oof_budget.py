#!/usr/bin/env python3
"""Step J/K: budget-selection baselines with deterministic 5-fold OOF on val497.

All rules choose k in {1,3,5,7} in D order; no candidate reordering, no GT at
inference time. Thresholds are fit on 4 folds and applied to the held-out fold.
"""
import hashlib
import json
import os
import random
import sys

import numpy as np

ROOT = "/home/bn/videoufo_research"
OUT = os.path.join(ROOT, "evcap_external/outputs")
NAME = "eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json"
KS = [1, 3, 5, 7]
sys.path.insert(0, os.path.join(ROOT, "evcap_external/evaluation"))
from pycocoevalcap.cider.cider import Cider  # noqa: E402

F = json.load(open(os.path.join(ROOT, "reports/val_budget_features.json")))
GT = {}
for line in open("/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2/val.caption.tsv"):
    p = line.split("\t", 1)
    GT[p[0].split("/")[-1][:-4]] = [x["caption"] for x in json.loads(p[1])]


def caps(k):
    d = f"eval_val497_D{k}"
    return {x["image_name"]: x["prediction"].strip() for x in json.load(open(os.path.join(OUT, d, NAME)))}


P = {k: caps(k) for k in KS}
Q = sorted(set.intersection(*[set(v) for v in P.values()]))
qid = lambda q: q.split("/")[-1][:-4]
Q = [q for q in Q if qid(q) in F]
g = {q: GT[qid(q)] for q in Q}
C = {k: dict(zip(Q, Cider().compute_score(g, {q: [P[k][q]] for q in Q})[1])) for k in KS}


def fold_of(q):
    return int(hashlib.md5(qid(q).encode()).hexdigest(), 16) % 5


FOLD = {q: fold_of(q) for q in Q}


def gap_k(f):
    gaps = [f[f"sim{i}"] - f[f"sim{i+1}"] for i in range(1, 7)]
    i = int(np.argmax(gaps)) + 1  # keep first i
    return 1 if i <= 1 else 3 if i <= 3 else 5 if i <= 5 else 7


def red_k(f, tau):
    best = 1
    for k in KS[1:]:
        if f[f"cap_cos_mean{k}"] <= tau:
            best = k
    return best


def qual_k(f, rho):
    best = 1
    for k in KS[1:]:
        if f[f"q{k}"] >= rho * f["q1"]:
            best = k
    return best


def sat_k(f, lam):
    scores = {1: f["sim1"] + f["q1"] + f["ground1"],
              3: f["sim_mean3"] + f["q3"] + f["ground_mean3"] - lam * f["cap_cos_mean3"],
              5: f["sim_mean5"] + f["q5"] + f["ground_mean5"] - lam * f["cap_cos_mean5"],
              7: f["sim_mean7"] + f["q7"] + f["ground_mean7"] - lam * f["cap_cos_mean7"]}
    return max(KS, key=lambda k: scores[k])


RULES = {
    "gap": (lambda f, p: gap_k(f), None),
    "redundancy": (red_k, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    "quality_tail": (qual_k, [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]),
    "saturation": (sat_k, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]),
}


def fit_rule(fn, grid, train):
    if grid is None:
        return None
    best, bp = -1e9, grid[0]
    for p in grid:
        v = np.mean([C[fn(F[qid(q)], p)][q] for q in train])
        if v > best:
            best, bp = v, p
    return bp


def oof(fn, grid):
    sel, params = {}, {}
    for f in range(5):
        tr = [q for q in Q if FOLD[q] != f]
        te = [q for q in Q if FOLD[q] == f]
        p = fit_rule(fn, grid, tr)
        for q in te:
            sel[q] = fn(F[qid(q)], p) if p is not None else fn(F[qid(q)], None)
        params[f] = p
    return sel, params


def write_caps(sel, tag):
    d = os.path.join(OUT, f"eval_val497_OOF_{tag}")
    os.makedirs(d, exist_ok=True)
    out = [{"split": "valid", "image_name": q, "captions": GT[qid(q)],
            "prediction": P[sel[q]][q]} for q in Q]
    json.dump(out, open(os.path.join(d, NAME), "w"))
    return d


rng = random.Random(42)
res = {"methods": {}, "folds": {}, "n": len(Q)}
fixed3 = np.mean([C[3][q] for q in Q])
fixed1 = np.mean([C[1][q] for q in Q])
for name, (fn, grid) in RULES.items():
    sel, params = oof(fn, grid)
    vals = np.array([C[sel[q]][q] for q in Q])
    dif = vals - np.array([C[3][q] for q in Q])
    boots = sorted(sum(dif[rng.randrange(len(dif))] for _ in range(len(dif))) / len(dif) for _ in range(10000))
    per_fold = {}
    for f in range(5):
        fs = [q for q in Q if FOLD[q] == f]
        per_fold[f] = round(float(np.mean([C[sel[q]][q] - C[3][q] for q in fs])), 4)
    res["methods"][name] = {
        "oof_cider": round(float(vals.mean()), 4),
        "gain_vs_D3": round(float(dif.mean()), 4),
        "ci95": [round(boots[250], 4), round(boots[9750], 4)],
        "mean_k": round(float(np.mean([sel[q] for q in Q])), 3),
        "k_dist": {str(k): sum(sel[q] == k for q in Q) for k in KS},
        "evidence_reduction_vs_D7": round(float(1 - np.mean([sel[q] for q in Q]) / 7), 4),
        "per_fold_gain": per_fold,
        "params": params,
    }
    write_caps(sel, name)
    print(name, res["methods"][name], flush=True)
res["fixed_D3"] = round(float(fixed3), 4)
res["fixed_D1"] = round(float(fixed1), 4)
json.dump(res, open(os.path.join(ROOT, "reports/oof_budget.json"), "w"), indent=2)
