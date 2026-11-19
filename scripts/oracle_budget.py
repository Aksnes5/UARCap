#!/usr/bin/env python3
"""Step F: per-query oracle context budget on val497 (no GPU, no regeneration)."""
import json
import os
import random
import sys

ROOT = "/home/bn/videoufo_research"
OUT = os.path.join(ROOT, "evcap_external/outputs")
NAME = "eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json"
KS = [1, 3, 5, 7]
sys.path.insert(0, os.path.join(ROOT, "evcap_external/evaluation"))
from pycocoevalcap.cider.cider import Cider  # noqa: E402


def caps(k):
    d = f"eval_val497_D{k}"
    return {x["image_name"]: x["prediction"].strip() for x in json.load(open(os.path.join(OUT, d, NAME)))}


GT = {}
for line in open("/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2/val.caption.tsv"):
    p = line.split("\t", 1)
    GT[p[0].split("/")[-1][:-4]] = [x["caption"] for x in json.loads(p[1])]

P = {k: caps(k) for k in KS}
Q = sorted(set.intersection(*[set(v) for v in P.values()]))
g = {q: GT[q.split("/")[-1][:-4]] for q in Q}
C = {}
for k in KS:
    C[k] = dict(zip(Q, Cider().compute_score(g, {q: [P[k][q]] for q in Q})[1]))

star = []
for q in Q:
    vals = [C[k][q] for k in KS]
    best = max(vals)
    star.append(KS[vals.index(best)])  # ties -> smaller k

dist = {k: star.count(k) for k in KS}
fixed = {k: sum(C[k][q] for q in Q) / len(Q) for k in KS}
best_fixed_k = max(KS, key=lambda k: fixed[k])
oracle = sum(max(C[k][q] for k in KS) for q in Q) / len(Q)
ident_all = sum(len({P[k][q] for k in KS}) == 1 for q in Q)
sens = len(Q) - ident_all

# paired bootstrap: oracle - fixed best
d = [max(C[k][q] for k in KS) - C[best_fixed_k][q] for q in Q]
rng = random.Random(42)
boots = sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(10000))
ci = (boots[250], boots[9750])

out = {
    "queries": len(Q),
    "k_star_distribution": dist,
    "k_star_ratio": {k: round(dist[k] / len(Q), 4) for k in KS},
    "fixed_mean_cider": {k: round(fixed[k], 4) for k in KS},
    "best_fixed_k": best_fixed_k,
    "oracle_adaptive_mean_cider": round(oracle, 4),
    "oracle_minus_best_fixed": round(oracle - fixed[best_fixed_k], 4),
    "oracle_minus_best_fixed_ci95": [round(ci[0], 4), round(ci[1], 4)],
    "all_k_identical_queries": ident_all,
    "context_sensitive_queries": sens,
}
json.dump(out, open(os.path.join(ROOT, "reports/oracle_budget_val497.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
