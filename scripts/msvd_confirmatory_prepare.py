#!/usr/bin/env python3
"""Route MSVD test670 adaptive captions from frozen similarity-gap rule."""
import json
import os

ROOT = "/home/bn/videoufo_research"
OUT = os.path.join(ROOT, "evcap_external/outputs")
NAME = "eval_evcap_top3_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_msvd_generated_captions.json"
KS = [1, 3, 5, 7]
S = json.load(open(os.path.join(ROOT, "external/external_msvd_sims.json")))


def caps(k):
    return json.load(open(os.path.join(OUT, f"eval_msvd_D{k}", NAME)))


P = {k: {x["image_name"]: x for x in caps(k)} for k in KS}
Q = sorted(set.intersection(*[set(v) for v in P.values()]))


def gap_k(key):
    sims = S[key][1][:7]
    gaps = [sims[i] - sims[i + 1] for i in range(6)]
    i = max(range(6), key=lambda j: gaps[j]) + 1
    return 1 if i <= 1 else 3 if i <= 3 else 5 if i <= 5 else 7


os.makedirs(os.path.join(OUT, "eval_msvd_Adaptive"), exist_ok=True)
rows = [P[gap_k(q)][q] for q in Q]
json.dump(rows, open(os.path.join(OUT, "eval_msvd_Adaptive", NAME), "w"))
sel = [gap_k(q) for q in Q]
print("msvd queries", len(Q), "mean_k", round(sum(sel) / len(sel), 3),
      "dist", {k: sel.count(k) for k in KS})
