#!/usr/bin/env python3
"""Assemble untouched2690 / full2990 evaluation files from test-k outputs.

Adaptive2690 uses the frozen similarity-gap selector (no test-time tuning); its
captions are routed from the stored deterministic D1/D3/D5/D7 outputs.
"""
import json
import os

ROOT = "/home/bn/videoufo_research"
OUT = os.path.join(ROOT, "evcap_external/outputs")
NAME = "eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json"
KS = [1, 3, 5, 7]
S = json.load(open(os.path.join(ROOT, "external/external_test_sims.json")))
PILOT = {l.strip() for l in open(os.path.join(ROOT, "dev300_queries.txt")) if l.strip()}


def caps(k):
    return {x["image_name"]: x for x in json.load(open(os.path.join(OUT, f"eval_test2990_D{k}", NAME)))}


P = {k: caps(k) for k in KS}
Q = sorted(set.intersection(*[set(v) for v in P.values()]))
print("test queries", len(Q), "pilot", len(PILOT), "untouched", len([q for q in Q if q.split('/')[-1][:-4] not in PILOT]))


def gap_k(key):
    sims = S[key][1][:7]
    gaps = [sims[i] - sims[i + 1] for i in range(6)]
    i = max(range(6), key=lambda j: gaps[j]) + 1
    return 1 if i <= 1 else 3 if i <= 3 else 5 if i <= 5 else 7


def write(variant, subset, outdir):
    os.makedirs(os.path.join(OUT, outdir), exist_ok=True)
    rows = []
    for q in subset:
        if variant == "Adaptive":
            k = gap_k(q)
            rows.append(P[k][q])
        else:
            rows.append(P[int(variant[1:])][q])
    json.dump(rows, open(os.path.join(OUT, outdir, NAME), "w"))


B = {x["image_name"]: x for x in json.load(open(
    "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/EVCapCode/outputs/"
    "eval_evcap_msrvtt_quality_text/"
    "eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json"))}
print("B rows", len(B), "intersect", len(set(B) & set(Q)))

untouched = [q for q in Q if q.split("/")[-1][:-4] not in PILOT]
for tag, subset in [("test2690", untouched), ("test2990", Q)]:
    for v in ["D1", "D3", "D5", "D7", "Adaptive"]:
        write(v, subset, f"eval_{tag}_{v}")
    os.makedirs(os.path.join(OUT, f"eval_{tag}_B"), exist_ok=True)
    json.dump([B[q] for q in subset], open(os.path.join(OUT, f"eval_{tag}_B", NAME), "w"))
    print(tag, "written", len(subset))
