#!/usr/bin/env python3
"""LOO Stage 1 analysis: per-evidence distracting effect on val150."""
import csv
import json
import os
import random
import sys

import numpy as np

ROOT = "/home/bn/videoufo_research"
OUT = os.path.join(ROOT, "evcap_external/outputs")
NAME = "eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json"
sys.path.insert(0, os.path.join(ROOT, "evcap_external/evaluation"))
from pycocoevalcap.cider.cider import Cider  # noqa: E402

qids = json.load(open(os.path.join(ROOT, "manifests/loo_val150.json")))["queries"]
Q = [f"datasets/MSRVTT-v2/videos/{q}.mp4" for q in qids]
GT = {}
for line in open("/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2/val.caption.tsv"):
    p = line.split("\t", 1)
    GT[p[0]] = [x["caption"] for x in json.loads(p[1])]

P = {}
base = {x["image_name"]: x for x in json.load(open(os.path.join(OUT, "eval_val497_D7", NAME)))}
P["base"] = {q: base[q]["prediction"].strip() for q in Q}
for i in range(7):
    d = json.load(open(os.path.join(OUT, f"eval_val150_loo{i}", NAME)))
    P[i] = {x["image_name"]: x["prediction"].strip() for x in d}
    assert set(P[i]) == set(Q)

gts, res = {}, {}
for q in Q:
    gts[f"{q}::base"] = GT[q]
    res[f"{q}::base"] = [P["base"][q]]
    for i in range(7):
        gts[f"{q}::{i}"] = GT[q]
        res[f"{q}::{i}"] = [P[i][q]]
C = dict(zip(gts, Cider().compute_score(gts, res)[1]))

rows = []
for q in Q:
    for i in range(7):
        rows.append({"query": q, "rank_removed": i + 1,
                     "effect": C[f"{q}::{i}"] - C[f"{q}::base"],
                     "cider_loo": C[f"{q}::{i}"], "cider_d7": C[f"{q}::base"]})

rng = random.Random(42)
by_rank = {}
for i in range(1, 8):
    d = np.array([r["effect"] for r in rows if r["rank_removed"] == i])
    boots = sorted(float(np.mean([d[rng.randrange(len(d))] for _ in range(len(d))])) for _ in range(5000))
    by_rank[i] = {"mean": round(float(d.mean()), 4), "median": round(float(np.median(d)), 4),
                  "p_harmful": round(float((d > 0).mean()), 4),
                  "ci95": [round(boots[125], 4), round(boots[4875], 4)]}
head = np.array([r["effect"] for r in rows if r["rank_removed"] <= 3])
tail = np.array([r["effect"] for r in rows if r["rank_removed"] >= 4])
summary = {"by_rank": by_rank,
           "head_mean": round(float(head.mean()), 4), "tail_mean": round(float(tail.mean()), 4),
           "head_p_harmful": round(float((head > 0).mean()), 4),
           "tail_p_harmful": round(float((tail > 0).mean()), 4),
           "n_queries": len(Q), "n_effects": len(rows)}
json.dump(summary, open(os.path.join(ROOT, "reports/loo_effects.json"), "w"), indent=2)
with open(os.path.join(ROOT, "reports/loo_effects.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(json.dumps(summary, indent=2))
