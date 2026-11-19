#!/usr/bin/env python3
"""Six metrics + paired tests for untouched2690 and full2990."""
import json
import os
import random
import sys

ROOT = "/home/bn/videoufo_research"
OUT = os.path.join(ROOT, "evcap_external/outputs")
NAME = "eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json"
SIX = [("Bleu_1", "B1"), ("Bleu_4", "B4"), ("METEOR", "M"), ("ROUGE_L", "R-L"),
       ("CIDEr", "CIDEr"), ("SPICE", "SPICE")]
VARS = ["B", "D1", "D3", "D5", "D7", "Adaptive"]
sys.path.insert(0, os.path.join(ROOT, "evcap_external/evaluation"))
from pycocoevalcap.cider.cider import Cider  # noqa: E402


def caps(d):
    return {x["image_name"]: x["prediction"].strip() for x in json.load(open(os.path.join(OUT, d, NAME)))}


def metrics(d):
    return json.load(open(os.path.join(OUT, d, "metrics.json")))


rng = random.Random(42)
res = {}
for tag in ["test2690", "test2990"]:
    ms = {v: metrics(f"eval_{tag}_{v}") for v in VARS}
    P = {v: caps(f"eval_{tag}_{v}") for v in VARS}
    K = sorted(set.intersection(*[set(p) for p in P.values()]))
    # GT from the eval files
    evalrows = json.load(open(os.path.join(OUT, f"eval_{tag}_D3", NAME)))
    GT = {x["image_name"]: x["captions"] for x in evalrows}
    g = {k: GT[k] for k in K}
    sc = {v: dict(zip(K, Cider().compute_score(g, {k: [P[v][k]] for k in K})[1])) for v in VARS}
    table = {v: {mk: round(float(ms[v][mk]) * 100, 3) for mk, _ in SIX} for v in VARS}
    paired = {}
    for a, b in [("D3", "D7"), ("D3", "D5"), ("D1", "D3"), ("Adaptive", "D3"),
                 ("Adaptive", "D7"), ("Adaptive", "B"), ("D3", "B")]:
        d = [sc[a][k] - sc[b][k] for k in K]
        obs = sum(d) / len(d)
        boots = sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(10000))
        cnt = sum(1 for _ in range(10000)
                  if abs(sum(x if rng.random() < 0.5 else -x for x in d) / len(d)) >= abs(obs))
        paired[f"{a}-{b}"] = {"diff": round(obs, 4), "ci95": [round(boots[250], 4), round(boots[9750], 4)],
                              "p_perm": round((cnt + 1) / 10001, 5),
                              "win": sum(x > 1e-6 for x in d), "loss": sum(x < -1e-6 for x in d),
                              "tie": sum(abs(x) <= 1e-6 for x in d)}
    res[tag] = {"n": len(K), "metrics": table, "paired": paired}
    print(tag, "n", len(K))
    for v in VARS:
        print(" ", v, table[v])
    for k, v in paired.items():
        print(" ", k, v)
json.dump(res, open(os.path.join(ROOT, "reports/test_confirmatory.json"), "w"), indent=2, default=float)
