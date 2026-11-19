#!/usr/bin/env python3
"""Formal paired stats for the OOF similarity-gap budget vs fixed budgets."""
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

GT = {}
for line in open("/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2/val.caption.tsv"):
    p = line.split("\t", 1)
    GT[p[0].split("/")[-1][:-4]] = [x["caption"] for x in json.loads(p[1])]


def caps(d):
    return {x["image_name"]: x["prediction"].strip() for x in json.load(open(os.path.join(OUT, d, NAME)))}


P = {f"D{k}": caps(f"eval_val497_D{k}") for k in KS}
P["gap"] = caps("eval_val497_OOF_gap")
Q = sorted(set.intersection(*[set(v) for v in P.values()]))
g = {q: GT[q.split("/")[-1][:-4]] for q in Q}
C = {m: dict(zip(Q, Cider().compute_score(g, {q: [P[m][q]] for q in Q})[1])) for m in P}

rng = random.Random(42)
out = {}
for base in ["D3", "D1", "D7", "D5"]:
    d = [C["gap"][q] - C[base][q] for q in Q]
    obs = sum(d) / len(d)
    boots = sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(10000))
    cnt = sum(1 for _ in range(10000)
              if abs(sum(x if rng.random() < 0.5 else -x for x in d) / len(d)) >= abs(obs))
    out[f"gap-{base}"] = {"diff": round(obs, 4), "ci95": [round(boots[250], 4), round(boots[9750], 4)],
                          "p_perm": round((cnt + 1) / 10001, 5),
                          "win": sum(x > 1e-6 for x in d), "loss": sum(x < -1e-6 for x in d),
                          "tie": sum(abs(x) <= 1e-6 for x in d)}
    print(base, out[f"gap-{base}"])
json.dump(out, open(os.path.join(ROOT, "reports/budget_stats.json"), "w"), indent=2, default=float)
