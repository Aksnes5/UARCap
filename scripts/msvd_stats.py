#!/usr/bin/env python3
"""MSVD external frozen validation: six metrics + paired stats."""
import json
import os
import random
import sys

ROOT = "/home/bn/videoufo_research"
OUT = os.path.join(ROOT, "evcap_external/outputs")
NAME = "eval_evcap_top3_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_msvd_generated_captions.json"
SIX = [("Bleu_1", "B1"), ("Bleu_4", "B4"), ("METEOR", "M"), ("ROUGE_L", "R-L"),
       ("CIDEr", "CIDEr"), ("SPICE", "SPICE")]
sys.path.insert(0, os.path.join(ROOT, "evcap_external/evaluation"))
from pycocoevalcap.cider.cider import Cider  # noqa: E402

VARS = ["D1", "D3", "D5", "D7", "Adaptive"]
ms = {v: json.load(open(os.path.join(OUT, f"eval_msvd_{v}", "metrics.json"))) for v in VARS}
P = {v: {x["image_name"]: x["prediction"].strip() for x in json.load(open(os.path.join(OUT, f"eval_msvd_{v}", NAME)))}
     for v in VARS}
Q = sorted(set.intersection(*[set(v) for v in P.values()]))
evalrows = json.load(open(os.path.join(OUT, "eval_msvd_D3", NAME)))
GT = {x["image_name"]: x["captions"] for x in evalrows}
g = {q: GT[q] for q in Q}
sc = {v: dict(zip(Q, Cider().compute_score(g, {q: [P[v][q]] for q in Q})[1])) for v in VARS}
rng = random.Random(42)
out = {"n": len(Q), "metrics": {v: {mk: round(float(ms[v][mk]) * 100, 3) for mk, _ in SIX} for v in VARS}, "paired": {}}
for a, b in [("D3", "D7"), ("D3", "D5"), ("D1", "D3"), ("Adaptive", "D3"), ("Adaptive", "D7")]:
    d = [sc[a][q] - sc[b][q] for q in Q]
    obs = sum(d) / len(d)
    boots = sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(10000))
    cnt = sum(1 for _ in range(10000)
              if abs(sum(x if rng.random() < 0.5 else -x for x in d) / len(d)) >= abs(obs))
    out["paired"][f"{a}-{b}"] = {"diff": round(obs, 4), "ci95": [round(boots[250], 4), round(boots[9750], 4)],
                                 "p_perm": round((cnt + 1) / 10001, 5),
                                 "win": sum(x > 1e-6 for x in d), "loss": sum(x < -1e-6 for x in d),
                                 "tie": sum(abs(x) <= 1e-6 for x in d)}
    print(a, b, out["paired"][f"{a}-{b}"], flush=True)
print(json.dumps(out["metrics"], indent=2))
json.dump(out, open(os.path.join(ROOT, "reports/msvd_stats.json"), "w"), indent=2, default=float)
