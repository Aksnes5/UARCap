#!/usr/bin/env python3
"""Per-scale budget stats + aggregate scale x budget table."""
import json
import os
import random
import sys

ROOT = "/home/bn/videoufo_research"
OUT = os.path.join(ROOT, "evcap_external/outputs")
NAME = "eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json"
KS = [1, 3, 5, 7]
SIX = [("Bleu_1", "B1"), ("Bleu_4", "B4"), ("METEOR", "M"), ("ROUGE_L", "R-L"),
       ("CIDEr", "CIDEr"), ("SPICE", "SPICE")]
sys.path.insert(0, os.path.join(ROOT, "evcap_external/evaluation"))
from pycocoevalcap.cider.cider import Cider  # noqa: E402

GT = {}
for line in open("/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2/val.caption.tsv"):
    p = line.split("\t", 1)
    GT[p[0].split("/")[-1][:-4]] = [x["caption"] for x in json.loads(p[1])]

agg = {}
rng = random.Random(42)
for scale in ["10k", "50k", "100k", "250k", "500k", "full"]:
    dirs = {k: (f"eval_val497_D{k}" if scale == "full" else f"eval_val497_{scale}_D{k}") for k in KS}
    if not all(os.path.exists(os.path.join(OUT, d, "metrics.json")) for d in dirs.values()):
        continue
    ms = {k: json.load(open(os.path.join(OUT, d, "metrics.json"))) for k, d in dirs.items()}
    P = {k: {x["image_name"]: x["prediction"].strip() for x in json.load(open(os.path.join(OUT, d, NAME)))}
         for k, d in dirs.items()}
    Q = sorted(set.intersection(*[set(v) for v in P.values()]))
    g = {q: GT[q.split("/")[-1][:-4]] for q in Q}
    C = {k: dict(zip(Q, Cider().compute_score(g, {q: [P[k][q]] for q in Q})[1])) for k in KS}
    paired = {}
    for a, b in [("D3", "D7"), ("D3", "D5"), ("D1", "D3")]:
        ka, kb = int(a[1]), int(b[1])
        d = [C[ka][q] - C[kb][q] for q in Q]
        obs = sum(d) / len(d)
        boots = sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(10000))
        paired[f"{a}-{b}"] = {"diff": round(obs, 4), "ci95": [round(boots[250], 4), round(boots[9750], 4)]}
    agg[scale] = {
        "n": len(Q),
        "metrics": {k: {mk: round(float(ms[k][mk]) * 100, 3) for mk, _ in SIX} for k in KS},
        "best_k": max(KS, key=lambda k: float(ms[k]["CIDEr"])),
        "paired": paired,
    }
    print(scale, "best_k", agg[scale]["best_k"], {k: agg[scale]["metrics"][k]["CIDEr"] for k in KS}, flush=True)
json.dump(agg, open(os.path.join(ROOT, "reports/scale_budget.json"), "w"), indent=2, default=float)
