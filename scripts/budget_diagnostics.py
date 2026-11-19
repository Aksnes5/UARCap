#!/usr/bin/env python3
"""Step I: which inference-time features predict context harm (diagnostic only)."""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

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

labels = {"d13": [C[3][q] - C[1][q] for q in Q],
          "d35": [C[5][q] - C[3][q] for q in Q],
          "d37": [C[7][q] - C[3][q] for q in Q]}
names = sorted(F[qid(Q[0])])


def auc(x, y):
    x = np.asarray(x); y = np.asarray(y, bool)
    pos, neg = x[y], x[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


out = {"n": len(Q), "spearman": {}, "auc": {}, "label_mean": {k: float(np.mean(v)) for k, v in labels.items()}}
for lab, y in labels.items():
    out["spearman"][lab] = {}
    for nm in names:
        rho, p = spearmanr([F[qid(q)][nm] for q in Q], y)
        out["spearman"][lab][nm] = {"rho": round(float(rho), 4), "p": float(p)}
    harm = [v < 0 for v in labels["d37"] if lab == "d37"]
for nm in names:
    out["auc"][nm] = {"d13_pos": round(auc([F[qid(q)][nm] for q in Q], [v > 0 for v in labels["d13"]]), 4),
                      "d37_harm": round(auc([F[qid(q)][nm] for q in Q], [v < 0 for v in labels["d37"]]), 4)}

json.dump(out, open(os.path.join(ROOT, "reports/budget_diagnostics.json"), "w"), indent=2)
for lab in labels:
    print(f"\n== {lab} (mean {out['label_mean'][lab]:+.4f})")
    top = sorted(out["spearman"][lab].items(), key=lambda kv: -abs(kv[1]["rho"]))[:12]
    for nm, v in top:
        print(f"  {nm:24s} rho {v['rho']:+.3f} p {v['p']:.2e}")
print("\nAUC (d13 benefit / d37 harm)")
for nm in sorted(names, key=lambda n: -abs(out["auc"][n]["d37_harm"] - 0.5))[:10]:
    print(f"  {nm:24s} {out['auc'][nm]['d13_pos']:.3f} / {out['auc'][nm]['d37_harm']:.3f}")
