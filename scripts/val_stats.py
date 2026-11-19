#!/usr/bin/env python3
"""val497 k-grid: six metrics, paired CIDEr bootstrap/permutation, k-utility curve."""
import json
import os
import random
import sys

ROOT = "/home/bn/videoufo_research"
OUT = os.path.join(ROOT, "evcap_external/outputs")
NAME = "eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json"
SIX = [("Bleu_1", "B1"), ("Bleu_4", "B4"), ("METEOR", "M"), ("ROUGE_L", "R-L"),
       ("CIDEr", "CIDEr"), ("SPICE", "SPICE")]
DIRS = {f"{t}{k}": f"eval_val497_{t}{k}" for t in ("CF", "D") for k in (1, 3, 5, 7)}
sys.path.insert(0, os.path.join(ROOT, "evcap_external/evaluation"))
from pycocoevalcap.cider.cider import Cider  # noqa: E402


def caps(d):
    return {x["image_name"]: x["prediction"].strip() for x in json.load(open(os.path.join(OUT, d, NAME)))}


def metrics(d):
    return json.load(open(os.path.join(OUT, d, "metrics.json")))


def main():
    ms = {k: metrics(v) for k, v in DIRS.items() if os.path.exists(os.path.join(OUT, v, "metrics.json"))}
    print("metric  " + "".join(f"{k:>8}" for k in DIRS))
    for mk, nm in SIX:
        print(f"{nm:7} " + "".join(f"{ms[k].get(mk, 0)*100:8.2f}" if k in ms else "       -" for k in DIRS))

    GT = {}
    for line in open("/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2/val.caption.tsv"):
        p = line.split("\t", 1)
        GT[p[0].split("/")[-1][:-4]] = [x["caption"] for x in json.loads(p[1])]
    P = {k: caps(v) for k, v in DIRS.items() if os.path.exists(os.path.join(OUT, v, NAME))}
    K = sorted(set.intersection(*[set(v) for v in P.values()]))
    g = {k: GT[k.split("/")[-1][:-4]] for k in K}
    sc = {m: dict(zip(K, Cider().compute_score(g, {k: [P[m][k]] for k in K})[1])) for m in P}

    rng = random.Random(42)

    def paired(a, b):
        d = [sc[a][k] - sc[b][k] for k in K]
        obs = sum(d) / len(d)
        boots = sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(10000))
        cnt = sum(1 for _ in range(10000)
                  if abs(sum(x if rng.random() < 0.5 else -x for x in d) / len(d)) >= abs(obs))
        w = sum(x > 1e-6 for x in d); l = sum(x < -1e-6 for x in d); t = len(d) - w - l
        return {"diff": round(obs, 4), "ci95": [round(boots[250], 4), round(boots[9750], 4)],
                "p_perm": round((cnt + 1) / 10001, 5), "win": w, "loss": l, "tie": t}

    pairs = [("D3", "D7"), ("D5", "D7"), ("D3", "D5"), ("D1", "D3"),
             ("CF3", "CF7"), ("CF5", "CF7"), ("CF1", "CF3"), ("CF1", "CF7"),
             ("D3", "CF3"), ("D3", "CF1"), ("D7", "CF7")]
    best = max((m for m in P if m.startswith("D")), key=lambda m: sum(sc[m][k] for k in K))
    res = {}
    for a, b in pairs:
        res[f"{a}-{b}"] = paired(a, b)
        print(f"{a}-{b}", res[f"{a}-{b}"], flush=True)
    print("best D by mean CIDEr:", best)
    json.dump({"metrics": {k: {mk: round(float(v) * 100, 4) for mk, v in ms[k].items()} for k in ms},
               "paired": res, "best_D": best, "queries": len(K)},
              open(os.path.join(ROOT, "reports/val_stats.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
