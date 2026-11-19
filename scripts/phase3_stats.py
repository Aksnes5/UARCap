#!/usr/bin/env python3
"""Paired bootstrap/permutation and per-query distribution for H5 vs baselines."""
import json
import os
import random
import sys

ROOT = "/home/bn/videoufo_research"
sys.path.insert(0, os.path.join(ROOT, "evcap_external/evaluation"))
from pycocoevalcap.cider.cider import Cider  # noqa: E402

DIRS = {"H5": "eval_dev300_UARH5", "H7": "eval_dev300_UARH7",
        "D": "eval_dev300_D", "CF": "eval_dev300_CF",
        "D100k": "eval_dev300_D100k", "B": "eval_dev300_B"}
NAME = "eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json"


def load(d):
    p = os.path.join(ROOT, "evcap_external/outputs", d, NAME)
    data = json.load(open(p))
    return {x["image_name"]: x["prediction"].strip() for x in data}


def main():
    P = {k: load(v) for k, v in DIRS.items()}
    GT = {}
    for line in open(os.path.join(ROOT, "evcap_external/dev300/dev300.caption.tsv")):
        p = line.split("\t", 1)
        GT[p[0].split("/")[-1][:-4]] = [x["caption"] for x in json.loads(p[1])]
    K = list(P["H5"])
    g = {k: GT[k.split("/")[-1][:-4]] for k in K}
    sc = {m: dict(zip(K, Cider().compute_score(g, {k: [P[m][k]] for k in K})[1])) for m in P}

    rng = random.Random(42)
    out = {}
    for base in ["D", "CF", "D100k", "B"]:
        d = [sc["H5"][k] - sc[base][k] for k in K]
        obs = sum(d) / len(d)
        boots = []
        for _ in range(10000):
            s = [d[rng.randrange(len(d))] for _ in range(len(d))]
            boots.append(sum(s) / len(s))
        boots.sort()
        ci = (boots[250], boots[9750])
        # sign-flip permutation
        cnt = 0
        for _ in range(10000):
            s = [x if rng.random() < 0.5 else -x for x in d]
            if abs(sum(s) / len(s)) >= abs(obs):
                cnt += 1
        pval = (cnt + 1) / 10001
        bins = {"gt+1": 0, "p0.2_1": 0, "pm0.2": 0, "m1_-0.2": 0, "lt-1": 0}
        for x in d:
            if x > 1: bins["gt+1"] += 1
            elif x > 0.2: bins["p0.2_1"] += 1
            elif x >= -0.2: bins["pm0.2"] += 1
            elif x >= -1: bins["m1_-0.2"] += 1
            else: bins["lt-1"] += 1
        entries = sorted(((sc["H5"][k] - sc[base][k], k) for k in K))
        out[base] = {"mean_diff": round(obs, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)],
                     "p_perm": round(pval, 5), "bins": bins,
                     "worst20": [{"query": k, "diff": round(v, 4),
                                  "H5": P["H5"][k], "base": P[base][k],
                                  "GT": GT[k.split("/")[-1][:-4]][:3]} for v, k in entries[:20]],
                     "best20": [{"query": k, "diff": round(v, 4),
                                 "H5": P["H5"][k], "base": P[base][k],
                                 "GT": GT[k.split("/")[-1][:-4]][:3]} for v, k in entries[-20:]]}
        print(base, out[base]["mean_diff"], out[base]["ci95"], out[base]["p_perm"], bins, flush=True)
    json.dump(out, open(os.path.join(ROOT, "reports/phase3_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
