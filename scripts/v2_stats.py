#!/usr/bin/env python3
"""FAITHFUL_V2 six metrics + paired per-query CIDEr bootstrap vs PRE_FIX references."""
import json
import os
import random
import sys

ROOT = "/home/bn/videoufo_research"
OUT = os.path.join(ROOT, "evcap_external/outputs")
NAME = "eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json"
sys.path.insert(0, os.path.join(ROOT, "evcap_external/evaluation"))
from pycocoevalcap.cider.cider import Cider  # noqa: E402

SIX = [("Bleu_1", "B1"), ("Bleu_4", "B4"), ("METEOR", "M"), ("ROUGE_L", "R-L"),
       ("CIDEr", "CIDEr"), ("SPICE", "SPICE")]
REF = "eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review"


def caps(d):
    return {x["image_name"]: x["prediction"].strip() for x in json.load(open(os.path.join(OUT, d, NAME)))}


def metrics(d):
    p = os.path.join(OUT, d, REF + "_metrics.json")
    if not os.path.exists(p):
        p = os.path.join(OUT, d, "metrics.json")
    return json.load(open(p))


DIRS = {"CF3v2": "eval_dev300_CF3v2", "D3v2": "eval_dev300_D3v2", "H3v2": "eval_dev300_H3v2",
        "H3pre": "eval_dev300_UARH3", "D7": "eval_dev300_D", "CF7": "eval_dev300_CF"}


def main():
    ms = {k: metrics(v) for k, v in DIRS.items()}
    print("metric  " + "".join(f"{k:>10}" for k in DIRS))
    for mk, nm in SIX:
        print(f"{nm:7} " + "".join(f"{ms[k].get(mk, 0)*100:10.2f}" for k in DIRS))

    GT = {}
    for line in open(os.path.join(ROOT, "evcap_external/dev300/dev300.caption.tsv")):
        p = line.split("\t", 1)
        GT[p[0].split("/")[-1][:-4]] = [x["caption"] for x in json.loads(p[1])]
    P = {k: caps(v) for k, v in DIRS.items()}
    K = list(P["H3v2"])
    g = {k: GT[k.split("/")[-1][:-4]] for k in K}
    sc = {m: dict(zip(K, Cider().compute_score(g, {k: [P[m][k]] for k in K})[1])) for m in P}

    rng = random.Random(42)
    out = {}
    for base in ["CF3v2", "D3v2", "H3pre", "CF7", "D7"]:
        d = [sc["H3v2"][k] - sc[base][k] for k in K]
        obs = sum(d) / len(d)
        boots = sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(10000))
        ci = (boots[250], boots[9750])
        cnt = sum(1 for _ in range(10000)
                  if abs(sum(x if rng.random() < 0.5 else -x for x in d) / len(d)) >= abs(obs))
        out[base] = {"mean_diff": round(obs, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)],
                     "p_perm": round((cnt + 1) / 10001, 5),
                     "wins": sum(x > 0 for x in d), "losses": sum(x < 0 for x in d)}
        print(f"H3v2-{base}: diff {obs:+.3f} CI [{ci[0]:+.3f},{ci[1]:+.3f}] p {out[base]['p_perm']} "
              f"W/L {out[base]['wins']}/{out[base]['losses']}", flush=True)
    json.dump({"metrics": {k: {mk: round(v * 100, 4) for mk, v in ms[k].items()} for k in ms},
               "paired": out}, open(os.path.join(ROOT, "reports/v2_stats.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
