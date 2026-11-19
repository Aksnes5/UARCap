#!/usr/bin/env python3
"""Paired A/B/C/D failure analysis on the fixed 300-query dev set."""
import json
import os
import sys

BASE = "/home/bn/videoufo_research/evcap_external/outputs"
NAME = "eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json"
EVAL = "/home/bn/videoufo_research/evcap_external/evaluation"
sys.path.insert(0, EVAL)
from pycocoevalcap.cider.cider import Cider  # noqa: E402
from pycocoevalcap.rouge.rouge import Rouge  # noqa: E402


def load(mode):
    p = os.path.join(BASE, f"eval_dev300_{mode}", NAME)
    d = json.load(open(p))
    return {x["image_name"]: x["prediction"].strip() for x in d}, {x["image_name"]: x["captions"] for x in d}


def main():
    preds, gts = {}, None
    for m in "ABCD":
        p, g = load(m)
        preds[m] = p
        gts = g
    keys = sorted(set(gts) & set(preds["A"]) & set(preds["B"]) & set(preds["C"]) & set(preds["D"]))
    print("queries", len(keys))
    scores = {}
    for m in "ABCD":
        g = {k: [{"caption": c} for c in gts[k]] for k in keys}
        r = {k: [{"caption": preds[m][k]}] for k in keys}
        cs = Cider().compute_score(g, r)[1]
        rs = Rouge().compute_score(g, r)[1]
        scores[m] = {"CIDEr": dict(zip(keys, cs)), "ROUGE_L": dict(zip(keys, rs))}
    ext = json.load(open("/home/bn/videoufo_research/external/external_reranked_full.json"))
    cases = []
    for k in keys:
        a, b, c, d = (scores[m]["CIDEr"][k] for m in "ABCD")
        tag = []
        if b > a + 0.02 and d < a - 0.02:
            tag.append("B_up_D_down")
        if c < a - 0.02 and d > a + 0.02:
            tag.append("C_down_D_recover")
        if d > b + 0.05:
            tag.append("D_much_better_than_B")
        if d < b - 0.05:
            tag.append("D_much_worse_than_B")
        if tag:
            cases.append({
                "query": k, "tags": tag,
                "CIDEr": {m: round(scores[m]["CIDEr"][k], 4) for m in "ABCD"},
                "ROUGE_L": {m: round(scores[m]["ROUGE_L"][k], 4) for m in "ABCD"},
                "GT": gts[k][:5],
                "A": preds["A"][k], "B": preds["B"][k], "C": preds["C"][k], "D": preds["D"][k],
                "top7": ext[k][0] if k in ext else [],
            })
    cases.sort(key=lambda x: -(x["CIDEr"]["D"] - x["CIDEr"]["B"]))
    with open("/home/bn/videoufo_research/reports/paired_cases.json", "w") as f:
        json.dump(cases, f, indent=1)
    print("cases", len(cases))
    for x in cases[:10]:
        print(x["query"], x["tags"], x["CIDEr"])


if __name__ == "__main__":
    main()
