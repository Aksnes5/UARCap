#!/usr/bin/env python3
"""Aggregate A/B/C/D metrics on the fixed 300-query dev set."""
import json
import os

ROOT = "/home/bn/videoufo_research/evcap_external/outputs"
M = {
    "A": "eval_dev300_A/eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review",
    "B": "eval_dev300_B/eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review",
    "C": "eval_dev300_C/eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review",
    "D": "eval_dev300_D/eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review",
}
KEYS = [("Bleu_1", "B1"), ("Bleu_4", "B4"), ("METEOR", "M"), ("ROUGE_L", "R-L"), ("CIDEr", "CIDEr"), ("SPICE", "SPICE")]
rows = {}
for k, base in M.items():
    p = os.path.join(ROOT, base + "_metrics.json") if os.path.exists(os.path.join(ROOT, base + "_metrics.json")) else os.path.join(ROOT, base + ".json")
    if not os.path.exists(p):
        p = os.path.join(ROOT, base + ".json")
    if os.path.exists(p):
        rows[k] = json.load(open(p))
print("metric   " + "".join(f"{k:>9}" for k in "ABCD"))
for mk, name in KEYS:
    print(f"{name:8} " + "".join(f"{rows.get(k,{}).get(mk,float('nan'))*100:9.2f}" for k in "ABCD"))
print("\ndeltas")
for a, b in [("C", "A"), ("D", "A"), ("D", "B"), ("D", "C")]:
    if a in rows and b in rows:
        print(a + "-" + b, {mk: round((rows[a].get(mk, 0) - rows[b].get(mk, 0)) * 100, 2) for mk, _ in KEYS})
