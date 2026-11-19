#!/usr/bin/env python3
"""Compare a dev300 prediction file with the original full-test predictions."""
import json
import sys

orig = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/EVCapCode/outputs/eval_repro_msrvtt/eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json"
a = json.load(open(sys.argv[1]))
b = {x["image_name"]: x["prediction"] for x in json.load(open(orig))}
n = same = 0
diff = []
for x in a:
    if x["image_name"] not in b:
        continue
    n += 1
    if x["prediction"].strip() == b[x["image_name"]].strip():
        same += 1
    elif len(diff) < 5:
        diff.append((x["image_name"], x["prediction"], b[x["image_name"]]))
print(f"compared {n} exact_match {same} ({same/max(1,n)*100:.1f}%)")
for d in diff:
    print("DIFF", d[0]); print("  new:", d[1]); print("  old:", d[2])
