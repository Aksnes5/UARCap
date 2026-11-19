#!/usr/bin/env python3
"""Compare pre-fix vs faithful-v2 X-CLIP retrieval (dev300)."""
import json
import os
import sys

ROOT = "/home/bn/videoufo_research"
sys.path.insert(0, os.path.join(ROOT, "evcap_external/evaluation"))
from pycocoevalcap.cider.cider import Cider  # noqa

Q = [l.strip() for l in open(os.path.join(ROOT, "dev300_queries.txt")) if l.strip()]
GT = {}
for line in open(os.path.join(ROOT, "evcap_external/dev300/dev300.caption.tsv")):
    p = line.split("\t", 1); GT[p[0].split("/")[-1][:-4]] = [x["caption"] for x in json.loads(p[1])]
A = json.load(open(os.path.join(ROOT, "external/external_sims_full.json")))
B = json.load(open(os.path.join(ROOT, "external/external_v2_sims.json")))
AR = json.load(open(os.path.join(ROOT, "external/external_reranked_full.json")))
BR = json.load(open(os.path.join(ROOT, "external/external_v2_reranked.json")))

# per-candidate CIDEr for each version
def cand_cider(S):
    gts, res = {}, {}
    for q in Q:
        for i, c in enumerate(S[f"datasets/MSRVTT-v2/videos/{q}.mp4"][0]):
            gts[f"{q}::{i}"] = GT[q]; res[f"{q}::{i}"] = [c]
    return dict(zip(gts, Cider().compute_score(gts, res)[1]))

CA, CB = cand_cider(A), cand_cider(B)
out = {"top1_changed": 0, "top20_overlap": [], "top3_overlap": [],
       "pre": {}, "post": {}}
for q in Q:
    k = f"datasets/MSRVTT-v2/videos/{q}.mp4"
    ca, cb = A[k][0], B[k][0]
    out["top1_changed"] += ca[0] != cb[0]
    out["top20_overlap"].append(len(set(ca) & set(cb)) / 20)
    out["top3_overlap"].append(len(set(ca[:3]) & set(cb[:3])) / 3)
for name, S, C, R in [("pre", A, CA, AR), ("post", B, CB, BR)]:
    t1 = [S[f"datasets/MSRVTT-v2/videos/{q}.mp4"][1][0] for q in Q]
    t20 = [sum(S[f"datasets/MSRVTT-v2/videos/{q}.mp4"][1]) / 20 for q in Q]
    o20 = [max(C[f"{q}::{i}"] for i in range(20)) for q in Q]
    o7 = [max(C[f"{q}::{i}"] for i in range(7)) for q in Q]
    o3 = [max(C[f"{q}::{i}"] for i in range(3)) for q in Q]
    top1c = [C[f"{q}::0"] for q in Q]
    rk = [R[f"datasets/MSRVTT-v2/videos/{q}.mp4"][0][0] for q in Q]
    rkc = [C[f"{q}::{S[f'datasets/MSRVTT-v2/videos/{q}.mp4'][0].index(c)}"] for q, c in zip(Q, rk)]
    out[name] = {"top1_sim": round(sum(t1) / 300, 4), "top20_sim": round(sum(t20) / 300, 4),
                 "oracle20": round(sum(o20) / 300, 4), "oracle7": round(sum(o7) / 300, 4),
                 "oracle3": round(sum(o3) / 300, 4), "top1_caption_cider": round(sum(top1c) / 300, 4),
                 "rerank_top1_cider": round(sum(rkc) / 300, 4)}
out["mean_top20_overlap"] = round(sum(out["top20_overlap"]) / 300, 4)
out["mean_top3_overlap"] = round(sum(out["top3_overlap"]) / 300, 4)
out.pop("top20_overlap"); out.pop("top3_overlap")
json.dump(out, open(os.path.join(ROOT, "reports/pre_post_v2.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
