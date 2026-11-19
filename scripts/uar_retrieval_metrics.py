#!/usr/bin/env python3
"""Retrieval-side metrics for UAR variants (diagnostic)."""
import json
import os
import sys
from collections import Counter

ROOT = "/home/bn/videoufo_research"
sys.path.insert(0, os.path.join(ROOT, "evcap_external/evaluation"))
from pycocoevalcap.cider.cider import Cider  # noqa: E402

QIDS = [l.strip() for l in open(os.path.join(ROOT, "dev300_queries.txt")) if l.strip()]
GT = {}
for line in open(os.path.join(ROOT, "evcap_external/dev300/dev300.caption.tsv")):
    p = line.split("\t", 1)
    GT[p[0].split("/")[-1][:-4]] = [x["caption"] for x in json.loads(p[1])]
S = json.load(open(os.path.join(ROOT, "external/external_sims_full.json")))
DIAG = json.load(open(os.path.join(ROOT, "external/external_top20_full.json")))

# per-candidate global-df CIDEr for all 300 x 20
gts, res = {}, {}
for q in QIDS:
    caps = S[f"datasets/MSRVTT-v2/videos/{q}.mp4"][0]
    for i, c in enumerate(caps):
        gts[f"{q}::{i}"] = GT[q]
        res[f"{q}::{i}"] = [c]
CS = dict(zip(gts, Cider().compute_score(gts, res)[1]))


def rl(a, b):
    x, y = a.lower().split(), b.lower().split()
    if not x or not y:
        return 0.0
    dp = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if x[i - 1] == y[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    l = dp[-1][-1]
    if not l:
        return 0.0
    p, r = l / len(x), l / len(y)
    return 2 * p * r / (p + r)


def run(v):
    cap = json.load(open(os.path.join(ROOT, f"external/external_uar_{v}_cap.json")))
    top1, o7, us, dup = [], [], [], []
    for q in QIDS:
        key = f"datasets/MSRVTT-v2/videos/{q}.mp4"
        caps = S[key][0]
        sel = cap[key][0]
        vals = [CS[f"{q}::{caps.index(c)}"] for c in sel] if sel else [0.0]
        top1.append(vals[0] if vals else 0.0)
        o7.append(max(vals) if vals else 0.0)
        ids = [cap[key][2][j][0] for j in range(len(sel))]
        us.append(len({i.rsplit(".", 1)[0] for i in ids}))
        vals2 = [rl(sel[i], sel[j]) for i in range(len(sel)) for j in range(i + 1, len(sel))]
        dup.append(sum(x > 0.5 for x in vals2) / len(vals2) if vals2 else 0.0)
    n = len(QIDS)
    return {"variant": v, "mean_top1_cider": round(sum(top1) / n, 4),
            "mean_oracle7": round(sum(o7) / n, 4),
            "mean_unique_sources_top7": round(sum(us) / n, 2),
            "near_dup_ratio": round(sum(dup) / n, 3)}


if __name__ == "__main__":
    out = []
    for v in ["E", "F", "Fv", "G", "H", "H1", "H3", "H5", "H7", "Ip", "Ia"]:
        out.append(run(v))
        print(out[-1], flush=True)
    json.dump(out, open(os.path.join(ROOT, "reports/uar_retrieval_metrics.json"), "w"), indent=2)
