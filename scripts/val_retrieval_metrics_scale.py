#!/usr/bin/env python3
"""Retrieval-side k-grid metrics for a given manifest scale (val497)."""
import json
import os
import sys

ROOT = "/home/bn/videoufo_research"
sys.path.insert(0, os.path.join(ROOT, "evcap_external/evaluation"))
from pycocoevalcap.cider.cider import Cider  # noqa: E402

SCALE = sys.argv[1]
DATA = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2"
S = json.load(open(os.path.join(ROOT, f"external/external_val_{SCALE}_sims.json")))
R = json.load(open(os.path.join(ROOT, f"external/external_val_{SCALE}_reranked.json")))
GT = {}
for line in open(os.path.join(DATA, "val.caption.tsv")):
    p = line.split("\t", 1)
    GT[p[0].split("/")[-1][:-4]] = [x["caption"] for x in json.loads(p[1])]


def rouge_l(a, b):
    x, y = a.lower().split(), b.lower().split()
    if not x or not y:
        return 0.0
    dp = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if x[i - 1] == y[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    l = dp[-1][-1]
    if l == 0:
        return 0.0
    p, r = l / len(x), l / len(y)
    return 2 * p * r / (p + r)


def cand_cider(seqs):
    gts, res = {}, {}
    for q, cs in seqs.items():
        for i, c in enumerate(cs):
            gts[f"{q}::{i}"] = GT[q]
            res[f"{q}::{i}"] = [c]
    return dict(zip(gts, Cider().compute_score(gts, res)[1]))


CF = {q.split("/")[-1][:-4]: S[q][0] for q in S}
CFS = {q.split("/")[-1][:-4]: S[q][1] for q in S}
DV = {q.split("/")[-1][:-4]: R[q][0] for q in R}
DR = {q.split("/")[-1][:-4]: R[q][1] for q in R}
qids = list(CF)
CC, DC = cand_cider(CF), cand_cider(DV)


def near_dup(caps):
    n = len(caps)
    if n < 2:
        return 0.0
    return sum(rouge_l(caps[i], caps[j]) >= 0.8 for i in range(n) for j in range(i + 1, n)) / (n * (n - 1) / 2)


def table(caps, scores, C, tag):
    rows = {}
    for k in (1, 3, 5, 7):
        rows[k] = {
            "top1_caption_cider": round(sum(C[f"{q}::0"] for q in qids) / len(qids), 4),
            "oracle": round(sum(max(C[f"{q}::{i}"] for i in range(k)) for q in qids) / len(qids), 4),
            "mean_selection_score": round(sum(sum(scores[q][:k]) / k for q in qids) / len(qids), 4),
            "unique_sources": round(sum(len({c.rsplit('.', 1)[0] for c in caps[q][:k]}) for q in qids) / len(qids), 3),
            "near_dup_ratio": round(sum(near_dup(caps[q][:k]) for q in qids) / len(qids), 4),
        }
        print(tag, k, rows[k], flush=True)
    return rows


out = {"scale": SCALE, "CF": table(CF, CFS, CC, "CF"), "D": table(DV, DR, DC, "D")}
json.dump(out, open(os.path.join(ROOT, f"reports/val_retrieval_grid_{SCALE}.json"), "w"), indent=2)
