#!/usr/bin/env python3
"""Oracle-evidence and distractor-tail variants of the val497 top-20 pool.

Oracle order: per-candidate CIDEr against GT references (DIAGNOSTIC ONLY).
Writes external_val_oracle{1,3,7}_*, goodtail7, hardwrong7, easywrong7.
"""
import json
import os
import sys

ROOT = "/home/bn/videoufo_research"
EXT = os.path.join(ROOT, "external")
sys.path.insert(0, os.path.join(ROOT, "evcap_external/evaluation"))
from pycocoevalcap.cider.cider import Cider  # noqa: E402

RAW = json.load(open(os.path.join(EXT, "external_val_sims.json")))
DIAG = json.load(open(os.path.join(EXT, "external_val_top20.json")))
GT = {}
for line in open("/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2/val.caption.tsv"):
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


def write(name, cap):
    json.dump(cap, open(os.path.join(EXT, f"external_val_{name}_cap.json"), "w"), indent=1)
    json.dump({k: [v[0], v[1]] for k, v in cap.items()},
              open(os.path.join(EXT, f"external_val_{name}_sims.json"), "w"), indent=1)


oracle7, goodtail, hardwrong, easywrong = {}, {}, {}, {}
for key, (caps, sims) in RAW.items():
    q = key.split("/")[-1][:-4]
    gt, res = {}, {}
    for i, c in enumerate(caps):
        gt[f"{q}::{i}"] = GT[q]
        res[f"{q}::{i}"] = [c]
    C = dict(zip(gt, Cider().compute_score(gt, res)[1]))
    orc = [C[f"{q}::{i}"] for i in range(20)]
    oorder = sorted(range(20), key=lambda i: (-orc[i], i))
    lists = [DIAG[key][i]["clip_ids"] for i in range(20)]
    oracle7[key] = [[caps[i] for i in oorder[:7]], [float(sims[i]) for i in oorder[:7]],
                    [lists[i] for i in oorder[:7]]]

    qual = [sum(rouge_l(caps[i], caps[j]) for j in range(20) if j != i) / 19 for i in range(20)]
    dorder = sorted(range(20), key=lambda i: (-qual[i], i))
    top3 = dorder[:3]
    rest = dorder[3:]
    low = sorted(rest, key=lambda i: orc[i])[:8]
    good = sorted(rest, key=lambda i: -orc[i])[:4]
    hard = sorted(low, key=lambda i: -sims[i])[:4]
    easy = sorted(low, key=lambda i: sims[i])[:4]
    for name, d in [("goodtail7", goodtail), ("hardwrong7", hardwrong), ("easywrong7", easywrong)]:
        idx = top3 + {"goodtail7": good, "hardwrong7": hard, "easywrong7": easy}[name]
        d[key] = [[caps[i] for i in idx], [float(sims[i]) for i in idx], [lists[i] for i in idx]]

write("oracle7", oracle7)
write("goodtail7", goodtail)
write("hardwrong7", hardwrong)
write("easywrong7", easywrong)
print("queries", len(oracle7))
