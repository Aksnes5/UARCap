#!/usr/bin/env python3
"""Build prompt-order and tail-quality variants of the val497 FULL D top-7.

Outputs external_val_{posrev,postop3last,midtail7,lowtail7}_{cap,sims}.json.
Same candidate set for position variants; tail variants replace ranks 4-7 with
ranks 8-11 / 17-20 of the quality ordering of the top-20.
"""
import json
import os

ROOT = "/home/bn/videoufo_research"
EXT = os.path.join(ROOT, "external")
CAP = json.load(open(os.path.join(EXT, "external_val_d7_cap.json")))
SIM = json.load(open(os.path.join(EXT, "external_val_d7_sims.json")))
RAW = json.load(open(os.path.join(EXT, "external_val_sims.json")))
DIAG = json.load(open(os.path.join(EXT, "external_val_top20.json")))


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


def write(name, cap, sim):
    json.dump(cap, open(os.path.join(EXT, f"external_val_{name}_cap.json"), "w"), indent=1)
    json.dump(sim, open(os.path.join(EXT, f"external_val_{name}_sims.json"), "w"), indent=1)


pos = {"posrev": [6, 5, 4, 3, 2, 1, 0], "postop3last": [3, 4, 5, 6, 0, 1, 2]}
for name, order in pos.items():
    cap, sim = {}, {}
    for key, v in CAP.items():
        cap[key] = [[v[0][i] for i in order], [v[1][i] for i in order], [v[2][i] for i in order]]
        s = SIM[key]
        m = dict(zip(s[0], s[1]))
        sim[key] = [[cap[key][0][i] for i in range(7)], [m[c] for c in cap[key][0]]]
    write(name, cap, sim)
    print(name, len(cap))

DORDER = {}
for key in RAW:
    raw_caps, raw_scores = RAW[key][0], RAW[key][1]
    q = [sum(rouge_l(raw_caps[i], raw_caps[j]) for j in range(len(raw_caps)) if j != i) / (len(raw_caps) - 1)
         for i in range(len(raw_caps))]
    order = sorted(range(len(raw_caps)), key=lambda i: (-q[i], i))
    capmap = {}
    for j, d in enumerate(DIAG[key]):
        capmap[d["caption"]] = (raw_scores[j], d["clip_ids"])

for name, tail_slice in [("midtail7", slice(7, 11)), ("lowtail7", slice(16, 20))]:
    cap, sim = {}, {}
    for key in CAP:
        raw_caps, raw_scores = RAW[key][0], RAW[key][1]
        q = [sum(rouge_l(raw_caps[i], raw_caps[j]) for j in range(len(raw_caps)) if j != i) / (len(raw_caps) - 1)
             for i in range(len(raw_caps))]
        order = sorted(range(len(raw_caps)), key=lambda i: (-q[i], i))
        idx = order[:3] + order[tail_slice]
        caps = [raw_caps[i] for i in idx]
        lists = [DIAG[key][i]["clip_ids"] for i in idx]
        scores = [float(raw_scores[i]) for i in idx]
        cap[key] = [caps, scores, lists]
        sim[key] = [caps, scores]
    write(name, cap, sim)
    print(name, len(cap))
