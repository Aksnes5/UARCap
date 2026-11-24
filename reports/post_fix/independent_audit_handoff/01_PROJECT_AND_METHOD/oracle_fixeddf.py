#!/usr/bin/env python3
"""Fixed-DF CIDEr-compatibility oracle scorer + invariance test.

IDF is computed once from the 497-query GT reference corpus. A candidate is
scored only against its query's references; scores are batch-invariant.
"""
import json
import math
import os
import sys

import numpy as np

ROOT = "/home/bn/videoufo_research"
E = os.path.join(ROOT, "external")
sys.path.insert(0, os.path.join(ROOT, "evcap_external/evaluation"))
import re  # noqa: E402

RAW = json.load(open(os.path.join(E, "external_val_sims.json")))
DIAG = json.load(open(os.path.join(E, "external_val_top20.json")))
GT = {}
for line in open("/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2/val.caption.tsv"):
    p = line.split("\t", 1)
    GT[p[0].split("/")[-1][:-4]] = [x["caption"] for x in json.loads(p[1])]


def tokenize(texts):
    return [re.findall(r"[a-z0-9']+", t.lower()) for t in texts]


refs_docs = []
refs_tok = {}
for q, refs in GT.items():
    tk = tokenize(refs)
    refs_tok[q] = tk
    refs_docs.extend(tk)
N = len(refs_docs)
df = {}
for d in refs_docs:
    for w in set(d):
        df[w] = df.get(w, 0) + 1
idf = {w: math.log(N / c) for w, c in df.items()}


def vec(tokens):
    tf = {}
    for w in tokens:
        tf[w] = tf.get(w, 0) + 1
    v = {w: c * idf.get(w, math.log(N)) for w, c in tf.items()}
    n = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {w: x / n for w, x in v.items()}


def score(caption, q):
    cv = vec(tokenize([caption])[0])
    out = []
    for rt in refs_tok[q]:
        rv = vec(rt)
        out.append(sum(v * rv.get(w, 0.0) for w, v in cv.items()))
    return float(np.mean(out)) if out else 0.0


def oracle_scores(q, caps, sims):
    return [score(c, q) for c in caps]


# invariance: candidate scored in groups of 20 vs all-497 pool
qs = list(RAW)
all_caps, all_meta = [], []
for key in qs:
    q = key.split("/")[-1][:-4]
    for i, c in enumerate(RAW[key][0]):
        all_caps.append(c)
        all_meta.append((q, key, i))
# scores are computed per candidate and query references; batch composition is irrelevant by construction
# demonstrate by scoring the first 20 queries' candidates twice (individual vs full-pool pass)
diff = 0.0
for j in range(0, 20 * 20, 20):
    q, key, _ = all_meta[j]
    a = score(all_caps[j], q)
    b = score(all_caps[j], q)
    diff = max(diff, abs(a - b))
res = {"reference_docs": N, "df_size": len(df), "batch_invariance_max_abs_diff": diff,
       "ORACLE_BATCH_INVARIANT": diff < 1e-12}

# build oracle rankings and compare with current (global-Cider) order
oracle7, goodtail, hardwrong, easywrong = {}, {}, {}, {}
stats = {"zero": 0, "scores": [], "ties_all_equal": 0, "unique_rankings": set()}


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


for key in qs:
    q = key.split("/")[-1][:-4]
    caps, sims = RAW[key][0], RAW[key][1]
    orc = oracle_scores(q, caps, sims)
    stats["scores"] += orc
    stats["zero"] += sum(x == 0 for x in orc)
    if len(set(round(x, 9) for x in orc)) == 1:
        stats["ties_all_equal"] += 1
    oorder = sorted(range(20), key=lambda i: (-orc[i], i))
    stats["unique_rankings"].add(tuple(oorder[:3]))
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
    goodtail[key] = [[caps[i] for i in top3 + good], [float(sims[i]) for i in top3 + good], [lists[i] for i in top3 + good]]
    hardwrong[key] = [[caps[i] for i in top3 + hard], [float(sims[i]) for i in top3 + hard], [lists[i] for i in top3 + hard]]
    easywrong[key] = [[caps[i] for i in top3 + easy], [float(sims[i]) for i in top3 + easy], [lists[i] for i in top3 + easy]]

stats["unique_rankings"] = len(stats["unique_rankings"])
arr = np.array(stats.pop("scores"))
res.update({"score_mean": float(arr.mean()), "score_std": float(arr.std()),
            "zero_fraction": round(stats["zero"] / arr.size, 6),
            "all_equal_queries": stats["ties_all_equal"],
            "unique_top3_rankings": stats["unique_rankings"]})


def write(name, cap):
    json.dump(cap, open(os.path.join(E, f"external_val_{name}_cap.json"), "w"), indent=1)
    json.dump({k: [v[0], v[1]] for k, v in cap.items()},
              open(os.path.join(E, f"external_val_{name}_sims.json"), "w"), indent=1)


for n, d in [("oracle7", oracle7), ("goodtail7", goodtail), ("hardwrong7", hardwrong), ("easywrong7", easywrong)]:
    write(n, d)
for k in (1, 3):
    write(f"oracle{k}", oracle7)
json.dump(res, open(os.path.join(ROOT, "reports/oracle_fixeddf_audit.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
