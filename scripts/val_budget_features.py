#!/usr/bin/env python3
"""Step H: inference-time features for each val497 query (no GT used)."""
import json
import os

import numpy as np

ROOT = "/home/bn/videoufo_research"
EXT = os.path.join(ROOT, "external")
S = json.load(open(os.path.join(EXT, "external_val_sims.json")))
R = json.load(open(os.path.join(EXT, "external_val_reranked.json")))
D = json.load(open(os.path.join(EXT, "external_val_top20.json")))
E = np.load(os.path.join(ROOT, "embeddings/videoufo_brief_text_fp16.npy"), mmap_mode="r")
IDS = open(os.path.join(ROOT, "embeddings/videoufo_brief_ids.txt")).read().splitlines()
id2row = {c: i for i, c in enumerate(IDS)}
XV = os.path.join(ROOT, "features/xclip_video")


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


def emb(cid):
    return np.asarray(E[id2row[cid]], np.float32)


rows = {}
for key in S:
    q = key.split("/")[-1][:-4]
    sims = S[key][1][:7]
    rc, rq, rl = R[key][0][:7], R[key][1][:7], R[key][2][:7]
    feats = {}
    for i in (1, 2, 3, 4, 5, 6, 7):
        feats[f"sim{i}"] = float(sims[i - 1])
    feats["sim1-sim2"] = sims[0] - sims[1]
    feats["sim1-sim3"] = sims[0] - sims[2]
    feats["sim3-sim5"] = sims[2] - sims[4]
    feats["sim5-sim7"] = sims[4] - sims[6]
    for k in (3, 5, 7):
        a = np.array(sims[:k]); feats[f"sim_mean{k}"] = float(a.mean()); feats[f"sim_std{k}"] = float(a.std())
    adj = np.diff(np.array(sims))
    feats["sim_max_adjacent_gap"] = float(np.max(adj))
    feats["sim_gap_pos"] = int(np.argmax(adj) + 1)  # gap after position g
    feats["sim_rel_decay"] = float((sims[0] - sims[6]) / (sims[0] + 1e-8))
    for i in (1, 2, 3, 5, 7):
        feats[f"q{i}"] = float(rq[i - 1])
    feats["q1-q3"] = rq[0] - rq[2]
    feats["q3-q5"] = rq[2] - rq[4]
    feats["q5-q7"] = rq[4] - rq[6]
    qa = np.array(rq)
    feats["q_mean7"] = float(qa.mean()); feats["q_std7"] = float(qa.std())
    feats["q_disp7"] = float(qa.std() / (abs(qa.mean()) + 1e-8))

    cids = [lst[0] for lst in rl]
    em = np.stack([emb(c) for c in cids])
    em = em / (np.linalg.norm(em, axis=1, keepdims=True) + 1e-8)
    cm = em @ em.T
    for k in (3, 5, 7):
        sub = cm[:k]
        iu = np.triu_indices(k, 1)
        feats[f"cap_cos_mean{k}"] = float(sub[iu].mean())
        feats[f"cap_cos_max{k}"] = float(sub[iu].max())
    feats["cap_cos_inc_1_3"] = feats["cap_cos_mean3"]
    feats["cap_cos_inc_3_5"] = feats["cap_cos_mean5"] - feats["cap_cos_mean3"]
    feats["cap_cos_inc_5_7"] = feats["cap_cos_mean7"] - feats["cap_cos_mean5"]

    def nd(cs):
        n = len(cs)
        if n < 2:
            return 0.0
        return sum(rouge_l(cs[i], cs[j]) >= 0.8 for i in range(n) for j in range(i + 1, n)) / (n * (n - 1) / 2)
    feats["near_dup7"] = nd(rc)
    feats["near_dup_inc_1_3"] = nd(rc[:3])
    feats["near_dup_inc_3_5"] = nd(rc[:5]) - nd(rc[:3])
    feats["near_dup_inc_5_7"] = nd(rc[:7]) - nd(rc[:5])
    srcs = [c.rsplit(".", 1)[0] for c in cids]
    feats["unique_sources7"] = len(set(srcs))
    feats["repeat_sources7"] = 7 - len(set(srcs))
    feats["source_div7"] = len(set(srcs)) / 7
    topics = [D[key][j].get("topics") for j in range(7)]
    feats["unique_topics7"] = len({str(t) for t in topics})

    gv, gv_ok = [], []
    for c, lst in zip(rc, rl):
        cid = lst[0]
        p = os.path.join(XV, cid + ".npy")
        t = emb(cid); t = t / (np.linalg.norm(t) + 1e-8)
        if os.path.exists(p):
            v = np.load(p).astype(np.float32); gv.append(float(np.dot(v / (np.linalg.norm(v) + 1e-8), t))); gv_ok.append(1)
        else:
            gv.append(0.0); gv_ok.append(0)
    ga = np.array(gv)
    feats["ground1"] = float(ga[0])
    for k in (3, 5, 7):
        feats[f"ground_mean{k}"] = float(ga[:k].mean())
    feats["ground_std7"] = float(ga.std())
    feats["ground_decay"] = float(ga[0] - ga[6])
    feats["ground_coverage"] = float(sum(gv_ok) / 7)
    rows[q] = feats

json.dump(rows, open(os.path.join(ROOT, "reports/val_budget_features.json"), "w"))
print("queries", len(rows), "features", len(next(iter(rows.values()))))
