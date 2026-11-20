#!/usr/bin/env python3
"""S7: source-clustered bootstrap for the primary paired comparisons."""
import json
import os
import random
import re
import sys

ROOT = "/home/bn/videoufo_research"
OUT = os.path.join(ROOT, "evcap_external/outputs")
sys.path.insert(0, os.path.join(ROOT, "evcap_external/evaluation"))
from pycocoevalcap.cider.cider import Cider  # noqa: E402

NAME7 = "eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json"
NAME_M = "eval_evcap_top3_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_msvd_generated_captions.json"


def yt(u):
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{6,})", u or "")
    return m.group(1) if m else None


def load(tag, variants, name):
    P = {}
    for v in variants:
        d = f"eval_{tag}_{v}"
        P[v] = {x["image_name"]: x["prediction"].strip() for x in json.load(open(os.path.join(OUT, d, name)))}
    rows = json.load(open(os.path.join(OUT, f"eval_{tag}_D3", name)))
    GT = {x["image_name"]: x["captions"] for x in rows}
    Q = sorted(set.intersection(*[set(p) for p in P.values()]))
    g = {q: GT[q] for q in Q}
    C = {v: dict(zip(Q, Cider().compute_score(g, {q: [P[v][q]] for q in Q})[1])) for v in variants}
    return Q, C


def clusters_msrvtt(Q):
    d = json.load(open("/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2/test_videodatainfo.json"))
    m = {v["video_id"]: yt(v.get("url")) for v in d["videos"]}
    return {q: m.get(q.split("/")[-1][:-4]) or q for q in Q}


def clusters_msvd(Q):
    return {q: q.split("/")[-1][:-4].rsplit("_", 2)[0] for q in Q}


def cluster_boot(Q, C, a, b, grp, rng, n=10000):
    d = {q: C[a][q] - C[b][q] for q in Q}
    by = {}
    for q in Q:
        by.setdefault(grp[q], []).append(q)
    srcs = list(by)
    obs = sum(d.values()) / len(d)
    boots = []
    for _ in range(n):
        sample = []
        for _ in range(len(srcs)):
            s = srcs[rng.randrange(len(srcs))]
            sample += by[s]
        boots.append(sum(d[q] for q in sample) / len(sample))
    boots.sort()
    return {"diff": round(obs, 4), "cluster_ci95": [round(boots[250], 4), round(boots[9750], 4)],
            "n_sources": len(srcs), "n_queries": len(Q)}


rng = random.Random(42)
out = {}
Q, C = load("test2690", ["D1", "D3", "D5", "D7", "Adaptive"], NAME7)
g = clusters_msrvtt(Q)
for a, b in [("D3", "D7"), ("D1", "D3"), ("D3", "D5"), ("Adaptive", "D3"), ("Adaptive", "D7")]:
    out[f"test2690:{a}-{b}"] = cluster_boot(Q, C, a, b, g, rng)
Qm, Cm = load("msvd", ["D1", "D3", "D5", "D7", "Adaptive"], NAME_M)
gm = clusters_msvd(Qm)
for a, b in [("D3", "D7"), ("D1", "D3"), ("D3", "D5"), ("Adaptive", "D3"), ("Adaptive", "D7")]:
    out[f"msvd:{a}-{b}"] = cluster_boot(Qm, Cm, a, b, gm, rng)
json.dump(out, open(os.path.join(ROOT, "reports/source_cluster_bootstrap.json"), "w"), indent=2)
for k, v in out.items():
    print(k, v)
