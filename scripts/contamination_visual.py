#!/usr/bin/env python3
"""Lightweight visual near-duplicate audit: MSR-VTT val videos vs retrieved VideoUFO clips.

dHash (64-bit) on 3 frames per video; min Hamming distance per query.
Random corpus sample provides the false-positive reference.
"""
import json
import os
import random
import sys

import cv2
import decord
import numpy as np

cv2.setNumThreads(1)
ROOT = "/home/bn/videoufo_research"
DATA = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2"
UFO = "/home/bn/datasets/VideoUFO/videos/VideoUFO"
OUT = os.path.join(ROOT, "reports/contamination_visual.json")
R = json.load(open(os.path.join(ROOT, "external/external_val_reranked.json")))


def dhash(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def hashes(path):
    try:
        vr = decord.VideoReader(path, num_threads=1)
        n = len(vr)
        if n < 3:
            return []
        idx = [int(n * 0.2), int(n * 0.5), int(n * 0.8)]
        return [dhash(f) for f in vr.get_batch(idx).asnumpy()]
    except Exception:
        return []


def ham(a, b):
    return bin(a ^ b).count("1")


def mindist(A, B):
    if not A or not B:
        return 999
    return min(ham(a, b) for a in A for b in B)


pairs = {}
for key in R:
    q = key.split("/")[-1][:-4]
    pairs[q] = [lst[0] for lst in R[key][2][:7] if lst]
uniq = {}
for q, cs in pairs.items():
    for c in cs:
        uniq.setdefault(c, None)

print("queries", len(pairs), "candidates", len(uniq), flush=True)
qh, ch = {}, {}
for i, q in enumerate(sorted(pairs)):
    qh[q] = hashes(os.path.join(DATA, "videos", q + ".mp4"))
for i, c in enumerate(sorted(uniq)):
    ch[c] = hashes(os.path.join(OUT and UFO, c + ".mp4"))
    if (i + 1) % 1000 == 0:
        print("cand", i + 1, "/", len(uniq), flush=True)

per = {}
for q, cs in pairs.items():
    ds = [mindist(qh[q], ch[c]) for c in cs]
    per[q] = {"min_d": min(ds), "n_le6": sum(d <= 6 for d in ds), "n_le10": sum(d <= 10 for d in ds)}

rng = random.Random(0)
allc = sorted(uniq)
rand = rng.sample(allc, min(500, len(allc)))
ctrl = []
for q in list(per)[:100]:
    ctrl.append(min(mindist(qh[q], ch[c]) for c in rand))

res = {
    "queries": len(per),
    "candidates": len(uniq),
    "query_min_d_hist": {t: sum(v["min_d"] <= t for v in per.values()) for t in (0, 4, 6, 8, 10)},
    "queries_with_candidate_le6": sum(v["n_le6"] > 0 for v in per.values()),
    "queries_with_candidate_le10": sum(v["n_le10"] > 0 for v in per.values()),
    "random_control_mean_min_d": float(np.mean(ctrl)),
    "per_query": per,
}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "per_query"}, indent=2))
