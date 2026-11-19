#!/usr/bin/env python3
"""Build Utility-Aware Reranking variant JSONs (training-free, deterministic)."""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/home/bn/videoufo_research"
SIMS = os.path.join(ROOT, "external/external_sims_full.json")
DIAG = os.path.join(ROOT, "external/external_top20_full.json")
EMB = os.path.join(ROOT, "embeddings/videoufo_brief_text_fp16.npy")
IDS = os.path.join(ROOT, "embeddings/videoufo_brief_ids.txt")
QEMB = os.path.join(ROOT, "embeddings/msrvtt_test_video_fp16.npy")
QIDS = os.path.join(ROOT, "embeddings/msrvtt_test_video_ids.txt")
ACT = os.path.join(ROOT, "external/uar_action_emb_fp16.npy")
ACTM = os.path.join(ROOT, "external/uar_action_meta.json")
XV = os.path.join(ROOT, "features/xclip_video")
OUT = os.path.join(ROOT, "external")
TOPK = 20
TOPN = 7


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def main():
    S = json.load(open(SIMS))
    D = json.load(open(DIAG))
    ids = open(IDS).read().splitlines()
    id2row = {c: i for i, c in enumerate(ids)}
    E = np.load(EMB, mmap_mode="r")
    qids = open(QIDS).read().splitlines()
    qpos = {q: i for i, q in enumerate(qids)}
    Q = np.load(QEMB)
    A = np.load(ACT)
    ameta = json.load(open(ACTM))
    apos = {c: i for i, c in enumerate(ameta["captions"])}

    variants = {
        "E": dict(a=1.0, b=0.0, g=0.0, lam=0.30, mu=0.50),
        "F": dict(a=1.0, b=0.50, g=0.0, lam=0.0, mu=0.0),
        "G": dict(a=1.0, b=0.30, g=0.30, lam=0.0, mu=0.0),
        "H": dict(a=1.0, b=0.30, g=0.30, lam=0.30, mu=0.50),
    }
    out = {k: {} for k in variants}
    out["I"] = {}
    conf_rows = []

    for key, (caps, rels) in S.items():
        q = key.split("/")[-1][:-4]
        diag = D[key]
        n = len(caps)
        tvec = []
        gvec = []
        srcs = []
        for j in range(n):
            cid = diag[j]["clip_ids"][0]
            row = id2row[cid]
            t = np.asarray(E[row], dtype=np.float32)
            tn = t / (np.linalg.norm(t) + 1e-8)
            tvec.append(tn)
            vp = os.path.join(XV, cid + ".npy")
            if os.path.exists(vp):
                v = np.load(vp).astype(np.float32)
                gvec.append(float(np.dot(v / (np.linalg.norm(v) + 1e-8), tn)))
            else:
                gvec.append(0.0)
            srcs.append(cid.rsplit(".", 1)[0])
        tvec = np.stack(tvec)
        qv = Q[qpos[q]].astype(np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-8)
        act = np.array([(float(np.dot(qv, A[apos[c]].astype(np.float32) /
                                   (np.linalg.norm(A[apos[c]].astype(np.float32)) + 1e-8)))
                         if ameta["phrases"][apos[c]] else 0.0)
                        for c in caps], dtype=np.float32)
        rel = norm(rels)
        actn = norm(act)
        gnd = norm(gvec)
        capmat = tvec @ tvec.T
        for name, w in variants.items():
            base = w["a"] * rel + w["b"] * actn + w["g"] * gnd
            chosen = []
            while len(chosen) < TOPN:
                best, bi = -1e9, -1
                for i in range(n):
                    if i in chosen:
                        continue
                    red = 0.0
                    if chosen:
                        red += w["lam"] * max(capmat[i, j] for j in chosen)
                        red += w["mu"] * (1.0 if srcs[i] in {srcs[j] for j in chosen} else 0.0)
                    sc = float(base[i]) - red
                    if sc > best + 1e-9:
                        best, bi = sc, i
                chosen.append(bi)
            sel = [caps[i] for i in chosen]
            out[name][key] = [sel, [float(base[i]) for i in chosen],
                              [diag[i]["clip_ids"] for i in chosen],
                              [float(rels[i]) for i in chosen]]
        # confidence for selective variant I
        margin = float(rel[0] - (rel[1] if n > 1 else 0.0))
        top7 = np.argsort(-rel)[:7]
        conf_rows.append((key, 0.5 * float(rel[0]) + 0.25 * margin +
                          0.15 * float(actn[top7].mean()) + 0.10 * float(gnd[top7].mean())))
    # variant I: H ranking + adaptive k
    confs = np.array([c for _, c in conf_rows], dtype=np.float32)
    q10, q25, q45, q70 = np.quantile(confs, [0.10, 0.25, 0.45, 0.70])
    for key, c in conf_rows:
        k = 0 if c <= q10 else 1 if c <= q25 else 3 if c <= q45 else 5 if c <= q70 else 7
        v = out["H"][key]
        out["I"][key] = [v[0][:k], v[1][:k], v[2][:k], v[3][:k]]

    for name in out:
        cap = {k: [v[0], v[1], v[2]] for k, v in out[name].items()}
        sims = {k: [v[0], v[3]] for k, v in out[name].items()}
        json.dump(cap, open(os.path.join(OUT, f"external_uar_{name}_cap.json"), "w"), indent=1)
        json.dump(sims, open(os.path.join(OUT, f"external_uar_{name}_sims.json"), "w"), indent=1)
        lens = [len(v[0]) for v in out[name].values()]
        print(name, "queries", len(cap), "mean_k", round(float(np.mean(lens)), 3))


if __name__ == "__main__":
    main()
