#!/usr/bin/env python3
"""FAITHFUL_V2 CF3/D3/H3 JSONs from v2 retrieval."""
import json
import os

import numpy as np

ROOT = "/home/bn/videoufo_research"
S = json.load(open(os.path.join(ROOT, "external/external_v2_sims.json")))
D = json.load(open(os.path.join(ROOT, "external/external_v2_top20.json")))
R = json.load(open(os.path.join(ROOT, "external/external_v2_reranked.json")))
EMB = os.path.join(ROOT, "embeddings/videoufo_brief_text_fp16.npy")
IDS = os.path.join(ROOT, "embeddings/videoufo_brief_ids.txt")
ACT = os.path.join(ROOT, "external/uar_action_union_emb_fp16.npy")
ACTM = os.path.join(ROOT, "external/uar_action_union_meta.json")
XV = os.path.join(ROOT, "features/xclip_video")
Q = np.load(os.path.join(ROOT, "embeddings/msrvtt_test_video_xclip_faithful_v2_video_fp16.npy"))
QIDS = open(os.path.join(ROOT, "embeddings/msrvtt_test_video_xclip_faithful_v2_ids.txt")).read().splitlines()
qpos = {q: i for i, q in enumerate(QIDS)}
OUT = os.path.join(ROOT, "external")
ids = open(IDS).read().splitlines(); id2row = {c: i for i, c in enumerate(ids)}
E = np.load(EMB, mmap_mode="r"); A = np.load(ACT)
ameta = json.load(open(ACTM)); apos = {c: i for i, c in enumerate(ameta["captions"])}


def z(x):
    x = np.asarray(x, np.float32); s = x.std()
    return np.zeros_like(x) if s < 1e-8 else np.clip((x - x.mean()) / s, -3, 3)


cf3, cf3s, d3, d3s, h3, h3s = {}, {}, {}, {}, {}, {}
for key, (caps, rels) in S.items():
    cf3[key] = [caps[:3], [0.0] * 3, [x["clip_ids"] for x in D[key][:3]]]
    cf3s[key] = [caps[:3], rels[:3]]
    rc = R[key][0][:3]; rq = R[key][1][:3]; rl = R[key][2][:3]
    d3[key] = [rc, rq, rl]
    m = {c: s for c, s in zip(caps, rels)}
    d3s[key] = [rc, [m.get(c, 0.0) for c in rc]]

    q = key.split("/")[-1][:-4]; n = len(caps)
    tv, gv, src, lists = [], [], [], []
    for j in range(n):
        cid = D[key][j]["clip_ids"][0]; row = id2row[cid]
        t = np.asarray(E[row], np.float32); tn = t / (np.linalg.norm(t) + 1e-8)
        tv.append(tn)
        vp = os.path.join(XV, cid + ".npy")
        if os.path.exists(vp):
            v = np.load(vp).astype(np.float32); gv.append(float(np.dot(v / (np.linalg.norm(v) + 1e-8), tn)))
        else:
            gv.append(0.0)
        src.append(cid.rsplit(".", 1)[0]); lists.append(D[key][j]["clip_ids"])
    capmat = np.stack(tv) @ np.stack(tv).T
    qv = Q[qpos[q]].astype(np.float32); qv = qv / (np.linalg.norm(qv) + 1e-8)
    act = np.array([(float(np.dot(qv, A[apos[c]].astype(np.float32) / (np.linalg.norm(A[apos[c]].astype(np.float32)) + 1e-8)))
                     if ameta["phrases"][apos[c]] else 0.0) for c in caps], np.float32)
    U = (z(rels) + z(act) + z(gv)) / 3
    chosen, cs = [], set()
    for _ in range(3):
        best, bi = -1e9, -1
        for i in range(n):
            if i in chosen: continue
            red = 0.0
            if chosen:
                red += 0.30 * max(capmat[i, j] for j in chosen)
                red += 0.50 * (1.0 if src[i] in cs else 0.0)
            sc = float(U[i]) - red
            if sc > best + 1e-9: best, bi = sc, i
        chosen.append(bi); cs.add(src[bi])
    h3[key] = [[caps[i] for i in chosen], [float(U[i]) for i in chosen], [lists[i] for i in chosen]]
    h3s[key] = [[caps[i] for i in chosen], [float(rels[i]) for i in chosen]]

for name, cap, sims in [("cf3", cf3, cf3s), ("d3", d3, d3s), ("h3", h3, h3s)]:
    json.dump(cap, open(os.path.join(OUT, f"external_v2_{name}_cap.json"), "w"), indent=1)
    json.dump(sims, open(os.path.join(OUT, f"external_v2_{name}_sims.json"), "w"), indent=1)
sel = set()
for cap in (cf3, d3, h3):
    for x in cap.values():
        for lst in x[2]:
            if lst: sel.add(lst[0])
open(os.path.join(OUT, "external_v2_selected_clips.txt"), "w").write("\n".join(sorted(sel)) + "\n")
print("written; selected clips", len(sel))
