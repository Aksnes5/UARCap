#!/usr/bin/env python3
"""Build UAR k=1/3/5/7 and R3/D3 for the 497 official val queries."""
import json
import os
import sys

import numpy as np

ROOT = "/home/bn/videoufo_research"
SIMS = os.path.join(ROOT, "external/external_val_sims.json")
DIAG = os.path.join(ROOT, "external/external_val_top20.json")
RER = os.path.join(ROOT, "external/external_val_reranked.json")
EMB = os.path.join(ROOT, "embeddings/videoufo_brief_text_fp16.npy")
IDS = os.path.join(ROOT, "embeddings/videoufo_brief_ids.txt")
QEMB = os.path.join(ROOT, "embeddings/msrvtt_val_video_fp16.npy")
QIDS = os.path.join(ROOT, "embeddings/msrvtt_val_video_ids.txt")
ACT = os.path.join(ROOT, "external/uar_action_emb_fp16.npy")
ACTM = os.path.join(ROOT, "external/uar_action_meta.json")
XV = os.path.join(ROOT, "features/xclip_video")
OUT = os.path.join(ROOT, "external")


def z(x):
    x = np.asarray(x, dtype=np.float32)
    s = x.std()
    return np.zeros_like(x) if s < 1e-8 else np.clip((x - x.mean()) / s, -3, 3)


def main():
    S = json.load(open(SIMS)); D = json.load(open(DIAG)); R = json.load(open(RER))
    ids = open(IDS).read().splitlines(); id2row = {c: i for i, c in enumerate(ids)}
    E = np.load(EMB, mmap_mode="r"); A = np.load(ACT)
    ameta = json.load(open(ACTM)); apos = {c: i for i, c in enumerate(ameta["captions"])}
    Q = np.load(QEMB); qids = open(QIDS).read().splitlines()
    out = {n: {} for n in ["H1", "H3", "H5", "H7", "R3", "D3"]}
    for ki, q in enumerate(qids):
        key = f"datasets/MSRVTT-v2/videos/{q}.mp4"
        caps, rels = S[key]; diag = D[key]; n = len(caps)
        tv, gv, src, lists = [], [], [], []
        for j in range(n):
            cid = diag[j]["clip_ids"][0]; row = id2row[cid]
            t = np.asarray(E[row], np.float32); tn = t / (np.linalg.norm(t) + 1e-8)
            tv.append(tn)
            vp = os.path.join(XV, cid + ".npy")
            if os.path.exists(vp):
                v = np.load(vp).astype(np.float32); gv.append(float(np.dot(v / (np.linalg.norm(v) + 1e-8), tn)))
            else:
                gv.append(0.0)
            src.append(cid.rsplit(".", 1)[0]); lists.append(diag[j]["clip_ids"])
        tv = np.stack(tv); capmat = tv @ tv.T
        qv = Q[ki].astype(np.float32); qv = qv / (np.linalg.norm(qv) + 1e-8)
        act = np.array([(float(np.dot(qv, A[apos[c]].astype(np.float32) / (np.linalg.norm(A[apos[c]].astype(np.float32)) + 1e-8)))
                         if ameta["phrases"][apos[c]] else 0.0) for c in caps], np.float32)
        zr, za, zg = z(rels), z(act), z(gv)
        U = (zr + za + zg) / 3
        def select(k):
            chosen, cs = [], set()
            for _ in range(k):
                best, bi = -1e9, -1
                for i in range(n):
                    if i in chosen: continue
                    red = 0.0
                    if chosen:
                        red += 0.30 * max(capmat[i, j] for j in chosen)
                        red += 0.50 * (1.0 if src[i] in cs else 0.0)
                    s = float(U[i]) - red
                    if s > best + 1e-9: best, bi = s, i
                chosen.append(bi); cs.add(src[bi])
            return chosen
        for k in (1, 3, 5, 7):
            ch = select(k)
            out[f"H{k}"][key] = [ch, U, rels, lists]
        out["R3"][key] = [caps[:3], [0.0] * 3, rels[:3], lists[:3]]
        rc = R[key][0][:3]; rq = R[key][1][:3]; rl = R[key][2][:3]
        m = {c: s for c, s in zip(caps, rels)}
        out["D3"][key] = [rc, rq, [m.get(c, 0.0) for c in rc], rl]
    for name in out:
        cap, sims = {}, {}
        for key, (ch, u, rels, lists) in out[name].items():
            if name in ("R3", "D3"):
                cap[key] = [ch, u, lists]
                sims[key] = [ch, rels]
            else:
                cap[key] = [[S[key][0][i] for i in ch], [float(u[i]) for i in ch], [lists[i] for i in ch]]
                sims[key] = [[S[key][0][i] for i in ch], [float(rels[i]) for i in ch]]
        json.dump(cap, open(os.path.join(OUT, f"external_val_uar_{name}_cap.json"), "w"), indent=1)
        json.dump(sims, open(os.path.join(OUT, f"external_val_uar_{name}_sims.json"), "w"), indent=1)
        print(name, len(cap), flush=True)


if __name__ == "__main__":
    main()
