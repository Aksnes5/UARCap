#!/usr/bin/env python3
"""UAR v1: utility = mean(z(relevance), z(action), z(grounding)); diversity separate."""
import json
import os

import numpy as np

ROOT = "/home/bn/videoufo_research"
SIMS = os.path.join(ROOT, "external/external_sims_full.json")
DIAG = os.path.join(ROOT, "external/external_top20_full.json")
EMB = os.path.join(ROOT, "embeddings/videoufo_brief_text_fp16.npy")
IDS = os.path.join(ROOT, "embeddings/videoufo_brief_ids.txt")
QEMB = os.path.join(ROOT, "embeddings/msrvtt_test_video_fp16.npy")
QIDS = os.path.join(ROOT, "embeddings/msrvtt_test_video_ids.txt")
ACT = os.path.join(ROOT, "external/uar_action_emb_fp16.npy")
ACTV = os.path.join(ROOT, "external/uar_verbonly_emb_fp16.npy")
ACTM = os.path.join(ROOT, "external/uar_action_meta.json")
XV = os.path.join(ROOT, "features/xclip_video")
OUT = os.path.join(ROOT, "external")


def z(x):
    x = np.asarray(x, dtype=np.float32)
    s = x.std()
    if s < 1e-8:
        return np.zeros_like(x)
    return np.clip((x - x.mean()) / s, -3.0, 3.0)


def main():
    S = json.load(open(SIMS))
    D = json.load(open(DIAG))
    ids = open(IDS).read().splitlines()
    id2row = {c: i for i, c in enumerate(ids)}
    E = np.load(EMB, mmap_mode="r")
    qpos = {q: i for i, q in enumerate(open(QIDS).read().splitlines())}
    Q = np.load(QEMB)
    A = np.load(ACT)
    AV = np.load(ACTV)
    ameta = json.load(open(ACTM))
    apos = {c: i for i, c in enumerate(ameta["captions"])}

    names = ["E", "F", "Fv", "G", "H", "H1", "H3", "H5", "H7", "Ip", "Ia"]
    out = {n: {} for n in names}
    conf = []
    for key, (caps, rels) in S.items():
        q = key.split("/")[-1][:-4]
        diag = D[key]
        n = len(caps)
        tvec, gvec, srcs, clip_lists = [], [], [], []
        for j in range(n):
            cid = diag[j]["clip_ids"][0]
            t = np.asarray(E[id2row[cid]], dtype=np.float32)
            tn = t / (np.linalg.norm(t) + 1e-8)
            tvec.append(tn)
            vp = os.path.join(XV, cid + ".npy")
            if os.path.exists(vp):
                v = np.load(vp).astype(np.float32)
                gvec.append(float(np.dot(v / (np.linalg.norm(v) + 1e-8), tn)))
            else:
                gvec.append(0.0)
            srcs.append(cid.rsplit(".", 1)[0])
            clip_lists.append(diag[j]["clip_ids"])
        tvec = np.stack(tvec)
        capmat = tvec @ tvec.T
        qv = Q[qpos[q]].astype(np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-8)
        act = np.array([(float(np.dot(qv, A[apos[c]].astype(np.float32) /
                                   (np.linalg.norm(A[apos[c]].astype(np.float32)) + 1e-8)))
                         if ameta["phrases"][apos[c]] else 0.0) for c in caps], dtype=np.float32)
        actv = np.array([(float(np.dot(qv, AV[apos[c]].astype(np.float32) /
                                    (np.linalg.norm(AV[apos[c]].astype(np.float32)) + 1e-8)))
                          if ameta["verb_only"][apos[c]] else 0.0) for c in caps], dtype=np.float32)
        zr, za, zav, zg = z(rels), z(act), z(actv), z(gvec)
        U = {
            "E": zr,
            "F": (zr + za) / 2,
            "Fv": (zr + zav) / 2,
            "G": (zr + za + zg) / 3,
            "H": (zr + za + zg) / 3,
        }
        div = {"E": True, "F": False, "Fv": False, "G": False, "H": True}

        def select(u, use_div, k):
            chosen, chosen_src = [], set()
            for _ in range(k):
                best, bi = -1e9, -1
                for i in range(n):
                    if i in chosen:
                        continue
                    red = 0.0
                    if use_div and chosen:
                        red += 0.30 * max(capmat[i, j] for j in chosen)
                        red += 0.50 * (1.0 if srcs[i] in chosen_src else 0.0)
                    s = float(u[i]) - red
                    if s > best + 1e-9:
                        best, bi = s, i
                chosen.append(bi)
                chosen_src.add(srcs[bi])
            return chosen

        for vname in ["E", "F", "Fv", "G", "H"]:
            ch = select(U[vname], div[vname], 7)
            out[vname][key] = [ch, U[vname], rels, clip_lists]
        for k in (1, 3, 5, 7):
            ch = select(U["H"], True, k)
            out[f"H{k}"][key] = [ch, U["H"], rels, clip_lists]
        # adaptive confidence (raw scales, no GT)
        margin = float(rels[0] - (rels[1] if n > 1 else 0.0))
        top5 = np.argsort(-np.asarray(rels))[:5]
        c_abs = float(rels[0]) + 0.3 * margin + 0.2 * float(act[top5].mean()) + 0.1 * float(np.mean(gvec))
        conf.append((key, c_abs, float(U["H"].max() - np.partition(U["H"], -2)[-2])))
    vals = np.array([c[1] for c in conf])
    q10, q25, q45, q70 = np.quantile(vals, [0.10, 0.25, 0.45, 0.70])
    for key, c, _ in conf:
        k = 0 if c <= q10 else 1 if c <= q25 else 3 if c <= q45 else 5 if c <= q70 else 7
        out["Ip"][key] = out["H"][key][:]
        out["Ip"][key][0] = out["H"][key][0][:k]
        # absolute version uses fixed thresholds from the unlabeled distribution
        k2 = 0 if c <= q10 else 1 if c <= q25 else 3 if c <= q45 else 5 if c <= q70 else 7
        out["Ia"][key] = out["H"][key][:]
        out["Ia"][key][0] = out["H"][key][0][:k2]

    for name in names:
        cap, sims = {}, {}
        for key, (ch, u, rels, lists) in out[name].items():
            cap[key] = [[S[key][0][i] for i in ch], [float(u[i]) for i in ch],
                        [lists[i] for i in ch]]
            sims[key] = [[S[key][0][i] for i in ch], [float(rels[i]) for i in ch]]
        json.dump(cap, open(os.path.join(OUT, f"external_uar_{name}_cap.json"), "w"), indent=1)
        json.dump(sims, open(os.path.join(OUT, f"external_uar_{name}_sims.json"), "w"), indent=1)
        print(name, "mean_k", round(float(np.mean([len(v[0]) for v in cap.values()])), 3))


if __name__ == "__main__":
    main()
