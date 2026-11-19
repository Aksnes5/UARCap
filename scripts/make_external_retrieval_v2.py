#!/usr/bin/env python3
"""FAITHFUL_V2 FULL external retrieval for dev300 with nearest-resize query features."""
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/home/bn/videoufo_research"
XCLIP = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/EvcapProject/X-CLIP"
CKPT = os.path.join(XCLIP, "modelpth/k600_16_16.pth")
EMB = os.path.join(ROOT, "embeddings/videoufo_brief_text_fp16.npy")
QEMB = os.path.join(ROOT, "embeddings/msrvtt_test_video_xclip_faithful_v2_video_fp16.npy")
QIMG = os.path.join(ROOT, "embeddings/msrvtt_test_video_xclip_faithful_v2_imgfeat_fp16.npy")
QIDS = os.path.join(ROOT, "embeddings/msrvtt_test_video_xclip_faithful_v2_ids.txt")
CSV_PATH = "/home/bn/datasets/VideoUFO/metadata/VideoUFO.csv"
PHASE1 = os.path.join(ROOT, "retrieval/top20_full_exact_raw.json")
OUT = os.path.join(ROOT, "external")
CHUNK = 100000
TOPK = 20


class _Log:
    def info(self, m):
        pass


def load_meta():
    csv.field_size_limit(10 ** 9)
    ids, brief = [], []
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f); h = next(r)
        i_id, i_b = h.index("ID"), h.index("Brief Caption")
        for row in r:
            if row:
                ids.append(row[i_id]); brief.append(row[i_b].strip())
    return ids, brief


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


def main():
    ids, brief = load_meta(); id2row = {c: i for i, c in enumerate(ids)}
    full_ids = [l.split("\t")[0] for l in open(os.path.join(ROOT, "manifests/clips_full.tsv"))
                if not l.startswith("clip_id")]
    groups, order = {}, []
    for cid in full_ids:
        key = brief[id2row[cid]]
        if key not in groups:
            groups[key] = []; order.append(key)
        groups[key].append(cid)
    for k in groups:
        groups[k].sort()
    rep = np.array([id2row[groups[k][0]] for k in order], dtype=np.int64)
    E = np.load(EMB, mmap_mode="r")
    X = torch.from_numpy(np.asarray(E[rep])).cuda().half()
    allq = open(QIDS).read().splitlines(); qpos = {q: i for i, q in enumerate(allq)}
    qids = [e["query"] for e in json.load(open(PHASE1))]
    Q = torch.from_numpy(np.load(QEMB)).cuda().half()
    Q = F.normalize(Q.float(), dim=1).half()
    QI = np.load(QIMG, mmap_mode="r")
    sys.path.insert(0, XCLIP); os.chdir(XCLIP)
    import clip  # noqa
    from models import xclip_all as xclip
    sd = torch.load(CKPT, map_location="cpu")["model"]
    model = xclip.build_model(sd, T=16, droppath=0., use_checkpoint=False, logger=_Log(),
                              prompts_alpha=0.1, prompts_layers=2, use_cache=True,
                              mit_layers=1, text_batch_size=0).cuda().eval()
    print("model loaded", flush=True)
    sims, diag = {}, {}
    for q in qids:
        k = qpos[q]; qv = Q[k]
        img = torch.from_numpy(np.asarray(QI[k], dtype=np.float32)).cuda().half().unsqueeze(0)
        bv = torch.full((TOPK,), -1e4, device="cuda"); bi = torch.zeros(TOPK, dtype=torch.long, device="cuda")
        for s in range(0, len(order), CHUNK):
            tf = X[s:s + CHUNK].unsqueeze(0)
            with torch.no_grad(), torch.cuda.amp.autocast():
                tf = tf + model.prompts_generator(tf, img)
                sc = qv @ F.normalize(tf[0], dim=1).T
            v, i = torch.topk(sc, min(TOPK, sc.numel())); i = i + s
            av, ai = torch.cat([bv, v]), torch.cat([bi, i])
            bv, sel = torch.topk(av, TOPK); bi = ai[sel]
        caps = [order[i] for i in bi.cpu().tolist()]; rels = [float(x) for x in bv.cpu().tolist()]
        key = f"datasets/MSRVTT-v2/videos/{q}.mp4"; sims[key] = [caps, rels]
        diag[key] = [{"caption": c, "score": s, "clip_ids": groups[c]} for c, s in zip(caps, rels)]
    reranked = {}
    for key, (caps, rels) in sims.items():
        q = [sum(rouge_l(caps[i], caps[j]) for j in range(len(caps)) if j != i) / (len(caps) - 1)
             for i in range(len(caps))]
        idx = sorted(range(len(caps)), key=lambda i: (-q[i], i))[:7]
        reranked[key] = [[caps[i] for i in idx], [q[i] for i in idx], [groups[caps[i]] for i in idx]]
    json.dump(sims, open(os.path.join(OUT, "external_v2_sims.json"), "w"), indent=1)
    json.dump(diag, open(os.path.join(OUT, "external_v2_top20.json"), "w"), indent=1)
    json.dump(reranked, open(os.path.join(OUT, "external_v2_reranked.json"), "w"), indent=1)
    print("DONE", len(sims), flush=True)


if __name__ == "__main__":
    main()
