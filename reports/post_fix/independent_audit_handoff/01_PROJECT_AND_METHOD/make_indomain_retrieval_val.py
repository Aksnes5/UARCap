#!/usr/bin/env python3
"""In-domain (MSR-VTT train) retrieval for val497 with the same X-CLIP machinery.

Corpus = unique MSR-VTT train captions (train.caption.tsv, no val/test).
Outputs external_val_indomain_{sims,top20,reranked}.json for D1/3/5/7.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/home/bn/videoufo_research"
DATA = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2"
XCLIP = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/EvcapProject/X-CLIP"
CKPT = os.path.join(XCLIP, "modelpth/k600_16_16.pth")
QEMB = os.path.join(ROOT, "embeddings/msrvtt_val_video_xclip_faithful_v2_video_fp16.npy")
QIMG = os.path.join(ROOT, "embeddings/msrvtt_val_video_xclip_faithful_v2_imgfeat_fp16.npy")
QIDS = os.path.join(ROOT, "embeddings/msrvtt_val_video_xclip_faithful_v2_ids.txt")
OUT = os.path.join(ROOT, "external")
TOPK, TOPN = 20, 7


class _Log:
    def info(self, m):
        pass


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
    caps, groups, order = [], {}, []
    for line in open(os.path.join(DATA, "train.caption.tsv")):
        p = line.split("\t", 1)
        vid = p[0]
        for x in json.loads(p[1]):
            c = x["caption"].strip()
            if not c:
                continue
            if c not in groups:
                groups[c] = []
                order.append(c)
            if vid not in groups[c]:
                groups[c].append(vid)
    for c in order:
        groups[c].sort()
    caps = order
    print("train unique captions", len(caps), flush=True)

    sys.path.insert(0, XCLIP)
    os.chdir(XCLIP)
    import clip  # noqa
    from models import xclip_all as xclip
    sd = torch.load(CKPT, map_location="cpu")["model"]
    model = xclip.build_model(sd, T=16, droppath=0., use_checkpoint=False, logger=_Log(),
                              prompts_alpha=0.1, prompts_layers=2, use_cache=True,
                              mit_layers=1, text_batch_size=0).cuda().eval()
    print("model loaded", flush=True)
    E = np.zeros((len(caps), 512), dtype=np.float16)
    for s in range(0, len(caps), 256):
        toks = clip.tokenize(caps[s:s + 256], context_length=77, cut=True).cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            e = model.encode_text(toks)
        E[s:s + 256] = e.float().cpu().numpy().astype(np.float16)
        if (s + 256) % 20000 == 0:
            print("enc", min(s + 256, len(caps)), flush=True)
    X = torch.from_numpy(E).cuda().half()
    qids = open(QIDS).read().splitlines()
    Q = F.normalize(torch.from_numpy(np.load(QEMB)).cuda().float(), dim=1).half()
    QI = np.load(QIMG, mmap_mode="r")

    sims, diag = {}, {}
    for k, q in enumerate(qids):
        img = torch.from_numpy(np.asarray(QI[k], dtype=np.float32)).cuda().half().unsqueeze(0)
        qv = Q[k]
        with torch.no_grad(), torch.cuda.amp.autocast():
            tf = X + model.prompts_generator(X.unsqueeze(0), img.float())
            sc = qv @ F.normalize(tf[0], dim=1).T
        v, i = torch.topk(sc, TOPK)
        cs = [caps[j] for j in i.cpu().tolist()]
        ss = [float(x) for x in v.cpu().tolist()]
        key = f"datasets/MSRVTT-v2/videos/{q}.mp4"
        sims[key] = [cs, ss]
        diag[key] = [{"caption": c, "score": s, "clip_ids": groups[c]} for c, s in zip(cs, ss)]
        if (k + 1) % 100 == 0:
            print(k + 1, "/", len(qids), flush=True)

    reranked = {}
    for key, (cs, ss) in sims.items():
        qual = [sum(rouge_l(cs[i], cs[j]) for j in range(len(cs)) if j != i) / (len(cs) - 1)
                for i in range(len(cs))]
        idx = sorted(range(len(cs)), key=lambda i: (-qual[i], i))[:TOPN]
        reranked[key] = [[cs[i] for i in idx], [qual[i] for i in idx], [groups[cs[i]] for i in idx]]
    json.dump(sims, open(os.path.join(OUT, "external_val_indomain_sims.json"), "w"), indent=1)
    json.dump(diag, open(os.path.join(OUT, "external_val_indomain_top20.json"), "w"), indent=1)
    json.dump(reranked, open(os.path.join(OUT, "external_val_indomain_reranked.json"), "w"), indent=1)
    print("DONE", len(sims), flush=True)


if __name__ == "__main__":
    main()
