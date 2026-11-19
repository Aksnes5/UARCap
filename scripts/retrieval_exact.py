#!/usr/bin/env python3
"""Exact X-CLIP retrieval sanity (VideoSpecificPrompt applied).

Matches XCLIP.forward: video feature L2-normalised, text features +
prompts_generator(text, mean frame tokens), text L2-normalised, inner product
(logit_scale is a positive constant and does not change ranking).
Chunked over the corpus so the full 1.09M corpus fits comfortably.
"""
import csv
import json
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/home/bn/videoufo_research"
XCLIP = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/EvcapProject/X-CLIP"
CKPT = os.path.join(XCLIP, "modelpth/k600_16_16.pth")
EMB = os.path.join(ROOT, "embeddings/videoufo_brief_text_fp16.npy")
IDS = os.path.join(ROOT, "embeddings/videoufo_brief_ids.txt")
QEMB = os.path.join(ROOT, "embeddings/msrvtt_test_video_fp16.npy")
QIMG = os.path.join(ROOT, "embeddings/msrvtt_test_imgfeat_fp16.npy")
QIDS = os.path.join(ROOT, "embeddings/msrvtt_test_video_ids.txt")
CSV_PATH = "/home/bn/datasets/VideoUFO/metadata/VideoUFO.csv"
MAN = os.path.join(ROOT, "manifests")
OUT = os.path.join(ROOT, "retrieval")
LEVELS = ["10k", "50k", "100k", "250k", "500k", "full"]
CHUNK = 100000
TOPK = 20


class _Log:
    def info(self, m):
        pass


def load_meta():
    csv.field_size_limit(10 ** 9)
    brief, topic = [], []
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        h = next(r)
        i_b, i_t = h.index("Brief Caption"), h.index("Topic")
        for row in r:
            if row:
                brief.append(row[i_b].strip())
                topic.append(row[i_t].strip().lower() or "unknown")
    return brief, topic


def main():
    ids = open(IDS).read().splitlines()
    id2row = {c: i for i, c in enumerate(ids)}
    brief, topic = load_meta()
    qids = open(QIDS).read().splitlines()
    Q = torch.from_numpy(np.load(QEMB)).cuda().half()
    Q = F.normalize(Q.float(), dim=1).half()
    QI = np.load(QIMG, mmap_mode="r")
    qi = np.linspace(0, len(qids) - 1, 300, dtype=int)
    Q = Q[qi]
    E = np.load(EMB, mmap_mode="r")

    import sys
    sys.path.insert(0, XCLIP)
    os.chdir(XCLIP)
    import clip  # noqa: F401
    from models import xclip_all as xclip
    sd = torch.load(CKPT, map_location="cpu")["model"]
    model = xclip.build_model(sd, T=16, droppath=0.0, use_checkpoint=False, logger=_Log(),
                              prompts_alpha=0.1, prompts_layers=2, use_cache=True,
                              mit_layers=1, text_batch_size=0).cuda().eval()
    print("model loaded", flush=True)

    summary = {}
    for level in LEVELS:
        t0 = time.time()
        lev = [l.split("\t")[0] for l in open(os.path.join(MAN, f"clips_{level}.tsv"))
               if not l.startswith("clip_id")]
        rows = np.array([id2row[c] for c in lev], dtype=np.int64)
        variants = {}
        variants["raw"] = (rows, [None, None])
        seen = {}
        for j, r in enumerate(rows):
            c = brief[r].lower()
            if c not in seen:
                seen[c] = j
        ucap = list(seen)
        urows = np.array([rows[seen[c]] for c in ucap], dtype=np.int64)
        rep_idx = [seen[c] for c in ucap]
        variants["dedup"] = (urows, [ucap, rep_idx])

        for vname, (vrows, extra) in variants.items():
            X = torch.from_numpy(np.asarray(E[vrows])).cuda().half()
            I_all = []
            V_all = []
            for k in range(len(Q)):
                q = Q[k]
                img = torch.from_numpy(np.asarray(QI[qi[k]], dtype=np.float32)).cuda().half().unsqueeze(0)
                best_v = torch.full((TOPK,), -1e4, device="cuda")
                best_i = torch.zeros(TOPK, dtype=torch.long, device="cuda")
                for s in range(0, len(vrows), CHUNK):
                    tf = X[s:s + CHUNK].unsqueeze(0)
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        tf = tf + model.prompts_generator(tf, img)
                        sc = q @ F.normalize(tf[0], dim=1).T
                    v, i = torch.topk(sc, min(TOPK, sc.numel()))
                    i = i + s
                    all_v = torch.cat([best_v, v]); all_i = torch.cat([best_i, i])
                    best_v, sel = torch.topk(all_v, TOPK)
                    best_i = all_i[sel]
                V_all.append(best_v.float().cpu().numpy())
                I_all.append(best_i.cpu().numpy())
            V_all = np.array(V_all); I_all = np.array(I_all)
            st = stats_for(level, lev, rows, I_all, V_all, brief, topic,
                           unique_caps=extra[0], rep_level_idx=extra[1])
            if extra[0] is not None:
                st["unique_captions"] = len(extra[0])
            variants[vname] = st
            with open(os.path.join(OUT, f"top20_{level}_exact_{vname}.json"), "w") as f:
                json.dump([{"query": qids[qi[k]],
                            "top20": [{"clip_id": lev[j], "source_id": lev[j].rsplit(".", 1)[0],
                                       "score": float(V_all[k][t]),
                                       **({"caption": extra[0][j]} if extra[0] else {"brief": brief[rows[j]], "topic": topic[rows[j]]})}
                                      for t, j in enumerate(I_all[k])]}
                           for k in range(len(Q))], f, indent=1)
            print(level, vname, "top1", st["top1_sim"], "sec", round(time.time() - t0, 1), flush=True)
        summary[level] = {k: variants[k] for k in ("raw", "dedup")}
        summary[level]["sec"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "retrieval_exact_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("DONE", flush=True)


def stats_for(level, level_ids, rows, idx, vals, brief, topic, unique_caps=None,
              rep_level_idx=None):
    nq = idx.shape[0]
    top1, top5, top20 = [], [], []
    usrc, maxsrc, repq = [], [], 0
    ucap, captopic = [], []
    for k in range(nq):
        js = idx[k]
        top1.append(float(vals[k][0]))
        top5.append(float(vals[k][:5].mean()))
        top20.append(float(vals[k].mean()))
        rep = [rep_level_idx[j] for j in js] if rep_level_idx is not None else list(js)
        srcs = [level_ids[j].rsplit(".", 1)[0] for j in rep]
        c = Counter(srcs)
        usrc.append(len(c))
        maxsrc.append(max(c.values()))
        repq += int(max(c.values()) > 1)
        if unique_caps is not None:
            caps = [unique_caps[j] for j in js]
        else:
            caps = [brief[rows[j]].lower() for j in js]
        ucap.append(len(set(caps)))
        captopic.append(len({topic[rows[j]] for j in rep}))
    return {
        "n_queries": nq, "corpus_clips": len(level_ids),
        "top1_sim": round(float(np.mean(top1)), 4),
        "top5_mean_sim": round(float(np.mean(top5)), 4),
        "top20_mean_sim": round(float(np.mean(top20)), 4),
        "mean_unique_sources_top20": round(float(np.mean(usrc)), 2),
        "mean_max_same_source": round(float(np.mean(maxsrc)), 2),
        "queries_with_repeated_source": int(repq),
        "mean_unique_captions_top20": round(float(np.mean(ucap)), 2),
        "caption_repetition_rate": round(float(1 - np.mean(ucap) / 20), 4),
        "mean_unique_topics_top20": round(float(np.mean(captopic)), 2),
    }


if __name__ == "__main__":
    main()
