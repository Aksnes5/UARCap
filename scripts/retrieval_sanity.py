#!/usr/bin/env python3
"""First-stage retrieval sanity: MSR-VTT queries -> VideoUFO caption corpus.

Scores use the base X-CLIP text embeddings.  The original forward adds a small
query-conditioned VideoSpecificPrompt residual; that exact rescoring is a
separate step (see report).  Two corpus variants are produced:
  raw   : every clip caption (duplicates preserved) - measures duplication
  dedup : unique captions (paper's no_duplicate corpus style)
"""
import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/home/bn/videoufo_research"
EMB = os.path.join(ROOT, "embeddings/videoufo_brief_text_fp16.npy")
IDS = os.path.join(ROOT, "embeddings/videoufo_brief_ids.txt")
QEMB = os.path.join(ROOT, "embeddings/msrvtt_test_video_fp16.npy")
QIDS = os.path.join(ROOT, "embeddings/msrvtt_test_video_ids.txt")
CSV_PATH = "/home/bn/datasets/VideoUFO/metadata/VideoUFO.csv"
MAN = os.path.join(ROOT, "manifests")
OUT = os.path.join(ROOT, "retrieval")
LEVELS = ["10k", "50k", "100k", "250k", "500k", "full"]


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ids = open(IDS).read().splitlines()
    id2row = {c: i for i, c in enumerate(ids)}
    brief, topic = load_meta()
    assert len(brief) == len(ids)
    qids = open(QIDS).read().splitlines()
    Q = torch.from_numpy(np.load(QEMB)).cuda().float()
    Q = F.normalize(Q, dim=1)
    qi = np.linspace(0, len(qids) - 1, a.queries, dtype=int)
    Q = Q[qi]
    E = np.load(EMB, mmap_mode="r")
    print("queries", len(qi), "corpus", len(ids), flush=True)

    summary = {}
    for level in LEVELS:
        t0 = time.time()
        level_ids = [l.split("\t")[0] for l in open(os.path.join(MAN, f"clips_{level}.tsv"))
                     if not l.startswith("clip_id")]
        rows = np.array([id2row[c] for c in level_ids], dtype=np.int64)
        X = torch.from_numpy(np.asarray(E[rows])).cuda().float()
        X = F.normalize(X, dim=1)
        # raw
        S = Q @ X.T
        vals, idx = torch.topk(S, 20, dim=1)
        vals, idx = vals.cpu().numpy(), idx.cpu().numpy()
        raw = stats_for(level, level_ids, rows, idx, vals, brief, topic)
        # dedup: first occurrence per normalized caption
        seen = {}
        for j, r in enumerate(rows):
            c = brief[r].lower()
            if c not in seen:
                seen[c] = j
        ucap = list(seen)
        urows = np.array([rows[seen[c]] for c in ucap], dtype=np.int64)
        Xu = torch.from_numpy(np.asarray(E[urows])).cuda().float()
        Xu = F.normalize(Xu, dim=1)
        Su = Q @ Xu.T
        vu, iu = torch.topk(Su, 20, dim=1)
        vu, iu = vu.cpu().numpy(), iu.cpu().numpy()
        rep_level_idx = [seen[c] for c in ucap]
        ded = stats_for(level, level_ids, rows, iu, vu, brief, topic,
                        unique_caps=ucap, rep_level_idx=rep_level_idx)
        ded["unique_captions"] = len(ucap)
        summary[level] = {"raw": raw, "dedup": ded, "sec": round(time.time() - t0, 1)}
        print(level, "raw", raw["top1_sim"], "dedup", ded["top1_sim"], "sec", summary[level]["sec"], flush=True)
        # save per-query top-20 (raw) for qualitative work
        with open(os.path.join(OUT, f"top20_{level}_raw.json"), "w") as f:
            json.dump([{
                "query": qids[qi[k]],
                "top20": [{
                    "clip_id": level_ids[j], "source_id": level_ids[j].rsplit(".", 1)[0],
                    "score": float(vals[k][t]), "brief": brief[rows[j]], "topic": topic[rows[j]],
                } for t, j in enumerate(idx[k])],
            } for k in range(len(qi))], f, indent=1)
    with open(os.path.join(OUT, "retrieval_sanity_summary.json"), "w") as f:
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
