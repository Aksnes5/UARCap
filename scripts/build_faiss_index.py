#!/usr/bin/env python3
"""Build exact and IVF FAISS indexes over X-CLIP VideoUFO caption embeddings.

The paper's retriever is query-conditioned (VideoSpecificPrompt), so FAISS is
used for candidate generation only; exact final scoring is done separately.
"""
import json
import os
import time

import faiss
import numpy as np

EMB = "/home/bn/videoufo_research/embeddings/videoufo_brief_text_fp16.npy"
OUT = "/home/bn/videoufo_research/indexes"
CHUNK = 100000


def main():
    os.makedirs(OUT, exist_ok=True)
    E = np.load(EMB, mmap_mode="r")
    n, d = E.shape
    print("embeddings", n, d, flush=True)
    t0 = time.time()
    flat = faiss.IndexFlatIP(d)
    for s in range(0, n, CHUNK):
        x = np.asarray(E[s:s + CHUNK], dtype=np.float32)
        faiss.normalize_L2(x)
        flat.add(x)
        print("flat add", min(s + CHUNK, n), flush=True)
    flat_time = time.time() - t0
    fp = os.path.join(OUT, "videoufo_brief_full_flat.faiss")
    faiss.write_index(flat, fp)
    print("flat_size_MB", round(os.path.getsize(fp) / 2**20), "build_sec", round(flat_time, 1), flush=True)

    t0 = time.time()
    quantizer = faiss.IndexFlatIP(d)
    ivf = faiss.IndexIVFFlat(quantizer, d, 4096, faiss.METRIC_INNER_PRODUCT)
    for s in range(0, n, CHUNK):
        x = np.asarray(E[s:s + CHUNK], dtype=np.float32)
        faiss.normalize_L2(x)
        if not ivf.is_trained:
            ivf.train(x)
        ivf.add(x)
        print("ivf add", min(s + CHUNK, n), flush=True)
    ivf_time = time.time() - t0
    ip = os.path.join(OUT, "videoufo_brief_full_ivf4096.faiss")
    faiss.write_index(ivf, ip)
    print("ivf_size_MB", round(os.path.getsize(ip) / 2**20), "build_sec", round(ivf_time, 1), flush=True)

    # latency + recall on 200 random synthetic queries drawn from the corpus
    rng = np.random.default_rng(42)
    qidx = rng.choice(n, 200, replace=False)
    xq = np.asarray(E[qidx], dtype=np.float32)
    faiss.normalize_L2(xq)
    t = time.time(); Df, If = flat.search(xq, 20); flat_ms = (time.time() - t) * 1000 / len(xq)
    res = {"n": n, "dim": d, "flat_size_MB": round(os.path.getsize(fp) / 2**20),
           "flat_build_sec": round(flat_time, 1), "flat_query_ms": round(flat_ms, 3),
           "ivf_size_MB": round(os.path.getsize(ip) / 2**20), "ivf_build_sec": round(ivf_time, 1),
           "ivf": {}}
    for nprobe in (1, 8, 16, 32):
        ivf.nprobe = nprobe
        t = time.time(); Di, Ii = ivf.search(xq, 20); ms = (time.time() - t) * 1000 / len(xq)
        recall = np.mean([len(set(Ii[i]) & set(If[i])) / 20 for i in range(len(xq))])
        res["ivf"][nprobe] = {"query_ms": round(ms, 3), "recall20": round(float(recall), 4)}
        print("nprobe", nprobe, res["ivf"][nprobe], flush=True)
    with open(os.path.join(OUT, "faiss_manifest.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
