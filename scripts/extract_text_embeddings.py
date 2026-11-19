#!/usr/bin/env python3
"""Extract X-CLIP text embeddings for every VideoUFO clip (Brief Caption).

This is the faithful retrieval-side feature for the paper's method:
X-CLIP retrieves captions (video->text), not corpus videos.
Writes chunked FP16 features + a merged .npy + manifest, resumable by chunk.
"""
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

XCLIP = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/EvcapProject/X-CLIP"
CKPT = os.path.join(XCLIP, "modelpth/k600_16_16.pth")
CSV_PATH = "/home/bn/datasets/VideoUFO/metadata/VideoUFO.csv"
OUT = "/home/bn/videoufo_research/embeddings"
CHUNK = 50000
BATCH = 2048

sys.path.insert(0, XCLIP)
os.chdir(XCLIP)
import clip  # noqa: E402
from models import xclip_all as xclip  # noqa: E402


class _Log:
    def info(self, m):
        pass


def build_model():
    sd = torch.load(CKPT, map_location="cpu")["model"]
    model = xclip.build_model(
        sd, T=16, droppath=0.0, use_checkpoint=False, logger=_Log(),
        prompts_alpha=0.1, prompts_layers=2, use_cache=True,
        mit_layers=1, text_batch_size=0,
    )
    msg = model.load_state_dict(sd, strict=False)
    assert not msg.missing_keys and not msg.unexpected_keys, msg
    return model.cuda().eval()


def load_captions():
    csv.field_size_limit(10 ** 9)
    ids, caps = [], []
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        h = next(r)
        i_id, i_cap = h.index("ID"), h.index("Brief Caption")
        for row in r:
            if not row:
                continue
            ids.append(row[i_id])
            caps.append(row[i_cap])
    return ids, caps


def main():
    os.makedirs(os.path.join(OUT, "text_chunks"), exist_ok=True)
    t0 = time.time()
    ids, caps = load_captions()
    n = len(ids)
    print("captions", n, "load_sec", round(time.time() - t0, 1), flush=True)
    with open(os.path.join(OUT, "videoufo_brief_ids.txt"), "w") as f:
        f.write("\n".join(ids) + "\n")

    model = build_model()
    print("model loaded", round(time.time() - t0, 1), flush=True)
    done = 0
    for ci, start in enumerate(range(0, n, CHUNK)):
        path = os.path.join(OUT, "text_chunks", f"chunk_{ci:04d}.npy")
        end = min(start + CHUNK, n)
        if os.path.exists(path):
            arr = np.load(path, mmap_mode="r")
            if arr.shape == (end - start, 512):
                done += end - start
                continue
        out = np.empty((end - start, 512), dtype=np.float16)
        for s in range(start, end, BATCH):
            e = min(s + BATCH, end)
            toks = clip.tokenize(caps[s:e], context_length=77, cut=True).cuda()
            with torch.no_grad(), torch.cuda.amp.autocast():
                emb = model.encode_text(toks)
            out[s - start:e - start] = emb.float().cpu().numpy().astype(np.float16)
        np.save(path, out)
        done += end - start
        print(f"chunk {ci} {end}/{n} elapsed {time.time()-t0:.1f}s", flush=True)

    merged = os.path.join(OUT, "videoufo_brief_text_fp16.npy")
    mm = np.lib.format.open_memmap(merged, mode="w+", dtype=np.float16, shape=(n, 512))
    for ci, start in enumerate(range(0, n, CHUNK)):
        end = min(start + CHUNK, n)
        mm[start:end] = np.load(os.path.join(OUT, "text_chunks", f"chunk_{ci:04d}.npy"))
    mm.flush()
    del mm

    def sha(p, limit=None):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()

    manifest = {
        "corpus": "VideoUFO",
        "caption_field": "Brief Caption",
        "num_embeddings": n,
        "dim": 512,
        "dtype": "float16",
        "row_order": "same as metadata/VideoUFO.csv data rows; also videoufo_brief_ids.txt",
        "xclip_repo": XCLIP,
        "xclip_checkpoint": CKPT,
        "xclip_checkpoint_sha256": sha(CKPT),
        "arch": "ViT-B/16",
        "num_frames": 16,
        "context_length": 77,
        "tokenize_cut": True,
        "feature": "model.encode_text (before VideoSpecificPrompt residual)",
        "merged_file": merged,
        "merged_sha256": sha(merged),
        "chunk_size": CHUNK,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT, "videoufo_brief_text_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("DONE", round(time.time() - t0, 1), "sha", manifest["merged_sha256"][:16], flush=True)


if __name__ == "__main__":
    main()
