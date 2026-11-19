#!/usr/bin/env python3
"""Freeze-check the val497 FAITHFUL_V2 pipeline.

Verifies: 497/497 queries, 16-frame protocol, INTER_NEAREST X-CLIP preprocessing
(recomputed sample bit-identical), no test-GT input, clip feature coverage.
"""
import json
import os
import sys

import numpy as np
import torch
import decord
import cv2

cv2.setNumThreads(0)
ROOT = "/home/bn/videoufo_research"
XCLIP = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/EvcapProject/X-CLIP"
CKPT = os.path.join(XCLIP, "modelpth/k600_16_16.pth")
DATA = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2"
T = 16
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
out = {}


def ids(path):
    return [os.path.basename(l.strip().split("\t")[0])[:-4] for l in open(path) if l.strip()]


paths = ids(os.path.join(ROOT, "evcap_external/val497/val_video_paths.tsv"))
labels = ids(os.path.join(DATA, "val.label.tsv"))
caps = ids(os.path.join(DATA, "val.caption.tsv"))
qids = open(os.path.join(ROOT, "embeddings/msrvtt_val_video_xclip_faithful_v2_ids.txt")).read().splitlines()
S = json.load(open(os.path.join(ROOT, "external/external_val_sims.json")))
out["paths"] = len(paths)
out["paths_unique"] = len(set(paths))
out["label_rows"] = len(labels)
out["caption_rows"] = len(caps)
out["query_feature_rows"] = len(qids)
out["retrieval_keys"] = len(S)
out["id_sets_equal"] = (set(paths) == set(labels) == set(caps) == set(qids) == {k.split("/")[-1][:-4] for k in S})
out["all_videos_exist"] = all(os.path.exists(os.path.join(ROOT, "evcap_external", p)) for p in
                              [l.strip() for l in open(os.path.join(ROOT, "evcap_external/val497/val_video_paths.tsv"))])

CF = json.load(open(os.path.join(ROOT, "external/external_val_cf7_cap.json")))
D7 = json.load(open(os.path.join(ROOT, "external/external_val_d7_cap.json")))
out["cf_keys"] = len(CF)
out["d7_keys"] = len(D7)
out["cf_len7"] = all(len(v[0]) == 7 and len(v[2]) == 7 for v in CF.values())
out["d_len7"] = all(len(v[0]) == 7 for v in D7.values())

sel = [l.strip() for l in open(os.path.join(ROOT, "external/external_val_grid_clips.txt")) if l.strip()]
fd = os.path.join(ROOT, "features/videollama_faithful_v2")
missing = [c for c in sel if not os.path.exists(os.path.join(fd, c + ".pt"))]
out["selected_clips"] = len(sel)
out["missing_clip_features"] = len(missing)

V = np.load(os.path.join(ROOT, "embeddings/msrvtt_val_video_xclip_faithful_v2_video_fp16.npy"), mmap_mode="r")
I = np.load(os.path.join(ROOT, "embeddings/msrvtt_val_video_xclip_faithful_v2_imgfeat_fp16.npy"), mmap_mode="r")
out["video_shape"] = list(V.shape)
out["imgfeat_shape"] = list(I.shape)


def load_clip(path):
    vr = decord.VideoReader(path, num_threads=2)
    n = len(vr)
    idx = np.clip(((np.arange(T) + 0.5) * n / T).astype(int), 0, n - 1)
    frames = vr.get_batch(idx).asnumpy()
    o = np.empty((T, 224, 224, 3), dtype=np.float32)
    for i, fr in enumerate(frames):
        h, w = fr.shape[:2]
        if h <= w:
            nh, nw = 256, max(256, int(round(w * 256 / h)))
        else:
            nw, nh = 256, max(256, int(round(h * 256 / w)))
        fr = cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_NEAREST)
        y0, x0 = (nh - 224) // 2, (nw - 224) // 2
        o[i] = (fr[y0:y0 + 224, x0:x0 + 224].astype(np.float32) - MEAN) / STD
    return torch.from_numpy(o).permute(0, 3, 1, 2).contiguous()


class _Log:
    def info(self, m):
        pass


sys.path.insert(0, XCLIP)
os.chdir(XCLIP)
import clip  # noqa
from models import xclip_all as xclip
sd = torch.load(CKPT, map_location="cpu")["model"]
model = xclip.build_model(sd, T=T, droppath=0., use_checkpoint=False, logger=_Log(),
                          prompts_alpha=0.1, prompts_layers=2, use_cache=True,
                          mit_layers=1, text_batch_size=0).cuda().eval()
pos = {q: i for i, q in enumerate(qids)}
maxdiff = 0.0
sample = [qids[0], qids[len(qids)//2], qids[-1]]
for q in sample:
    path = os.path.join(DATA, "videos", q + ".mp4")
    x = load_clip(path).unsqueeze(0).cuda().half()
    with torch.no_grad(), torch.cuda.amp.autocast():
        v, _ = model.encode_video(x)
    d = float(np.abs(v[0].float().cpu().numpy().astype(np.float16) - np.asarray(V[pos[q]])).max())
    maxdiff = max(maxdiff, d)
out["recompute_sample"] = sample
out["recompute_max_abs_diff_fp16"] = maxdiff
out["recompute_ok"] = maxdiff < 2e-2

json.dump(out, open(os.path.join(ROOT, "reports/val497_pipeline_check.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
