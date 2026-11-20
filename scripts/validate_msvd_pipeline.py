#!/usr/bin/env python3
"""Freeze-check the MSVD test670 FAITHFUL_V2 query pipeline."""
import json
import os
import sys

import numpy as np
import torch
import decord
import cv2

cv2.setNumThreads(0)
ROOT = "/home/bn/videoufo_research"
MSVD = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSVD"
XCLIP = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/EvcapProject/X-CLIP"
CKPT = os.path.join(XCLIP, "modelpth/k600_16_16.pth")
T = 16
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
out = {}

paths = [l.strip().split("\t")[0] for l in open(os.path.join(MSVD, "test.caption.tsv")) if l.strip()]
ids = [os.path.basename(p)[:-4] for p in paths]
qids = open(os.path.join(ROOT, "embeddings/msvd_test_video_xclip_faithful_v2_ids.txt")).read().splitlines()
sp = os.path.join(ROOT, "external/external_msvd_sims.json")
S = json.load(open(sp)) if os.path.exists(sp) else None
out["queries"] = len(ids)
out["unique_ids"] = len(set(ids))
out["feature_rows"] = len(qids)
out["retrieval_keys"] = len(S) if S else None
out["id_sets_equal"] = (set(ids) == set(qids) == ({k.split("/")[-1][:-4] for k in S} if S else set(qids)))
out["all_videos_exist"] = all(os.path.exists(os.path.join(MSVD, "videos", i + ".avi")) for i in ids)
V = np.load(os.path.join(ROOT, "embeddings/msvd_test_video_xclip_faithful_v2_video_fp16.npy"), mmap_mode="r")
I = np.load(os.path.join(ROOT, "embeddings/msvd_test_video_xclip_faithful_v2_imgfeat_fp16.npy"), mmap_mode="r")
out["video_shape"] = list(V.shape)
out["imgfeat_shape"] = list(I.shape)
out["finite"] = bool(np.isfinite(np.asarray(V[:5], np.float32)).all())


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
sample = [ids[0], ids[len(ids) // 2], ids[-1]]
maxdiff = 0.0
for q in sample:
    x = load_clip(os.path.join(MSVD, "videos", q + ".avi")).unsqueeze(0).cuda().half()
    with torch.no_grad(), torch.cuda.amp.autocast():
        v, _ = model.encode_video(x)
    maxdiff = max(maxdiff, float(np.abs(v[0].float().cpu().numpy().astype(np.float16) - np.asarray(V[pos[q]])).max()))
out["recompute_sample"] = sample
out["recompute_max_abs_diff_fp16"] = maxdiff
out["recompute_ok"] = maxdiff < 2e-2
json.dump(out, open(os.path.join(ROOT, "reports/msvd_pipeline_check.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
