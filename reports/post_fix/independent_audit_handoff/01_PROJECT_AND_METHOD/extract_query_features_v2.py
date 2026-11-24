#!/usr/bin/env python3
"""FAITHFUL_V2 MSR-VTT query features: 16 center frames, INTER_NEAREST.

Usage: extract_query_features_v2.py <label_tsv> <out_prefix>
Writes <out_prefix>_video_fp16.npy, <out_prefix>_imgfeat_fp16.npy, <out_prefix>_ids.txt
"""
import json
import os
import sys

import torch
import numpy as np
import decord
import cv2

cv2.setNumThreads(0)
XCLIP = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/EvcapProject/X-CLIP"
CKPT = os.path.join(XCLIP, "modelpth/k600_16_16.pth")
OUT = "/home/bn/videoufo_research/embeddings"
T = 16
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


class _Log:
    def info(self, m):
        pass


def load_clip(path):
    vr = decord.VideoReader(path, num_threads=2)
    n = len(vr)
    idx = np.clip(((np.arange(T) + 0.5) * n / T).astype(int), 0, n - 1)
    frames = vr.get_batch(idx).asnumpy()
    out = np.empty((T, 224, 224, 3), dtype=np.float32)
    for i, fr in enumerate(frames):
        h, w = fr.shape[:2]
        if h <= w:
            nh, nw = 256, max(256, int(round(w * 256 / h)))
        else:
            nw, nh = 256, max(256, int(round(h * 256 / w)))
        fr = cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_NEAREST)
        y0, x0 = (nh - 224) // 2, (nw - 224) // 2
        fr = fr[y0:y0 + 224, x0:x0 + 224]
        out[i] = (fr.astype(np.float32) - MEAN) / STD
    return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()


def main():
    label, prefix = sys.argv[1], sys.argv[2]
    vids = []
    for line in open(label):
        p = line.split("\t", 1)[0]
        vids.append((os.path.basename(p)[:-4], "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/" + p))
    n = len(vids)
    sys.path.insert(0, XCLIP)
    os.chdir(XCLIP)
    import clip  # noqa
    from models import xclip_all as xclip
    sd = torch.load(CKPT, map_location="cpu")["model"]
    model = xclip.build_model(sd, T=T, droppath=0., use_checkpoint=False, logger=_Log(),
                              prompts_alpha=0.1, prompts_layers=2, use_cache=True,
                              mit_layers=1, text_batch_size=0).cuda().eval()
    vf = np.zeros((n, 512), dtype=np.float16)
    imf = np.zeros((n, 196, 512), dtype=np.float16)
    for i, (vid, path) in enumerate(vids):
        try:
            x = load_clip(path).unsqueeze(0).cuda().half()
            with torch.no_grad(), torch.cuda.amp.autocast():
                v, im = model.encode_video(x)
                im = im.mean(dim=1)
            vf[i] = v[0].float().cpu().numpy().astype(np.float16)
            imf[i] = im[0].float().cpu().numpy().astype(np.float16)
        except Exception as e:
            print("FAIL", vid, repr(e)[:80], flush=True)
        if (i + 1) % 200 == 0:
            print(i + 1, "/", n, flush=True)
    np.save(os.path.join(OUT, prefix + "_video_fp16.npy"), vf)
    np.save(os.path.join(OUT, prefix + "_imgfeat_fp16.npy"), imf)
    open(os.path.join(OUT, prefix + "_ids.txt"), "w").write("\n".join(v for v, _ in vids) + "\n")
    print("DONE", n, flush=True)


if __name__ == "__main__":
    main()
