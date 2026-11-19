#!/usr/bin/env python3
"""X-CLIP 512-d video features for a list of VideoUFO clips (resumable)."""
import json
import os
import sys
import time

import torch
import numpy as np
import decord
import cv2

cv2.setNumThreads(0)

XCLIP = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/EvcapProject/X-CLIP"
CKPT = os.path.join(XCLIP, "modelpth/k600_16_16.pth")
VIDEO_DIR = "/home/bn/datasets/VideoUFO/videos/VideoUFO"
OUT = "/home/bn/videoufo_research/features/xclip_video"
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
    clips = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    os.makedirs(OUT, exist_ok=True)
    done = {f[:-3] for f in os.listdir(OUT) if f.endswith(".npy")}
    todo = [c for c in clips if c not in done]
    print("clips", len(clips), "todo", len(todo), flush=True)
    sys.path.insert(0, XCLIP)
    os.chdir(XCLIP)
    import clip  # noqa: F401
    from models import xclip_all as xclip
    sd = torch.load(CKPT, map_location="cpu")["model"]
    model = xclip.build_model(sd, T=T, droppath=0.0, use_checkpoint=False, logger=_Log(),
                              prompts_alpha=0.1, prompts_layers=2, use_cache=True,
                              mit_layers=1, text_batch_size=0).cuda().eval()
    print("model loaded", flush=True)
    t0 = time.time()
    for s, c in enumerate(todo):
        try:
            x = load_clip(os.path.join(VIDEO_DIR, c + ".mp4")).unsqueeze(0).cuda().half()
            with torch.no_grad(), torch.cuda.amp.autocast():
                v, _ = model.encode_video(x)
            np.save(os.path.join(OUT, c + ".npy"), v[0].float().cpu().numpy().astype(np.float16))
        except Exception as e:
            print("FAIL", c, repr(e)[:100], flush=True)
        if (s + 1) % 100 == 0:
            print(f"{s+1}/{len(todo)} elapsed {time.time()-t0:.1f}s", flush=True)
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump({"clips": len(clips), "features": len([c for c in clips if os.path.exists(os.path.join(OUT, c + '.npy'))]),
                   "dim": 512, "dtype": "float16", "frames": T,
                   "mean": MEAN.tolist(), "std": STD.tolist()}, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
