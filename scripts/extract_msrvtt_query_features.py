#!/usr/bin/env python3
"""Extract X-CLIP query-side features for MSR-VTT test videos.

Faithful to EvcapProject/X-CLIP val pipeline:
  DecordInit -> uniform 16 frames -> Resize(short side 256) -> CenterCrop(224)
  -> Normalize(ImageNet mean/std, RGB) -> NCHW
Outputs per video: 512-d video feature and mean frame-token features
(196 x 512) used by X-CLIP's VideoSpecificPrompt at scoring time.
"""
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
LABEL = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2/test.label.tsv"
OUT = "/home/bn/videoufo_research/embeddings"
CHUNK = 100
T = 16
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

sys.path.insert(0, XCLIP)
os.chdir(XCLIP)
import clip  # noqa: E402
from models import xclip_all as xclip  # noqa: E402


class _Log:
    def info(self, m):
        pass


def load_video(path):
    vr = decord.VideoReader(path, num_threads=2)
    n = len(vr)
    if n <= 0:
        raise RuntimeError("empty video")
    idx = np.clip(((np.arange(T) + 0.5) * n / T).astype(int), 0, n - 1)
    frames = vr.get_batch(idx).asnumpy()  # T,H,W,3 RGB uint8
    out = np.empty((T, 224, 224, 3), dtype=np.float32)
    for i, fr in enumerate(frames):
        h, w = fr.shape[:2]
        if h <= w:
            nh, nw = 256, max(256, int(round(w * 256 / h)))
        else:
            nw, nh = 256, max(256, int(round(h * 256 / w)))
        fr = cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        y0, x0 = (nh - 224) // 2, (nw - 224) // 2
        fr = fr[y0:y0 + 224, x0:x0 + 224]
        out[i] = (fr.astype(np.float32) - MEAN) / STD
    return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()


def main():
    os.makedirs(os.path.join(OUT, "query_chunks"), exist_ok=True)
    videos = []
    with open(LABEL) as f:
        for line in f:
            p = line.split("\t", 1)[0]
            vid = os.path.basename(p)[:-4]
            videos.append((vid, "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/" + p))
    n = len(videos)
    with open(os.path.join(OUT, "msrvtt_test_video_ids.txt"), "w") as f:
        f.write("\n".join(v for v, _ in videos) + "\n")

    sd = torch.load(CKPT, map_location="cpu")["model"]
    model = xclip.build_model(
        sd, T=T, droppath=0.0, use_checkpoint=False, logger=_Log(),
        prompts_alpha=0.1, prompts_layers=2, use_cache=True,
        mit_layers=1, text_batch_size=0,
    )
    model.load_state_dict(sd, strict=True)
    model = model.cuda().eval()
    print("model loaded", flush=True)

    valid = []
    t0 = time.time()
    for ci, start in enumerate(range(0, n, CHUNK)):
        end = min(start + CHUNK, n)
        vp = os.path.join(OUT, "query_chunks", f"v_{ci:04d}.npy")
        ip = os.path.join(OUT, "query_chunks", f"i_{ci:04d}.npy")
        if os.path.exists(vp) and os.path.exists(ip):
            valid.extend(range(start, end))
            continue
        vf = np.zeros((end - start, 512), dtype=np.float16)
        imf = np.zeros((end - start, 196, 512), dtype=np.float16)
        ok = 0
        for j, (vid, path) in enumerate(videos[start:end]):
            try:
                x = load_video(path).unsqueeze(0).cuda().half()
                with torch.no_grad(), torch.cuda.amp.autocast():
                    v, im = model.encode_video(x)
                    im = im.mean(dim=1)
                vf[j] = v[0].float().cpu().numpy().astype(np.float16)
                imf[j] = im[0].float().cpu().numpy().astype(np.float16)
                ok += 1
            except Exception as e:
                print("FAIL", vid, repr(e)[:100], flush=True)
        np.save(vp, vf)
        np.save(ip, imf)
        valid.extend(range(start, end))
        print(f"chunk {ci} {end}/{n} ok={ok} elapsed {time.time()-t0:.1f}s", flush=True)

    vf = np.lib.format.open_memmap(os.path.join(OUT, "msrvtt_test_video_fp16.npy"),
                                   mode="w+", dtype=np.float16, shape=(n, 512))
    imf = np.lib.format.open_memmap(os.path.join(OUT, "msrvtt_test_imgfeat_fp16.npy"),
                                    mode="w+", dtype=np.float16, shape=(n, 196, 512))
    for ci, start in enumerate(range(0, n, CHUNK)):
        end = min(start + CHUNK, n)
        vf[start:end] = np.load(os.path.join(OUT, "query_chunks", f"v_{ci:04d}.npy"))
        imf[start:end] = np.load(os.path.join(OUT, "query_chunks", f"i_{ci:04d}.npy"))
    vf.flush(); imf.flush(); del vf, imf
    with open(os.path.join(OUT, "msrvtt_test_query_manifest.json"), "w") as f:
        json.dump({
            "videos": n, "dim": 512, "img_tokens": 196, "dtype": "float16",
            "frames": T, "resize_short": 256, "crop": 224,
            "mean": MEAN.tolist(), "std": STD.tolist(),
            "order": "test.label.tsv line order",
            "elapsed_sec": round(time.time() - t0, 1),
        }, f, indent=2)
    print("DONE", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
