#!/usr/bin/env python3
"""FAITHFUL_V2 Video-LLaMA features: 16 frames at linspace(0,N-1,16), 224, ImageNet."""
import json
import os
import sys
import time

import torch
import numpy as np
import decord
import cv2

cv2.setNumThreads(0)
ROOT = "/home/bn/videoufo_research"
CODE = os.path.join(ROOT, "evcap_external")
CKPT = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/EVCapCode/outputs/train_evcap_msrvtt_quality_text/train_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review.pt"
VICUNA = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/models/vicuna-7b-v1.3"
VIDEO_DIR = "/home/bn/datasets/VideoUFO/videos/VideoUFO"
OUT = os.path.join(ROOT, "features/videollama_faithful_v2")
T = 16
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_clip(path):
    vr = decord.VideoReader(path, num_threads=2)
    n = len(vr)
    idx = np.clip(np.linspace(0, n - 1, T).astype(int), 0, n - 1)
    frames = vr.get_batch(idx).asnumpy()
    out = np.empty((T, 224, 224, 3), dtype=np.uint8)
    for i, fr in enumerate(frames):
        h, w = fr.shape[:2]
        if h <= w:
            nh, nw = 224, max(224, int(round(w * 224 / h)))
        else:
            nw, nh = 224, max(224, int(round(h * 224 / w)))
        fr = cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_NEAREST)
        y0, x0 = (nh - 224) // 2, (nw - 224) // 2
        out[i] = fr[y0:y0 + 224, x0:x0 + 224]
    x = torch.from_numpy(out).float().permute(0, 3, 1, 2) / 255.0
    return (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)


def main():
    clips = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    os.makedirs(OUT, exist_ok=True)
    todo = [c for c in clips if not os.path.exists(os.path.join(OUT, c + ".pt"))]
    print("clips", len(clips), "todo", len(todo), flush=True)
    sys.path.insert(0, CODE); os.chdir(CODE)
    from models.evcap_fix_video_llama_xclip_msrvtt_smallcap_quality_fusion_gate_prompt5 import EVCap
    model = EVCap(ext_path="", vit_model="eva_clip_g",
                  q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
                  img_size=224, vit_precision="fp16", freeze_vit=True, freeze_qformer=True,
                  num_query_token=32, topn=7, llama_model=VICUNA, prompt_path="prompts/prompt_evcap_msrvtt.txt",
                  max_txt_len=128, low_resource=False, bs=4, frames=T, retriever=False,
                  num_video_query_token=32, freeze_video_qformer=True, load_video_feature=False,
                  use_fusion=False, deterministic=True)
    sd = torch.load(CKPT, map_location="cpu")["model"]
    model.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}, strict=False)
    model = model.cuda().eval()
    print("model loaded", flush=True)
    t0 = time.time(); BS = 4
    for s in range(0, len(todo), BS):
        batch = todo[s:s + BS]; xs = []; ok = []
        for c in batch:
            try:
                xs.append(load_clip(os.path.join(VIDEO_DIR, c + ".mp4"))); ok.append(c)
            except Exception as e:
                print("DECODE_FAIL", c, repr(e)[:80], flush=True)
        if not xs: continue
        x = torch.cat(xs).cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            q, _, _ = model.encode_img(x, ok, None, bs=len(ok), fusion=False, raw_video_feature=True)
        q = q.float().cpu()
        for i, c in enumerate(ok):
            torch.save(q[i], os.path.join(OUT, c + ".pt"))
        if (s + BS) % 200 == 0:
            print(f"{min(s+BS,len(todo))}/{len(todo)} elapsed {time.time()-t0:.1f}s", flush=True)
    print("DONE", len([c for c in clips if os.path.exists(os.path.join(OUT, c + '.pt'))]), "/", len(clips), flush=True)


if __name__ == "__main__":
    main()
