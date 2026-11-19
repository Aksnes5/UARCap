#!/usr/bin/env python3
"""Write frozen_method_config.json (no arguments; pins the final Phase-4 method)."""
import hashlib
import json
import os
import subprocess
from datetime import datetime

ROOT = "/home/bn/videoufo_research"
CKPT = ("/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/EVCapCode/outputs/"
        "train_evcap_msrvtt_quality_text/"
        "train_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review.pt")
VICUNA = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/models/vicuna-7b-v1.3"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return p if not p.startswith(ROOT) else os.path.relpath(p, ROOT)


cfg = {
    "name": "similarity-gap adaptive context budget",
    "frozen_at": datetime.now().isoformat(timespec="seconds"),
    "git_commit": subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"]).decode().strip(),
    "selection_split": "official MSR-VTT val497 (FAITHFUL_V2), 5-fold OOF by video id",
    "ranking": {
        "candidate_order": "ROUGE-L quality reranking of X-CLIP top-20 (original D)",
        "k_set": [1, 3, 5, 7],
    },
    "selector": {
        "inputs": "X-CLIP similarities s1..s7 of the top-7 ranked candidates",
        "rule": "largest adjacent gap g_i = s_i - s_{i+1}, i=1..6; budget = g_index mapped:"
                " i<=1 -> 1; i in {2,3} -> 3; i in {4,5} -> 5; i=6 -> 7",
        "fitted_parameters": [],
        "uses_ground_truth": False,
        "uses_generated_metrics": False,
    },
    "preprocessing": {
        "query_xclip": "16 center-of-bin frames, Resize short side 256, CenterCrop 224,"
                       " cv2.INTER_NEAREST, ImageNet mean/std",
        "fusion_videollama": "16 frames at linspace(0,N-1,16), 224, ImageNet mean/std",
        "dataset_frames": 16,
        "frame_sampling": "uniform linspace endpoints (EVCap dataset decoder)",
    },
    "model": {
        "checkpoint": rel(CKPT), "checkpoint_sha256": sha(CKPT),
        "vicuna": rel(VICUNA),
        "prompt": "evcap_external/prompts/prompt_evcap_msrvtt_detail_smallcap_prompt5.txt",
        "prompt_sha256": sha(os.path.join(ROOT, "evcap_external/prompts/prompt_evcap_msrvtt_detail_smallcap_prompt5.txt")),
        "beam_width": 5, "seed": 42, "deterministic_clip_choice": True, "fusion": True,
    },
    "corpus": {
        "name": "VideoUFO FULL", "clips": 1091712,
        "manifest": "manifests/clips_full.tsv",
        "manifest_sha256": sha(os.path.join(ROOT, "manifests/clips_full.tsv")),
        "metadata": "/home/bn/datasets/VideoUFO/metadata/VideoUFO.csv",
    },
    "query_features": {
        "val": {"file": "embeddings/msrvtt_val_video_xclip_faithful_v2_video_fp16.npy",
                "sha256": sha(os.path.join(ROOT, "embeddings/msrvtt_val_video_xclip_faithful_v2_video_fp16.npy"))},
        "test": {"file": "embeddings/msrvtt_test_video_xclip_faithful_v2_video_fp16.npy",
                 "sha256": sha(os.path.join(ROOT, "embeddings/msrvtt_test_video_xclip_faithful_v2_video_fp16.npy"))},
    },
    "test_policy": "untouched test2690 evaluated once after this freeze; pilot300 excluded from confirmatory claims",
}
json.dump(cfg, open(os.path.join(ROOT, "frozen_method_config.json"), "w"), indent=2)
print(json.dumps(cfg, indent=2)[:600])
