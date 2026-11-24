#!/usr/bin/env python3
"""POST_FIX val497 frame sanity: 16 real temporal positions, not static frames."""
import argparse
import csv
import hashlib
import json
import os
import sys

import cv2
import numpy as np

ROOT = "/home/bn/videoufo_research"
sys.path.insert(0, os.path.join(ROOT, "evcap_external"))
os.chdir(os.path.join(ROOT, "evcap_external"))
from dataset.msrvtt.vl_dataloader import build_dataset  # noqa: E402

OUT = os.path.join(ROOT, "reports/post_fix")
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)
MSR = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/datasets/MSRVTT-v2/videos"

args = argparse.Namespace(img_res=224, patch_size=16, img_feature_dim=512, max_num_frames=16,
                          use_asr=False, decoder_sampling_strategy="uniform", debug_speed=False,
                          effective_batch_size=1, add_od_labels=False, on_memory=False)
ds = build_dataset(args, "val497/val16_video.yaml", None, is_train=False)
ids = [it["image_id"] for it in ds]
want = [f"datasets/MSRVTT-v2/videos/video{q}.mp4" for q in
        ["6518", "6513", "6637", "6722", "7011"]]
# deterministic 20-video sample across the list
step = max(1, len(ids) // 20)
want += [ids[i] for i in range(0, len(ids), step)][:20]
want = [k for k in dict.fromkeys(want) if k in ids]


def montage(img, path):
    frames = []
    for i in range(img.shape[0]):
        f = img[i].transpose(1, 2, 0) * 255.0
        f = np.clip(f, 0, 255).astype(np.uint8)
        f = cv2.resize(f, (128, 128))
        cv2.putText(f, str(i), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        frames.append(f)
    cv2.imwrite(path, np.concatenate(frames, axis=1)[:, :, ::-1])


rows = []
for key in want:
    idx = ids.index(key)
    it = ds[idx]
    img = it["image"].numpy().astype(np.float32)
    hashes = [hashlib.md5(np.round(img[i] * 255).astype(np.uint8).tobytes()).hexdigest()[:8]
              for i in range(img.shape[0])]
    diff = [float(np.abs(img[i] - img[0]).mean()) for i in range(img.shape[0])]
    vid = os.path.basename(key)[:-4]
    cap = cv2.VideoCapture(os.path.join(MSR, vid + ".mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    dur = n / fps if fps else 0
    cap.release()
    ts = [round(dur * i / 15, 3) for i in range(16)]
    rows.append({
        "video": vid, "duration_s": round(dur, 3),
        "timestamps_s": ts,
        "unique_frame_hashes": len(set(hashes)),
        "identical_to_frame0": int(sum(d < 1e-9 for d in diff)),
        "max_abs_diff_vs_frame0": round(max(diff), 6),
        "mean_abs_diff_vs_frame0": round(float(np.mean(diff)), 6),
        "static_gate_pass": not (dur > 2.0 and (len(set(hashes)) < 8)),
    })
    if vid in {"6518", "6513", "6637", "6722", "7011"} or len(rows) <= 5:
        montage(img, os.path.join(OUT, "frames", f"{vid}_montage.png"))

with open(os.path.join(OUT, "val497_frame_sanity.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
summary = {
    "checked": len(rows),
    "static_gate_failures": [r["video"] for r in rows if not r["static_gate_pass"]],
    "max_identical_to_frame0": max(r["identical_to_frame0"] for r in rows),
    "min_unique_hashes": min(r["unique_frame_hashes"] for r in rows),
    "video6518": next(r for r in rows if r["video"] == "video6518"),
    "rows": rows,
}
json.dump(summary, open(os.path.join(OUT, "val497_frame_sanity.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
print("GATE", "PASS" if not summary["static_gate_failures"] else "FAIL")
