# POST_FIX val497 frame sanity report

## Fix
`evcap_external/dataset/msrvtt/vision_language_tsv.py:338`
changed eval branch from `start, end = 0, 0` to `start, end = None, None`.
With 0/0, `pyav_decode` set `video_end_pts=0`, keeping 1–2 frames; the temporal
sampler then produced 15 copies of frame 0 + 1 copy of frame 1. With `None`,
the whole video is decoded and `temporal_sampling` selects 16 distinct positions.

## Sanity (real dataloader, epoch 0)
- Checked: 23 val497 videos (20 deterministic sample + targeted cases).
- Unique frame hashes: **16/16 for every video** (min 16).
- Frames identical to frame 0: **1/16** (frame 0 itself); no static replication.
- Max per-frame mean-abs difference vs frame 0: 1.15 (video6518).
- Static-gate failures: **none**; gate = PASS.
- Timestamps sampled at `duration·i/15`, e.g. video6518 (26.03 s):
  0, 1.74, 3.47, …, 24.29, 26.03 s.
- video6518 montage confirms frame 0 is a bank/logo overlay and later frames
  show the wrestling match; prior frozen prediction "advertisement for bank of
  america" was frame-0 content.

## Artifacts
- `reports/post_fix/val497_frame_sanity.json`
- `reports/post_fix/val497_frame_sanity.csv`
- `reports/post_fix/frames/{video6518,video6513,video6537,video6637,video6722}_montage.png`

## Scope
test2690, MSVD670 and dev300 use pre-extracted frame TSVs and are unaffected
(`get_frames_from_tsv` ignores start/end); no changes made to those inputs.
