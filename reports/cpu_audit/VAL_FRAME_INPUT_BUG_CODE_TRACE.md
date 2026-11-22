# val497 static-first-frame bug — source-level code trace

Status: **VERIFIED**. This document fixes the provenance of the val497 visual-input
bug. No code was modified and no val run was re-executed.

## 1. One-paragraph summary

The val497 caption generator is fed through a **video-path YAML**
(`val497/val16_video.yaml`). For evaluation, `VisionLanguageTSVDataset.__getitem__`
sets `start, end = 0, 0` and passes them to the on-the-fly PyAV decoder. In
`pyav_decode`, non-`None` `start`/`end` set `video_start_pts = 0` and
`video_end_pts = 0`, so `pyav_decode_stream` keeps only the first frame (plus one
buffered frame). `decode()` then calls `temporal_sampling` on this 1–2 frame pool,
and `torch.linspace(0, 1, 16).long()` yields fifteen 0s and one 1 — i.e. **15 copies
of frame 0 plus 1 copy of frame 1**. The generator therefore sees a static first
frame, not the 16-frame video.

## 2. Exact call chain and branches

All paths are relative to the repository root. Line numbers are from the frozen
checkout at commit `c31dc82`.

### 2.1 YAML selects the on-the-fly video path

`evcap_external/val497/val16_video.yaml`

```
1  img: /home/bn/videoufo_research/evcap_external/val497/val_video_paths.tsv
2  label: .../MSRVTT-v2/val.label.tsv
3  caption: .../MSRVTT-v2/val.caption.tsv
```

`val_video_paths.tsv` has one column per row (a video path), so the loader's
`row[0] == row[-1]` test in `get_visual_data` is true.

### 2.2 Eval branch sets `start, end = 0, 0`

`evcap_external/dataset/msrvtt/vision_language_tsv.py`

```
330  def __getitem__(self, idx):
...
335      if not self.is_train: # by lijian
336          data_sample = self.get_caption_and_timeinfo_wrapper(img_idx, cap_idx)
337          caption = [cap['caption'] for cap in data_sample]
338          start, end = 0, 0
...
346      raw_frames, is_video = self.get_visual_data(img_idx, start, end)
```

This is the decisive line: for eval, the temporal window is
`[start, end] = [0, 0]`, not `[None, None]`.

### 2.3 `get_visual_data` takes the video-decode branch

`evcap_external/dataset/msrvtt/vision_language_tsv.py`

```
309  def get_visual_data(self, idx, start=None, end=None):
310      row = self.get_row_from_tsv(self.visual_tsv, idx)
...
313      if row[0] == row[-1]:
314          return self.decode_and_get_frames(row[-1], start, end), True
315      # if the input is a video tsv with frames pre-extracted,
316      # return a video-frame tensor
317      elif len(row) >= self.decoder_num_frames + 2:
318          return self.get_frames_from_tsv(row[2:]), True
```

The val video-path TSV has `row[0] == row[-1]`, so line 314 is taken and
`start=0, end=0` are forwarded. The test/MSVD/dev300 frame TSVs have
`len(row) >= 16 + 2`, so line 318 is taken instead.

### 2.4 `decode_and_get_frames` forwards `start`/`end`

`evcap_external/dataset/msrvtt/vision_language_tsv.py`

```
291  def decode_and_get_frames(self, clip_path_name, start=None, end=None):
...
302      frames, video_max_pts = extract_frames_from_video_path(
303          resolved_video_path, self.decoder_target_fps,
304          self.decoder_num_frames, self.decoder_multi_thread_decode,
305          self.decoder_sampling_strategy, self.decoder_safeguard_duration,
306          start, end)
```

### 2.5 `extract_frames_from_video_path` forwards to the PyAV decoder

`evcap_external/dataset/msrvtt/data_utils/video_ops.py`

```
65  def extract_frames_from_video_path(
66          video_path, target_fps=3, num_frames=3,
67          multi_thread_decode=False, sampling_strategy="rand",
68          safeguard_duration=False, start=None, end=None):
...
71      frames, video_max_pts = extract_frames_from_video_binary(
72          in_mem_bytes_io, target_fps=target_fps, num_frames=num_frames,
73          multi_thread_decode=multi_thread_decode,
74          sampling_strategy=sampling_strategy,
75          safeguard_duration=safeguard_duration,
76          start=start, end=end)
```

`extract_frames_from_video_binary` (line 83) passes the same `start`/`end` into
`decoder.decode(...)` at lines 141–147.

### 2.6 `pyav_decode` turns 0/0 into a one-frame decode window

`evcap_external/dataset/msrvtt/data_utils/video_decoder.py`

```
137  def pyav_decode(...):
...
202      if clip_idx == -2:
203          # the sampled clip will be the entire video
204          clip_size = frames_length
205          sample_clip_idx = 0
206          sample_num_clips = 1
207      start_idx, end_idx = get_start_end_idx(...)
...
218      if start!=None and end!=None and duration!=None:
219          timebase = duration / frames_length
220          video_start_pts = int(start * fps * timebase)
221          video_end_pts = int(end * fps * timebase)
```

With `start = end = 0`, `video_start_pts = video_end_pts = 0`. This branch is
evaluated **after** the `clip_idx == -2` block, so it overrides the intended
whole-video window with `[0, 0]`.

### 2.7 `pyav_decode_stream` retains only 1–2 frames

`evcap_external/dataset/msrvtt/data_utils/video_decoder.py`

```
95   def pyav_decode_stream(container, start_pts, end_pts, stream, stream_name, buffer_size=0):
...
122      for frame in container.decode(**stream_name):
...
124          if frame.pts < start_pts:
125              continue
126          if frame.pts <= end_pts:
127              frames[frame.pts] = frame
128          else:
129              buffer_count += 1
130              frames[frame.pts] = frame
131              if buffer_count >= buffer_size:
132                  break
133      result = [frames[pts] for pts in sorted(frames)]
```

With `start_pts = end_pts = 0` and `buffer_size = 0`, only the first frame
(`pts <= 0`) and the first buffered frame (`pts > 0`) are kept — two frames.

### 2.8 `decode()` replicates frame 0 to fill 16 slots

`evcap_external/dataset/msrvtt/data_utils/video_decoder.py`

```
261  def decode(...):
...
331      if clip_idx == -2:
332          clip_size = len(frames)
333          sample_clip_idx = 0
334          sample_num_clips = 1
335
336      start_idx, end_idx = get_start_end_idx(
337          len(frames), clip_size,
338          sample_clip_idx if decode_all_video else 0,
339          sample_num_clips if decode_all_video else 1,
340      )
343      frames = temporal_sampling(frames, start_idx, end_idx, num_frames)
```

For the 2-frame pool, `len(frames) = 2`, `clip_size = 2`, and
`get_start_end_idx(2, 2, 0, 1)` gives `delta = max(2-2, 0) = 0`, `start_idx = 0`,
`end_idx = 1`. Then `temporal_sampling` (lines 11–27) does:

```
24   index = torch.linspace(start_idx, end_idx, num_samples)   # linspace(0, 1, 16)
25   index = torch.clamp(index, 0, len(frames)-1).long()       # [0]*15 + [1]
26   frames = [frames[idx] for idx in index]
```

Hence **15 copies of frame 0 and 1 copy of frame 1**. This matches the empirical
per-frame differences exactly (15 zeros, then a small non-zero at index 15).

## 3. Empirical verification through the real dataloader

`scripts/cpu_audit/frame_input_evidence.py` builds the dataset with
`build_dataset(args, yaml, None, is_train=False)` and inspects
`ds[idx]["image"]`. It is CPU-only and does not load the model.

| split | item | frames identical to frame 0 | max per-frame mean abs diff |
|---|---|---:|---:|
| MSR-VTT val497 (`val497/val16_video.yaml`) | `video6513` | **15 / 16** | 0.112 (frame 15 only) |
| MSR-VTT test2990 (`test2990/test2990_16frames.yaml`) | `video7010` | 1 / 16 (frame 0 itself) | 0.93 |
| MSR-VTT test2990 | `video7011` | 1 / 16 | 0.94 |
| MSVD670 (`datasets/MSVD/test_16frames.yaml`) | 3 sampled videos | 1 / 16 | 0.47–1.47 |

The val rows show exactly the 15-identical-plus-1 pattern predicted by the code
trace. The test/MSVD rows show distinct frames.

Visual confirmation: `reports/cpu_audit/frames/video6518_montage.png` (frame 0 is
a Bank of America logo; the rest is wrestling), `video6637_montage.png` (frame 0 is
a book close-up; the rest is a classroom), `video6722_montage.png` (frame 0 is a
football game; the rest is BMX). The frozen val497 D3 predictions for these videos
are "an advertisement for bank of america", "a person is reading a book", and
"a football game is being played" respectively — i.e. frame-0 content.

## 4. Why test2690 / MSVD / dev300 are unaffected

Their YAMLs point at **pre-extracted frame TSVs**, so `get_visual_data` takes the
`elif len(row) >= self.decoder_num_frames + 2` branch (line 317) and calls
`get_frames_from_tsv(row[2:])` (line 264). That function ignores `start`/`end` and
samples `self.decoder_num_frames` frames with:

```
276      def sampling(start,end,n):
...
283          step = (end-start)/float(n-1)
284          return [int(round(start+x*step)) for x in range(n)]
...
288      for i in sampling(0, len(binary_frms)-1, self.decoder_num_frames):
```

The `start=0, end=0` values from `__getitem__` are never used on this branch.

YAML provenance:

| YAML | `img:` field | affected? |
|---|---|---|
| `evcap_external/val497/val16_video.yaml` | `val497/val_video_paths.tsv` (video paths) | **yes** |
| `evcap_external/val150/val150_video.yaml` | `val150/val_video_paths.tsv` (video paths) | **yes** |
| `evcap_external/test2990/test2990_16frames.yaml` | `test_16frm/frame_tsv/test_16frames_img_size224.img.tsv` | no |
| `evcap_external/dev300/dev300_16frames.yaml` | `dev300.img.tsv` (pre-extracted) | no |
| `evcap_external/datasets/MSVD/test_16frames.yaml` | `frame_tsv/test_16frames_img_size256.img.tsv` | no |

## 5. Git provenance

- **UARCap**: `git log --all --oneline -S "start, end = 0, 0" -- evcap_external/dataset/msrvtt/vision_language_tsv.py`
  returns exactly one commit:
  `6efccfe chore: initial UARCap research snapshot` (2026-11-19).
  `git blame -L 332,346` attributes line 338 to `^6efccfe (Aksnes5 2026-11-19)`.
  The same commit is the only commit touching this file in UARCap history.
- **Upstream EVCap tree**: the local checkout at
  `lj/EVCapCode/dataset/msrvtt/vision_language_tsv.py` is **byte-identical**
  (`diff` exit 0), but the file is untracked in that checkout
  (`git status --short` → `??`), so no upstream commit can be assigned locally.
  Its configured remote is `https://github.com/Jiaxuan-Li/EVCap.git`.
- **Affected generation script version**:
  `evcap_external/eval_evcap_fix_video_llama_xclip_msrvtt_fusion_gate_prompt5.py`
  (the MSR-VTT eval driver used for every val497/val150 run). The MSVD driver
  `eval_evcap_fix_video_llama_xclip_msrvtt_fusion_gate_prompt5_msvd.py` would be
  affected only if given a video-path YAML; the actual MSVD run uses a frame TSV,
  so it is not.
- **Affected outputs**: every `eval_val497_*` and `eval_val150_*` directory
  (six-scale D/CF grids, adaptive/OOF, in-domain control, Oracle and tail
  conditions, factorial, position controls, replicates, LOO). Not affected:
  `eval_test*`, `eval_msvd_*`, `eval_dev300_*`.

## 6. What was and was not done

- Verified: source path, branch conditions, `start,end = 0,0`, frame replication,
  empirical 15/16 identical-frame result, and unaffected test/MSVD paths.
- Not done: code fix; val497 re-run; any regeneration; any modification of frozen
  outputs.
- The bug's provenance is now fixed in this document for a later, GPU-enabled
  repair and re-run.
