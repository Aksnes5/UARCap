# N2 — VideoUFO ↔ MSR-VTT contamination audit

Date: 2026-11-20. Corpus: VideoUFO FULL (1,091,712 clips, 345,437 YouTube sources).
MSR-VTT: 10,000 videos (train/val/test) with `url` + start/end times in
`train_val_videodatainfo.json` / `test_videodatainfo.json`.

## 1. Source-level overlap (decisive)

VideoUFO clip IDs embed the YouTube source ID (`<yt_id>.<segment>`), and the CSV
carries segment start/end times. MSR-VTT metadata carries YouTube URLs and the
segment times.

| Split | MSR-VTT videos | same YouTube source in VideoUFO | same source AND ≥3 s time overlap |
|---|---|---|---|
| train | 6,513 | **0** | 0 |
| val | 497 | **0** | 0 |
| test | 2,990 | **0** | 0 |

Retrieved val497 top-7 candidates: 0 clips share a source ID with their query
video. **No video/source-level leakage exists.**

## 2. Caption text overlap

Exact normalized brief-caption intersection with VideoUFO's 812,014 unique
captions:

| Split | unique MSR-VTT captions | exact matches | ratio |
|---|---|---|---|
| train | 110,025 | 93 | 0.08% |
| val | 8,823 | 7 | 0.08% |
| test | 51,579 | 41 | 0.08% |

These are generic templates ("a man is talking", "a woman is walking") with no
source linkage; the 0.08% rate is identical across splits, i.e. baseline
vocabulary overlap rather than clip duplication.

## 3. Visual near-duplicate check

dHash (64-bit) on 3 frames per video; 497 val queries vs their 3,140 distinct
retrieved candidates:

- queries with any candidate at Hamming ≤ 6: **1/497**
- ≤ 10: 7/497
- random-corpus control mean min distance: 14.64

Manual inspection of the flagged queries shows graphics/static content
(Minecraft gameplay, cartoon cat on blue background, Cartoon Network logo,
dark cave scenes, website screenshots). dHash is unreliable on flat/graphical
frames; combined with the zero source-level overlap these are false positives,
not duplicated footage.

## Verdict

**No severe contamination.** No exclusion or experiment recomputation required.
This audit will be repeated for the final retrieved test candidates before
submission (same source-ID method, cheap and decisive).
