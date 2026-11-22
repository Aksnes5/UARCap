# val497 vs test2690 absolute CIDEr gap audit

**Status: EXPLAINED.** CPU-only, read-only audit of frozen artifacts. No generation,
no CUDA, no frozen output or frozen metric was modified.

## 1. Bottom line

The ~17–19 point CIDEr gap between MSR-VTT val497 (≈50–55) and untouched test2690
(≈68–73) is **not** a scorer / reference / IDF / split-membership / split-difficulty
artifact. It is caused by a **visual-input dataloader bug specific to the val497
YAML**: the val split is fed to the caption generator through a video-path TSV, and
the eval loader passes `start=0, end=0` to the on-the-fly PyAV decoder. With
`start == end == 0` only the first two decoded frames are kept, and the temporal
sampler replicates the first frame 15 times (and the second frame once). The
generator therefore saw, in effect, a **static first frame**, not the 16-frame video.

The retrieval path is unaffected: X-CLIP query features were pre-extracted with the
correct 16-frame protocol, so retrieved evidence quality is normal. Every test/MSVD/
dev300 evaluation uses pre-extracted frame TSVs and is unaffected.

Because all val497 conditions share the same broken visual input, **paired within-val
contrasts remain internally valid**, but **absolute val497 scores are not comparable**
to test2690, MSVD, or the published EVCap numbers. A counterfactual re-run of val497
with correct 16-frame input requires generation and is **BLOCKED** (GPU unavailable).

## 2. Frozen reported values vs audit recomputation (kept separate)

| condition | frozen val497 CIDEr | frozen test2690 CIDEr | gap |
|---|---:|---:|---:|
| D1 | 51.77 | 71.21 | 19.44 |
| D3 | 53.14 | 70.08 | 16.94 |
| D5 | 50.51 | 69.48 | 18.97 |
| D7 | 50.05 | 68.27 | 18.22 |
| in-domain B/D1 | 54.16 | 73.09 | 18.93 |

Frozen values are from each run's `metrics.json` and are unchanged.

Audit re-evaluation (same `pycocoevalcap` PTB tokenizer + `Cider` implementation as
`evaluation/cocoeval.py`, run in-process):

| artifact | frozen | audit recompute |
|---|---:|---:|
| `eval_val497_D3` | 0.531361 | 0.531361 |
| `eval_test2690_D3` | 0.700757 | 0.700757 |
| `eval_test2990_D1` | 0.703108 | 0.703108 |

Recomputation is bit-for-bit at 4 decimals; the scorer is not the source of the gap.
(The audit recomputation is a separate record, not a replacement of the frozen values.)

## 3. Evaluation-pipeline checklist

| Check | Result | Verdict |
|---|---|---|
| Reference files | val.caption.tsv / test.caption.tsv; both verified to match `MSR_VTT.json` official annotations exactly (mean Jaccard = 1.0) | consistent |
| References per video | 20 for val497 and 20 for test2690 | consistent |
| Reference length (PTB tokens) | 9.226 (val) vs 9.314 (test) | consistent |
| Reference consensus (mean pairwise ROUGE-L) | 0.2348 (val) vs 0.2349 (test) | consistent |
| Leave-one-out reference CIDEr ceiling | mean 0.5177 / max 1.1126 (val) vs mean 0.5074 / max 1.0884 (test) | val is, if anything, slightly easier |
| CIDEr scorer implementation | same `pycocoevalcap/cider/cider_scorer.py` for both | consistent |
| PTB tokenization | same `PTBTokenizer` path | consistent |
| DF / IDF construction | corpus-local (`document_frequency` over the refs in the evaluated batch) | scope differs; quantified below |
| Scorer invocation | same `evaluation/cocoeval.py` via `scripts/run_metrics.sh` | consistent |
| Evaluation batch scope | 497 single predictions vs 2690 single predictions; one reference set per query | consistent |
| Candidate formatting | `prediction` string; refs from the same generated JSON `captions` field | consistent |
| Video IDs / split membership | val = video6513–video7009; test2690 = video7011–video9998; overlap 0; union of pilot300 + 2690 = full 2990 test | correct |
| val uses a different reference subset? | No — each val item carries its own official 20 references | consistent |
| Accidental metric implementation difference | none detected | consistent |
| Query-video input path | **val: on-the-fly PyAV decode from `val_video_paths.tsv`; test: pre-extracted 16-frame TSV** | **ROOT CAUSE** |
| Split difficulty | references and retrieval-candidate compatibility are matched (below) | not supported as the main cause |

### IDF scope is not the explanation

Scoring the same frozen predictions against the same references while swapping the
document-frequency corpus:

| prediction set | native DF | other-split DF | combined DF |
|---|---:|---:|---:|
| val497 D3 | 0.5314 | 0.5208 (test DF) | 0.5202 |
| test2690 D3 | 0.7008 | 0.7182 (val DF) | 0.6992 |

Under a common combined DF the gap is **0.179 (17.9 points)**, i.e., IDF scope
accounts for at most ~1 point and does not explain the gap.

### Retrieval candidate quality is matched

Fixed-DF reference compatibility of the supplied candidates (same estimator for both):

| statistic | val497 | test2690 |
|---|---:|---:|
| top-1 candidate compatibility (mean) | 0.1869 | 0.1899 |
| top-3 mean compatibility | 0.1511 | 0.1515 |
| top-7 mean compatibility | 0.1066 | 0.1047 |

The val candidate pool is not worse than the test pool. Checkpoints are identical
(sha256 `a8b5af79d55a2c7e684771475d645be6373208bdaae5a4211bee556581eb8002`), and
generation arguments match except for the dataset YAML.

## 4. Root cause: static first frame in val497 generation

### Code path

- `evcap_external/dataset/msrvtt/vision_language_tsv.py:338` (eval branch):
  `start, end = 0, 0` before `get_visual_data(...)`.
- `evcap_external/dataset/msrvtt/data_utils/video_decoder.py:218–227`:
  when `start` and `end` are not `None`, they set `video_start_pts` / `video_end_pts`;
  with 0/0 this becomes a decode window of `[0, 0]`.
- `video_decoder.py:246–252` (`pyav_decode_stream`): only frames with
  `start_pts <= pts <= end_pts` are kept, plus one buffered frame; `temporal_sampling`
  (line 343) then samples 16 indices from this 1–2 frame pool.
- `temporal_sampling` with a 2-frame pool and `linspace(0,1,16)` truncates to
  15 copies of frame 0 and 1 copy of frame 1.

The val YAML uses `img: .../val_video_paths.tsv` (one path per row), so
`get_visual_data` takes the on-the-fly branch (`row[0] == row[-1]`). The test2690
YAML uses a pre-extracted frame TSV, whose branch (`get_frames_from_tsv(row[2:])`)
ignores `start`/`end` and returns 16 distinct frames.

The generation script reaches this path: `make_data_loader(..., is_train=False)`
builds the same `VisionLanguageTSVYamlDataset` used in the empirical check below, and
`__getitem__` takes the `not self.is_train` branch that sets `start, end = 0, 0`.

### Empirical proof

End-to-end through the real dataloader (`build_dataset(...)`; CPU only):

| split | item | frames identical to frame 0 | max per-frame mean abs diff |
|---|---|---:|---:|
| MSR-VTT val497 | video6513 | **15 / 16** | 0.112 (frame 15 only) |
| MSR-VTT test2990 | video7010, video7011, video7110 | 1 / 16 (frame 0 itself) | 1.4–1.9 |
| MSVD test670 | 3 sampled videos | 1 / 16 (frame 0 itself) | 0.5–1.5 |

A direct call to `extract_frames_from_video_path(..., start=0, end=0)` reproduces the
same result: 15 identical copies of frame 0 plus one near-copy. With
`start=None, end=None` the same function returns 16 distinct, uniformly spaced frames.
Evidence JSON: `reports/cpu_audit/val_frame_input_bug_evidence.json`.

### Visual confirmation

Sample val497 videos (decoded frames, CPU) show the model's output tracking the
**first frame**, not the annotated content or the retrieved evidence:

| video | frame 0 content | rest of video | references | D3 prediction |
|---|---|---|---|---|
| video6518 | Bank of America logo/ad | wrestling match | wrestling | "an advertisement for bank of america" |
| video6637 | close-up of a book on a desk | woman in a classroom | classroom explanation | "a person is reading a book" |
| video6722 | football game | BMX rider | sports-blooper montage | "a football game is being played" |
| video7011 (test) | paint-can animation | same paint-can animation | paint / colors | "an advertisement for pink paint" |

For video6518, both the 20 references and all seven retrieved captions are about
wrestling, yet the output describes the logo in frame 0. This is the signature of a
static-first-frame input, not a retrieval or metric failure.

## 5. Consequence and manuscript implication

- **All val497 absolute numbers** (the six-scale D/CF grid, Oracle1/3/7, Good/Hard/Easy,
  Text×Visual, in-domain control, position/tail controls, replicates, LOO) were produced
  under this static-first-frame regime. They must not be quoted as absolute performance
  and must not be compared with test2690, MSVD, or the original EVCap paper.
- **Paired within-val comparisons remain internally valid**, because every val497
  condition used the same YAML and therefore the same (broken) visual input; only the
  textual evidence / ordering / budget changed between conditions. This is why the
  val-internal effects are directionally consistent with the unaffected test/MSVD runs.
  Their *magnitudes* may not transfer: under a static visual input the generator is
  more text-dominated, and the val D3−D7 effect (+3.11) is indeed larger than the
  test2690 effect (+1.81). This should not be read as a split difference.
- The **val/test gap is a pipeline artifact**, not evidence about external retrieval or
  split difficulty. It should not be interpreted as "val is harder".
- The **adaptive selector** was frozen using val497 out-of-fold outputs produced under
  this bug. The selector itself uses retrieval similarity gaps, so its test2690
  application is unaffected, but the OOF val497 estimate (+1.03) is not a clean estimate
  of the selector's value under normal visual input.
- **Fix needed for publication-grade absolute val numbers:** build a val 16-frame
  224 px TSV (analogous to `test_16frm/frame_tsv/test_16frames_img_size224.img.tsv`)
  and re-run val generation with it. This requires the frozen generator and is
  **BLOCKED (GPU unavailable)**. Until then, val497 should be labelled as a
  static-first-frame diagnostic split, or absolute val numbers should be dropped.

## 6. Verdict

**EXPLAINED.** The evaluation pipeline (references, tokenizer, CIDEr/DF/IDF, scorer
invocation, batch scope, candidate formatting, split membership) is consistent and
does not produce the gap. The gap is produced by the val497 generator input path
(static first frame). The counterfactual "val497 with 16 distinct frames" score cannot
be measured in this audit and remains **BLOCKED_REQUIRES_GENERATION**.

## 7. Files

- `reports/cpu_audit/cpu_audit_valtest_forensics.json` — reference/IDF/retrieval controls
- `reports/cpu_audit/val_frame_input_bug_evidence.json` — dataloader frame evidence
- `scripts/cpu_audit/valtest_forensics.py`, `scripts/cpu_audit/frame_input_evidence.py`
- `reports/cpu_audit/frames/*.png` — sample decoded frames
