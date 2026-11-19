# val497 pipeline — FAITHFUL_V2, frozen 2026-11-19

Purpose: independent model-selection split for the context-budget experiment.
No test-split ground truth is read anywhere in this pipeline.

## Integrity check (`reports/val497_pipeline_check.json`)

| Check | Result |
|---|---|
| val paths / label / caption rows | 497 / 497 / 497 |
| id sets identical (paths = label = caption = query features = retrieval keys) | **true** |
| videos exist on disk | true |
| query feature shape | 497 × 512 (video), 497 × 196 × 512 (frame tokens) |
| recompute 3 videos (video6513, video6761, video7009) | max abs diff **0.0** (bit-identical fp16) |
| CF7 / D7 cap entries per query | 7 / 7 |
| selected clips / missing faithful Video-LLaMA features | 4,949 / **0** |

## Protocol

- 16-frame protocol; X-CLIP query preprocessing `INTER_NEAREST`, short side 256,
  center crop 224, ImageNet mean/std, center-of-bin temporal sampling
  (same code path as test FAITHFUL_V2).
- Video-LLaMA fusion features: `linspace(0, N-1, 16)` endpoints, 224,
  ImageNet norm — `features/videollama_faithful_v2/`.
- Corpus: VideoUFO FULL (1,091,712 clips, 93 parquet), retrieval identical to
  dev300/FULL (same X-CLIP checkpoint, same prompt generator).
- Generation: same EVCap checkpoint, prompt, fusion branch, beam width 5,
  deterministic clip choice (first clip id), `topn = k`.

## Candidate sets

- `CF7` = raw X-CLIP top-7 (retrieval order); CFk = first k.
- `D7` = ROUGE-L quality rerank of the top-20; Dk = first k.
- Both share the same retrieval JSON (`external_val_sims/top20/reranked.json`);
  only `--topn` changes between k values.

## Retrieval-side grid (`reports/val_retrieval_grid.json`)

| k | CF oracle@k | D oracle@k | CF top1 CIDEr | D top1 CIDEr | D near-dup |
|---|---|---|---|---|---|
| 1 | 0.0800 | 0.1176 | 0.0800 | 0.1176 | 0.000 |
| 3 | 0.1314 | 0.1481 | 0.0800 | 0.1176 | 0.352 |
| 5 | 0.1535 | 0.1720 | 0.0800 | 0.1176 | 0.224 |
| 7 | 0.1698 | 0.1835 | 0.0800 | 0.1176 | 0.148 |

Candidate coverage rises monotonically with k for both orderings; D's reranker
already lifts top-1 caption CIDEr by +0.038 over raw retrieval. Near-duplicate
pressure (pairwise ROUGE-L ≥ 0.8) is highest at k=3 for D (0.352) and drops as
more slots are filled.

## Freeze rule

k, ordering and all thresholds are chosen on this split only. After the method
is frozen, evaluation moves to untouched test2690, then full test2990.
