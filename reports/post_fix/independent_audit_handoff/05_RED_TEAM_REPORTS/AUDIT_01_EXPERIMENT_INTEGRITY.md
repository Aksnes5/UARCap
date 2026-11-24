# AUDIT 01 — Experiment integrity (independent re-check)

Verdict: **PASS** for formal configs; **PASS with scope note** for retrieval-side bug exposure.

## 1. Configuration completeness
Independent re-scan of `evcap_external/outputs/eval_postfix_*`:
- expected formal configs: 71 (48 main + 4 in-domain + 2 T×V + 3 Oracle + 2 position + 5 tail/distractor + 7 LOO);
- verified: **71/71**;
- for each: generated_captions exists, row count 497 (or 150 for LOO), unique image IDs,
  non-empty predictions, metrics.json present, IDs equal to the val497/val150 list;
- smoke `smoke_val20_D3` (20 rows) exists and is excluded from formal statistics.
Evidence: `AUDIT_MAIN_GRID_STATS.json`, `POST_FIX_FINAL_RESULTS_LEDGER.csv`,
`reports/post_fix/manifests/*.json` (71 files).

## 2. No PRE_FIX mixing
All formal POST_FIX predictions live under `eval_postfix_*` with commit
`2b3da9d` code; PRE_FIX directories are separate (`eval_val497_*`). Aggregations
read only `eval_postfix_*`. PASS.

## 3. Per-config identity
- 48 main configs use the expected `external_val_{scale}_{cf7,d7}_{cap,sims}.json`;
  FULL uses `external_val_{cf7,d7}`; in-domain uses `external_val_indomain_d7`;
  Oracle/position/tail use their own cap/sims files;
- all runs use the same checkpoint (`a8b5af79…`), prompt SHA, beam 5, seed 42,
  deterministic clip choice, and `video_feature_dir` faithful cache. PASS by
  command-line/manifest inspection.
- LOO uses 150-query YAML and remove-one cap files (6 entries), no refill. PASS.

## 4. Retrieval-side exposure to the frame bug (previously unverified)
The `start,end=0,0` bug lives only in the evaluation dataset loader
(`vision_language_tsv.py`, eval branch). Retrieval and its caches do **not** use
this loader:
- val query X-CLIP features: `extract_val_query_features.py` / `extract_query_features_v2.py`
  decode the whole video with decord and sample center-of-bin frames
  (`scripts/extract_query_features_v2.py:33`), no start/end window;
- VideoUFO caption embeddings: text-only, unaffected by video decode;
- candidate X-CLIP groundings: `extract_xclip_video_features.py` uses decord over
  the whole clip and center sampling, not the eval loader;
- Video-LLaMA candidate features: `extract_external_features_v2.py` uses
  `linspace(0,N-1,16)` over the whole clip.
Therefore the val retrieval top-20, query vectors and fusion-feature caches are
**not** contaminated by the 0/0 bug. The bug affected only the generator's frame
input in val497/val150 generation. This is supported by code paths, not inference.

## 5. Residual risks
- PRE_FIX val-derived choices (adaptive selector, k mapping) remain methodologically
  invalid even though retrieval itself was sound.
- MSR-VTT test2690/MSVD670 use pre-extracted frame TSVs and are unaffected; this
  was re-verified by YAML `img:` fields and `get_frames_from_tsv` behavior.

Status: formal results PASS; no evidence of hidden PRE_FIX mixing.
