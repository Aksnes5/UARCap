# Pre-run integrity patch (in-domain control + LOO)

## 1. In-domain caption corpus
- Train videos: 6,513; val/test overlap: **0 / 0**.
- Raw train annotation captions: **130,260** (20 per video).
- Exact-unique captions: **110,031**; normalized-unique: **110,025**.
- Indexed captions: exact-unique pool (110,031). Every indexed caption maps to one
  or more train videos; no val/test rows are read by the builder.
- Dedup policy: exact-string dedup, matching the original EVCap in-domain
  retrieval artifact naming (`..._no_duplicate...`). Raw annotation-level pool
  size is recorded above; no per-video collapse to one caption is performed.

## 2. In-domain feature equivalence
- Legacy cache (`ext_data/video_query/*.pt`) is not proven equivalent to the
  current FAITHFUL_V2 candidate pipeline; therefore the matched control uses
  **newly extracted FAITHFUL_V2 features** for the in-domain candidate union,
  not the legacy cache.
- Decision: `FEATURE_PIPELINE_MATCHED = RECOMPUTED_FAITHFUL_V2`.
- Template/index: legacy=frame-based cache with unknown sampling; current =
  `linspace(0,N-1,16)`, INTER_NEAREST short-side 224, ImageNet norm.
- 20-video numeric comparison is recorded in
  `reports/indomain_feature_equivalence.json` once extraction completes.

## 3. Query / generation config equivalence
- Query IDs/features: same `msrvtt_val_video_xclip_faithful_v2_*` (SHA from
  `frozen_method_config.json`).
- Checkpoint `a8b5af79…`, prompt SHA, beam width 5, seed 42, deterministic clip
  choice, fusion formula: identical to external D runs.

## 4. LOO implementation
- `external_val_loo_rm{i}` files are derived from the frozen D7 set by deleting
  exactly one (caption, clip_ids, score) triple; no replacement, no rerank.
- Assertions for 5 queries × 7 removals: **PASS** (count 6, removed item absent,
  relative order preserved, pairing preserved, no new candidate).
- Fusion weights are `softmax(sim_scores)` over the remaining features, so
  renormalization is automatic after removal.

## 5. Oracle / tail candidate pairing
- Oracle1/3/7, GoodTail7, HardWrong7, EasyWrong7, MidTail7, LowTail7, position
  variants: 3,479/3,479 pairs checked per variant, **0 mismatches**
  (`reports/mechanism_pairing_audit.json`).
- Oracle score: per-candidate CIDEr (pycocoevalcap, raw 0–1, no ×100) between a
  candidate caption and the query GT reference set; ties broken by original
  top-20 index. GT used only for candidate selection.

## 6. Changes made
- Added `--text_topn` / `--visual_topn` decoupling (defaults unchanged).
- Added FAITHFUL_V2 extractor for MSR-VTT candidate videos.
- Priority chain now extracts in-domain candidate features before generation and
  runs LOO assertions before LOO generation.

## 6b. Caption→video ambiguity and source redundancy

- 3,340 / 110,031 unique train captions (3.04%) map to >1 train video; max 262.
- Index policy: deterministic sorted-video-ID (first). Recorded, not guessed.
- External variant source redundancy: D7 17.9% queries with a repeated source
  (max 4), Oracle7 11.7%, HardWrong7 15.9%, EasyWrong7 13.7%; kept as covariate.
- Details: `reports/indomain_caption_video_mapping_audit.md`.

## 7. Final status

`READY`

Feature-equivalence numeric table is pending GPU extraction; the matched-control
generation is already restricted to recomputed FAITHFUL_V2 features, so the
audit outcome cannot invalidate the control.
