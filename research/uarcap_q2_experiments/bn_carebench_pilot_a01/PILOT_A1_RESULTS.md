# Pilot A1 — CaReBench E2 real BN execution results

**Date:** 2026-11-26 (UTC+8) · **Verdict:** `PASS` (`REAL_RUNTIME_PASS`)
· **Run ID:** `pilot_a1_carebench_v1`

## Summary

| Item | Value |
| --- | --- |
| Dataset | CaReBench (E2 primary), 1,000 records / 913 FineAction parents |
| Video archive | 3,918,994,506 B, SHA-256 `957706c2…203` (HF LFS = ModelScope) |
| Frozen sample | 30 unique parents / 30 subcategories, seed 20261125 |
| Sampling manifest SHA-256 | `26fbf6abaf7ed12309ca5be40ccc583c068e5bd2c9b427f0753c4600d5b5bd21` |
| Model | frozen Video-LLaMA/EVCap `videollama_evcap_frozen_v08` |
| Checkpoint SHA-256 | `a8b5af79d55a2c7e684771475d645be6373208bdaae5a4211bee556581eb8002` |
| Retriever | VideoUFO X-CLIP (`videoufo_xclip_faithful_v2_subprocess`), top-20 → D rerank |
| Arms | `ARM_A0`, `ARM_A1`, `ARM_A7` |
| Generations | **90 / 90**, 0 failed, 0 empty (rate 0.0000) |
| Wall time | 467.3 s total (A0 111.0 s, A1 115.6 s, A7 143.2 s) |
| Peak VRAM | 17.82 GB (per-arm maxima 15.75 / 15.84 / 16.60 GiB) |
| GPU lock | cooperative lock held; no foreign GPU processes |

## Integrity (independent audit, `AUDIT_PILOT_A1.md`)

All checks passed:

- 90 unique predictions, all non-empty, 30 × 3 arms.
- Prediction ↔ actual-input-trace 1:1 SHA-256 linkage.
- `ARM_A0` strictly zero text/visual external evidence (also verified by the
  forward-hook `fused_nonzero == 0`).
- `ARM_A1` evidence is an exact prefix of `ARM_A7` (sizes 1 and 7); scores match
  on the shared prefix.
- Same per-query visual input SHA-256 across all three arms; 16 frame indices.
- Retrieval evidence: 30/30 queries, top-20 original + 20 D-reranked, A7
  representatives equal the top-7 reranked candidates; `rerank_rank` is a
  contiguous permutation.
- Frozen manifest query IDs equal the run's `query_manifest.json`.

## Pre-run data work

- 376 of the top-20 representative VideoUFO clips had no cached `faithful_v2`
  Video-LLaMA visual feature. They were extracted with
  `/home/bn/videoufo_research/scripts/extract_external_features_v2.py`
  (same pipeline as the existing 45,677-clip cache); all are `(32, 768)` and
  finite. Retrieval ranking was **not** altered (no candidate filtering).

## Artifacts

Archived under `pilot_a1_run_artifacts/` (also live at
`/home/bn/uarcap_q2_runtime/outputs/pilot_a1/pilot_a1_carebench_v1/`):

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `predictions.jsonl` | 200,978 | `f958c87c83d31b7f28468e6637c22a3f45df338f36fc94f36c317f034a119428` |
| `input_traces.jsonl` | 174,546 | `cbf789fe4261f0431d4949c398eba424036c8c2d01400210112ef8dca2bbd133` |
| `retrieval_candidates.jsonl` | 305,073 | `14f29dde5ae9f5484b5ae404946b42a695e468eaf24821df34d32aa0dc86c2dd` |
| `run_receipt.json` | 1,159 | `e792d35ba5b9d3e8a45d5b7278a6d3542961426797a8be906f3e19b0b5703ae3` |
| `PILOT_A1_GO_NO_GO.md` | 202 | `591bd69d54bd4a4cbf3957ec47bf41953bf6b87063c794bcecd9ad2f42bbf4ec` |

## Observations / next steps

- A0 and A1 captions are frequently identical for short clips; this pilot tests
  runtime compatibility and trace integrity, not effect size. Pilot A2 (human
  alignment / core-event audit) is the next gate and requires its own
  authorization receipt (`PILOT_A2`); A3 and Formal B must not start from this
  state.
- The shared feature cache is now additive by 376 files; record this delta when
  re-freezing historical assets.
