# 50K / 250K scale-points — preflight (2026-11-20)

Frozen nested manifests are used as-is (no new subsets).

| scale | manifest | sha256 (16) | retrieval | rerank | candidate union | feature cache | missing | generation |
|---|---|---|---|---|---|---|---|---|
| 50K | manifests/clips_50k.tsv (1.64 MB) | b89fe74a52e23b1f | **absent** (runner ready) | absent | computable after retrieval | audit script ready | — | 0/4 |
| 250K | manifests/clips_250k.tsv (8.21 MB) | 77ad43769d75fd5e | **absent** (runner ready) | absent | computable after retrieval | audit script ready | — | 0/4 |

No 50K/250K video features exist or will be extracted: only the D1/3/5/7
candidate union needs faithful Video-LLaMA features, and only after retrieval.

## Prepared artifacts

- `scripts/make_external_retrieval_val_scale.py <scale>` — caption-corpus
  retrieval against the frozen manifest, same query features / X-CLIP model.
- `scripts/build_val_grid_scale.py <scale>` — CF7/D7 cap+sim grids.
- `scripts/audit_candidate_features.py <prefix> <D|CF>` — unique candidate
  clips, cache hit rate, missing list (`/tmp/missing_<prefix>_<track>.txt`).
- `scripts/scale_track_chain.sh <scale> <D|CF> [ks]` — resume-safe chain:
  retrieval → grid → cache audit → missing-feature extraction → generation →
  metrics → retrieval metrics.
- `scripts/run_remaining_gpu_queue.sh` — global orchestrator (state markers in
  `logs/queue_state/`).

## Runner dry-runs completed

- Cache-audit on 10K CF: 1,796 unique clips, 100% cache hit.
- Adaptive routing on test2990 sims: mean k 2.356
  (k1 1485 / k3 1068 / k5 352 / k7 85); routing equality verified for all
  queries whose selected k ∈ {1,3,7} (2,638) against stored outputs.
- MSVD static preflight: checkpoint 147,336,938 B, 670 test rows, 1,970 avi,
  670-row frame TSV, yaml and feature dir present, runner/eval external args
  present (`reports/msvd_preflight.json`).

## GPU queue (auto)

D5 validation → MSVD test670 → 50K D-grid → 250K D-grid → CF grids
(10K, 100K, 500K) → 50K/250K CF if GPU remains. Each stage has a completion
marker and is skipped on re-run; D5 integrity failure halts the queue.
