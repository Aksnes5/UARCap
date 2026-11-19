# UARCap — Utility-Aware Retrieval for Video Captioning

Research workspace for **large-scale external retrieval for video captioning**,
built on top of the EVCap codebase. The original EVCap checkpoint, prompts and
evaluation protocol are used as the fixed captioning baseline; no new captioning
module is trained in this repository.

- Baseline paper / code: EVCap: "Retrieval-augmented captioning with video-text
  features" (Displays 93 (2026) 103420) and its public implementation.
- External corpus: **VideoUFO** (200 tar shards, 1,091,712 clips, 93 parquet
  metadata files, CC BY 4.0), mounted read-only from the 8 TB volume.

## What this repository contains

| Path | Contents |
|---|---|
| `manifests/` | Frozen nested corpus manifests `clips_{10k,50k,100k,250k,500k,full}.tsv`, `manifest_stats.json`, `SHA256SUMS` |
| `scripts/` | Manifest builder, X-CLIP / Video-LLaMA feature extractors, retrieval builders, UAR builders, metric and bootstrap scripts |
| `retrieval/` | Phase-1 top-20 retrieval outputs (raw + exact rescoring) per corpus scale |
| `external/` | Phase-2/3 external-retrieval results: similarity/rerank/top-20 JSON, UAR variant selections, selected clip lists, action-phrase metadata |
| `reports/` | Phase reports, failure taxonomy, oracle diagnostics, paired-statistics JSON |
| `evcap_external/` | Writable copy of EVCap with our patches: external-retrieval hooks, deterministic clip selection, no-retrieval mode, faithful preprocessing extractors, dev300 / val497 dataset configs |
| `PRE_FIX_DIAGNOSTIC/` | Provenance labels for results produced before the FAITHFUL_V2 preprocessing fixes (numbers in `reports/` predating the fix) |

## What is deliberately **not** in the repository

Git history must stay small and free of raw data. Excluded (see `.gitignore`):

- VideoUFO videos and metadata (datasets, `*.mp4`, `*.parquet`)
- X-CLIP / Video-LLaMA / EVCap weights (`*.pt`, `*.pth`, `*.bin`, `*.safetensors`)
- Embeddings and feature caches (`embeddings/`, `features/`, `*.npy`, `*.npz`)
- FAISS indexes (`indexes/`, `*.faiss`, `*.index`)
- Raw frame tables (`*.img.tsv`, 138 MB for dev300 alone)
- SPICE cache, Stanford CoreNLP jars/models, METEOR paraphrase data, third-party
  demo assets and the vendored EgoInstructor repository
- Run logs, checkpoints symlinked from the read-only disk, `__pycache__`

Generated captions (`*_generated_captions.json`) and `metrics.json` **are**
kept: they are small and are the actual experimental record.

## Research phases

1. **Phase 1 — external retrieval.** Audit EVCap, freeze nested VideoUFO
   manifests, extract X-CLIP caption features for 1,091,712 clips, build FAISS
   indexes, retrieve top-20 for 300 dev queries, sanity-check oracle recall.
2. **Phase 2 — scale study.** Retrieval at 10K / 100K / 500K / FULL with the
   same query set; naive external retrieval (CF), quality reranking (D), and
   paired per-query analysis against the paper baseline (B) and no-retrieval (A).
3. **Phase 3 — UAR.** Training-free Utility-Aware Reranking combining
   `S_relevance`, `S_action`, `S_grounding` and redundancy control, with fixed-k
   and adaptive-k variants (E/F/Fv/G/H/H1/H3/H5/H7/I), validated with paired
   bootstrap/permutation tests.
4. **FAITHFUL_V2 — preprocessing fidelity.** Two confirmed deviations were fixed
   before any new claims: MSR-VTT X-CLIP query resize switched
   `INTER_LINEAR → INTER_NEAREST` (official X-CLIP protocol), and VideoUFO
   Video-LLaMA fusion features switched to a fixed faithful temporal sampling
   (`linspace(0, N-1, 16)`). Query features were re-extracted under
   `*_xclip_faithful_v2*`, FULL retrieval recomputed, and CF3/D3/H3 rerun for a
   like-for-like comparison. Results produced before the fix are prefixed
   `PRE_FIX_DIAGNOSTIC` and must not be mixed with FAITHFUL_V2 numbers.

## Reproduction

```bash
# 1. corpus manifests (deterministic: source-hash order, round-robin by topic)
python3 scripts/build_manifests.py

# 2. X-CLIP caption embeddings for the VideoUFO brief captions
/home/bn/anaconda3/envs/videollama2/bin/python scripts/extract_text_embeddings.py

# 3. MSR-VTT query features (FAITHFUL_V2: 16 center frames, INTER_NEAREST)
/home/bn/anaconda3/envs/videollama2/bin/python scripts/extract_query_features_v2.py \
    <test.label.tsv> msrvtt_test_video_xclip_faithful_v2

# 4. FAISS index + retrieval sanity
/media/.../repro_evcap/venv/bin/python scripts/build_faiss_index.py
/home/bn/anaconda3/envs/videollama2/bin/python scripts/retrieval_sanity.py

# 5. external retrieval / UAR json
python3 scripts/make_external_retrieval_v2.py
python3 scripts/build_v2_variants.py

# 6. generation + metrics (needs EVCap checkpoint + Vicuna weights, not in repo)
bash scripts/run_v2.sh CF3v2   # likewise D3v2, H3v2
bash scripts/run_metrics.sh <generated_captions.json> <metrics.json>
```

`extract_msrvtt_query_features.py` must import torch before cv2 (OpenMP
conflict otherwise: cv2-before-torch segfaults at `model.cuda()`).

### Rebuilding the large manifests

`manifests/*.tsv` are generated from `VideoUFO.csv` plus the frozen manifest
builder: source-hash ordering, round-robin within topic, proportional topic
allocation, strict nesting. `manifests/manifest_stats.json` records row counts,
topic distribution and nesting checks; `manifests/SHA256SUMS` pins the exact
files. If a checkout ever lacks the large TSVs, rerunning `build_manifests.py`
on the same `VideoUFO.csv` reproduces them byte-identically (verified by
`SHA256SUMS`).

## Notes on retrieval protocol

EVCap §3.3 retrieves **captions** with X-CLIP (video-to-text); corpus videos are
never embedded for retrieval. The 1,091,712 VideoUFO caption features are the
faithful retrieval artifact. Corpus video features (Video-LLaMA 32×768) are only
needed for cross-gating fusion of the top-K retrieved clips and are computed on
demand under `features/`.

The paper reports MSR-VTT 86.69 / 50.11 / 32.47 / 66.64 / 72.20 and MSVD
91.53 / 69.67 / 47.04 / 84.04 / 158.32 (B@1/B@4/M/R/CIDEr, ×100).

## Status

Phase 1–2 complete; Phase 3 UAR complete on PRE_FIX features; FAITHFUL_V2
preprocessing fixes applied and the CF3/D3/H3 comparison rerun in progress.
Numbers under `reports/` that predate the fix are explicitly marked
`PRE_FIX_DIAGNOSTIC` and are not paper claims.
