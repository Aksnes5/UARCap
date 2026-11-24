# BN asset gaps and blockers

Checked at **2026-11-24T16:10:00**. Status per item: `VERIFIED`, `UNVERIFIED`, `MISSING`, `MISMATCH`,
`NOT_CHECKED`.

## Resolved (no blocker)

- MSR-VTT test2990 roles: 2,990 rows in image/label/caption TSVs, exact ID order, 20 references per
  query, canonical reference digest `204563f74d9e4354863289760f65a767dde8d9e189d7e99dc201505847b15271`.
- All three TSV offset files exist, are monotonic and stay inside their TSVs; a 106-row frame sample
  has 18 fields and non-empty frame payloads.
- Manifests: `msrvtt_test2990.tsv` 2990 rows and `msrvtt_untouched2690.tsv`
  2690 rows; SHA-256 values match the frozen archive.
- Checkpoint and three historical slots: SHA-256
  `a8b5af79d55a2c7e684771475d645be6373208bdaae5a4211bee556581eb8002`.
- Vicuna-7B-v1.3: config, tokenizer, index and two shards present; shard list equals the index.
- Video-LLaMA Q-Former, EVA-ViT-G and BLIP2 Q-Former caches present and readable.
- Frozen D cap/sims and top-20 retrieval files: all SHA-256 values match.
- A1 feature coverage: 2,539 unique clips, canonical ID digest matches, 0 missing.
- A7 feature coverage: 15,213 unique clips, canonical ID digest matches, 0 missing.
- Six frozen source files and nine runtime dependency files match their frozen SHA-256 values.
- GPU idle: 0% utilization, 15 MiB used, no compute client; GPU lock unheld.

## Blockers / deviations

| ID | Severity | Item | Status | Consequence |
| --- | --- | --- | --- | --- |
| G1 | P0 | GPU execution authorization | **NO** (policy) | No smoke, full generation, scoring or model load is permitted. |
| G2 | P1 | Live environment vs archived receipt | **MISMATCH** | Python `3.9.13` vs `3.9.18`; PyTorch `2.1.2+cu121` vs `2.1.0`. The venv is a hybrid of the TTP Conda base and the Display venv. Freeze the actual runtime hash set and obtain an explicit deviation decision before execution. |
| G3 | P0 | Historical per-job runtime trace | **MISSING** | D1/D3/D7 remain conditional analogs; exact reuse stays 0. Re-run the v0.6 arms rather than reusing historical outputs. |
| G4 | P0 | A0 prepared inputs (`A_k0_cap.json`, `A_k0_sims.json`) | **MISSING** | Must be generated CPU-side from the frozen empty-array contract before the A0 smoke. |
| G5 | P0 | B prepared inputs (`B_text3_cap`, `B_text7_hq_cap`, `B_text7_lq_cap`, `B_shared_visual_sims`, `B_visual_contract`, `B_eligible_queries`) | **MISSING** | Deterministically regenerable from verified top-20/D files on CPU, but not staged. B cannot start. |
| G6 | P0 | Real-model parity / A smoke / B smoke | **NOT_CHECKED** | No model was loaded. B formal execution remains NO-GO until A full and B smoke pass. |
| G7 | P2 | Display disk mount | **VERIFIED (read-only)** | Reads are fine; all new outputs must go to the read-write project filesystem, never Display. |
| G8 | P2 | Image TSV byte-level digest | `NOT_CHECKED` | 1.38 GB `test_16frames_img_size224.img.tsv` was pinned by metadata, lineidx SHA-256 and a row sample. `deploy_guard.py` intentionally uses mtime for this role. |
| G9 | P2 | Main project working tree | 107 dirty entries | Do not deploy from the dirty main checkout; use a dedicated clean worktree. |
| G10 | P3 | BERT HF cache | tokenizer/config only | No BERT weight file is cached; historical runs used the cached tokenizer and BLIP2 Q-Former. Confirm at smoke time. |

## Not inferred

No asset was marked missing merely because it is absent from Git. Each status above comes from a
live filesystem probe. No large model weight, video, feature tensor or TSV was copied into Git.
