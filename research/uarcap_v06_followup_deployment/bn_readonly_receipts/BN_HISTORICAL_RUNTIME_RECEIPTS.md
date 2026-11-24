# BN historical runtime receipts (D1 / D3 / D7)

Checked at **2026-11-24T16:10:00**. Read-only reconstruction from the live filesystem; no generation was run.

## Prediction bundles

| Bundle | Rows | Manifest order | Prediction SHA-256 | CIDEr |
| --- | ---: | :---: | --- | ---: |
| test2990_D1 | 2990 | True | `dd4c8ea536852be9efa3135e7799bf76b629610e96a61a17465588a64a17b865` | 0.703108 |
| test2990_D3 | 2990 | True | `183240102470116be7c2a648704edfce0bedc1d61bf20ef0d894623c30f7164b` | 0.694052 |
| test2990_D7 | 2990 | True | `8aa52ffa5d7de3e287a1e7b872b8279965e6df2cd430d46b07aa34f1a4af89fc` | 0.674569 |
| test2690_D1 | 2690 | True | `e11ea796544ac4bab0cf4a87a979338a30bb29383737f0f9b1ee34130d1eec26` | 0.712085 |
| test2690_D3 | 2690 | True | `a6e61c52c0b08bccc63f62b57f1fa142562fb464f0bf967ff1c1f115f7dbd1cd` | 0.700757 |
| test2690_D7 | 2690 | True | `05671e09d411e12382b2a52fdf024b252134de4e29796bced6cd0b1da2cc4838` | 0.682740 |

All six prediction files are byte-identical to the frozen values in
`research/uarcap_v06_followup_deployment/ARCHIVE_EVIDENCE.json`, with matching byte sizes.
The 2,990-row bundles follow `manifests/msrvtt_test2990.tsv` order; the 2,690-row bundles follow
`manifests/msrvtt_untouched2690.tsv` order. Reference lists use the canonical ordered digest
`204563f74d9e4354863289760f65a767dde8d9e189d7e99dc201505847b15271`.

## Command provenance

`scripts/run_test.sh` SHA-256 is `42d7d58e534bfcb798f01fc99ee0a10d70e600aa41af05431848ff7d49b2c88f`
(expected `42d7d58e534bfcb798f01fc99ee0a10d70e600aa41af05431848ff7d49b2c88f`). The generation logs record:

| Arm | `topn` | `test_yaml` | `ext_path` | `sim_scores_path` | `video_feature_dir` | seed / frames |
| --- | ---: | --- | --- | --- | --- | --- |
| D1 | 1 | `test2990/test2990_16frames.yaml` | `/home/bn/videoufo_research/external/external_test_d7_cap.json` | `/home/bn/videoufo_research/external/external_test_d7_sims.json` | `/home/bn/videoufo_research/features/videollama_faithful_v2` | 42 / 16 |
| D3 | 3 | `test2990/test2990_16frames.yaml` | `/home/bn/videoufo_research/external/external_test_d7_cap.json` | `/home/bn/videoufo_research/external/external_test_d7_sims.json` | `/home/bn/videoufo_research/features/videollama_faithful_v2` | 42 / 16 |
| D7 | 7 | `test2990/test2990_16frames.yaml` | `/home/bn/videoufo_research/external/external_test_d7_cap.json` | `/home/bn/videoufo_research/external/external_test_d7_sims.json` | `/home/bn/videoufo_research/features/videollama_faithful_v2` | 42 / 16 |

| Log | Bytes | SHA-256 |
| --- | ---: | --- |
| `logs/gen_test_D1.log` | 3511065 | `e8daf1204f4d6775a01ec28db0442ffcb7e3a5bab09c32e26b952d330a606178` |
| `logs/gen_test_D3.log` | 3514922 | `acd5284001d83576ecccbbd577966250cf341ed88b19a037df8ca5f32361db76` |
| `logs/gen_test_D7.log` | 3525653 | `24473c05193df6808d6391805528431429a2b7a5dd1c7021e293a9e5097fefde` |

Each log also names the per-run checkpoint slot (`outputs/train_test2990_D{k}/...pt`) and the
Vicuna path `/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/models/vicuna-7b-v1.3`. All three slots are symlinks to one canonical
checkpoint with SHA-256 `a8b5af79d55a2c7e684771475d645be6373208bdaae5a4211bee556581eb8002` (147,337,150 bytes), verified again during
this audit.

## Runtime trace and environment

- **Execution-bound runtime trace: `MISSING`.** `ARCHIVE_EVIDENCE.json` states that no historical
  query frame/feature/prompt/weight trace exists. Therefore D1/D3/D7 remain
  **conditionally reusable only**; exact-reuse exemptions remain **0**.
- **Per-job environment receipt: `UNVERIFIED`.** No historical `pip freeze` or interpreter receipt
  was written. The live interpreter that the archived runner points to is recorded in
  `BN_LIVE_ENVIRONMENT.md`; its Python/PyTorch versions differ from the archived expectation.
- Source hashes for the six frozen files and all nine runtime dependencies match the frozen
  bindings (see `BN_DEPLOYMENT_PATH_BINDINGS.json`).

## Verdict

Prediction bytes, row counts, order, metrics and command arguments are **VERIFIED**. Runtime
provenance is **PARTIAL**: no per-job trace or environment snapshot. The historical bundles are
evidence and comparison receipts, not exact-reuse substitutes for the v0.6 arms.
