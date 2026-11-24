# GPU resource estimate

## Generation budget

The pinned `reports/runtime_ledger.csv` records 8,521 s for historical test2990_D7 (2.37 GPU-hours). The nonzero 497-query historical entries span 834–2,777 s, or roughly 1.68–5.59 s/query. They inform a planning range across different run settings. All formal arms generate the same 2,990 rows in frozen order, including rows that are excluded from the primary scoring cohort.

| Required work | Arms | Queries | Rate-based range | D7 direct anchor |
| --- | ---: | ---: | ---: | ---: |
| One full arm | 1 | 2,990 | 1.39–4.64 h | 2.37 h |
| Full A or B | 3 | 8,970 | 4.2–13.9 h | 7.10 h |
| Both full families before or after current reuse decision | 6 | 17,940 | 8.4–27.9 h | 14.20 h |
| Counterfactual if all three historical analogs later satisfy strict reuse | 3 | 8,970 | 4.2–13.9 h | 7.10 h |
| Required smoke, eight queries per arm | 6 starts | 48 | 81–268 s of generation | 137 s of generation |

The 8.4 h lower planning estimate excludes smoke startup, model loading, feature cache I/O differences and trace overhead. There are six real smoke starts on the current matrix; each loads the model and checks the same prefix within its family. Neither initialization nor BN GPU throughput has been measured here. Completed arms may be kept across approved sessions if receipts, context and hashes still match the logical run and family checks.

## Trace overhead and peak memory

The trace copies fingerprints of each query frame (`[16,3,224,224]` float32, about 9.2 MiB), Q-Former input and mask groups (e.g. `[16,257,1408]` image encoder states, about 11 MiB fp16 or 22 MiB fp32), each visual feature actually used, intermediate fusion tensors, and full prompt embeddings. It then stores small digests and metadata only. A CPU-only check of five similarly shaped frame/Q-Former tensors and one Torch CPU thread measured ~42 ms median per full fingerprint pass after warm-up. This includes CPU finite checks and SHA-256, with no GPU synchronization or device transfer measurement. The frozen hook and overlay together do additional work. BN smoke must report measured wall time, trace volume and peak CUDA allocated/reserved bytes. Current runtime telemetry is aggregate.

Keep exact per-query digests for the signatures needed by B. Sample verbose diagnostic logs if needed. Hashing only a sampled subset of B queries would change the acceptance criteria. Raw tensors normally stay transient and are excluded from disk logs. A7's seven `[32,768]` cached feature tensors occupy ~0.33 MiB at fp16 or ~0.66 MiB at fp32 per stack, exclusive of intermediates. Vicuna, fixed encoders/fusion, CUDA workspace and beam-generation state dominate device memory. B7 may need more decoder memory with a longer prompt. BN free VRAM and the total peak have not been measured.

## Disk and interruptions

The three archived full prediction files are 3.56–3.57 MB each. At roughly 5–12 KiB of trace per query as a planning envelope, 17,940 queries would create roughly 90–220 MB trace, plus ~22 MB at archived prediction sizes and launcher logs. Reserve at least 500 MB for new outputs and one failed-arm attempt, separate from existing assets, checkout and retained repeated attempts. Confirm free space and actual byte rates at smoke. No new frame corpus, index, checkpoint copy or visual cache rebuild is needed. Retain separate logs, telemetry and hashes for failures. A completed and sealed arm can be reused inside its logical execution cohort after validating all receipts.
