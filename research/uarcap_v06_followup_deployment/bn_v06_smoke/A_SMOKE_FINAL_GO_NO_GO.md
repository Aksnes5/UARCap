# UARCap v0.6 — A smoke final GO / NO-GO

**Executed:** 2026-11-24 20:03–20:08 on BN RTX A6000  
**Authorization:** `USER_APPROVED_A_SMOKE_20261124`  
**Scope:** A0, A1, A7 × first 8 frozen queries = 24 generations.

| Gate | Result | Evidence |
| --- | --- | --- |
| `A0_SMOKE` | **PASS (8/8)** | 8 rows, zero external text/features/softmax, zero aggregate |
| `A1_SMOKE` | **PASS (8/8)** | 8 rows, 1 text/feature, 1 softmax weight |
| `A7_SMOKE` | **PASS (8/8)** | 8 rows, 7 texts/features, 7 softmax weights |
| `MODEL_LOADED` | **YES** | checkpoint shards loaded in all three logs |
| `CUDA_INITIALIZED` | **YES (child processes)** | CUDA 12.1, driver 580.178.04, per-query CUDA RNG signatures |
| `BN_VALIDATOR` | **PASS** | `validate_bundle.py --family A --mode smoke`, `trace_parity=true` |
| `WINDOWS_VALIDATOR` | **PENDING** | Windows has not run the official gate |
| `A_FULL_READY` | **PENDING** | waiting for Windows gate acceptance and full-run authorization |
| `A_FULL_EXECUTION_APPROVED` | **NO** | not authorized |
| `B_SMOKE_READY` | **NO** | A full gate, B smoke authorization, visual parity gate all outstanding |
| `B_EXECUTION` | **NOT RUN** | out of scope |
| `GPU_IDLE_AFTER` | **YES** | 15 MiB used, 0% util, no compute client, lock free |
| `P0_BLOCKERS` | **NONE for this A smoke** | no failure in the executed scope |

## What passed

- **24 real model generations** completed.
- A0: 0 text evidence, 0 feature loads, 0 softmax weights, zero external aggregate (`3a3ed164…e6e4`), attention K/V equal to zero aggregate.
- A0 did not fall back to k=7; launcher log records `topn=0`, `text_topn=0`, `visual_topn=0`.
- A0 used the frozen query video, two `<VideoHere>` prompt markers and `retriever=True`, `use_fusion=True`, `no_retrieval=False`.
- Frame digests, image/video Q-Former inputs, Q-Former query, query projection, visual mask, module-training state and CPU/CUDA RNG signatures are identical per query across A0/A1/A7.
- Prompt template hash is identical across arms; text counts are 0/1/7 as designed.
- Dtypes: frames `float32`, wrapped prompt `float16`, prompt mask `int32` all ones, visual mask `int64`, aggregate `float32`.
- No NaN, non-finite tensor, shape error, missing feature or truncated trace.

## What is not yet true

- Windows official validation is **not complete**.
- A full 2,990-query generation is **not authorized** and was not run.
- B smoke/full is **not authorized** and was not run.
- No scientific scoring or significance testing was performed on BN.

## Decision

`A_SMOKE_BN_PASS = YES`  
`A_FULL_READY = PENDING`  
`A_FULL_EXECUTION_APPROVED = NO`  
`B_EXECUTION = NO-GO (not authorized; A full gate absent)`

The correct next step is for Windows to fetch the committed raw bundles and run:

```text
validate_bundle.py --family A --mode smoke \
  --arm-directories <A0/A1/A7 raw directories> \
  --archive-scratch <Windows frozen replay scratch> \
  --gate-out <Windows A smoke gate>
```

Only after that gate is accepted may an A full execution authorization be requested.
