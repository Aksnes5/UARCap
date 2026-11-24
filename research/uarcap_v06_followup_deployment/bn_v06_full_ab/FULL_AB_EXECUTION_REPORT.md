# UARCap v0.6 — Full A/B GPU execution report

**Executed on BN:** 2026-11-24 20:43 through 2026-11-25 07:53 local  
**Authorization:** `USER_APPROVED_ALL_AB_FOLLOWUP_20261124`  
**Scope:** A0/A1/A7 full 2,990 queries; B3/B7HQ/B7LQ 8-query smoke; B3/B7HQ/B7LQ full 2,990 queries.

All runs were serial on the single RTX A6000. No other GPU job was started, preempted or stopped. No training, indexing or protocol change was performed.

## A full results

| Arm | Rows | Wall time | Peak allocated | Prediction SHA-256 | Trace SHA-256 | Raw ZIP SHA-256 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| A0 | 2,990 | 71.5 min | 16.98 GB | `8c003aea…b66d8` | `f05a055d…6f5a` | `708c7250…aa431` |
| A1 | 2,990 | 77.0 min | 17.12 GB | `86d336b8…11d6a` | `94910eb6…84be0` | `04d5b317…0bb49` |
| A7 | 2,990 | 143.4 min | 18.71 GB | `fa7d5205…4fd5` | `8f02e05e…41cc` | `33bc167b…457456` |

- **8,970** A generations completed.
- `A_FULL_VALIDATED_GATE.json`: **PASS**, `trace_parity=true`, `rows_per_arm=2990`.
- A0 full check: 2,990/2,990 queries with zero external text, zero feature loads, no visual softmax, zero external aggregate, and attention K/V equal to the zero aggregate.
- A0/A1/A7 full parity: 2,990/2,990 identical query frames, image/video Q-Former inputs, Q-Former query, query projection, visual mask, module-training state, CPU RNG and CUDA RNG. Gating inputs differ only because external visual evidence differs by design.

## B smoke and B full

| Arm | Mode | Rows | Wall time | Peak allocated | Prediction SHA-256 | Trace SHA-256 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| B3 | smoke | 8 | 82.5 s | — | `e2d58a5b…e91f` | `6831ba3f…42fd` |
| B7HQ | smoke | 8 | 85.9 s | — | `077c1af8…ed2c` | `b7fa351e…eea` |
| B7LQ | smoke | 8 | 86.6 s | — | `6e5646e6…d030` | `a1b2f8e6…a08d` |
| B3 | full | 2,990 | 99.8 min | 17.59 GB | `4814a153…50` | `b96cf5c7…c6` |
| B7HQ | full | 2,990 | 137.1 min | 18.43 GB | `de7453b5…75` | `3aaf8556…c9` |
| B7LQ | full | 2,990 | 136.2 min | 18.44 GB | `b9f21942…47` | `11671b21…e9` |

- B smoke: **24** generations; `B_SMOKE_VALIDATED_GATE.json` **PASS**, `trace_parity=true`, `rows_per_arm=8`.
- B full: **8,970** generations; `B_FULL_VALIDATED_GATE.json` **PASS**, `trace_parity=true`, `rows_per_arm=2990`.
- The B gates verify per-query visual parity across B3/B7HQ/B7LQ: query frames, clip IDs, loaded feature tensors, raw similarities, softmax weights, visual aggregate, attention K/V, gating inputs, gated visual projection, module state and RNG. Only the designed text evidence and prompt/token changes are allowed.
- B full still evaluates only the frozen **2,618** eligible queries; all 2,990 rows and traces were retained, including fallback and pilot rows.

## Gate hashes

| Gate | SHA-256 |
| --- | --- |
| A full | `bef8ce36341a4ae6712525e1777d76940cb5a0ffd05ded3f004d55c8e9414ec1` |
| B smoke | `acd95c7aadcf7be588ab0f1bf4b03cff4f3c64342f8ab0b08a5da257c31e4408` |
| B full | `f9255d9cfe884ec26c89accaaa2031802fdbb92f84dd25a71557008b9975c01e` |
| Accepted A smoke independent gate | `b07ad358b4efa1ab3d64b3bad5273b27445f1f4c7c7a7f3bf0c99e7b806a9082` |

## Totals

| Metric | Value |
| --- | ---: |
| Full generations (six arms) | **17,940** |
| B smoke generations | **24** |
| This run total | **17,964** |
| Full GPU hours | **11.0846 h** |
| B smoke GPU hours | **0.0708 h** |
| This run GPU hours | **11.1554 h** |
| Prior A smoke GPU hours | 0.0691 h |
| All A/B GPU hours including prior A smoke | **11.2245 h** |
| Peak allocated VRAM | **18.71 GB** (A7) |
| Failures | **0** |
| Retries | **0** |

TensorFlow emitted non-fatal oneDNN/cuDNN/cuFFT/cuBLAS/TF-TRT startup warnings. No traceback, OOM, missing feature, NaN/non-finite tensor or incomplete trace occurred.

## Runtime

- Python 3.9.13
- PyTorch 2.1.2+cu121
- CUDA runtime 12.1
- Transformers 4.28.0
- Driver 580.178.04
- GPU UUID `GPU-66d83fcc-2695-33a7-33aa-af30b7e3d26d`

The known package-version deviations from the archive remain documented. No environment modification was made.

## Evidence delivered

- Full raw per-arm ZIPs: `raw_zips/A0.zip`, `A1.zip`, `A7.zip`, `B3.zip`, `B7HQ.zip`, `B7LQ.zip`.
- Each ZIP contains the complete prediction JSON, `input_trace.jsonl`, `run_receipt.json`, `runtime_telemetry.json`, `arm_completed.json` and `launcher.log`.
- B smoke raw files: `b_smoke_raw/{B3,B7HQ,B7LQ}/`.
- Gates and configs: `gates/`, `configs/`.
- Runtime receipts/status: `FULL_AB_RUNTIME_RECEIPTS.json`, `FULL_AB_GATES.json`, `FULL_AB_FINAL_STATUS.json`.
- Scorer manifests: `SCORER_MANIFEST_A.json`, `SCORER_MANIFEST_B.json`, `FULL_AB_SCORER_MANIFEST.json`.
- Runner evidence: `runner_status.json`, `runner_events.jsonl`, `runner.log`.

Formal CIDEr, source-cluster bootstrap, sign-flip and Holm analysis were **not** run on BN. Windows can start the frozen statistical stage directly from the delivered predictions, traces, receipts and scorer manifest.
