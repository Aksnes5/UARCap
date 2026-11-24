# UARCap v0.6 — Experiment A GPU smoke execution report

**Executed on BN:** 2026-11-24, 20:03–20:08 local  
**Authorization:** `USER_APPROVED_A_SMOKE_20261124`  
**Scope:** A0, A1, A7; first **8** frozen test queries each; **24 generations total**  
**Not authorized:** A full 2,990-query run, B smoke, B full, training, indexing.

## Deployment

| Item | Value |
| --- | --- |
| Deployment commit | `f69ab3d090f50c02c58d81c1480b5f5b69e9e83f` |
| Authorized config | `/home/bn/uar_v06_runtime/deployment/deployment_uarcapv06_A_smoke_authorized.json` |
| Config SHA-256 | `a08c18ab86e5622c43121449a9f0ef396b8e25f10a721c797150c89ae03d91a9` |
| Isolated root | `/home/bn/uar_v06_runtime/evcap_work` |
| Inputs | `/home/bn/uar_v06_runtime/inputs` |
| GPU | RTX A6000, UUID `GPU-66d83fcc-2695-33a7-33aa-af30b7e3d26d` |
| GPU lock | acquired by `deploy_guard.py` for each arm; released after each process |

The authorized config adds only `approval_id=USER_APPROVED_A_SMOKE_20261124` to the already-validated deployment; `server_data_ready` remains `true`. No frozen code or input was modified.

## Command pattern

```bash
python -B deploy_guard.py \
  --evcap-root /home/bn/uar_v06_runtime/evcap_work \
  --deployment .../deployment_uarcapv06_A_smoke_authorized.json \
  --archive-scratch /home/bn/uar_v06_runtime/inputs \
  --arm A0 \
  --mode smoke \
  --gpu-approval-id USER_APPROVED_A_SMOKE_20261124
```

The same command was run serially for `A1` and `A7`, in that order. No arm was run concurrently.

## Results

| Arm | Rows | Wall time | Peak allocated | Peak reserved | Prediction SHA-256 | Trace SHA-256 | Exit |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| A0 | 8 | 80.05 s | 16,883,267,584 B | 17,425,235,968 B | `f3ba8881…c5ba4` | `3e15a6d8…fbba1` | 0 |
| A1 | 8 | 79.96 s | 16,982,871,552 B | 17,548,967,936 B | `b02f1d0b…0ecb` | `d403ec90…251b` | 0 |
| A7 | 8 | 88.87 s | 17,841,610,240 B | 19,121,831,936 B | `490193fc…07c9c` | `b64e1601…b6f98` | 0 |

All 24 predicted captions are non-empty. Prediction row counts and query IDs match the first eight entries of `manifests/msrvtt_test2990.tsv`.

## Model and runtime

The three `launcher.log` files contain the expected checkpoint load and two-shard load sequence:

```text
load: outputs/train_v06_smoke_uarcapv06_attempt1_<arm>/<checkpoint>.pt
Loading checkpoint shards: 100%|██████████| 2/2
```

| Runtime field | Value |
| --- | --- |
| Python | 3.9.13 |
| PyTorch | 2.1.2+cu121 |
| CUDA runtime | 12.1 |
| Transformers | 4.28.0 |
| Driver | 580.178.04 |
| GPU UUID | `GPU-66d83fcc-2695-33a7-33aa-af30b7e3d26d` |

The model loaded successfully in all three processes. TensorFlow emitted startup warnings about cuDNN/cuFFT/cuBLAS factories and TF-TRT; there was no traceback and all evaluator processes exited 0. This is recorded as an observation, not a smoke failure.

## A0 zero-evidence checks

| Check | Result |
| --- | --- |
| `text_count` for all 8 queries | 0 |
| External feature tensors loaded | 0 for all 8 |
| Softmax visual weighting | `None` for all 8 |
| External visual aggregate | zero tensor, all 8 |
| Aggregate SHA-256 | `3a3ed164e42500a1c5b2d0093f0a813d27dc50d038f330cc100a7e70ece2e6e4` |
| Attention K/V | equal to the zero aggregate |
| `topn` / `text_topn` / `visual_topn` in launcher log | 0 / 0 / 0 |
| `retriever` / `use_fusion` / `no_retrieval` | `True` / `True` / `False` |
| Legacy `no_retrieval` branch | not used |
| Query video path | retained |
| `<VideoHere>` slots | prompt file contains two markers; frozen prompt hash unchanged |
| Final fusion projection | allowed to be non-zero; only the external visual aggregate is pinned to zero |

## Cross-arm parity

The following are identical per query across A0/A1/A7:

- query IDs and order
- query frame tensor digest: `b88d2033…79ee` at query 0
- image Q-Former inputs
- video Q-Former inputs
- Q-Former query tensor
- query projection
- visual mask, shape `[1, 32]`, `torch.int64`
- module-training flags
- CPU RNG signature
- CUDA RNG signature
- prompt template SHA-256: `4719acfb…d2ec`

Expected differences only:

- `text_count`: 0 / 1 / 7
- loaded visual features: 0 / 1 / 7
- visual softmax: none / 1 / 7 weights
- external visual aggregate: zero for A0; non-zero for A1/A7
- prompt token length: A0 100; A1 108–121; A7 180–259
- gating inputs: differ across A arms because external visual evidence differs by design; A-family parity does not pin them. They must be checked for B, which was not run.

## Dtypes and masks

| Tensor | dtype |
| --- | --- |
| Query frames | `torch.float32` |
| Wrapped prompt embeddings | `torch.float16` |
| Prompt attention mask | `torch.int32`, all ones |
| Visual mask | `torch.int64` |
| External aggregate | `torch.float32` |

No NaN/non-finite tensor was observed; the trace layer rejects non-finite floating tensors.

## Validation

- BN-side frozen `validate_bundle.py --family A --mode smoke` was executed against the complete raw bundles and returned **PASS** with `trace_parity=true`.
- BN gate: `A_SMOKE_BN_GATE.json`.
- **Windows gate: `WINDOWS_GATE_PENDING`.** The official Windows validation has not been run. BN-side PASS is not a substitute.

## Raw artifacts

Complete raw files are committed under `bn_v06_smoke/raw/{A0,A1,A7}/`:

- `*generated_captions.json`
- `input_trace.jsonl`
- `run_receipt.json`
- `runtime_telemetry.json`
- `arm_completed.json`
- `launcher.log`

Their sizes and SHA-256 values are in `A_SMOKE_ARTIFACT_MANIFEST.json`.
