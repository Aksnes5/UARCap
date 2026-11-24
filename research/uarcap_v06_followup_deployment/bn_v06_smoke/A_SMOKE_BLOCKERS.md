# UARCap v0.6 — A smoke blockers

**Status: no P0 blocker for the executed A smoke.**

## Resolved / pass

- GPU UUID matched the deployment config; no competing compute process; shared lock was free before launch.
- Checkpoint SHA-256 matched `a8b5af79…eb8002`; all six frozen source files and nine runtime dependencies matched.
- All 8 input/feature/prepared objects matched the transfer receipt.
- A0, A1 and A7 each ran exactly 8 queries.
- Model loaded and generated in all three arms.
- A0 external text count = 0, feature loads = 0, softmax = none, external aggregate = zero tensor.
- A0 retained the query video path, two `<VideoHere>` slots, original fusion/gating path and `no_retrieval=False`.
- Frame, image/video Q-Former input, qformer query, query projection, visual mask and RNG parity passed.
- No NaN, non-finite tensor, mask-shape error, missing feature or truncated trace.
- BN-side frozen `validate_bundle.py --family A --mode smoke` passed.

## P1 / observations

| ID | Item | Effect | Action |
| --- | --- | --- | --- |
| O1 | TensorFlow startup warnings (cuDNN/cuFFT/cuBLAS factory already registered; TF-TRT missing) | Noise only; all processes exited 0 and generated 8 captions | Recorded; no change made |
| O2 | `video_qformer.bert.training=True` and `attention.training=True` | The frozen model retains Q-Former dropout; this is the inherited module state, not a new change | Preserved; do not force `eval()` |
| O3 | Environment versions differ from archive (Python 3.9.13 / torch 2.1.2+cu121 / torchvision 0.16.2 / NumPy 1.26.4 / Pillow 11.3.0 / OpenCV 4.10.0.84) | Known deviation; no reinstall was authorized | Documented; smoke generated successfully |
| O4 | Gating inputs differ across A0/A1/A7 | Expected because external visual evidence differs by design; A parity does not pin gating inputs | Not a failure; B must separately test gating parity, and B was not run |

## Pending / not authorized

- **Windows official gate:** `WINDOWS_GATE_PENDING`. BN-side validator passed, but Windows must run `validate_bundle.py --family A --mode smoke`.
- **A full 2,990-query run:** not authorized and not started.
- **B3/B7HQ/B7LQ smoke:** not authorized and not started.
- **B full:** not authorized and not started.
- **GPU full execution:** `GPU_EXECUTION_APPROVED=NO` for anything beyond this smoke.

## Failure rule

If Windows validation returns a mismatch, no A full run may start. The smoke must be diagnosed and, if a deployment fix is needed, rerun under a new authorization and attempt. No threshold may be lowered to obtain PASS.
