# UARCap v0.6 — Full A/B blockers and failure history

**P0 blockers: none.** All six full arms and the B smoke completed under the frozen protocol.

## Failures and retries

- Full-arm failures: **0**
- Retries: **0**
- Interrupted arms: **0**
- OOM events: **0**
- Missing features: **0**
- NaN / non-finite tensors: **0**
- Incomplete traces: **0**

`runner_events.jsonl` contains `ALL COMPLETE`; `grep -c FAILED` = 0.

## Non-blocking observations

| ID | Observation | Action |
| --- | --- | --- |
| O1 | TensorFlow startup warnings about cuDNN/cuFFT/cuBLAS factory registration and missing TF-TRT | Non-fatal; all processes exited 0 |
| O2 | Python/PyTorch/torchvision/NumPy/Pillow/OpenCV versions differ from the archived expectation | Documented; no reinstall performed |
| O3 | `video_qformer.bert.training=True` and `attention.training=True` | Inherited frozen model state; preserved |
| O4 | A7/B7HQ/B7LQ full runs take longer because of 7 visual evidence items | Expected, not a failure |
| O5 | Formal statistical inference not run on BN | Correctly deferred to Windows |

## Gate status

- A full: PASS
- B smoke: PASS
- B full: PASS
- All three gates report `trace_parity=true` and the exact row counts.
- A0 zero-evidence check over all 2,990 queries: 0 failures.

No research design, cohort, eligible list, prompt or checkpoint was changed. No B candidate was re-matched.
