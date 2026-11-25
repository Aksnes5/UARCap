# UARCap Q2 Pilot A1 — execution report

**Status: NOT EXECUTED — `PILOT_A01_BLOCKED`.**

| Item | Result |
| --- | --- |
| DREAM-1K domains sampled | 0 of 5 |
| Videos selected | 0 of 30 |
| Arms executed | 0 of 3 |
| Real generations | **0 of 90** |
| X-CLIP retrieval calls | 0 |
| Video-LLaMA forward calls | 0 |
| GPU time | 0 h |
| Peak VRAM | not applicable |
| OOM | none |
| Retries | 0 |

## Why no generation was run

1. The five dataset-side Python modules required by `run_preflight`, `stage_assets`, and the runner are absent from commit `350c63e3019d569d8cf84c8f7873aa32493181f1`. The import failure occurs before any data or model access.
2. DREAM-1K is gated and the current BN shell has no HF token; the metadata endpoint returns HTTP 403 (`not in the authorized list`). No metadata or video archive is present on BN.

## What must be preserved

No output, prediction, trace, retrieval candidate, or run receipt was fabricated. The blocker receipt and raw preflight/stage logs are committed with this report. The GPU lock was never acquired and the GPU remained idle.
