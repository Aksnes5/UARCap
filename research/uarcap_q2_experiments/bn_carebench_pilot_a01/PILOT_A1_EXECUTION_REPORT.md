# UARCap Q2 CaReBench Pilot A1 — execution report

**Status: NOT EXECUTED — `PILOT_A01_BLOCKED` by CaReBench video download.**

| Item | Result |
| --- | ---: |
| Metadata records | 1,000 |
| Unique FineAction parents | 913 |
| Pilot preview selected | 30 |
| Unique pilot parents | 30 |
| Category allocation | 10 / 8 / 6 / 6 |
| Videos acquired | 0 |
| Videos decoded | 0 / 30 |
| Arms executed | 0 / 3 |
| Real generations | **0 / 90** |
| X-CLIP retrieval calls | 0 |
| Video-LLaMA forward calls | 0 |
| GPU hours | 0 |
| OOM / retries | 0 / 0 |

## What exists

- A real, deterministic 30-record sampling preview from the real 1,000-record CaReBench metadata, seed `20261125`.
- All 30 selected records have distinct FineAction parent IDs.
- No videos, predictions, traces, retrieval candidates or run receipts were fabricated.

## What blocked execution

`videos/videos.zip` is 3,918,994,506 bytes. Only 20,971,520 bytes were received; all available BN routes were too slow or timed out. The pilot runner correctly stopped at `check-ready` because 1,000 videos were missing.

## Follow-up

Use `CAREBENCH_DOWNLOAD_TRANSFER_FALLBACK.md`. After the archive is verified and normalized, rerun the dry-run and the full A0/A1/A7 pilot. A2 remains blocked until a real `REAL_RUNTIME_PASS` A1 receipt exists.
