# UARCap Q2 CaReBench Pilot A0/A1 — GO / NO-GO

> **SUPERSEDED (2026-11-26):** this document records the historical download block.
> The block was resolved via the SHA-256-verified ModelScope mirror; real Pilot A1
> then completed with `REAL_RUNTIME_PASS` (90/90). See
> [`PILOT_A1_RESULTS.md`](PILOT_A1_RESULTS.md) and
> [`PILOT_A1_RUNBOOK.md`](PILOT_A1_RUNBOOK.md). The receipt below is kept unchanged
> as historical evidence.

**Final status at the time of the block: `PILOT_A01_BLOCKED`.**

| Gate | Result | Reason |
| --- | --- | --- |
| Fixed commit pulled | **YES** | `647f7a2eace0c124302b3a4f70874e05169bb742` |
| Dataset modules restored | **YES** | all five plus `carebench.py` present |
| Bootstrap | **YES** | existing Python 3.10.8 environment |
| Preflight / historical assets | **YES** | `bn_live_ready=true` |
| Checkpoint SHA | **YES** | `a8b5af79…eb8002` |
| CaReBench metadata | **YES** | 1,000 records / 913 parents |
| 30-video stratified sampling | **YES (preview)** | 30 parents, 10/8/6/6 categories |
| CaReBench video archive | **NO** | 20,971,520 / 3,918,994,506 bytes partial |
| 30 videos decoded | **0 / 30** | archive absent |
| 90 real generations | **0 / 90** | blocked before model load |
| GPU used | **NO** | idle, lock free |
| OOM / retries | **0 / 0** | no run |
| Integrity checks | **NOT REACHED** | no predictions/traces |

## P0 blocker

CaReBench's public metadata is available, but the 3.9 GB video archive cannot be retrieved at usable throughput from BN:

- Official `huggingface.co`: timeout.
- `hf-mirror.com` metadata: works.
- `hf-mirror.com` video: Xet bridge read timeout after ~10–20 MB.
- HTTP Range: ~66 KB/s, parallel ranges stalled.
- Current partial file: 20,971,520 bytes; expected 3,918,994,506 bytes.

The 2-hour pilot budget cannot accommodate this transfer. Continuing would not be a pilot run; it would be an uncontrolled multi-hour download.

## What is ready for a later run

- Fixed code and CaReBench adapter.
- Metadata and deterministic 30-parent sampling manifest.
- Historical VideoUFO, X-CLIP, Video-LLaMA and checkpoint assets.
- Authorization receipt scoped to `PILOT_A0`/`PILOT_A1`.
- Preflight and failed-download evidence.

## Unblock

Use the Windows/offline transfer route in `CAREBENCH_DOWNLOAD_TRANSFER_FALLBACK.md`, then rerun:

```text
python deployment/stage_assets.py prepare ... --transfer-manifest ...
bash scripts/run_pilot_a01.sh --dry-run
bash scripts/run_pilot_a01.sh
```

No A2/A3/Formal B work may begin from this state. The GPU was not used and the shared lock was not acquired.
