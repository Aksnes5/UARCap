# UARCap Q2 Pilot A0/A1 — GO / NO-GO

**Final status: `PILOT_A01_BLOCKED`.**

| Gate | Result | Reason |
| --- | --- | --- |
| Fixed commit pulled | **YES** | `350c63e3019d569d8cf84c8f7873aa32493181f1` |
| Historical assets reachable | **YES** | VideoUFO, X-CLIP, Video-LLaMA, Vicuna, checkpoint, features all resolve |
| Q2 CPU env created | **NO** | conda-forge network timeout |
| Q2 code importable | **NO** | Five `uarcap_q2/datasets/*` modules omitted by `.gitignore` |
| Preflight | **FAILED** | `ModuleNotFoundError: uarcap_q2.datasets.dream1k` |
| DREAM-1K metadata | **NO** | gated; HTTP 403 |
| DREAM-1K videos | **NO** | gated; 0 bytes acquired |
| 5-domain sampling | **0/30** | no metadata/videos |
| 90 real generations | **0/90** | blocked before model load |
| GPU used | **NO** | idle; no lock held |
| OOM / failures / retries | **0 / 0 / 0** | no run |
| Integrity checks | **NOT REACHED** | no predictions/traces |

## P0 blockers

1. **DREAM-1K gated access.** The BN account has no HF token and is not on the authorized list. Metadata and video archive are unavailable.
2. **Incomplete GitHub commit.** The dataset modules required by the pipeline and preflight are absent from the commit. This must be fixed in the source delivery; it cannot be inferred from the existing tree.

## P1 blockers

- Isolated CPU env creation failed because `conda.anaconda.org` is unreachable from BN. An existing Python 3.10.8 environment with the required CPU packages is available, but the requested `uarcap_q2_cpu` env was not created.

## What was completed

- Fixed-commit checkout in a new worktree.
- Bootstrap directories and `bn_paths.local.yaml`.
- Authorization receipt for `PILOT_A0`/`PILOT_A1`.
- Verification that all historical frozen assets are readable.
- Non-destructive GPU/lock check.
- Exact blocker capture, including preflight and stage-assets import logs.
- No GPU work, no data mutation, no model download.

## What is required to unblock

1. Restore the five missing dataset modules from the Windows working copy and push them to a fixed commit.
2. Obtain DREAM-1K gated access using the official Hugging Face page and a read token supplied only through the shell environment.
3. Re-run `stage_assets.py prepare` under the A0/A1 receipt.
4. Re-run `run_preflight.sh` and then `run_pilot_a01.sh`.

Until both P0 items are resolved, no A1 pilot result exists and no A2/A3/Formal B work may begin.
