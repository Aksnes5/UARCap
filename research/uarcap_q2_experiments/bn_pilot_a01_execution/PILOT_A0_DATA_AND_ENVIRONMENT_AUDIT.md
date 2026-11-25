# UARCap Q2 Pilot A0 — data and environment audit

**Status: `PILOT_A01_BLOCKED`.** No GPU model was loaded and no generation was run.

- Fixed commit: `350c63e3019d569d8cf84c8f7873aa32493181f1`
- Worktree: `/home/bn/uarcap_q2_repo`
- Runtime root: `/home/bn/uarcap_q2_runtime`
- Authorization receipt: `/home/bn/uarcap_q2_runtime/authorizations/pilot_a01_auth.json`
- Approval ID: `USER_APPROVED_UARCAP_Q2_PILOT_A01_20261125`
- GPU: RTX A6000 idle, 15 MiB used, no compute client, shared lock unheld.

## 1. Code checkout

The requested commit was fetched over the existing SSH GitHub remote and checked out in a detached worktree. `git rev-parse HEAD` returned exactly the requested SHA.

## 2. Bootstrap

`bash scripts/bootstrap_bn.sh` created:

- `/home/bn/uarcap_q2_runtime/{data,outputs,authorizations,...}`
- `configs/bn_paths.local.yaml`
- verified `/home/bn/videoufo_research/gpu.lock`

It failed only at isolated CPU env creation:

```text
CondaHTTPError: HTTP 000 CONNECTION FAILED
https://conda.anaconda.org/conda-forge/noarch/repodata.json
ReadTimeoutError
```

An existing Python 3.10.8 environment with `numpy`, `scipy`, `yaml`, `PIL`, `av`, and `pytest` exists at:

```text
/home/bn/anaconda3/envs/videollama2/bin/python
```

It was used for the preflight attempt; it is not the requested isolated `uarcap_q2_cpu` env.

## 3. Historical asset bindings

All path bindings in `configs/bn_paths.local.yaml` resolve on BN:

- VideoUFO root, CSV, clips TSV, text embeddings
- X-CLIP repo and `k600_16_16.pth`
- `evcap_external`, frozen `repro_evcap/venv` Python
- frozen checkpoint, Vicuna, feature cache
- `gpu.lock`

No historical data, model or environment was modified.

## 4. P0 code packaging blocker

The fixed commit cannot import its dataset layer. The root `.gitignore` contains:

```text
datasets/
```

That pattern excluded the Q2 package's own `uarcap_q2/datasets/` sources. The commit tree contains only `data_preparer.py` and `formal_sampling.py`; the following required modules are absent from the commit and from every fetched `antigravity/*` branch:

```text
uarcap_q2/datasets/__init__.py
uarcap_q2/datasets/asset_transfer.py
uarcap_q2/datasets/dream1k.py
uarcap_q2/datasets/vidhalluc.py
uarcap_q2/datasets/video_decoder.py
```

Observed failures:

```text
preflight:    ModuleNotFoundError: uarcap_q2.datasets.dream1k
stage_assets: ModuleNotFoundError: uarcap_q2.datasets.asset_transfer
```

The documented 59-test result cannot be reproduced from this commit because the test suite imports the same missing modules. This is a repository packaging error, not a BN path issue. The files must be restored from the Windows local working copy and committed (or the ignore rule must be scoped so they can be added).

## 5. DREAM-1K access

No DREAM-1K metadata or video archive exists on BN.

- Official page: `https://huggingface.co/datasets/omni-research/DREAM-1K`
- `huggingface.co` direct connection: timed out from BN.
- `https://hf-mirror.com` responds, but the gated metadata URL returns **HTTP 403**:

```text
Access to dataset omni-research/DREAM-1K is restricted and you are not in the authorized list.
```

- No `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` is present in the execution shell.
- Bytes acquired: **0**.
- Videos acquired: **0/30**.

## 6. GPU state

The GPU was not used:

- allocated: 15 MiB
- utilization: 0%
- compute clients: none
- lock: free

No model was loaded, no OOM occurred, no retries were attempted, and no partial predictions exist.
