# UARCap Q2 CaReBench Pilot A0 — data and environment audit

**Status: `PILOT_A01_BLOCKED` before GPU use.**

- Fixed commit: `647f7a2eace0c124302b3a4f70874e05169bb742`
- Worktree: `/home/bn/uarcap_q2_carebench`
- Runtime root: `/home/bn/uarcap_q2_runtime`
- Python: `/home/bn/anaconda3/envs/videollama2/bin/python` (3.10.8)
- Authorization: `/home/bn/uarcap_q2_runtime/authorizations/pilot_a01_auth.json`
- GPU: RTX A6000, 1 MiB used, 0% utilization, no compute client, lock free.

## 1. Code and bootstrap

The fixed commit was fetched over the existing SSH remote and checked out in a separate worktree. The five previously missing dataset modules are present, including `carebench.py`.

`bash scripts/bootstrap_bn.sh` passed using the existing Python 3.10.8 environment. No Conda environment was created and no package was changed.

`run_preflight.sh --e2-dataset carebench --verify-ckpt-sha` returned:

- `bn_live_ready = true`
- checkpoint SHA-256 matches `a8b5af79d55a2c7e684771475d645be6373208bdaae5a4211bee556581eb8002`
- all historical VideoUFO / X-CLIP / Video-LLaMA / Vicuna paths exist
- disk free 1,672.81 GB
- shared GPU lock readable and unheld
- direct `huggingface.co` metadata probe: **not accessible** from BN (`URLError`)

## 2. Metadata and sampling

CaReBench metadata was fetched successfully through `hf-mirror.com` using the installed `huggingface_hub`, not through the broken direct endpoint.

- Metadata path: `/home/bn/uarcap_q2_runtime/data/carebench/json/metadata.json`
- Metadata size: 3,696,714 bytes
- SHA-256: `740f43b71a1f8131521e2a6cde95a59c3db10a5c4bdd554dc64959b48b0fc951`
- Records: **1,000**
- Unique FineAction parents: **913**
- Dropped same-parent siblings: **87**
- Empty reference descriptions: **0**

The fixed-seed sampling preview (`seed=20261125`) selected **30** records with **30 unique parents**:

- Socializing, Relaxing: **10**
- Sports, Excercise: **8**
- Personal Care: **6**
- Household Activities: **6**

This matches the documented 10/8/6/6 allocation. The preview is not an executed pilot and has no predictions.

## 3. Video archive download — P0 blocker

Expected archive:

- `videos/videos.zip`
- Expected size: **3,918,994,506 bytes**
- Partial local file: **20,971,520 bytes** remaining as an incomplete HF cache entry
- Fully extracted videos: **0**
- `check-ready`: **false**, `n_missing_or_empty_videos=1000`

Attempts:

| Route | Result |
| --- | --- |
| `huggingface.co` direct | timeout / URLError |
| `hf-mirror.com` metadata via `huggingface_hub` | **success** |
| `hf-mirror.com` video via `huggingface_hub` | Xet bridge read timeout after ~10–20 MB |
| `hf-mirror.com` HTTP Range | ~66 KB/s; 16 parallel ranges stalled |
| Direct official archive | not reachable |

At the observed throughput, 3.9 GB would take tens of hours, far beyond the 2-hour pilot budget. The download was stopped rather than consuming resources indefinitely.

## 4. Dry-run

`bash scripts/run_pilot_a01.sh --dry-run` passed preflight and then stopped at readiness:

```text
ready: false
n_metadata_records: 1000
n_matched_videos: 0
n_missing_or_empty_videos: 1000
```

No model was loaded.

## 5. GPU and integrity

- Real generations: **0 / 90**
- Videos decoded: **0 / 30**
- X-CLIP retrieval calls: 0
- Video-LLaMA forward calls: 0
- GPU hours: **0**
- OOM: none
- Retries: 0
- GPU lock: not acquired

No mock outputs, no placeholder predictions and no partial predictions were created.
