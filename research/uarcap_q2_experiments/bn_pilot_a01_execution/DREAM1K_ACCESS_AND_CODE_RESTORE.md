# DREAM-1K access and missing-module recovery

## 1. Official gated access

1. Sign in to Hugging Face and open:
   `https://huggingface.co/datasets/omni-research/DREAM-1K`
2. Accept the dataset terms and submit the access request for the account that will download the data.
3. Wait for the dataset owner's approval. The BN response currently is:
   `Access to dataset omni-research/DREAM-1K is restricted and you are not in the authorized list.`
4. Create a **read-only** token at `https://huggingface.co/settings/tokens`.
5. Do not write the token into a file, shell history, Git, or any report. In the BN shell only:

```bash
read -rsp 'HF token: ' HF_TOKEN; echo
export HF_TOKEN
export HF_ENDPOINT=https://hf-mirror.com   # per-command endpoint only; do not change global network
```

6. From the fixed code checkout, after the missing modules are restored:

```bash
cd /home/bn/uarcap_q2_repo/research/uarcap_q2_experiments
python deployment/stage_assets.py prepare \
  --data-root /home/bn/uarcap_q2_runtime/data \
  --dataset dream1k \
  --allow-download \
  --auth-receipt /home/bn/uarcap_q2_runtime/authorizations/pilot_a01_auth.json
```

The direct `huggingface.co` endpoint timed out from BN; `hf-mirror.com` returned HTTP 200 for the host but HTTP 403 for the gated file without authorization. Do not bypass the gate.

## 2. Windows / transfer fallback

If BN direct access remains unreliable after approval:

1. Download `json/metadata.json` and `video/video.zip` on Windows with the approved account.
2. Split the archive with the Windows copy of the same tool:

```text
python deployment/stage_assets.py split --src video.zip --out-dir dream1k_chunks --chunk-mb 512
```

3. Transfer `dream1k_chunks/` and `metadata.json` to BN.
4. On BN, import using the transfer manifest:

```text
python deployment/stage_assets.py prepare \
  --data-root /home/bn/uarcap_q2_runtime/data \
  --dataset dream1k \
  --transfer-manifest /home/bn/uarcap_q2_runtime/staging/video.zip.transfer_manifest.json
```

## 3. Missing source modules

The fixed commit is not importable because the root `.gitignore` pattern `datasets/` excluded the Q2 dataset package. Restore these exact files from the Windows working copy, then commit them to the fixed branch:

```text
research/uarcap_q2_experiments/uarcap_q2/datasets/__init__.py
research/uarcap_q2_experiments/uarcap_q2/datasets/asset_transfer.py
research/uarcap_q2_experiments/uarcap_q2/datasets/dream1k.py
research/uarcap_q2_experiments/uarcap_q2/datasets/vidhalluc.py
research/uarcap_q2_experiments/uarcap_q2/datasets/video_decoder.py
```

Recommended ignore fix (scoped, not a global unignore):

```gitignore
!research/uarcap_q2_experiments/uarcap_q2/datasets/
!research/uarcap_q2_experiments/uarcap_q2/datasets/**
```

Then verify on BN:

```bash
cd /home/bn/uarcap_q2_repo
git rev-parse HEAD
cd research/uarcap_q2_experiments
python -c "from uarcap_q2.datasets.dream1k import load_dream1k_records; print('imports ok')"
bash scripts/run_preflight.sh --verify-ckpt-sha --check-hf-network
```

Only after both the data gate and import gate pass may `run_pilot_a01.sh` be executed. No GPU authorization is consumed by this blocked state.
