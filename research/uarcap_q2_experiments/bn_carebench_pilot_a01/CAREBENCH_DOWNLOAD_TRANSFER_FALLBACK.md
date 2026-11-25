# CaReBench download / offline transfer fallback

## Current block

Metadata succeeded through `hf-mirror.com`, but the **3,918,994,506-byte** `videos/videos.zip` could not be transferred at usable speed from BN:

- `huggingface.co`: connection timeout
- `hf-mirror.com` via `huggingface_hub`: Xet bridge read timeout after ~10–20 MB
- HTTP Range: ~66 KB/s; 16 parallel ranges stalled
- Partial file: 20,971,520 bytes
- Extracted videos: 0

This is a network/throughput block, not a code or asset-path block.

## Route B — Windows download and chunk transfer

### 1. On Windows (repository checkout at the fixed commit)

```powershell
cd D:\UARCap\research\uarcap_q2_experiments

# Preview only
python scripts/download_carebench_stage.py `
  --stage-dir D:\UARCap_LocalData\carebench_staging `
  --hf-endpoint https://hf-mirror.com `
  --dry-run

# Actual public download (no HF token required; CaReBench is ungated, MIT)
python scripts/download_carebench_stage.py `
  --stage-dir D:\UARCap_LocalData\carebench_staging `
  --hf-endpoint https://hf-mirror.com `
  --confirm-download
```

Expected outputs:

```text
D:\UARCap_LocalData\carebench_staging\json\metadata.json
D:\UARCap_LocalData\carebench_staging\videos\videos.zip
D:\UARCap_LocalData\carebench_staging\bn_transfer_chunks\
D:\UARCap_LocalData\carebench_staging\videos\videos.zip.transfer_manifest.json
```

The tool refuses to fetch the 3.65 GiB archive without `--confirm-download`.

### 2. Transfer to BN

Transfer the metadata and `bn_transfer_chunks/` to:

```text
/home/bn/uarcap_q2_runtime/staging/carebench/
```

Keep `videos.zip.transfer_manifest.json` next to the chunks.

### 3. Import and verify on BN

```bash
mkdir -p /home/bn/uarcap_q2_runtime/data/carebench/json
cp /home/bn/uarcap_q2_runtime/staging/carebench/json/metadata.json \
   /home/bn/uarcap_q2_runtime/data/carebench/json/metadata.json

cd /home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments
/home/bn/anaconda3/envs/videollama2/bin/python deployment/stage_assets.py prepare \
  --data-root /home/bn/uarcap_q2_runtime/data \
  --dataset carebench \
  --transfer-manifest /home/bn/uarcap_q2_runtime/staging/carebench/videos.zip.transfer_manifest.json \
  --auth-receipt /home/bn/uarcap_q2_runtime/authorizations/pilot_a01_auth.json
```

Then verify:

```bash
/home/bn/anaconda3/envs/videollama2/bin/python deployment/stage_assets.py check-ready \
  --data-root /home/bn/uarcap_q2_runtime/data --dataset carebench
```

Required readiness:

```text
ready: true
n_metadata_records: 1000
n_unique_fineaction_parents: 913
n_missing_or_empty_videos: 0
```

### 4. Run the pilot only after readiness

```bash
cd /home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments
export PYTHON_BIN=/home/bn/anaconda3/envs/videollama2/bin/python
export E2_DATASET=carebench
bash scripts/run_pilot_a01.sh --dry-run
bash scripts/run_pilot_a01.sh
```

Do not modify the shared GPU lock or any network/proxy configuration. Do not use mock outputs. The pilot authorization is limited to `PILOT_A0`/`PILOT_A1`.
