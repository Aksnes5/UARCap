# Pilot A1 (CaReBench E2) — BN runtime runbook

Written 2026-11-26 after the first real BN GPU execution. Every step below is
reproducible and every external input is SHA-256 pinned.

## 1. Data acquisition (official HF source → trusted ModelScope mirror)

`huggingface.co` is unreachable from BN and the `hf-mirror.com` Xet bridge
stalls at 10–20 MB. The ModelScope copy of `MCG-NJU/CaReBench` is registered as
a trusted mirror and is bit-identical to the HF LFS object:

| File | Bytes | SHA-256 (HF `lfs.oid` == ModelScope `Sha256`) |
| --- | --- | --- |
| `videos/videos.zip` | 3,918,994,506 | `957706c201e669ae2085994da4def1baceb2c096b854aac613171bac67cd1203` |
| `json/metadata.json` | 3,696,714 | `740f43b71a1f8131521e2a6cde95a59c3db10a5c4bdd554dc64959b48b0fc951` |

```bash
python deployment/stage_assets.py mirror-check --dataset carebench
python deployment/stage_assets.py prepare \
  --data-root /home/bn/uarcap_q2_runtime/data --dataset carebench
```

The downloader (`download_parallel_ranges`) uses 16–24 parallel resumable Range
requests, keeps chunk parts across retries, assembles deterministically, and
refuses the file unless the full SHA-256 matches. Measured: 3.9 GB in 177 s
(~22 MB/s) versus ~11 KB/s on the HF Xet bridge.

## 2. Frozen 30-video sampling manifest

```bash
python scripts/freeze_pilot_a1_sampling.py \
  --metadata /home/bn/uarcap_q2_runtime/data/carebench/json/metadata.json \
  --video-root /home/bn/uarcap_q2_runtime/data/carebench/videos \
  --out bn_carebench_pilot_a01/PILOT_A1_FROZEN_SAMPLING_MANIFEST.json \
  --preview bn_carebench_pilot_a01/sampling_preview.json
```

- Seed `20261125`, 30 unique FineAction parents, 30 unique subcategories.
- Category allocation: 6 Household, 6 Personal Care, 10 Socializing, 8 Sports.
- Manifest SHA-256: `26fbf6abaf7ed12309ca5be40ccc583c068e5bd2c9b427f0753c4600d5b5bd21`.
- The derived sample is verified ID-for-ID against the A0 preview.

## 3. Retrieval candidate visual features

Text retrieval covers all 1,091,712 VideoUFO clips; the shared Video-LLaMA
feature cache only contains clips retrieved by earlier experiments. Running the
pilot on new CaReBench queries therefore requires adding the missing
`faithful_v2` features for exactly the top-20 representatives:

```bash
bash scripts/ensure_pilot_a1_features.sh
```

- Discovery: `scripts/discover_pilot_a1_candidates.py` (same X-CLIP search,
  feature-existence check disabled).
- Extraction: `/home/bn/videoufo_research/scripts/extract_external_features_v2.py`
  with `HUGGINGFACE_HUB_CACHE`/`HF_HOME` pointed at `repro_evcap/hf_cache`
  (offline; the default `~/.cache/huggingface` snapshot is incomplete).
- Pilot A1 required **376** new features (all shape `(32, 768)`, finite).
- Retrieval semantics are unchanged: candidates are not filtered or re-ranked.

## 4. Runtime fixes applied on BN (2026-11-26)

1. **Re-entrant cooperative GPU lock** (`uarcap_q2/common/gpu_lock.py`):
   `fcntl.flock` conflicts with a second `open()` in the same process. The
   orchestrator holds the lock while the isolated X-CLIP wrapper acquired it
   again, making every real run fail with `GPULockBusyError`. Nested
   acquisitions now reuse the process-held lock; cross-process exclusion is
   still enforced (covered by an explicit test).
2. **X-CLIP import compatibility** (`uarcap_q2/retrieval/videoufo_retriever.py`):
   the frozen X-CLIP fork needs `import clip` before `models.*` (circular
   import) and a `sys.modules["timm.layers"] = timm.models.layers` alias for
   the installed timm 0.6.13.
3. **certifi CA bundle** for `urllib` in the conda env (stale OpenSSL paths).

## 5. Execute + audit

```bash
export PYTHON_BIN=/home/bn/anaconda3/envs/videollama2/bin/python
export E2_DATASET=carebench
bash scripts/run_pilot_a01.sh --dry-run   # GPU untouched
bash scripts/run_pilot_a01.sh

python scripts/audit_pilot_a1_run.py \
  --run-dir /home/bn/uarcap_q2_runtime/outputs/pilot_a1/pilot_a1_carebench_v1 \
  --frozen-manifest bn_carebench_pilot_a01/PILOT_A1_FROZEN_SAMPLING_MANIFEST.json
```

Artifacts under `outputs/pilot_a1/pilot_a1_carebench_v1/`:
`predictions.jsonl`, `input_traces.jsonl`, `retrieval_candidates.jsonl`,
`failures.jsonl`, `query_manifest.json`, `run_receipt.json`,
`PILOT_A1_EXECUTION_REPORT.md`, `PILOT_A1_INTEGRITY_REPORT.md`,
`PILOT_A1_GO_NO_GO.md`, plus `AUDIT_PILOT_A1.{json,md}`.
