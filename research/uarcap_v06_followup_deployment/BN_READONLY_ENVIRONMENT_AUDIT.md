# BN read-only environment audit

**Status: UNVERIFIED.** Windows did not reach BN. The existing endpoint name did not resolve. The local check found no configured BN alias, matching known-host record, or alternate authenticated endpoint in the inspected materials. The only connection attempt was:

```text
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=4 -o ConnectionAttempts=1 bn@bn true
exit 255: Could not resolve hostname bn
```

No system query ran on BN. GPU occupancy, driver, CUDA, live jobs, Python/Conda, free storage, filesystem paths and access permissions are all unverified. The historical plan names an RTX A6000-class 48 GB device. The archived environment file lists Python 3.9.18, PyTorch 2.1.0, Transformers 4.28.0 and timm 0.6.13. These describe archived expectations. The historical runner explicitly used the existing Display reproduction venv and local cache directories. The active BN interpreter and any Conda alternatives remain to be identified.

| Scope | Archived route | Required read-only check |
| --- | --- | --- |
| Project, runner, shared lock | `/home/bn/videoufo_research`, `scripts/run_test.sh`, `gpu.lock` | Canonical paths, permissions, exactly one cooperating lock, actual source/checkout digests. |
| Checkpoint | Display `lj/EVCapCode/outputs/train_evcap_msrvtt_quality_text/` | Original basename, resolved target, bytes, trusted SHA or one approved off-peak sequential hash. |
| Vicuna and visual initialization | Display `repro_evcap` model/tokenizer and offline caches, evaluator-relative Video-LLaMA checkpoint | Model shard manifest, tokenizer/config bytes, BERT cache, EVA/BLIP2 cache files and Video-LLaMA weights. Use trusted large-file receipts with metadata. |
| Dataset path | Archived test2990 YAML routes to pre-extracted image TSV with caption/label TSV and offsets | Bind YAML roles to canonical physical paths, exact 2,990 order, 20-reference lists, offsets and image bytes. Hash small objects. Establish trusted image-digest provenance before approving any large direct read. |
| External evidence | Frozen FULL D and Top20 Git blobs, existing faithful feature directory | Compare small archived files by SHA. Check required clip paths individually, with size/readability and existing digests where trustworthy. Use byte-level runtime tensor signatures in future A/B tracing. |
| GPU and free space | A6000-class historical description | Management queries for UUID, driver, total/free VRAM and compute processes, plus disk free and live process inventory. Check runtime CUDA metadata without initializing CUDA. |

`bn_readonly_probe.sh` is a prepared, syntax-checked probe using the explicit project, Display, dataset and interpreter paths. Run it only through an authenticated existing endpoint. It hashes objects up to 16 MiB and reports larger files by metadata. It queries management interfaces, without importing PyTorch, launching CUDA or writing remotely. The probe is an initial inventory. Cross-check cached weights, shard manifest, offsets, per-clip paths, frame order and the canonical asset receipts before setting `SERVER_DATA_READY`. The adjacent machine inventory separates archived facts from live fields.

For any confirmed absence, check approved existing BN roots first, then the pinned Git archive and Windows inputs, then consider only a targeted transfer of the exact documented files. Record unresolved paths and halt deployment if a required asset remains unauthenticated. No assets were classified missing based on this failed connection.
