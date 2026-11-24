# Execution and rollback plan

**Current scope: Windows preparation and read-only BN access.** There is no approval to start smoke or GPU inference. The default launcher command only prints the plan. Keep the manuscript, historical output tree, prior branch, protocol and original server files frozen.

## Before any request to execute

1. Run the Windows archive replay and save the pinned files and receipts. Confirm local CPU test results and dry-run plans. The original B eligible cohort is an immutable input.
2. Once authenticated BN access is available, run the readonly probe. Reconcile the physical files through `SERVER_DATA_AND_CHECKPOINT_INVENTORY.json`. Match YAML route, canonical dataset paths, physical frame/label/caption TSVs and offsets, selected cache entries, checkpoint hash, Vicuna config/shards/tokenizer and vision initialization assets. Validate suitable free VRAM, available compute UUID, shared lock and disk headroom. Establish Python/CUDA/PyTorch/timm compatibility and offline caches. Escalate a large-file hash separately from inexpensive metadata queries.
3. Obtain explicit approval for the named experiment arms, GPU UUID and time window. Use a single existing device exclusively for this admitted job. If `nvidia-smi` reports any active compute client or the shared project lock is held, stop the new launch. Do not interfere with jobs.

## Deployment fields and logs

After approval, stage an isolated inference checkout containing the pinned original code and its relative runtime dependencies. Keep original assets mounted read-only for the run. Reuse the byte-verified archived CAP/SIM, transfer the Windows-built B and A0 files byte for byte, and use the associated `expected_feature_ids_<arm>.txt` lists. Verify all bytes upon transfer. Use the same existing interpreter and local model/encoder caches. Create a run-specific checkpoint symlink to the inspected original checkpoint inside the isolated checkout (no duplicate checkpoint and no historical path edits). The dry-run JSON gives the exact checkpoint slot for a full arm. Smoke uses the same pattern with `smoke` in place of `full`.

Deployment JSON consumed by `deploy_guard.py`:

- `run_id` is a safe cohort ID without the substring `eval`, due to the pinned evaluator's checkpoint naming. `attempt` is a positive integer. Use one logical run ID across all six conditions and both stages.
- `approval_id` refers to the explicit user authorization. `server_data_ready` becomes true after the live asset audit. `device` is `cuda:0` within the process and `gpu_uuid` is the admitted physical UUID.
- `original_evcap_root`, `canonical_checkpoint`, `checkpoint_name`, `checkpoint_sha256`, `shared_gpu_lock`, `llm_model`, `video_feature_dir` describe the inspected files. The isolated root cannot be the historical inference tree or share its outputs path.
- `test_yaml`, `test_yaml_sha256`, and `frame_receipts` bind the three YAML roles. A frame receipt includes `role`, `path`, `size`, `sha256`, `lineidx_sha256`, `mtime_ns` and `digest_status`. At runtime the image TSV is bound by its trusted receipt plus current size/mtime and offset digest, to avoid repeatedly streaming the image TSV. Label and caption bytes are hashed at each launch.
- `llm_manifest`, `llm_manifest_sha256`, `python_environment`, `python_environment_sha256` point to the operator-verified model/tokenizer and environment manifests. Record canonical executable and paths, versions, shard/cache file metadata and available hashes. `offline_cache_dirs` maps `TORCH_HOME` and `HUGGINGFACE_HUB_CACHE` to the verified existing directories. Launcher requires local HF and timm assets.
- `artifacts` maps the frozen input keys in `MINIMAL_GPU_RUN_MATRIX.json` and `B_visual_contract` to absolute paths. `smoke_report` maps family A/B to its gate report. `A_full_report` refers to the validated A full report required by both B stages.

Future launch wrapper, with placeholder operands only:

```text
python -B deploy_guard.py --evcap-root <isolated-evcap> --deployment <verified-receipt> --archive-scratch <pinned-feature-lists> --arm A0
python -B deploy_guard.py --evcap-root <isolated-evcap> --deployment <verified-receipt> --archive-scratch <pinned-feature-lists> --arm A0 --mode smoke --gpu-approval-id <approved-id>
```

The first line is dry-run, with no runtime imports. The second requires new authorization. Both use the archive evaluator with identical fixed parameters. Smoke places a bounded iterator around the ordered loader after the first eight queries are selected by the original dataset path. Use `--mode full` only after the family smoke passes. Each attempt records the full raw prediction object, observed traces, launcher log, runtime plus checkpoint and input receipts, model-environment versions, SHA-256 completion marker and wall-time/trace/peak-VRAM telemetry. Archive external scheduler/stdout logs too when used. Stage into separate output directories for smoke and full. No frozen result is overwritten.

## Order and local gates

1. A smoke: A0, A1, A7, sequentially. Copy outputs and logs to Windows. Run the three-arm validation and verify zero-evidence behavior, dtype/masks, query equality and bound inputs. Return the smoke gate to the authorized isolated run workspace.
2. A full: A0, A1, A7 sequentially, in separately scheduled windows if necessary. Validate all native-order outputs and receipts on Windows and stage the A full gate. Validate inputs/receipts and schema only until both families are done.
3. B smoke: B3, B7HQ, B7LQ sequentially. Validate strict visual parity on Windows and check eligible plus fallback prompt counts. Stop if even one observation fails. After its gate, B full uses the same arm order and exact per-query contract for all 2,990 rows.
4. For each stage, map arm names to the local copied directories in JSON and run:

   ```text
   python -B validate_bundle.py --family <A-or-B> --mode <smoke-or-full> --arm-directories <arm-directory-json> --archive-scratch <windows-pinned-scratch> --gate-out <new-gate.json>
   ```

   Full validation alone can also write `--scorer-manifest <new-manifest.json>`. Its per-arm keys are `predictions`, `trace` and `receipt`. Recheck captured hashes before scoring and keep the files immutable. Staged reports for future drives must preserve the same config binding, overlay version, cohort and runtime. Overlay source hashes use LF-normalized bytes across Windows and BN. Input artifact and inherited runtime code checks use exact raw bytes. Gate each prerequisite before starting the dependent work.
5. Windows only: call the frozen `evaluate_followup.py` using the paired manifest, original audit folder/metadata, pinned PTB jar/vendor/Java, source manifests and `--eligible <pinned-B_eligible_queries.json>` for B. Score A1−A0 and A7−A0 as one family, B7HQ−B3, B7LQ−B3 and B7HQ−B7LQ as the other. Retain official PTB CIDEr in points ×100, source clustering, 10,000 source-level sign flips and cluster bootstraps, Holm within families and the fixed 1.0-point interpretation reference. Evaluate the locked follow-up as such. Earlier research decisions informed the design. HQ/LQ qualify X-CLIP relevance.

## Failure and rollback

- No approval, missing asset, busy GPU, ambiguous identity or checksum mismatch: stop before running the model. Inspect existing paths or obtain exact archived inputs. No data regeneration, cache replacement or forced worker stop.
- Failed or interrupted run: retain the output directory, launcher log, partial trace and provenance. No completion marker is admissible for a partial attempt. Relaunch only the failed arm with the same logical cohort and incremented attempt number after diagnosing the failure. The pinned evaluator writes its complete output at the end; restart the failed arm from query one. Preserve already sealed arms if bindings and runtime match.
- Any real-data, Q-Former, cache, prompt, K/V, weight or B gate mismatch invalidates the affected bundle. If the fix changes the runtime, allowed paths or instrumentation, revalidate smoke and regenerate any incompatible arm within the locked condition. No tuning or subgroup changes based on realized outputs, and no interim significance checks. Keep score products isolated and score only fully verified bundles.
