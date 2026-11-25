# PILOT A2 AUTOMATIC STAGE — INDEPENDENT AUDIT

- **All checks passed**: `True`

| Check | Result | Details |
| --- | --- | --- |
| `eval_rows_complete_90` | PASS | n_rows=90; missing=[] |
| `all_parse_status_ok` | PASS | parse_failures=0; statuses=['OK'] |
| `identical_captions_have_identical_verdicts` | PASS | n_unique_caption_groups=50; violations=[]; n_duplicate_reuse_sum=40 |
| `per_arm_claim_ids_unique` | PASS | violations=[] |
| `reference_fields_consistent` | PASS | violations=[]; fields=[('caption', 'spatial_caption', 'temporal_caption')] |
| `core_event_verdicts_aligned` | PASS | violations=[] |
| `blind_bundles_clean_and_complete` | PASS | bundles={'A1': {'n_tasks': 30, 'n_captions': 90, 'n_tasks_with_16_frames': 30}, 'A2': {'n_tasks': 30, 'n_captions': 90, 'n_tasks_with_16_frames': 30}}; violations=[] |
| `blind_key_maps_all_conditions` | PASS | n_blind_captions=180; per_annotator={'A1': 90, 'A2': 90}; arms=['ARM_A0', 'ARM_A1', 'ARM_A7']; n_key_claims=496; n_auto_claims=248 |
| `frame_previews_complete` | PASS | n_frame_dirs=30; bad=[] |
| `raw_judge_outputs_saved` | PASS | n_calls=150; empty=0; retries=0; stages=['3WAY_REF_NLI', 'CORE_EVENT_NLI', 'EXTRACT_CLAIMS'] |
| `core_event_lock_precedes_judge` | PASS | lock_time=2026-11-25T17:38:31.221133+00:00; judge_time=2026-11-25T17:40:38.050284+00:00; n_entries=30 |
| `a1_inputs_sha_verified` | PASS | a1_status=REAL_RUNTIME_PASS; metadata_match=True |
| `judge_receipt_consistent` | PASS | judge=RedHatAI/gemma-3-27b-it-FP8-dynamic; family=gemma; unique=50 |
| `status_human_validation_pending` | PASS | status=PILOT_A2_HUMAN_VALIDATION_PENDING |

## Artifacts

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `a2_ref_fact_eval.jsonl` | 283532 | `63747be652051a65ac9513abf5b8a1d39a2fcc738c83dacb167cdf3917594968` |
| `a2_judge_raw_calls.jsonl` | 304611 | `db837335c12de570a858a2934e8e8352f359097cfc5370d7de619e5d3cd9ff90` |
| `a2_judge_run_receipt.json` | 1517 | `918239ed2ad2ef29a3e680f211accd61e0743ed6f16a253b5f36458bc7dc8505` |
| `a2_input_verification.json` | 2460 | `56ee5cd9b27eef6ed3a6b5e942b788d91418c29b7d2e0eca8ecbf1844b133b5b` |
| `a2_core_event_lock.json` | 116784 | `2deb0464f47bfa70066d8fe5626c969743c97b46e6d2ec6a67b26135e58aa90e` |
| `a2_bundle_manifest.json` | 1527 | `bb007b931cb136546422635fcc631dbc5e5dc404ed40dfec25d689765a691764` |
| `calibration_receipt.json` | 1307 | `031da7e070695e6a5260c4a411668a4f0a8ec20f7cda756528e867e51e160e75` |
| `a2_auto_diagnostics.json` | 35197 | `c2ca3793a4bd4bc57e9d35d1a9da934d755fe7bef54b6371b73f09ef928c1404` |
| `PILOT_A2_AUTO_DIAGNOSTICS.md` | 10148 | `4e9b2608d5cd3ed2ef7710a08e829154b9a521a3dda860a4766cae73a4a04936` |
| `blind_annotation_package/blind_tasks_e2_A1.json` | 141118 | `74e164d367051c857372d870c0c4b25bfe75ae6252977e746db64018ab4e6e0d` |
| `blind_annotation_package/blind_tasks_e2_A2.json` | 141118 | `0b2e7f18d236a3a48ef6e0587eb2a79bbf9dec22d3b8006f21c183f641cf8370` |
