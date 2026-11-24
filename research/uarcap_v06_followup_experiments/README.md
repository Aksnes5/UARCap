# Follow-up preparation for UARCap v0.6

Two additive interventions are specified after the standalone manuscript commit. The protocols are [A, matched k=0](EXPERIMENT_A_K0_PROTOCOL.md) and [B, fixed-visual text extension](EXPERIMENT_B_FIXED_VISUAL_3TO7_PROTOCOL.md). See the [statistical plan](FOLLOWUP_STATISTICAL_ANALYSIS_PLAN.md), [execution gates](FOLLOWUP_EXECUTION_READINESS.md), [implementation audit](FOLLOWUP_IMPLEMENTATION_AUDIT.md), [resource assumptions](FOLLOWUP_RESOURCE_AND_RUNTIME_PLAN.md), and [decision rules](FOLLOWUP_RESULT_DECISION_TREE.md).

## Windows CPU reproduction

Use Python 3.11 and NumPy for the tests. Stage exact binary Git blob contents from the `basis_commit` in `FROZEN_SOURCE_BINDINGS.json` to a scratch folder. PowerShell users should capture stdout with Python `subprocess.check_output` and write bytes so line endings remain unchanged. Use these mappings.

| Scratch filename | Frozen Git object path |
| --- | --- |
| `top20.json` | `external/external_test_top20.json` |
| `cap.json` | `external/external_test_d7_cap.json` |
| `sims.json` | `external/external_test_d7_sims.json` |
| `full.tsv` | `manifests/msrvtt_test2990.tsv` |
| `pilot_excluded.tsv` | `manifests/msrvtt_untouched2690.tsv` |

From this directory run the following reads and CPU operations. Keep materialized caption inputs in scratch storage. Commit only the preflight summary and hashes.

```text
python -B -m unittest -v
python -B prepare_inputs.py --top20 <scratch/top20.json> --cap <scratch/cap.json> --sims <scratch/sims.json> --full-ids <scratch/full.tsv> --test-ids <scratch/pilot_excluded.tsv> --out <scratch/prepared> --report <scratch/preflight.json>
```

Compare the rebuilt summary against `CPU_PREFLIGHT_SUMMARY.json`. Its hashes bind object content using canonical JSON and pin source files by raw byte hash. The top20, D CAP, D SIM and manifests can be read from Git independently of BN. Baseline runtime receipts and physical `.pt` caches are a separate step.

## Deployment package

For a future authorized run, create a deployment JSON beside the scratch inputs. It supplies a fresh alphanumeric `run_id`, `device`, absolute `test_yaml`, `test_yaml_sha256`, pinned `checkpoint_name` basename and `checkpoint_sha256`, absolute `llm_model`, absolute `video_feature_dir`, and `llm_manifest_sha256` and `python_environment_sha256` from verified inventories. Supply `frame_receipts` as three objects with `role` (`img`, `label`, `caption`), absolute `path`, `size` (bytes) and `sha256`. Align these receipts with the YAML and check the offsets during deployment. Supply an `artifacts` map from scratch object names without `.json` to absolute paths. Include `frozen_D_cap` and `frozen_D_sims` pointing to the untouched archived files. Do not substitute any artifacts from earlier validation experiments.

Use a dedicated checkout at the pinned code commit and a prepared runtime with the inherited model caches. The unmodified evaluator looks for `<checkpoint_name>` under `outputs/train_v06_<run_id>_<arm>/`. Use read-only copies or links there, and confirm all six arms point to the same hash. Record the original offline model/cache environment outside Git and set it as in the archived runner. Without the approval argument, `python launch_pinned.py --evcap-root <checkout/evcap_external> --deployment <path> --arm A0` simply prints the exact arguments. A separate explicit approval and a verified server-data receipt are required for a future execute invocation. This repository contains no verified deployment record and no current approval.

After a complete authorized run, give `evaluate_followup.py` a JSON mapping every required arm to its `predictions`, `trace` and `receipt` paths. It verifies one common runtime/weight/annotation receipt across arms, full query order, and the actual measured visual and textual input contracts. Pass the original two query manifests, the independent audit directory, frozen vendor scorer path, exact Java and PTB JAR, the audit's hash-verified MSR-VTT source metadata and a new output folder. For B additionally pass the generated `B_eligible_queries.json`. The scorer computes formal PTB CIDEr only after the trace and reference checks. No prediction or metric file from either follow-up exists in this commit.
