# PILOT A1 INDEPENDENT AUDIT

- **All checks passed**: `True`
- **Run receipt status**: `REAL_RUNTIME_PASS`
- **Go/No-Go**: `PASS`
- **Frozen sampling manifest SHA-256**: `26fbf6abaf7ed12309ca5be40ccc583c068e5bd2c9b427f0753c4600d5b5bd21`

| Check | Result | Details |
| --- | --- | --- |
| `predictions_complete_90_unique` | PASS | n_predictions=90; n_unique=90; n_expected=90; missing=[]; extra=[] |
| `prediction_trace_hash_linkage` | PASS | n_traces=90; mismatches=[] |
| `all_generations_nonempty_ok` | PASS | bad=[]; n_bad=0 |
| `a0_strict_zero_evidence` | PASS | violations=[] |
| `a1_prefix_of_a7` | PASS | violations=[] |
| `visual_input_consistency_16_frames` | PASS | violations=[]; n_violations=0 |
| `retrieval_evidence_complete` | PASS | n_rows=30; violations=[] |
| `frozen_manifest_matches_run` | PASS | n_manifest=30; n_run=30; missing=[]; extra=[] |
| `gpu_timing_recorded` | PASS | run_wall_seconds=467.3246; sum_generation_seconds=369.903; peak_vram_bytes=17823170560; peak_vram_gib=16.6 |

## Artifacts

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `predictions.jsonl` | 200978 | `f958c87c83d31b7f28468e6637c22a3f45df338f36fc94f36c317f034a119428` |
| `input_traces.jsonl` | 174546 | `cbf789fe4261f0431d4949c398eba424036c8c2d01400210112ef8dca2bbd133` |
| `retrieval_candidates.jsonl` | 305073 | `14f29dde5ae9f5484b5ae404946b42a695e468eaf24821df34d32aa0dc86c2dd` |
| `query_manifest.json` | 1231 | `79c01549be28bc4ce62c8893c73716736ed01f452c6f81e2e3dab7225395cc53` |
| `run_receipt.json` | 1159 | `e792d35ba5b9d3e8a45d5b7278a6d3542961426797a8be906f3e19b0b5703ae3` |
| `PILOT_A1_EXECUTION_REPORT.md` | 512 | `cf28d751e04a269698e454023d8d70ab9e54f4dbd5e80febb5b630a475d8a0ff` |
| `PILOT_A1_INTEGRITY_REPORT.md` | 485 | `366b1cf200ffdda00eae9cfdc77ae2577ae66d1dcbd22a464068df44adde4fc1` |
| `PILOT_A1_GO_NO_GO.md` | 202 | `591bd69d54bd4a4cbf3957ec47bf41953bf6b87063c794bcecd9ad2f42bbf4ec` |
