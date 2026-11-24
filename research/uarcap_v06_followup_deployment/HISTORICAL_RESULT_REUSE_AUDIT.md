# Historical result reuse audit

| Arm | Historical witness | Status | Formal generation now |
| --- | --- | --- | --- |
| A0 | No matched zero-evidence archive | `NOT_REUSABLE` (new condition) | Required |
| A1 | FULL D1 | `CONDITIONALLY_REUSABLE` | Required |
| A7 | FULL D7 | `CONDITIONALLY_REUSABLE` | Required |
| B3 | FULL D3 | `CONDITIONALLY_REUSABLE` | Required |
| B7HQ | No fixed-visual HQ-tail archive | `NOT_REUSABLE` (new condition) | Required |
| B7LQ | No fixed-visual LQ-tail archive | `NOT_REUSABLE` (new condition) | Required |

`CONDITIONALLY_REUSABLE` means the archived design maps to the planned condition and decisive runtime evidence is incomplete. `EXACT_REUSABLE` requires matching model, checkpoint, code, dataset bytes, effective inputs and generation settings tied to the historical process. An observed incompatibility disqualifies an analog. `UNVERIFIED` applies where even the decisive design evidence is unavailable.

## Verified outcome lineage

Git archive `52b5ed3a69e4b84a8e16ed747f141eefb5b1e2ec` contains both the full test2990 prediction file and test2690 prediction file for each of D1, D3, D7. All three full files exactly match the frozen query order. Removing the same 300 pilot queries produces every complete archived test2690 record in the same order, including prediction strings, split labels and references. The full files each have 59,800 ordered reference strings (20 per query) with identical canonical SHA-256 `204563f74d9e4354863289760f65a767dde8d9e189d7e99dc201505847b15271`. File-specific sizes, SHA-256 and path strings are in `ARCHIVE_EVIDENCE.json`. This establishes result and cohort lineage. The import commit identifies the archived bytes, with historical producer revision still unknown.

## Effective settings and remaining proof

| Field | Design-side evidence | Historical runtime gap |
| --- | --- | --- |
| Test branch and query frames | `scripts/run_test.sh` points to the ordered test2990 evaluator and pre-extracted 16-frame YAML. Archived predictions, query order and refs align exactly. | No process-bound YAML/TSV/offset hashes or query frame tensors. |
| CAP/SIM and budgets | Runner passes FULL `external_test_d7_cap.json` and `external_test_d7_sims.json`. Archived D1/D7 outer k resolves both inner prefixes. A1 and A7 point to those byte-verified blobs and explicitly pin the same inner budgets. | No run receipt binds the loaded source bytes to each historical process. |
| Prompt, flags, beam and seed | Model runner and evaluator support the same prompt skeleton, `bs=1`, seed 42, 16 frames, 32 query tokens, default beam 5, deterministic representative selection, cached features, freeze/share flags, and cross-gating. | No historical rendered prompt/embedding and execution argument receipt tied to each output. |
| Checkpoint | Runner names the pinned checkpoint. `reports/audit_checkpoint_usage.md` records 147,337,150 bytes and SHA-256 `a8b5af79d55a2c7e684771475d645be6373208bdaae5a4211bee556581eb8002`. | Individual D1/D3/D7 jobs lack process-bound evidence of the bytes consumed and load result. |
| Vicuna/tokenizer, feature cache and producer code | Runner points to Vicuna-7B-v1.3 and faithful Video-LLaMA feature root. Exact archived evaluator/model/prompt/decoder hashes are recorded. | No per-run model/tokenizer/cache, runtime environment or producer revision receipt. |

## B3 versus D3

All 2,990 B3 inputs preserve the first three D caption strings, representative `clip_group[0]` and raw SIM lookup values. The legacy D3 loader uses only the first three text and visual entries. B3 uses explicit inner (3,3) with outer `topn=7`; archived model references to `self.topn` only assign the stored value and set defaults via `topn` when inner args are absent. The changed raw CAP/SIM serialization is therefore consistent with the same model-visible prefix. B3 has three groups and no tails. Newly added B7 tails use empty groups and do not enter the visual prefix. The CPU check proves the representation-level contract and mock tensor behavior. The comparison still lacks historical runtime feature, weight, aggregate, intermediate and wrapped-prompt integrity evidence.

No historical condition meets the exact-reuse bar. A new smoke will validate the new deployment for those queries and cannot certify how all 2,990 historical rows were produced. Count the six formal arms and zero GPU savings in the current resource plan. If one or more historical runs later obtain conclusive runtime proof, obtain a fresh reuse adjudication before changing this run matrix or permitting the scorer to consume historical outputs.
