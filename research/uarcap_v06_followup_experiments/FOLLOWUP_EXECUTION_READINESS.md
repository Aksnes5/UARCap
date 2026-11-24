# Follow-up execution readiness

| Gate | State | Evidence and remaining work |
| --- | --- | --- |
| `DESIGN_READY` | YES for A and B | Arm matrix, estimands, source families, queries, margins, failure rules, and ordered interpretation fixed after the v0.6 commit. |
| `IMPLEMENTATION_READY` | YES, Windows static and CPU preflight | Eight CPU unit tests, full-pool construction against 2,990 records, 2,618 eligible B queries, exact D head lookup, deterministic fingerprints and mock tensor hooks. Future real-model smoke still required. |
| `SERVER_DATA_READY` | UNVERIFIED | No BN connection. Verify physical test TSVs/offsets, checkpoint, tokenizer, model cache, selected clip features, runtime environment and device allocation. Git availability of JSON inputs does not establish server cache availability. |
| `GPU_EXECUTION_APPROVED` | NO | No inference launched. The parameter-free default of the launcher prints the argv, without importing torch, allocating a GPU, or running the evaluator. |

The corpus and exact run settings in v0.6 remain frozen. The CPU preflight only builds input plans and tests code. It generates neither captions nor new performance metrics. The staged JSON has input hashes and eligibility only. Treat any GPU smoke as execution requiring future approval. An actual approved deployment must fill its data and runtime receipts, verify the existing cache and weights, and use a dedicated checkout and fresh run ID. Never write new outputs into historical directories.

First validate the launcher and decoder environment without interpreting any metrics. Inspect the paired data fingerprints, visual parity and deterministic output ordering. If the real run produces a visual, frame, architecture-flag, source, reference, prompt or ID mismatch, stop and diagnose the implementation within the locked design. Do not use any partial result for a manuscript claim.
