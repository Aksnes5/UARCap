# Final pre-execution decision

| Experiment | Decision | Rationale |
| --- | --- | --- |
| A: matched k=0 | **CONDITIONAL GO** | Local implementation passes. Execution requires verified BN assets, GPU permission and actual A smoke with exact zero evidence and intact query/fusion route. |
| B: fixed-visual Text3→Text7 | **NO-GO for formal execution** | Fully specified and locally verified. Requires verified BN assets, explicit GPU permission, completed trace-verified A and an authorized B smoke proving identical visual inputs and operations across B arms. |

| State | Value |
| --- | --- |
| `DESIGN_READY` | YES, A and B |
| `IMPLEMENTATION_READY` | YES for Windows source, input and CPU deployment preflight. BN runtime integration remains unverified. |
| `SERVER_DATA_READY` | UNVERIFIED |
| `REAL_MODEL_PARITY_VERIFIED` | NO |
| `GPU_EXECUTION_APPROVED` | NO |

**P0 gates in order.** Reach the existing authenticated BN host and verify the full data, cache, checkpoint, environment and GPU inventory. Obtain explicit authorization for execution. Run and validate the real A smoke, then complete and validate the full A bundle. Run and validate a real B smoke before any B full generation. Fail closed on any B tensor/content mismatch. Score full, trace-verified bundles on Windows using the frozen analysis plan.

`MINIMAL_GPU_RUN_MATRIX.json` requires A0/A1/A7 and B3/B7HQ/B7LQ. D1/D3/D7 remain conditional historical analogs, with zero exact-reuse exemptions. The run matrix is 17,940 full-order generations (minimum historical range estimate about 8.4 GPU-hours plus smoke/overhead, direct anchor 14.2 h). Windows CPU tests and dry runs are complete. No captions, scores or GPU work were produced.
