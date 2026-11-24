# Deployment dry-run report

Windows local checkout at protocol commit `ee7d3b9876dea94a1ee9eeb2626b92102c85d458`. The archive replay and deployment adapter run on Windows without GPU approval.

| Check | Result |
| --- | --- |
| Frozen Python unit suite | 8/8 passed. |
| Deployment Python CPU suite | 6/6 passed with PyTorch `2.6.0+cpu` in an isolated Windows temporary directory. The tests use selected function/class method bodies extracted from the pinned Git source and CPU fixture tensors/tokenizer/encoders, including active query-side dropout and a check for RNG drift, with no model loader or decoder. |
| Source binding and real input preparation | Verified all five input source SHA-256 digests and six pinned source hashes, the original YAML, both ordered cohorts, three prediction archive pairs and all rebuilt frozen fingerprints. The archive witness also pins selected data-loader, model and Q-Former dependencies for future staging. |
| Six-arm launcher rehearsal | A0 (0/0/0), A1 (1/1/1), A7 (7/7/7), B3 (7/3/3), B7HQ and B7LQ (7/7/3), in outer/text/visual order. Verified CAP/SIM paths against real CPU-built or immutable archived inputs. Default dry-run returned JSON and left the proposed execution root absent. |
| Disallowed execution and repeats | Tested refusal before asset access without approval; compared separate full, smoke and numbered retry paths. Retry retains logical run ID but gets independent checkpoint slot and output path. Real mode rejects an existing output path and a second GPU lock holder. |
| Output and statistics schema | Tested trace rows against the unchanged scorer's `check_trace` and the extended gates on the CPU. Altered observations were rejected. The bundle validator emits the unchanged scorer's three per-arm keys after validating the real future completion markers/hashes, receipts, data, prediction order and reference lists. No fabricated prediction rows and no end-to-end real scoring were produced. |
| Remote probe | `bash -n` passes. Existing BN endpoint could not be resolved. Device locking, runtime imports, offsets, data-root availability, GPU and model loading are awaiting server verification. |

CPU reproduction from the repository root:

```text
python -B research/uarcap_v06_followup_deployment/audit_archives.py --repo . --scratch <new-scratch>
python -B -m unittest discover -v -s research/uarcap_v06_followup_experiments
python -B -m unittest discover -v -s research/uarcap_v06_followup_deployment
python -B research/uarcap_v06_followup_deployment/dry_run_matrix.py --archive-scratch <new-scratch>
```

Set `UARCAP_ARCHIVE_SCRATCH` to the first command's scratch path and make a local Torch CPU build available for the deployment suite. The archive replay refuses to replace a pre-existing scratch directory. No new caption, metric, statistical output or real smoke was created.
