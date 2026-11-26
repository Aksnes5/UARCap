# Anti-Gravity next prompt — UARCap Q2 laptop development

You are taking over **code development only** for UARCap Q2 on a Windows laptop with no
A6000, no videos, no model weights and no large feature caches. The BN server keeps
running the real GPU work independently.

## Context to read first

Repo: `https://github.com/Aksnes5/UARCap`
Branch: `antigravity/uarcap-q2-bn-to-laptop-handoff-20261126`

Read, in this order:

1. `research/uarcap_q2_experiments/handoff_20261126/CURRENT_PROJECT_STATUS.md`
2. `.../COMPLETED_EXPERIMENTS_INDEX.json`
3. `.../RUNNING_TASKS.json`
4. `.../BN_RUNTIME_INTERFACE.md`
5. `.../KNOWN_BUGS_AND_SCIENTIFIC_RISKS.md`
6. `.../REMAINING_DEVELOPMENT_TASKS.md`
7. `.../RETURN_DEPLOYMENT_GUIDE.md`

Then inspect the actual code, tests and samples. Do not trust summaries alone.

## Mission

Finish the P0 code tasks, add CPU tests, and prepare GPU tasks that BN can run later.
Do **not** claim any scientific result is validated.

## P0 acceptance criteria

### P0.1 Candidate-specific neutral evidence for formal E3
- Add deterministic neutral assignment from a pre-declared pool.
- Validate textually: zero content-token overlap with f+/f−, natural, mutually exclusive.
- Add an independent VLM visual-absence verification interface (do not run it on laptop).
- A new approved file must be produced; never mutate `E3_FORMAL_APPROVED.json`.
- Reject candidates whose neutral is visible; emit BLOCKED if fewer than 150 remain.
- The existing formal `E3_NEU` rows are invalid and must stay marked invalid.

### P0.2 E3-Bridge verifier
- Implement `vlm_statement_verify.py` for the 180 jobs:
  `uarcap_q2_runtime/outputs/supplementary/E3_BRIDGE_VERIFICATION_JOBS.json`.
- CPU-testable parser and merge; GPU verifier must be a separate task.
- Approval requires: f+ entailed + visible yes; f− contradicted; neutral not entailed;
  every verdict parsed. <30 valid => `BLOCKED_INSUFFICIENT_VERIFIED_CANDIDATES`.
- Never use an external caption as “truth”.

### P0.3 Human validation tooling
- Ensure the 360-caption, 60-video package can be annotated blindly by two annotators.
- Export the exact `uarcap_q2.human_primary_endpoints.v1` schema.
- Refuse incomplete imports; no default inclusion probabilities; no fabricated agreement.
- Add tests for: missing annotator, wrong schema, out-of-set query, wrong inclusion
  probability, missing stratum, calibration leakage.

### P0.4 Supplementary generation deployment
- Prepare queue-task JSON fragments (not executed on laptop) for:
  - A3/A5: 2400 generations (600 × 2 × 2).
  - length control: 360 Qwen (120 × 3).
  - visual sensitivity: 300 Qwen (150 × 2 frame settings).
- Add a generic Gemma-eval + Llama-cross task spec for these outputs.
- Ensure resume keys and output dirs are unique.

### P0.5 E3 Qwen completion
- After neutrals, queue Qwen E3 generation and dual-target evaluation.
- Keep `expressed=None` on parse failure.
- Tests must prove no coercion to 0.

## Hard constraints

- Do not download VideoUFO, CaReBench, VidHalluc, or model weights.
- Do not require a GPU for tests.
- Do not modify the frozen 3600 formal predictions.
- Do not commit videos/frames/features/weights/blind keys/credentials.
- Do not set any human gate to true without a real signed receipt.
- Do not change primary endpoints or the Holm family post hoc.
- Preserve `ARM_A0` empty-evidence and A1⊂A3⊂A5⊂A7 prefix invariants.
- Keep the existing 73 tests passing; add more.

## Deliverable

- A branch/PR with code + tests + updated handoff docs.
- A short report:
  - what changed,
  - tests run and results,
  - GPU tasks BN must run (exact commands, expected generations/time),
  - what remains blocked or awaiting human.
- Do not push to a branch BN is using without a PR.

Start by running the CPU tests and reading the actual `E3_FORMAL_APPROVED.json` and
`E3_BRIDGE_VERIFICATION_JOBS.json` samples in the handoff folder.
