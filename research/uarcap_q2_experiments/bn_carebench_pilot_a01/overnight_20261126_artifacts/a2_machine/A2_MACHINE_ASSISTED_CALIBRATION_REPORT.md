# Pilot A2 — Machine-Assisted Annotation Report

> **MACHINE-ASSISTED / UNCALIBRATED.** These labels are not human Gold and do not satisfy the human validation gate.
- Generated: 2026-11-25T18:19:51.693514+00:00
- Unique claims: 119; require human review: 17

## Label sources
- Gemma 3 27B FP8 + Llama 3.3 70B W4A16 text judges; Qwen3.8 visual verification available for 0 flagged claims.

## Per-arm machine labels (with 90-row experimental weights preserved)

| Arm | Claims | Labels / review |
| --- | --- | --- |
| ARM_A0 | 85 | {'total': 85, 'entailed': 61, 'REQUIRES_HUMAN_REVIEW': 11, 'contradicted': 6, 'unmentioned': 7} |
| ARM_A1 | 80 | {'total': 80, 'entailed': 59, 'REQUIRES_HUMAN_REVIEW': 11, 'contradicted': 6, 'unmentioned': 4} |
| ARM_A7 | 83 | {'total': 83, 'entailed': 58, 'REQUIRES_HUMAN_REVIEW': 10, 'contradicted': 6, 'unmentioned': 9} |

## Human review
See `A2_HUMAN_REVIEW_REQUIRED.json` — **17 items**, each with video path, 16 frame paths, the reference, all model opinions, and a single final-label choice.

## Integrity notes
- No human annotator name, signature, agreement or adjudication is generated.
- `UNMENTIONED` is never counted as a factual error.
- Qwen visual outputs are auxiliary and are not independent confirmation when Qwen also generates A3 text.
