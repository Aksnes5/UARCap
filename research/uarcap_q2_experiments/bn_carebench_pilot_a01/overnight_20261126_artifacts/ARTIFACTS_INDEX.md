# Overnight UARCap Q2 artifacts (2026-11-26)

All labels here are **machine-assisted / exploratory / unvalidated**. No human annotator
signatures, agreement, adjudication, or human-calibration completion are included.
A2 remains `PILOT_A2_HUMAN_VALIDATION_PENDING`.

## Formal CaReBench candidate generation (`formal/`)

- `FORMAL_CAREBENCH_ANALYSIS_MANIFEST.json`: pre-frozen 600-candidate manifest with 60
  independent validation IDs, pilot-parent exclusion, parent dedup audit, near-duplicate
  audit, strata and inclusion probabilities. Manifest SHA-256 in `FORMAL_MANIFEST_RECEIPT.json`.
- `formal_candidates.jsonl`: 600 selected candidates (query, parent, category, reference,
  core-event candidate, `in_e2_validation`).
- `formal_core_lock.json`: references/core events frozen before generation.
- `formal_retrieval.jsonl`: 600 × VideoUFO top-20 + D rerank.
- `formal_frame_indices.jsonl`: 600 × 16 frames + sampled indices.
- `FORMAL_FEATURES_RECEIPT.json`: faithful_v2 feature top-up audit.
- `videollama/{predictions,input_traces,run_receipt}.jsonl/json`: 1800 frozen Video-LLaMA
  A0/A1/A7 predictions (`FORMAL_CANDIDATE_GENERATION_UNVALIDATED`).
- `qwen/{predictions,input_traces,run_receipt}.jsonl/json`: 1800 Qwen3.8 A0/A1/A7 predictions.
- `eval_old/`, `eval_qwen/`: Gemma 3 27B three-way automatic fact evaluations (1800 rows each).
- `cross_old/`, `cross_qwen/`: Llama 3.3 70B cross-judge labels and agreement summaries.

## A2 / A3 (`a2_machine/`, `a3/`)

- A2 cross-judge agreement/labels, machine annotations, human-review JSON, machine-assisted report.
- A2 Qwen 16-frame visual verification results/summary.
- A3 Qwen3.8 and Qwen3.5 90-generation predictions/traces/receipts, Gemma eval rows and
  Llama cross-judge labels/agreement; A3 Qwen visual verification.

## VidHalluc / E3 (`vidhalluc/`)

- Dataset source audit, 30 pilot candidates, text and 16-frame visual screening, approved/
  review lists, 180-candidate formal E3 pool, extraction receipt.
- E3 generation/evaluation artifacts are added after the pipeline completes.

## Audit (`audit/`)

- `FORMAL_INTEGRITY_AUDIT.{json,md}`: 20/20 checks (counts, trace linkage, A0 zero evidence,
  A1⊂A7 prefix, 16 frames/arm hash, constant checkpoint, retrieval prefix, features,
  decode, pilot/validation isolation, near-dup scan).
- `DESCRIPTIVE_RESULTS.{json,md}`: per-model/arm support/conflict/unmentioned/core coverage
  and Gemma-vs-Llama differential offsets by retrieval budget (descriptive only).
- `USER_REVIEW_TODO_FINAL.{json,md}`: 117 minimal human-review items, including 20 random
  judge-agreed audit samples.

## Deliberately not committed

Raw videos/frames, model weights/embeddings/features, blind annotation key, incomplete human
answers, `work/` TSV bundles, and multi-hundred-MB raw judge-call logs. These remain under
`/home/bn/uarcap_q2_runtime/` for local audit.
