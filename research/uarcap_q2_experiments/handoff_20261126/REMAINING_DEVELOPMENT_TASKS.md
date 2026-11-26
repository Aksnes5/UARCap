# Remaining development tasks for Anti-Gravity (P0/P1/P2)

Statuses are based on actual files and artifacts, not on prior plans. The BN server keeps
the real GPU only; the laptop has no A6000, no videos and no model weights.

## P0 — must finish before any new GPU science

### P0.1 Candidate-specific neutral evidence for formal E3
- **Status:** PARTIALLY_IMPLEMENTED; formal E3_NEU invalid and Qwen BLOCKED.
- Problem: `E3_FORMAL_APPROVED.json` has `neutral_text=null` for all 159 candidates.
  The already-running Video-LLaMA job used a constant fallback
  `"reading a book in a library"`; Qwen was blocked by a strict missing-neutral gate.
- Work: implement deterministic neutral assignment from a pre-declared pool with
  (a) zero content-token overlap with f+/f−, (b) text naturalness/exclusivity screen,
  (c) independent VLM visual-absence verification, (d) rejection when invalid.
- Files: `uarcap_q2/final_analysis/supplementary.py` (`NEUTRAL_POOL`, candidate builder),
  `/home/bn/uarcap_q2_runtime/overnight/20261126/tasks/e3_formal_generate.py`.
- Then: write a **new** approved JSON (never mutate the frozen original), regenerate only
  missing NEU and Qwen E3, and re-run dual-target evaluation.

### P0.2 E3-Bridge independent verification and approval
- **Status:** PARTIALLY_IMPLEMENTED. 60 candidates and 180 jobs exist; no VLM verifier.
- Work: implement the statement verifier (Qwen 16-frame, `entailed/contradicted/unverifiable`
  + visible flag), run 180 jobs, merge, and require f+ entailed & visible, f− contradicted,
  neutral not entailed, all parsed. If <30 valid, emit `BLOCKED` — do not relax.
- Files: `supplementary.merge_e3_bridge_verification`, `E3_BRIDGE_VERIFICATION_JOBS.json`,
  `scripts/run_supplementary.py --e3-bridge-merge`.
- Missing: verifier task script (planned `vlm_statement_verify.py`).

### P0.3 Human endpoint import and validation UI
- **Status:** IMPLEMENTED code / AWAITING_HUMAN.
- The 60-video package has 360 real captions and a blind key kept off GitHub.
- Work: annotator UI flow for 360 captions (phase-1 core events, phase-2 claims), two
  independent annotators, adjudication export in
  `uarcap_q2.human_primary_endpoints.v1`, signature/hash receipt, then
  `run_final_analysis.py --human-calibrated --human-annotations <file>`.
- Files: `pipelines.verify_human_primary_endpoints`, `run_final_analysis.py`,
  `uarcap_q2/annotation/package_builder.py`.
- Do not fabricate inclusion probabilities; they are in the frozen manifest.

### P0.4 Deploy and evaluate supplementary generation
- **Status:** IMPLEMENTED_PENDING_RUN.
- A3/A5: 2400 gen (600 × 2 arms × 2 models).
- Length control: 360 Qwen gen on 120 videos (detailed prompt).
- Visual sampling: 300 Qwen gen on 150 videos (32/64 frames).
- Missing: queue entries, generic Gemma eval + Llama cross tasks for these outputs,
  and analysis of length/frame effects with uncertainty.
- Script: `/home/bn/uarcap_q2_runtime/overnight/20261126/tasks/supp_generate.py`.
- Important: Video-LLaMA is frozen to 16 frames/constant prompt; do not change its
  contract for length/frame experiments.

### P0.5 E3 formal evaluation and Qwen completion
- **Status:** PARTIAL. 1272 eval rows are Video-LLaMA only; Qwen has 0 predictions.
- Work after P0.1: regenerate Qwen, re-run evaluation, ensure `expressed=None` on parse
  failure (already patched), and recompute cross-judge only over complete pairs.
- Files: `tasks/e3_formal_generate.py`, `tasks/e3_formal_evaluate.py`.

## P1 — scientific completeness

### P1.1 Text-only / visual-only / fusion fairness experiment
- **Status:** AUDIT ONLY. No controlled modality-removal generation exists.
- Work: add explicit `text_only`, `visual_only`, `text+visual` conditions with matched
  prompts for Qwen; document that frozen EVCap cannot be ablated without breaking v0.8.
- Files: `uarcap_q2/generators/qwen3_vllm_adapter.py`,
  `uarcap_q2/generators/modern_vlm_adapter.py`, `supplementary.fusion_fairness_audit`.

### P1.2 Length-confounding statistical model
- **Status:** DESCRIPTIVE ONLY (Spearman + query-FE OLS + quintile stratification).
- Work: pre-register a length-controlled primary model (e.g., within-video length strata or
  spline adjustment), with cluster uncertainty and human-calibrated endpoints. Do not
  promote post-hoc coefficients to confirmatory claims.

### P1.3 Historical MSR-VTT/MSVD reverse bridge
- **Status:** PARTIAL / BLOCKED ON LABELS.
- 424 historical metric files were found without rerun, but no historical fact-level
  annotations. Only directional discussion is possible; absolute values are not comparable
  (multi-reference vs single-reference).
- Work: locate/curate the valid historical eval directories, freeze a manifest, and either
  add fact labels or define the bridge as metric-direction-only.

### P1.4 SPICE
- **Status:** NOT_RUN (`SKIPPED`).
- Work: run SPICE in the frozen `repro_evcap` environment with Java, or keep it explicitly
  missing. Never interpolate.

### P1.5 CaReBench source provenance
- **Status:** BLOCKED partially. Parent IDs are anonymized; cross-dataset join is
  `NOT_JOINABLE`. Do not invent YouTube IDs.
- Work: obtain an official CaReBench source map if one exists; otherwise keep the caveat.

### P1.6 E3 full-video vs 16-frame sensitivity
- **Status:** NOT_IMPLEMENTED (Qwen frame sensitivity 32/64 planned only).
- Work: define full-video evaluation (frame budget ≠ full video), measure visibility and
  answer stability, and report the 16-frame contract as a limitation.

### P1.7 Retrieval relevance ground truth
- **Status:** DESCRIPTIVE lexical/X-CLIP proxies only.
- Work: human relevance labels for top-7 evidence on a stratified subset; report
  inter-annotator agreement. Do not call lexical Jaccard “relevance”.

## P2 — engineering and paper support

- E3-Bridge human review UI and adjudication.
- Generic resume/verification tooling for supplementary outputs.
- Paper figure/table regeneration from the final calibrated analysis (currently figures
  include PENDING markers).
- Cost model refinement (beyond measured s/generation).
- Repository cleanup: historical metric scan currently walks large trees and returns 424
  duplicate sources; curate to a frozen list.
- Remove/retire stale BLOCKED queue tasks or mark them SKIPPED with reasons.

## Explicitly NOT authorized

- Confirmatory E2/E3 statistics.
- Human calibration claims, human agreement, arbitration, or E3 human screening.
- Re-running the frozen 3600 formal captions.
- New large model/video downloads.
