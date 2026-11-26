# UARCap Q2 — current project status at BN→laptop handoff

- Date: 2026-11-26 ~14:45 CST (UTC+8)
- Repo: `https://github.com/Aksnes5/UARCap`
- Branch: `antigravity/uarcap-q2-bn-to-laptop-handoff-20261126`
- Last pre-handoff fixed baseline: `43b642d`
- Scientific status: **machine-assisted / exploratory only. No human validation has been completed. No confirmatory claim is authorized.**
- Handoff status: `BN_TO_LAPTOP_HANDOFF_READY`

## 1. What the project is

UARCap Q2 studies retrieval-evidence budgets and controlled evidence conflicts for
video captioning factuality, using:

- **E2**: 600 frozen CaReBench videos; arms A0/A1/A7; frozen EVCap Video-LLaMA v0.8 and
  Qwen3.8-27B-FP8; Gemma-3-27B + Llama-3.3-70B judges.
- **E3**: VidHalluc-ACH controlled Base/Sup/Con/Neu interventions.
- **A2/A3**: 90-caption pilots for automatic fact evaluation, cross-judge reliability and
  modern-model behaviour.

## 2. Real completed results (verified from files)

### A1 / A2 / A3 pilots
- A1: 90/90 captions and frozen inputs.
- A2 automatic evaluation: 90/90 captions, 0 parse failures; cross-judge 119/119 unique
  claims, Gemma–Llama agreement 0.857, Cohen κ 0.671.
- A2 visual verification (Qwen 16 frames): 41 flagged claims; 28 entailed / 11 contradicted
  / 2 unverifiable; 39/41 fully visible.
- A3 Qwen3.8: 90/90 generations; eval 90 rows; Llama cross 260 claims, agreement 0.804,
  κ 0.604; Qwen visual 120/120.
- A3 Qwen3.5 backup: 90/90 generations and evaluation completed.
- A3 remains `EXPLORATORY_COMPLETED_UNCALIBRATED`; A2 remains
  `PILOT_A2_HUMAN_VALIDATION_PENDING`.

### Formal CaReBench (600 pre-frozen candidates, 60-video audit subset)
- Video-LLaMA: 1800/1800 predictions (600 × A0/A1/A7), 0 failures, 6525 s.
- Qwen3.8: 1800/1800 predictions, 0 failures, 1910 s.
- Gemma evaluation: 1800 rows for each model, 0 parse failures.
- Llama cross-judge: 2660 Video-LLaMA claims (agreement 0.8774, κ 0.703); 5098/5099 Qwen
  claims (agreement 0.8723, κ 0.710), one claim pending.
- Formal integrity audit: **20/20 checks passed** (counts, trace linkage, A0 zero evidence,
  A1⊂A7 prefix, frame hash, checkpoint, retrieval, feature, decode, pilot/validation isolation).
- Sampling: 913 parents → 87 same-parent siblings dropped → 30 pilot parents excluded →
  883 eligible → 600 formal → 540 main + 60 validation, parent isolation verified.
- **Residual risk:** CaReBench parent IDs are anonymized; YouTube-level source independence
  cannot be proven.

### E3 pilot (VidHalluc-ACH)
- 16.75 GB downloaded, SHA-256 verified, extracted; source dedup and exclusion recorded.
- 30 candidates → text screen 30/30 → 16-frame visual screen 30/30 → **20 approved**.
- 160/160 generations (20 × 4 arms × 2 models).
- 320 dual-target eval rows / 160 pairs; Gemma–Llama agreement 0.9125 (true target) and
  0.9875 (conflict target).
- Exploratory only; no significance claim.

### E3 formal (partial; current checkpoint)
- 180 candidates → **159 machine-approved** (≥150 gate).
- Video-LLaMA: **636/636** predictions (159 × 4 arms), 0 failures.
  - BASE/SUP/CON are usable machine-assisted conditions.
  - **E3_NEU is INVALID for scientific use**: the frozen approved file has
    `neutral_text=null`; the already-running generator used the constant fallback
    `"reading a book in a library"`.
- Qwen formal E3: **0 predictions / 636 failures** because a strict
  `MISSING_NEUTRAL_TEXT` gate was added before its run. Status: BLOCKED.
- Dual-target evaluation: **1272 rows / 636 pairs, Video-LLaMA only**; 0 parse failures;
  true-target agreement 0.9575, conflict-target agreement 0.9733.
- No Qwen formal E3 evaluation exists. No human screening.

### New CPU results after `43b642d`
- **E2-Metric-Bridge** on the same 600 videos / 3600 captions, 7200 scored rows.
  - R-Full is a long human description (mean 228.1 words, median 218), so BLEU/CIDEr are
    ~0 and must not be interpreted as caption quality.
  - R-Short is a 100%-valid human substring (mean 14.5 words).
  - R-Short CIDEr: Video-LLaMA A0/A1/A7 = 0.3103/0.3534/0.3664; Qwen =
    0.3393/0.3423/0.3419.
  - SPICE was **SKIPPED**, not completed.
  - Paired video-level CIDEr-vs-fact relations are near zero (|ρ| ≤ 0.10); traditional
    metrics do not track fact support in this dataset. Post-hoc, single-reference.
- **Length analysis**: longer captions have lower support rate (A7 Spearman
  −0.272 Video-LLaMA; query-fixed-effect word coefficient −0.0112). Length is a confound.
- **Retrieval relevance**: 600 queries, mean top-1 X-CLIP score 0.4654, top-7 lexical
  overlap with core event 0.1144, with full reference 0.0241. Lexical overlap is not
  human relevance.
- **Natural evidence provenance**: A0 has zero text/visual evidence; A1/A7 have 1/7 text
  evidence; Video-LLaMA also receives visual clip evidence, Qwen does not
  (`PROMPT_TEXT_ONLY`). This is not an architecture-isolated comparison.
- **Contamination/source isolation**: validation∩main = 0, pilot∩formal = 0, parent
  isolation true, 159 unique VidHalluc sources in formal E3. Cross-dataset join is
  **NOT_JOINABLE** because CaReBench IDs are anonymized.
- **Cost Pareto**: formal Video-LLaMA 6525 s/1800 = 3.63 s/gen; Qwen 1910 s/1800 =
  1.06 s/gen; Gemini/Gemma eval projection marked extrapolated.
- **Historical MSR-VTT/MSVD bridge**: 424 existing metric files found (no rerun), but no
  historical fact-level labels; only directions are discussable. Status is partial.
- **Strict 60-video validation package**: 60 videos × 2 models × 3 arms = **360 caption
  records**, all backed by real frozen predictions/evals/frames; 0 missing-claim records;
  per-qid inclusion probabilities 0.0976–0.1034; blind key NOT committed.
- **E3-Bridge**: 60 natural-conflict candidates and 180 verification jobs built. VLM
  verification and approval are **not run**; candidate approval is BLOCKED pending it.

## 3. Code state

Uncommitted development completed immediately before handoff (committed in this handoff):

- P0 gate fixes: real receipts, no hard-coded gates, input SHA-256 version binding.
- Strict `(model_id, query_id, arm)` validation package.
- Primary endpoints restored: `n_contra_ref`, `r_core_ref`; Holm family primary-only.
- Cross-model budget differential: `(Qwen_A7−Qwen_A0) − (VideoLLaMA_A7−VideoLLaMA_A0)`.
- Two-phase IPW correction uses `uarcap_q2/statistics/two_phase_correction.py`; old
  precision/recall correction removed.
- E3-Bridge v2 candidate/verification/approval code with explicit BLOCKED status.
- ARM_A5 added to schema and arm builder (prefix A3⊂A5⊂A7); A3/A5 supplementary generation
  script implemented, not run.
- Supplementary CPU analyses (`scripts/run_supplementary.py`).
- E2 metric bridge (`scripts/e2_metric_bridge.py`).
- E3 parse-failure policy fixed to `expressed=None`, never coerced to 0.
- CPU test suite: **73/73 passed** at handoff.

## 4. What is NOT done / not authorized

- Any human annotation, adjudication, human calibration or human E3 screening.
- Any confirmatory E2/E3 statistic or paper claim.
- A3/A5 budget extension (code ready; 2400 generations not run).
- Length-control generation (code ready; 360 Qwen generations not run).
- Visual-sampling 32/64 sensitivity (code ready; 300 Qwen generations not run).
- E3-Bridge VLM verification (180 jobs built; not run).
- Qwen formal E3 generation and evaluation (BLOCKED on candidate-specific neutrals).
- SPICE.
- A valid candidate-specific neutral design for formal E3.

## 5. Current server state at handoff

- Orchestrator `run_overnight.py` PID 1779396: alive, **idle** (no GPU task).
- Supervisor PID 1779399: alive; started 02:09 with `--duration-hours 14`, so it will
  naturally exit around 16:09 local unless restarted.
- Report writer PID 1789311: alive.
- Download watcher PID 1787162: alive.
- GPU 0% utilization, 2 MiB used at last heartbeat — no inference is being interrupted.
- Task state: 60 tasks → 38 COMPLETED, 12 BLOCKED, 9 PENDING, 1 RUNNING (orchestrator).
- No STOP_REQUESTED file.
