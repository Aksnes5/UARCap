# Known bugs, invalid artifacts and scientific risks

## A. Invalid or partial artifacts (do not use as final results)

1. **E3 formal `E3_NEU` (Video-LLaMA) is invalid.**
   - `E3_FORMAL_APPROVED.json` has `neutral_text=null` for all 159 candidates.
   - The generator process that was already running fell back to the constant
     `"reading a book in a library"`.
   - All 159 `E3_NEU` rows therefore test one constant unrelated action, not a
     candidate-specific neutral condition. BASE/SUP/CON remain usable as machine-assisted
     conditions.
   - Source fallback was removed in
     `/home/bn/uarcap_q2_runtime/overnight/20261126/tasks/e3_formal_generate.py`, but the
     generated rows cannot be repaired without regenerating.

2. **Formal E3 Qwen is empty.**
   - 0 predictions / 636 failures, all `MISSING_NEUTRAL_TEXT`, because the strict gate was
     applied to every arm. The E3 formal evaluation (1272 rows) is Video-LLaMA only.
   - Do not describe it as a two-model E3 result.

3. **E3-Bridge is not approved.**
   - 60 candidates / 180 jobs exist; no VLM verification has run. `eligible=false`,
     status `PENDING_VERIFICATION`. Do not treat candidates as valid.
   - `source_id` is `None` / `UNKNOWN_OPAQUE` for CaReBench-derived bridge candidates.

4. **SPICE is missing.** `E2_METRIC_BRIDGE_RECEIPT.json` says `SKIPPED`. Do not report a
   SPICE number.

5. **Historical reverse bridge is not a fact-level bridge.**
   - `HISTORICAL_REVERSE_BRIDGE.json` found 424 metric files, many duplicates. No
     historical fact labels exist, so only metric directions are discussable.

6. **Validation package has no human data.**
   - 360 real caption records exist and the blind key is withheld, but no annotation,
     adjudication or calibration exists. All human gates are false.

## B. Code defects / interfaces to fix

1. `uarcap_q2/final_analysis/supplementary.py`
   - `build_e3_bridge_candidates_v2` chooses neutral by an arbitrary hash-min rule; it is
     deterministic but not scientifically validated.
   - `merge_e3_bridge_verification` expects verdict rows with
     `bridge_id, role, parse_ok, video_truth, visible_in_16_frames`; the verifier that
     produces this format is not yet implemented.
   - `historical_reverse_bridge` recursively scans large trees and returns duplicate metric
     files; curate a frozen source list.
   - `e3_authenticity_audit` reports the currently available files; it will silently show 0
     eval rows when the evaluation directory is absent. Treat zeros as “missing”, not
     “passed”.

2. `uarcap_q2/final_analysis/pipelines.py`
   - `verify_human_primary_endpoints` requires an external adjudication receipt file and its
     SHA-256, ≥2 annotators, exact pre-registered inclusion probabilities, and ≥1 audited
     unit per stratum. This is intentionally strict but has no dedicated unit tests yet.
   - `prepare_validation_package` assumes the frozen formal predictions already cover all
     60 validation videos. That is true for the current freeze; re-check if sampling
     changes.

3. `scripts/run_final_analysis.py`
   - `gates()` requires a real `A3_HUMAN_VALIDATION_RECEIPT.json` and
     `FORMAL_B_AUTHORIZATION.json`; neither exists. Modes correctly refuse.
   - `--human-calibrated` additionally requires
     `calibration_receipt.json["human_calibration_completed"] is True`, which is false.
   - The `stats.json` schema changed (primary/auxiliary nested). Any downstream consumer of
     the old flat schema must be updated.

4. `uarcap_q2/common/schemas.py` + `retrieval/arm_builder.py`
   - `ARM_A5` was added. CPU tests pass (73/73), but **A5 has not been executed on GPU** and
     no A5 prefix-invariant test was added for the new arm. Add one before trusting A5.

5. `tasks/supp_generate.py`
   - Length-control and 32/64-frame visual sensitivity rely on Qwen only. Video-LLaMA is
     frozen to 16 frames and a fixed prompt; any attempt to reuse it for those conditions
     is a protocol violation.
   - The script's `--qids-file` accepts either JSONL or JSON with `query_ids`; this branch is
     not covered by tests.
   - No evaluation/cross-judge jobs for these outputs are deployed yet.

6. E3 evaluation
   - Parse failures are now `expressed=None` and excluded from pairs. Rows are appended;
     later retry rows win during aggregation. Do not count `n_rows` as independent pairs.
   - `n_pairs=636` in the current receipt is Video-LLaMA only.

7. `e2_metric_bridge.py`
   - R-Full references are long descriptions (mean 228 words), not captions; BLEU/CIDEr
     collapse to ~0. This is a reference-format mismatch, not a model failure.
   - Single-reference CaReBench CIDEr/BLEU are not comparable to multi-reference
     MSR-VTT/MSVD numbers.

## C. Scientific risks

1. **No human validation anywhere.** A2 is `PILOT_A2_HUMAN_VALIDATION_PENDING`; A3 and E3
   are exploratory/unvalidated. Machine judges are not human Gold.
2. **Judge bias.** Llama vs Gemma differences are systematic (≈ −0.04 entailed,
   +0.04–0.06 contradicted) and stable across budgets; this is a judge bias, not a budget
   effect.
3. **Model-system contrast, not architecture.** Video-LLaMA receives EVCap text+visual
   fusion; Qwen receives text-in-prompt + raw frames (`PROMPT_TEXT_ONLY`). Pretraining,
   prompts and injection interfaces differ. No isolated architectural causal claim.
4. **16 uniformly sampled frames ≠ full video.** Visibility/answers may change with frame
   coverage; the full-video sensitivity is not implemented.
5. **Length confounding.** Longer captions have lower support in the current data
   (A7 Spearman −0.272; query-FE word coefficient −0.0112). Do not attribute budget effects
   to retrieval without length adjustment.
6. **Retrieval relevance is unlabeled.** Lexical overlap and X-CLIP scores are proxies; no
   human relevance labels exist.
7. **CaReBench source independence unprovable** because parent IDs are anonymized. Paper
   must state this.
8. **E3 candidates were machine-screened**, not human-verified. The 159 “approved” are
   machine-approved only.
9. **Post-hoc analyses.** Metric bridge, length, retrieval, contamination and Pareto
   analyses were added after initial results; label them exploratory.
10. **Single reference / dataset shift** for traditional metrics; absolute values must not
    be compared across datasets.
