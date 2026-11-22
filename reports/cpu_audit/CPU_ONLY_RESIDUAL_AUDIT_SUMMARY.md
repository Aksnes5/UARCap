# CPU-only residual scientific audit — summary

Date: 2026-11-22. GPU unavailable. No CUDA call, no generation, no training, no
X-CLIP embedding, no frozen generation output or frozen statistic was modified.
All analyses below are read-only re-evaluations of existing artifacts.

## 1. Status table

| Issue | Status | Key finding | Manuscript implication |
|---|---|---|---|
| Style/length drift | **RESOLVED** (descriptive) | Raising k monotonically increases output length and evidence copying. k=1→7: 3-gram copy rate 0.142→0.334 (test2690), 0.138→0.387 (MSVD); exact copies 1.1%→6.0%, 0%→3.9%. In the same sweep SPICE rises (8.82→9.12; 9.08→9.25) while CIDEr falls (71.21→68.27; 156.39→141.59). | Safe as a descriptive Results paragraph + Supplement table (test2690/MSVD). Do **not** write causal "copying causes the drop"; do **not** use val497 to support it. |
| Oracle pool depletion | **PARTIAL** | Ranks 4–7 are 25.3% lower in reference compatibility than ranks 1–3 (0.1031 vs 0.1380; 497/497 queries positive; 3→4 p<1e-4). But the decay is smooth (3→4 drop 0.0092 < 1→2 drop 0.0209), with no 3→4 cliff. Per-query depletion does not predict the per-query Oracle3−Oracle7 CIDEr loss (Spearman −0.04; high- vs low-depletion halves 1.71 vs 2.19 points). | Supplement / Limitations. Candidate depletion is a real contributor but cannot be claimed as the main explanation; integration overload cannot be excluded. |
| Val−test CIDEr gap | **RESOLVED** (EXPLAINED) | A val497 dataloader bug feeds the generator a **static first frame**: eval branch sets `start, end = 0, 0`; the PyAV decoder then keeps 1–2 frames and the temporal sampler replicates frame 0 15 times. Test/MSVD/dev300 use pre-extracted frame TSVs and are unaffected. References, PTB, scorer, IDF scope, retrieval candidate quality, checkpoint, and split membership were all checked and are consistent. | **Mandatory correction/disclosure.** val497 absolute numbers (≈50–55) are not comparable to test2690/MSVD or the original paper. Paired within-val contrasts remain internally valid because all conditions share the same visual input. |
| Source×budget interaction | **RESOLVED** | Δ_external = +3.08 [0.70, 5.68], p=0.0138; Δ_indomain = −0.29 [−2.31, 1.69], p=0.78; interaction = **+3.37 [0.30, 6.69], p=0.0365**, source-cluster CI [0.27, 6.88]; n=497, 361 sources. | Secondary analysis for v0.5.1 / Supplement: the D3→D7 penalty is external-specific in the matched control. Does not replace the two original separate tests. |
| Matched k=0 | **BLOCKED_REQUIRES_GENERATION** | The only topn=0 run is `eval_dev300_A` (CIDEr 64.18), which is **PRE_FIX_DIAGNOSTIC**, pilot300 only, with `external_empty_cap.json` and `video_feature_dir=None`. No val497 / test2690 / MSVD k=0 output exists. | Do not use `eval_dev300_A` as matched k=0, and do not use Fang et al. 70.8. A matched k=0 requires new generation. |

## 2. The val/test gap is a generator-input bug, not a metric or split effect

This is the highest-impact finding. Evidence:

- `vision_language_tsv.py:338` sets `start, end = 0, 0` for eval; `video_decoder.py`
  turns 0/0 into a decode window of one frame; `temporal_sampling` replicates frame 0
  15 times (empirically: frames 0–14 byte-identical, frame 15 near-identical).
- End-to-end dataloader check: val497 item `video6513` → 15/16 frames identical to
  frame 0; test2990 and MSVD items → distinct frames.
- Sample val497 outputs describe frame 0 rather than the annotated content:
  video6518 frame 0 = Bank of America logo → prediction "an advertisement for bank of
  america" (references and all seven retrieved captions are about wrestling);
  video6637 frame 0 = book close-up → "a person is reading a book"; video6722
  frame 0 = football → "a football game is being played" (rest of video is BMX).
- Checked and excluded: reference files (match `MSR_VTT.json` exactly), 20 refs per
  video, reference length/consensus/LOO ceiling (val is slightly easier), PTB, CIDEr
  implementation, DF/IDF scope (common-DF gap still 17.9 points), retrieval candidate
  compatibility (val 0.1869 / 0.1511 / 0.1066 vs test 0.1899 / 0.1515 / 0.1047),
  checkpoint sha256, split membership, evaluation scope/format.

Verdict: **EXPLAINED**. The exact counterfactual "val497 with 16 distinct frames"
score is not measurable without generation → **BLOCKED**.

## 3. What to put in v0.5.1, Supplement, and Limitations

### Worth writing into v0.5.1

1. **Explicit val497 disclosure / correction.** State that the val497 generator saw a
   static first frame due to the video-path dataloader path, that this explains the
   ≈18-point val/test gap, and that val497 absolute scores are not comparable to
   test2690/MSVD or prior work. Paired val contrasts remain internally valid.
2. **Length/copying gradient (descriptive).** One Results paragraph + compact
   Supplement table using test2690 and MSVD: k↑ → length↑, evidence copying↑;
   SPICE↑ and CIDEr↓ co-occur. No causal claim.
3. **Source×budget interaction (secondary).** +3.37 points, 95% CI [0.30, 6.69],
   p=0.0365, source-cluster CI [0.27, 6.88]. This strengthens the external-specific
   reading of the D3→D7 penalty without touching the original tests.
4. **Oracle depletion as partial evidence.** Ranks 4–7 are systematically lower in
   compatibility, but there is no discontinuity and no query-level link to CIDEr;
   report as a candidate-quality contributor, not a decisive explanation.

### Supplement / Limitations only

- Full rank table (1–20), band summaries, adjacent-drop table, and the
  depletion-vs-CIDEr null result.
- The val497 static-frame details and the complete evaluation-pipeline checklist.
- The within-condition vs between-k sign reversal in the length/overlap correlations.
- The in-domain control's source/size confound in the interaction analysis.

### Reviewer attacks weakened

- *"SPICE up while CIDEr down is unexplained."* Now a quantified, replicated
  length/copying gradient with condition-level SPICE/CIDEr co-occurrence.
- *"The val/test gap suggests cherry-picking or a different metric."* Resolved: the
  metric pipeline is consistent; the gap is a dataloader input bug, now documented.
- *"Oracle7 < Oracle3 proves integration overload."* Weakened: candidate depletion is
  a real average contributor, so the Oracle7 result is not clean evidence of pure
  integration overload.
- *"The external budget effect is a split artifact."* Weakened: the source×budget
  interaction is positive with CIs excluding 0 under both query and source-cluster
  bootstrap, and the external D3→D7 direction replicates on test2690 and MSVD.
- *"Val numbers are comparable to the original EVCap paper."* Corrected: they are
  not, and the manuscript must say so.

### Still unresolved without GPU

- **Matched k=0**: BLOCKED_REQUIRES_GENERATION.
- **Counterfactual val497 re-run with a pre-extracted 16-frame 224 TSV**: BLOCKED.
  Until it exists, the magnitude (not direction) of every val-internal effect under
  normal visual input is unknown.
- Per-query SPICE (optional; only condition-level SPICE was used).
- Any new condition, rerun, or embedding.

## 4. Files generated by this audit

New reports:

- `reports/cpu_audit/VAL_TEST_CIDER_GAP_AUDIT.md`
- `reports/cpu_audit/STYLE_LENGTH_COPYING_AUDIT.md`
- `reports/cpu_audit/ORACLE_POOL_DEPLETION_AUDIT.md`
- `reports/cpu_audit/SOURCE_BUDGET_INTERACTION_AUDIT.md`
- `reports/cpu_audit/CPU_ONLY_RESIDUAL_AUDIT_SUMMARY.md`

Machine-readable / supporting:

- `reports/cpu_audit/style_length_copying_summary.csv`
- `reports/cpu_audit/style_length_copying_full.json`
- `reports/cpu_audit/cpu_audit_valtest_forensics.json`
- `reports/cpu_audit/val_frame_input_bug_evidence.json`
- `reports/cpu_audit/oracle_pool_depletion.json`
- `reports/cpu_audit/oracle_rank_compatibility.csv`
- `reports/cpu_audit/source_budget_interaction.json`
- `reports/cpu_audit/source_budget_interaction_per_query.csv`
- `reports/cpu_audit/frames/*.png` (sample decoded frames)

Audit scripts (new, not imported by any frozen pipeline):

- `scripts/cpu_audit/perquery_cider.py`
- `scripts/cpu_audit/cider_fixed.py`
- `scripts/cpu_audit/style_audit.py`
- `scripts/cpu_audit/valtest_forensics.py`
- `scripts/cpu_audit/frame_input_evidence.py`
- `scripts/cpu_audit/oracle_depletion.py`
- `scripts/cpu_audit/source_budget_interaction.py`

## 5. Repository / commit status

The audit was performed against the frozen reference commit
`e42c6953b29f9983141e814b17922cb2f7d99c5a` (the branch base is the later
`origin/main` commit `c31dc82`, which only adds documentation). The reports and
scripts in this directory are the formally committed copy on branch
`audit/cpu-residual-scientific-audit-20261122` of `Aksnes5/UARCap`. No tracked file,
frozen generation output, or frozen statistic was modified; the audit is additive
only.
