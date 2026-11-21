# Paper claim matrix (final, valid results only)

All CIDEr values are points (×100). Invalid/recomputed artifacts are listed at
the end and are not cited.

| # | Claim | Status | Experiment / evidence | Effect, CI, p | Result file |
|---|---|---|---|---|---|
| 1 | Large context budgets are rarely optimal across scales and rerankers | SUPPORTED | 6 scales × D/CF k-grid | k=7 optimal 0/12; k=1/3 10/12; k=5 only at 10K; k7 penalty mean D 2.06 / CF 1.60; combined 1.83 | `reports/scale_budget.md`, `reports/scale_budget.json` |
| 2 | Beyond 10K, every best budget is k=1 or k=3 | SUPPORTED | same grid | 10/12 best settings; exact optimum varies (1 vs 3) with scale/track | same |
| 3 | Best k is monotone in corpus size | NOT SUPPORTED | same grid | sequence 5/3/1/1/3/3 (D) and 5/3/1/3/3/1 (CF) | same |
| 4 | Textual context count drives degradation | STRONG | Text×Visual 2×2 | Text main −3.24 (CI [−5.68,−0.97], p=0.007); T3V7−T7V3 +3.37 (p=0.006) | `reports/text_visual_2x2_factorial_corrected.md` |
| 5 | Visual context count independently hurts | NOT SUPPORTED (current setting) | 2×2 | Visual main +0.13 (CI [−0.34,+0.63], p=0.62); interaction −0.78 (ns) | same |
| 6 | Prompt position/primacy explains the drop | NOT SUPPORTED | posrev, top3-last vs D7 | +0.09 / −0.09, p≈0.89 | `reports/variant_stats.json` |
| 7 | Evidence selection is a real bottleneck | SUPPORTED | fixed-DF Oracle1/3/7 | Oracle1−D1 +3.27 (CI [1.40,5.28], p=0.0008); Oracle7−D7 +3.13 (p=0.0036); Oracle3−D3 +2.05 (p=0.053) | `reports/oracle_distractor_stats.json`, `reports/oracle_fixeddf_audit.md` |
| 8 | Seven contexts overload even with oracle-selected evidence | STRONG | Oracle7 vs Oracle3 | −2.03 (CI [−3.95,−0.23], p=0.033) | same |
| 9 | External evidence overload vs matched in-domain | SUPPORTED | matched in-domain D1/3/5/7 | in-domain 54.16/55.14/55.00/55.43, D3−D7 −0.27 (ns); external 51.77/53.14/50.51/50.05, D3−D7 +3.11 (p=0.012) | `reports/in_domain_vs_external_budget_control.md` |
| 10 | The effect is generic to the generator regardless of corpus | NOT SUPPORTED | same control | generator absorbs 7 in-domain contexts without loss | same |
| 11 | A small set of toxic low-ranked items explains k=7 harm | NOT SUPPORTED | LOO 150×7 | head +0.66 vs tail +0.98, medians 0, CIs overlap; expansion gate not met | `reports/evidence_level_distracting_effect.md` |
| 12 | Plausible-but-wrong distractors are more harmful than obviously weak ones | MIXED | Good/Hard/Easy tails | Good +2.64 (p=0.006); Hard −2.02 (p=0.022); Easy −1.04 (p=0.21); Hard−Easy −0.97 (p=0.28) | `reports/oracle_distractor_stats.json` |
| 13 | Context overload generalizes to another benchmark (MSVD) | STRONG | MSVD670 frozen validation | D1 156.39 > D3 151.45 > D5 144.39 > D7 141.59; D3−D7 +9.86 (p=1e-4) | `reports/msvd_external_frozen_validation.md` |
| 14 | Frozen adaptive budget beats fixed D3 | MIXED | 5-fold OOF / untouched2690 / MSVD | val OOF +1.03 (CI [−0.30,+2.52]); test2690 +0.58 (CI [0.02,1.14], p=0.046); vs D1 −0.56 (ns); MSVD +0.49 (ns) | `reports/adaptive_budget_decision.md`, `reports/msrtt_test2690_confirmatory.md` |
| 15 | Adaptive budget beats the best fixed external budget | NOT SUPPORTED | test2690, MSVD | loses to D1 on test (−0.56 ns); ≈D3 on MSVD | same |
| 16 | External retrieval replaces in-domain retrieval | NOT SUPPORTED | test2690 | D3−B −3.02 (p<1e-4); Adaptive−B −2.44 | `reports/msrtt_test2690_confirmatory.md` |
| 17 | First-version UAR reranking adds independent gain | NOT SUPPORTED | FAITHFUL_V2 CF3/D3/H3 | H3 66.32 vs D3 67.09; H3−D3 −0.77 (p=0.43) | `reports/faithful_v2.md` |
| 18 | Retrieval score and downstream utility are strongly aligned | NOT SUPPORTED / weak | scale churn + diagnostics | evidence churn large while ΔCIDEr ≤0.014; feature–harm |ρ|≤0.12 | `reports/nested_scale_evidence_survival.md`, `reports/budget_diagnostics.json` |

## Status labels

- **STRONG**: multiple independent, significant results; survives oracle/matched controls.
- **SUPPORTED**: significant primary result with caveats.
- **MIXED**: direction present, direct contrast or replication not significant.
- **NOT SUPPORTED**: data contradict or fail to support the claim.
- **LIMITATION**: acknowledged confound, not a claim (see below).

## Limitations (not claims)

- In-domain control confounds evidence source with corpus size (6,513 vs 1.09 M).
- Selection/adaptive results come from one val497 selection split; test2690 is
  confirmatory but the adaptive gain over D3 is marginal.
- Tail-quality and distractor conditions use GT-CIDEr oracle for candidate
  selection (diagnostic only, never inference).
- Source-level bootstrap assumes independent sources; MSR-VTT test2690 has 2,004
  sources.

## Invalid / superseded artifacts (not cited)

- `INVALID_ORACLE_SCORER`: Oracle1/3/7, Good/Hard/Easy built before fixed-DF
  scoring were deleted and regenerated.
- `SUPERSEDED_FACTORIAL_STATISTICS`: unscaled ±6.48 text contrast replaced by
  averaged −3.24.
- Failed LOO runs before lineidx fix; failed in-domain/replicate attempts; all
  rerun and superseded by the results above.
