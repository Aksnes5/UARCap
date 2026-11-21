# UARCap — one-page final status

## What the paper can safely claim

1. **Large budgets are consistently suboptimal.** Across 12 scale×reranking
   settings, k=7 was never optimal; k=1/3 won 10/12; k=5 only at 10K. Mean k=7
   penalty is 1.83 CIDEr (D 2.06, CF 1.60). The exact optimum (1 vs 3) is
   scale- and track-dependent (not monotone).
2. **The degradation is textual, not visual.** Text×Visual 2×2: text main effect
   −3.24 CIDEr (p=0.007); visual main +0.13 (ns); interaction −0.78 (ns).
   T3V7 (53.66) is not below T3V3 (53.14).
3. **Evidence selection is a real bottleneck.** GT-CIDEr oracle selection beats
   the deployed ranking by +2.0 to +3.3 CIDEr (Oracle1−D1 +3.27, p=0.0008).
4. **Set-level integration overload survives oracle selection.** Even with
   oracle-selected evidence, 7 contexts are significantly worse than 3
   (Oracle7−Oracle3 −2.03, p=0.033).
5. **The overload is external-evidence-specific in this setting.** Matched
   in-domain D1/3/5/7 are flat (54.16/55.14/55.00/55.43; D3−D7 ns), while
   external FULL degrades (D3−D7 +3.11, p=0.012).
6. **It generalizes across benchmarks.** MSVD670 external: D1 156.39 → D3 151.45
   → D5 144.39 → D7 141.59; D3−D7 +9.86 (p=1e-4). External D1 is within 1.9
   CIDEr of the in-domain reference (158.32).
7. **Retrieval similarity alone is a weak proxy for downstream utility.** Scale
   churn is large (top-1 survival 0.19–0.54) while ΔCIDEr stays within ±0.014;
   feature–harm correlations are weak (|ρ|≤0.12).

## What it cannot claim

- That external retrieval replaces in-domain retrieval (test2690: D3−B −3.02,
  p<1e-4).
- That the adaptive similarity-gap budget is a superior method: it is ≈D3
  (MSVD +0.49 ns), beats D3 marginally on test2690 (+0.58, p=0.046) but loses to
  fixed D1 (−0.56 ns), and never significantly exceeds the best fixed budget.
- That first-version UAR reranking adds independent gain (H3−D3 −0.77, p=0.43).
- That visual context can never hurt, or that visual fusion is generally
  harmless — only that no independent visual 3→7 negative effect was detected
  here.
- That a few toxic tail items explain the effect: LOO head/tail difference is
  ~0.3 CIDEr with overlapping CIs and zero medians.
- That plausible distractors are decisively worse than easy ones: direction is
  consistent (Hard −2.02 p=0.022; Easy −1.04 ns) but hard−easy −0.97 is not
  significant (MIXED).

## Remaining limitations

- In-domain control confounds evidence source with corpus size (6,513 vs 1.09 M).
- Adaptive selection was fit on one 497-query split; test2690 is confirmatory but
  its edge over D3 is marginal.
- Oracle and hard/easy conditions use ground-truth CIDEr for candidate selection
  (diagnostic only; never used for inference).
- Source-cluster bootstrap assumes independent sources (2,004 sources in 2,690
  test queries); conclusions unchanged.

## Canonical evidence

`reports/paper_claim_matrix.md` (claim → experiment → effect/CI/p → file),
`reports/mechanism_summary.md`, `reports/scale_budget.md`,
`reports/in_domain_vs_external_budget_control.md`,
`reports/oracle_fixeddf_audit.md`, `reports/msvd_external_frozen_validation.md`,
`reports/msrtt_test2690_confirmatory.md`.
