# FINAL CLAIM MATRIX

All effects in CIDEr points (×100). Status: STRONG / SUPPORTED / MIXED /
NOT_SUPPORTED.

| Claim | Evidence | Effect | CI/p | Dataset | Status | Safe wording |
|---|---|---|---|---|---|---|
| C1 Large context budgets are consistently suboptimal across scale/reranking | 6 scales × D/CF k-grid | k=7 penalty mean 1.83 (D 2.06, CF 1.60) | — | MSR-VTT val497 | SUPPORTED | "Large context budgets are consistently suboptimal across scales and rerankers." |
| C2 k=7 is never optimal across 12 settings | same | 0/12 best at k=7; k=1/3 10/12; k=5 only 10K | — | val497 | SUPPORTED | "k=7 was never optimal; beyond 10K every optimum used k=1 or k=3." |
| C3 Retrieval coverage and generation utility decouple | oracle@k vs CIDEr; nested churn | oracle@7 0.184 vs top1 0.118; ΔCIDEr across scales ≤0.014 despite top1 survival 0.19–0.54 | — | val497 | SUPPORTED | "Coverage rises with k and scale while downstream utility does not." |
| C4 Context overload generalizes to MSVD | frozen external MSVD670 | D1 156.39 > D3 151.45 > D5 144.39 > D7 141.59; D3−D7 +9.86 | p=1e-4 | MSVD test | STRONG | "The overload pattern replicates on MSVD." |
| C5 Frozen adaptive budgeting mitigates but does not beat best fixed k | OOF val497; untouched2690; MSVD | val +1.03 (ns); test +0.58 vs D3; vs D1 −0.56 (ns); MSVD +0.49 (ns) | test p=0.046; others ns | val/test/MSVD | MIXED | "The frozen selector mitigates large-k loss but does not universally outperform the best fixed budget." |
| C6 Textual context count, not visual count, dominates | Text×Visual 2×2 | text main −3.24; visual +0.13; interaction −0.78 | text CI [−5.68,−0.97] p=0.007 | val497 | STRONG | "Under this setting, degradation is textual; no independent visual effect was detected." |
| C7 Prompt-position bias does not explain degradation | reverse / top3-last | +0.09 / −0.09 vs D7 | p≈0.89 | val497 | NOT_SUPPORTED | "Prompt ordering does not explain the drop." |
| C8 Matched in-domain retrieval does not degrade at large k | in-domain D1/3/5/7 | 54.16/55.14/55.00/55.43; D3−D7 −0.27 (ns) | ns | val497 | SUPPORTED | "The overload is specific to external evidence in this matched control." |
| C9 Oracle selection improves all corresponding D conditions | fixed-DF oracle | +3.27 (1 vs D1), +2.05 (3 vs D3), +3.13 (7 vs D7) | p=0.0008/0.053/0.0036 | val497 | SUPPORTED | "Substantial selection headroom exists inside the top-20." |
| C10 Oracle7 < Oracle3: integration overload persists | oracle k-grid | −2.03 | CI [−3.95,−0.23], p=0.033 | val497 | SUPPORTED | "Even oracle-selected seven contexts are worse than three." |
| C11 GoodTail improves D7 | goodtail vs D7 | +2.64 | CI [+0.78,+4.55], p=0.006 | val497 | SUPPORTED | "Replacing the tail with oracle-good evidence improves generation." |
| C12 Hard distractors harm; hard-vs-easy inconclusive | hard/easy vs D7 | hard −2.02; easy −1.04; hard−easy −0.97 | hard p=0.022; easy p=0.21; contrast p=0.28 | val497 | MIXED | "Plausible-wrong evidence is harmful; the direct hard-vs-easy contrast is not significant." |
| C13 No single rank-dependent toxic tail | LOO 150×7 | head +0.66, tail +0.98, medians 0 | CIs overlap | val497 | NOT_SUPPORTED | "A single toxic low-rank item does not explain the effect." |
| C14 Candidate identity churns with scale without proportional gain | nested transitions | top1 survival 0.19–0.54; ΔCIDEr ≤0.014 | — | val497 | SUPPORTED | "Scaling replaces stable evidence without downstream improvement." |
| C15 Source-cluster bootstrap preserves conclusions | source-level resampling | test D3−D7 +1.81 (cluster CI [0.88,2.75]); MSVD +9.86 ([5.38,14.84]) | — | test2690/MSVD | SUPPORTED | "Main paired conclusions survive source clustering." |
