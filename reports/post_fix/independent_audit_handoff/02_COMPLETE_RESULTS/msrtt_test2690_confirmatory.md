# MSR-VTT untouched test2690 — confirmatory results

Frozen method (val497 similarity-gap selector) evaluated once on the 2,690
queries never used during exploration. Full2990 reported separately.

## Six metrics (×100), untouched2690

| | B1 | B4 | MET | R-L | CIDEr | SPICE |
|---|---|---|---|---|---|---|
| B in-domain | 86.87 | 50.30 | 32.60 | 66.91 | **73.09** | 8.62 |
| D1 | 86.85 | 48.73 | 32.45 | 66.17 | 71.21 | 8.82 |
| D3 | 86.64 | 47.93 | 32.50 | 66.10 | 70.08 | 8.88 |
| D5 | 86.37 | 47.35 | 32.55 | 65.76 | 69.48 | 9.00 |
| D7 | 85.50 | 45.82 | 32.32 | 65.24 | 68.27 | 9.12 |
| Adaptive (frozen) | 86.90 | 48.34 | 32.51 | 66.13 | 70.67 | 8.89 |

## Paired per-query CIDEr (10k bootstrap / 10k sign-flip)

| comparison | Δ | 95% CI | p | W/L/T |
|---|---|---|---|---|
| D3 − D7 | +1.81 | [+0.87, +2.75] | 0.0002 | 570/469/1651 |
| D1 − D3 | +1.14 | [+0.34, +1.95] | 0.0052 | 454/401/1835 |
| D1 − D7 | +2.96 | [+1.85, +4.08] | <1e-4 | — |
| D3 − D5 | +0.60 | [−0.20, +1.41] | 0.151 | 392/372/1926 |
| Adaptive − D3 | +0.58 | [+0.02, +1.14] | **0.046** | 294/237/2159 |
| Adaptive − D7 | +2.40 | [+1.43, +3.35] | 0.0001 | 639/493/1558 |
| Adaptive − D1 | −0.56 | [−1.19, +0.08] | 0.10 | — |
| Adaptive − B | −2.44 | [−3.60, −1.36] | <1e-4 | 614/651/1425 |
| D3 − B | −3.02 | [−4.15, −1.92] | <1e-4 | 599/687/1404 |

Full2990 gives the same ordering (D1 70.31 > Adaptive 69.76 > D3 69.41 > D5
68.78 > D7 67.46; D3−D7 +1.96, p<1e-4).

## Interpretation

1. **Context overload is confirmed on untouched data.** D3 > D7 (+1.81, p=0.0002)
   and even D1 > D3 (+1.14, p=0.005); generation utility decreases monotonically
   with k on this split, while retrieval coverage rises (oracle@k monotone).
2. **Frozen adaptive transfers but is not the best fixed budget.** It beats D3
   (+0.58, p=0.046) and D7 (+2.40), but loses to fixed D1 by 0.56 (n.s.) and to
   in-domain B by 2.44 (p<1e-4).
3. The val497 choice of D3 as the reference budget does not transfer as the
   absolute optimum: on test, k=1 is best. The frozen selector still captures
   most of that gain (mean k 2.36, 66% fewer contexts than k=7) without
   test-time tuning.
4. **No external configuration beats in-domain retrieval** on this benchmark
   (+2.44 to +4.8 CIDEr gap remaining). External retrieval remains a
   characterization result, not a replacement.
5. Claim discipline: the adaptive selector is a generic similarity-gap baseline;
   its val gain over D3 was +1.03 (CI crossing 0), and its test gain is marginal
   (+0.58). It should be reported as context-budget policy evidence, not as a
   novel method.
