# POST_FIX claim-evidence matrix

| Claim | Evidence | Effect (CIDEr) | CI / p | Status | Write in abstract? |
|---|---|---|---|---|---|
| Large budgets are suboptimal | 12 scale×track settings | k7 0/12 best; penalty 1.7–3.5 | descriptive | SUPPORTED | Yes (descriptive) |
| k=1 is the dominant best budget | same | 11/12 best | — | SUPPORTED (scope-limited) | With scope qualifier |
| Scale does not improve generation monotonically | 6-scale k1 | CF1 69.89–71.17; D1 68.89–70.59 | small ns | SUPPORTED | Yes |
| Candidate churn decouples from utility | 10K→50K/100K | 80.7/88.7% top1 replaced; CF1 Δ −0.56/−0.61 on replaced | — | SUPPORTED (raw CF) | Yes |
| Selection headroom | Oracle3−D3 | +3.04 | [+0.74,+5.37], p=.0096 | SUPPORTED | Yes |
| Integration overload under oracle | Oracle7−Oracle3 | −2.78 | [−4.92,−0.77], p=.0076 | SUPPORTED | Yes |
| Oracle1 helps | Oracle1−D1 | +1.67 | [−0.55,+3.85], p=.13 | MIXED | No |
| Text context harms | Text main 2×2 | −1.40 | [−4.05,+1.25], p=.31 | MIXED (direction only) | No (direction only in text) |
| Visual context null | Visual main | −0.04 | p=.86 | NOT_SIGNIFICANT_NULL | No |
| Text×visual interaction | 2×2 | +0.16 | p=.69 | NOT SUPPORTED | No |
| Position/order explains drop | reverse/top3-last | −0.24 / +0.02 | p≈0.9 | NOT SUPPORTED | Negative claim OK |
| In-domain responds differently | DiD | +2.77 | [−0.61,+6.39], p=.13 | MIXED (direction) | Only as direction |
| Tail quality dose response | mid/low tails | +1.23 / −0.61 | ns | NOT SUPPORTED | No |
| GoodTail improves | goodtail−D7 | −0.56 | p=.69 | NOT SUPPORTED (reversed) | No |
| Hard distractor harmful | hard−D7; hard−easy | −1.42; −1.05 | p=.18; p=.38 | MIXED | No |
| Single toxic tail item | LOO | head +0.60, tail −0.41, medians 0 | — | NOT SUPPORTED | Negative claim OK |
| Adaptive beats fixed budget | test2690/MSVD | +0.58 vs D3; ≈D3 MSVD | marginal/ns | NOT SUPPORTED | No |
| k=0 baseline | missing | — | — | BLOCKED | No |
