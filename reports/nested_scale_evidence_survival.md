# Nested-scale evidence survival and candidate churn (S5)

Adjacent scale transitions, val497, top-20 retrieval sets.

| transition | top1 surv | top3 surv | top7 surv | J@7 | mean rank disp | new cand@7 | D3 output change | Δ CIDEr D3 |
|---|---|---|---|---|---|---|---|---|
| 10K→50K | 0.193 | 0.210 | 0.207 | 0.124 | 7.98 | 0.793 | 32.2% | +0.0097 |
| 50K→100K | 0.535 | 0.518 | 0.520 | 0.377 | 4.79 | 0.480 | 28.2% | +0.0015 |
| 100K→250K | 0.427 | 0.419 | 0.423 | 0.287 | 5.80 | 0.578 | 33.0% | −0.0100 |
| 250K→500K | 0.519 | 0.523 | 0.525 | 0.379 | 4.75 | 0.475 | 28.0% | +0.0135 |
| 500K→FULL | 0.515 | 0.519 | 0.519 | 0.375 | 4.82 | 0.482 | 29.6% | +0.0019 |

## Interpretation

- Evidence identity is highly unstable under scaling: only 19–54% of top-1
  candidates survive an adjacent transition; about half of the 7-slot context
  is replaced even between 250K and 500K.
- Despite that churn, fixed-k=3 generation changes on only 28–33% of queries and
  CIDEr moves by at most ±0.014. Scaling mostly substitutes competitively
  ranked alternatives without improving downstream utility.
- This is consistent with the coverage–utility decoupling claim: better
  retrieval similarity at larger scale does **not** translate into better
  evidence for the frozen generator.
- No claim that churn is harmful per se; the association with generation
  change is weak and the quality delta is flat.
