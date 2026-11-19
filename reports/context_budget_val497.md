# Context-budget experiment on official val497 (FAITHFUL_V2)

Date: 2026-11-20. 497 queries, same EVCap checkpoint/prompt/fusion/beam/preprocessing,
FULL VideoUFO corpus. Only k and ordering (CF vs D) vary.

## 1. Generation metrics (×100)

| | B1 | B4 | M | R-L | CIDEr | SPICE |
|---|---|---|---|---|---|---|
| CF1 | 79.55 | 38.51 | 28.49 | 59.47 | **53.20** | 7.63 |
| CF3 | 79.57 | 38.35 | 28.37 | 59.31 | 52.42 | 7.68 |
| CF5 | 79.64 | 37.73 | 28.45 | 59.14 | 51.29 | 7.59 |
| CF7 | 78.72 | 36.88 | 28.28 | 58.79 | 51.13 | 7.73 |
| D1 | 79.40 | 39.05 | 28.32 | 59.71 | 51.77 | 7.55 |
| D3 | 80.12 | 39.23 | 28.57 | 59.81 | **53.14** | **7.89** |
| D5 | 79.36 | 37.43 | 28.34 | 59.27 | 50.51 | 7.66 |
| D7 | 77.82 | 36.04 | 28.41 | 58.61 | 50.05 | 7.88 |

## 2. k-utility curve vs candidate coverage

| k | CF oracle@k | D oracle@k | CF gen CIDEr | D gen CIDEr |
|---|---|---|---|---|
| 1 | 0.0800 | 0.1176 | 53.20 | 51.77 |
| 3 | 0.1314 | 0.1481 | 52.42 | **53.14** |
| 5 | 0.1535 | 0.1720 | 51.29 | 50.51 |
| 7 | 0.1698 | 0.1835 | 51.13 | 50.05 |

Coverage increases monotonically with k for both orderings, while generation
utility does not: CF peaks at k=1 and declines; D peaks at k=3 and then falls
sharply. This is the inverted-U / context-overload pattern on an independent
split.

## 3. Paired per-query CIDEr (10k bootstrap, 10k sign-flip)

| Comparison | Δ CIDEr | 95% CI | p | W/L/T |
|---|---|---|---|---|
| D3 − D7 | **+3.11** | [+0.72, +5.70] | **0.012** | 112/88/297 |
| D3 − D5 | **+2.65** | [+0.64, +4.78] | **0.012** | 92/62/343 |
| D5 − D7 | +0.46 | [−1.14, +2.13] | 0.583 | 88/72/337 |
| D1 − D3 | −1.38 | [−3.62, +0.73] | 0.229 | 85/96/316 |
| CF1 − CF7 | +2.11 | [−0.36, +4.66] | 0.098 | 125/123/249 |
| CF1 − CF3 | +0.78 | [−1.06, +2.72] | 0.421 | 94/86/317 |
| CF3 − CF7 | +1.33 | [−0.92, +3.60] | 0.252 | 111/114/272 |
| CF5 − CF7 | +0.19 | [−1.70, +2.11] | 0.843 | 89/84/324 |
| D3 − CF3 | +0.74 | [−1.53, +3.03] | 0.544 | 104/93/300 |
| D3 − CF1 | −0.05 | [−2.37, +2.36] | 0.966 | 111/106/280 |
| D7 − CF7 | −1.04 | [−3.47, +1.35] | 0.392 | 104/106/287 |

`D3 > D5` and `D3 > D7` are the only significant effects; the ordering (CF vs D)
does not matter at the optimum (D3 vs CF1 indistinguishable).

## 4. Tie structure (important)

Because beam search is deterministic and many queries ignore the extra context,
a large fraction of outputs is identical across k:

| Pair | identical generated captions |
|---|---|
| D3 vs D7 | 295/497 (59.4%) |
| CF3 vs CF7 | 271/497 (54.5%) |
| D3 vs CF3 | 300/497 (60.4%) |
| D1 vs D3 | 316/497 (63.6%) |

On the queries where output actually changes, the effect is much larger:
D3 − D7 = **+7.64** CIDEr (202 queries), CF3 − CF7 = **+2.93** (226 queries).
The aggregate +3.11 / +1.33 are diluted by the 55–60% identical outputs.
Any future contextual method must be judged on changed-query behavior, and
context sensitivity itself is a meaningful signal for adaptive budgeting.

## 5. Conclusion

`context-overload finding reproduces on the independent validation split`:

- candidate coverage rises with k (oracle@1 → oracle@7: CF 0.080→0.170, D 0.118→0.184);
- generation utility peaks at k=3 for quality reranking (D3 = 53.14, significantly
  above D5 and D7, p = 0.012) and at k=1 for raw retrieval (CF1 = 53.20);
- the two optima are statistically indistinguishable (D3 − CF1 = −0.05, p = 0.97);
- the apparent dev300 optimum k=3 is corroborated for D, while raw CF does not
  benefit from any additional context.

Therefore the second-round method is justified: **Adaptive Context Budgeting**
(decide how much retrieved context each query receives), not another reranking
score. All thresholds/stopping policies must be fit on val497 only; the
untouched test2690 is for final evaluation, full test2990 for paper comparison.
