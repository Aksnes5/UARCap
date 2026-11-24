# POST_FIX LOO Stage 1 (val150, frame loader fixed)

Baseline = POST_FIX FULL D7 predictions on the same 150 queries.

| removed rank | mean effect (CIDEr pts) | median | P(effect>0) |
|---|---|---|---|
| 1 | +2.00 | 0.0 | 0.200 |
| 2 | −0.49 | 0.0 | 0.167 |
| 3 | +0.29 | 0.0 | 0.200 |
| 4 | +0.69 | 0.0 | 0.167 |
| 5 | −0.78 | 0.0 | 0.147 |
| 6 | −0.13 | 0.0 | 0.167 |
| 7 | −1.42 | 0.0 | 0.167 |

Head (1–3) +0.60, P(harmful) 0.189; tail (4–7) −0.41, P(harmful) 0.162.
Medians are 0 at every rank; head/tail harmful probabilities are nearly equal.
POST_FIX still does not support a single rank-dependent toxic tail item; rank 7
removal slightly hurts (evidence is useful), not helps.
