# POST_FIX Priority-1 results and PRE_FIX comparison

## PRE vs POST (CIDEr points, same val497/configs)
| scale | PRE CF1 | POST CF1 | Δ | PRE D1 | POST D1 | Δ |
|---|---|---|---|---|---|---|
| 10K | 51.22 | 70.43 | +19.21 | 50.49 | 69.21 | +18.72 |
| 50K | 52.72 | 69.98 | +17.26 | 51.85 | 68.89 | +17.04 |
| 100K | 52.17 | 69.89 | +17.72 | 52.90 | 69.61 | +16.71 |
| 250K | 52.22 | 70.50 | +18.28 | 53.20 | 70.29 | +17.09 |
| 500K | 52.36 | 70.50 | +18.14 | 52.83 | 70.59 | +17.76 |
| FULL | 53.20 | 71.17 | +17.97 | 51.77 | 69.97 | +18.20 |

Absolute level moves up ~17–19 points; val is now aligned with test2690.

## What is preserved
- Large-k degradation: k7 penalty 1.70–3.46 per setting; mean D 2.86 / CF 2.56.
- k=7 is still **never optimal** (0/12); k=1 wins 11/12, k=3 once (100K CF).
- Scale still does not monotonically improve generation; k1 differences are
  small and non-monotone.
- Direction of the external-vs-in-domain interaction is preserved
  (external declines with k; in-domain rises, k7 = 73.76).
- Text direction is still negative; visual remains null.

## What changes
- PRE_FIX's "10K best k=5" no longer holds: POST_FIX best is k1 for both tracks.
- Magnitudes shrink: Text main −3.24 → **−1.40** (p 0.007 → 0.31);
  External D3−D7 +3.11 → **+1.44** (p 0.012 → 0.29);
  DiD +2.77 (p=0.13).
- Several previously significant mechanism effects are now directional only;
  claims must be downgraded unless replicated in Priority 2/3.

## Status
Priority 1 only. Oracle, position, tail/distractor and LOO POST_FIX runs are
not yet executed; old PRE_FIX mechanism results remain marked PRE_FIX and must
not be mixed with these tables.
