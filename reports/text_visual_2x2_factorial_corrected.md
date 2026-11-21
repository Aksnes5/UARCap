# Text × Visual 2×2 factorial statistics — corrected

Per-query CIDEr on val497 (497/497 aligned, no duplicates/missing).

## Elementary paired effects

| effect | Δ CIDEr | 95% CI | p (perm) |
|---|---|---|---|
| Text 3→7 at Video=3 (T7V3−T3V3) | −2.85 | [−5.45, −0.43] | 0.026 |
| Text 3→7 at Video=7 (T7V7−T3V7) | −3.63 | [−6.06, −1.41] | 0.0015 |
| Visual 3→7 at Text=3 (T3V7−T3V3) | +0.52 | [−0.25, +1.44] | 0.253 |
| Visual 3→7 at Text=7 (T7V7−T7V3) | −0.26 | [−0.82, +0.14] | 0.407 |

## Averaged marginal effects (paper scale)

| effect | Δ CIDEr | 95% CI | p |
|---|---|---|---|
| **Text main (averaged)** | **−3.24** | [−5.68, −0.97] | **0.007** |
| Visual main (averaged) | +0.13 | [−0.34, +0.63] | 0.616 |
| Interaction (difference-in-differences) | −0.78 | [−1.84, +0.13] | 0.123 |

The previously reported −6.48 / +0.27 were **unscaled factorial contrasts**
(sums of the two pairwise effects), not averaged marginal effects; they are
superseded (`SUPERSEDED_FACTORIAL_STATISTICS`).

## Interpretation (conservative)

Under val497 / FULL / D ordering, **textual evidence overload is the dominant
channel**: adding text contexts 3→7 lowers CIDEr by 2.9–3.6 points in both
visual conditions. No independent negative visual effect was detected at this
sample size (+0.13, CI crossing 0), and the interaction CI crosses 0. This does
not establish that visual context can never hurt in other settings.
