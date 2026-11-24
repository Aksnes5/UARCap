# POST_FIX matched in-domain vs external budget interaction

| k | External FULL D | In-domain D |
|---|---|---|
| 1 | 69.97 | 71.93 |
| 3 | 68.63 | 72.42 |
| 5 | 67.31 | 71.07 |
| 7 | 67.22 | 73.76 |

Paired (497 queries):
- External D3−D7 = **+1.44** (CI [−1.20, +4.13], p=0.29)
- In-domain D3−D7 = **−1.33** (CI [−3.57, +0.85], p=0.24)
- Difference-in-differences = **+2.77** (CI [−0.61, +6.39], p=0.13)

Direction of the original interaction is preserved: in-domain prefers larger k
(k7 highest at 73.76), external declines at large k. However, after fixing the
val frame loader, the external D3−D7 effect and the DiD are no longer
statistically significant at 497 queries; the evidence is directional, not
confirmatory.
