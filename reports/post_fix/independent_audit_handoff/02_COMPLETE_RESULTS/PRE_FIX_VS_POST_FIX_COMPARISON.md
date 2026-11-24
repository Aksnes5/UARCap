# PRE_FIX (frame-bug) vs POST_FIX comparison

All PRE_FIX val497 numbers are marked `PRE_FIX_FRAME_BUG` and must not be mixed
with POST_FIX tables. CIDEr points.

## Level shift
Across scales, POST_FIX is +17 to +19 CIDEr above PRE_FIX at k=1 (e.g. FULL CF1
53.20→71.17; 10K D1 50.49→69.21). Val is now aligned with test2690.

## Preserved directions
- k=7 never optimal; k7 penalty remains 1.7–3.5 (POST mean D 2.86 / CF 2.56).
- k=1 wins 11/12 POST settings (PRE: mixed best-k 5/3/1/1/3/3 and 5/3/1/3/3/1).
- Scale does not monotonically improve generation.
- Text direction negative, visual null.
- External declines with k while in-domain does not (DiD direction +2.77).
- Oracle selection helps; Oracle7 < Oracle3 (POST −2.78, p=0.0076).

## Weakened / overturned
- "10K best k=5" overturned: POST best k=1 for both tracks.
- Text main effect: PRE −3.24 (p=0.007) → POST **−1.40 (p=0.31)**.
- External D3−D7: PRE +3.11 (p=0.012) → POST **+1.44 (p=0.29)**.
- In-domain vs external DiD: POST +2.77 (p=0.13).
- GoodTail +2.64 (p=0.006) → POST −0.56 (p=0.69).
- HardWrong −2.02 (p=0.022) → POST −1.42 (p=0.18); hard−easy −1.05 (p=0.38).
- Oracle1 +3.27 (p=0.0008) → POST +1.67 (p=0.13).

## Implication
The frame bug inflated apparent effects and depressed absolute scores. The
descriptive structure (large-k suboptimal; text vs visual; external vs
in-domain; oracle integration overload) survives in direction, but several
PRE_FIX significance claims do not survive the fixed loader at n=497 and must
be downgraded or reported as directional.
