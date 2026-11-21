# Mechanism backbone (frozen)

Valid results only; oracle-dependent numbers use the fixed-DF scorer.

## 1. Selection bottleneck — SUPPORTED
Oracle1/3/7 all exceed matching normal D: +3.27 (Oracle1−D1, p=0.0008),
+2.05 (Oracle3−D3, p=0.053), +3.13 (Oracle7−D7, p=0.0036). Substantially better
evidence exists inside the top-20 than the deployed ranking selects.

## 2. External textual integration overload — STRONG
Text main effect −3.24 CIDEr (CI [−5.68,−0.97], p=0.007); visual main effect
+0.13 (n.s.); interaction −0.78 (n.s.). Oracle7 is significantly below Oracle3
(−2.03, CI [−3.95,−0.23], p=0.033), so seven textual contexts remain worse than
three even when evidence is selected by GT. External D3→D7 drops significantly
while the matched in-domain k=1/3/5/7 curve is flat (54.16/55.14/55.00/55.43).

## 3. Position-bias explanation — NOT SUPPORTED
posrev and top3-last are indistinguishable from normal D7 (+0.09 / −0.09,
p≈0.89).

## 4. Visual-fusion-overload explanation — NOT SUPPORTED (current setting)
T3V7 (53.66) is not below T3V3 (53.14); visual main effect is not significantly
negative.

## 5. Single toxic tail / rank-dependent explanation — NOT SUPPORTED
LOO head (ranks 1–3) +0.66 vs tail (4–7) +0.98 CIDEr points, medians 0, CIs
overlap; MidTail7 (+0.42) and LowTail7 (+0.98) are not worse than D7.

## 6. Conservative mechanism statement
"Large-scale external retrieval exposes two complementary bottlenecks:
imperfect evidence selection and set-level textual integration overload. Better
evidence selection substantially improves generation, but increasing even
oracle-selected textual context from three to seven still reduces utility. The
degradation is not explained by visual context count, prompt position, or a
small set of consistently toxic low-ranked items."

## 7. Plausible distractor hypothesis — MIXED
GoodTail7 52.71 (+2.64 vs D7, p=0.006) and HardWrong7 48.01 (−2.02, p=0.022)
are significantly different from D7; EasyWrong7 48.98 (−1.04, p=0.21) is not.
HardWrong − EasyWrong = −0.97 (CI [−2.79, +0.70], p=0.28): the direction is
consistent with plausible-but-wrong evidence being more damaging, but the direct
hard-vs-easy contrast is not significant. Status: MIXED.

## 8. Scope
No new GPU experiments beyond the already queued tasks and replicates.
