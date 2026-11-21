# Evidence-level leave-one-out causal influence — Stage 1 (val150)

150 deterministically selected queries (frozen manifest), seven remove-one
conditions each (1,050 counterfactuals), baseline = stored D7 outputs.

`DistractingEffect_i = CIDEr(D7 \ {i}) − CIDEr(D7)`; positive = removing item
improved generation. **All effects below are in CIDEr points (×100)**; the JSON/CSV
keep raw 0–1 values.

| removed rank | mean effect | median | P(effect>0) | 95% CI |
|---|---|---|---|---|
| 1 | +1.46 | 0.00 | 0.200 | [−1.15, +4.54] |
| 2 | −0.72 | 0.00 | 0.160 | [−3.39, +1.67] |
| 3 | +1.22 | 0.00 | 0.180 | [−1.19, +4.29] |
| 4 | +0.71 | 0.00 | 0.180 | [−1.72, +3.59] |
| 5 | +1.25 | 0.00 | 0.187 | [−1.65, +4.59] |
| 6 | +1.23 | 0.00 | 0.180 | [−1.37, +4.24] |
| 7 | +0.74 | 0.00 | 0.213 | [−2.31, +4.03] |

Head (ranks 1–3) mean **+0.66 CIDEr points**, P(harmful) 0.18; tail (4–7)
mean **+0.98**, P(harmful) 0.19; head−tail difference ≈ **0.32 CIDEr points**. Median effect is exactly 0 for every rank: most single-item
removals do not change the generated caption at all.

## Gate decision

No coherent rank-related signal (head vs tail difference ~0.003 CIDEr, CIs
overlap zero). Per the pre-registered expansion gate, **LOO was not expanded to
all 497 queries**.

Interpretation (conservative): the results do **not** support "low-ranked tail
evidence is the main harmful source", nor "a single specific evidence item is
the main culprit". Individual items can have negative downstream utility, but
harmful removals are not concentrated in ranks 4–7. The pattern is more
consistent with a **set-level / collective textual-context effect** than with
one toxic candidate.

Combined with the other mechanism results (text 3→7 harms; visual 3→7 has no
independent effect; prompt order is null; mid/low tails are not worse; LOO has no
rank-dependent harmful item), the best current explanation is **semantic
interference among multiple textual contexts**, not a single tail item and not
visual fusion.
