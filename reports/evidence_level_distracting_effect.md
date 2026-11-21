# Evidence-level leave-one-out causal influence — Stage 1 (val150)

150 deterministically selected queries (frozen manifest), seven remove-one
conditions each (1,050 counterfactuals), baseline = stored D7 outputs.

`DistractingEffect_i = CIDEr(D7 \ {i}) − CIDEr(D7)`; positive = removing item
improved generation.

| removed rank | mean effect | median | P(effect>0) | 95% CI |
|---|---|---|---|---|
| 1 | +0.0146 | 0.00 | 0.200 | [−0.0115, +0.0454] |
| 2 | −0.0072 | 0.00 | 0.160 | [−0.0339, +0.0167] |
| 3 | +0.0122 | 0.00 | 0.180 | [−0.0119, +0.0429] |
| 4 | +0.0071 | 0.00 | 0.180 | [−0.0172, +0.0359] |
| 5 | +0.0125 | 0.00 | 0.187 | [−0.0165, +0.0459] |
| 6 | +0.0123 | 0.00 | 0.180 | [−0.0137, +0.0424] |
| 7 | +0.0074 | 0.00 | 0.213 | [−0.0231, +0.0403] |

Head (ranks 1–3) mean +0.0066, P(harmful) 0.18; tail (4–7) mean +0.0098,
P(harmful) 0.19. Median effect is exactly 0 for every rank: most single-item
removals do not change the generated caption at all.

## Gate decision

No coherent rank-related signal (head vs tail difference ~0.003 CIDEr, CIs
overlap zero). Per the pre-registered expansion gate, **LOO was not expanded to
all 497 queries**.

Interpretation: the k=7 degradation is not attributable to any single
identifiable tail evidence item; output is insensitive to removing one item,
consistent with a context-count / composition effect rather than one toxic
candidate.
