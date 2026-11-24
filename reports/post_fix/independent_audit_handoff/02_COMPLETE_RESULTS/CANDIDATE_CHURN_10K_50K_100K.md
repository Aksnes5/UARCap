# POST_FIX candidate churn: 10K → 50K / 100K (CPU only)

## k=1 paired CIDEr (10K − larger, per-query bootstrap 10k)
| track / comparison | Δ CIDEr | 95% CI | p |
|---|---|---|---|
| CF1 10K − 50K | +0.45 | [−1.47, +2.32] | 0.63 |
| CF1 10K − 100K | +0.55 | [−1.45, +2.59] | 0.61 |
| D1 10K − 50K | +0.32 | [−1.74, +2.43] | 0.77 |
| D1 10K − 100K | −0.40 | [−2.60, +1.74] | 0.72 |

Scale 10K vs 50K/100K is statistically indistinguishable at k=1.

## Top-1 candidate replacement
| transition | replaced | fraction | sim increased among replaced | mean sim Δ |
|---|---|---|---|---|
| 10K→50K | 401/497 | 80.7% | 400/401 | +0.0134 |
| 10K→100K | 441/497 | 88.7% | 440/441 | +0.0163 |

## Generation change on replaced queries
| transition | CF1 Δ CIDEr (replaced) | D1 Δ CIDEr (replaced) |
|---|---|---|
| 10K→50K | **−0.56** | +0.26 |
| 10K→100K | **−0.61** | +0.86 |

## Interpretation
Expanding the corpus replaces the top-1 candidate for ~81–89% of queries.
Among replaced queries, X-CLIP similarity almost always **increases**
(400/401 and 440/441), while CF1 generation on those same queries
**decreases** (−0.56 / −0.61 CIDEr). This is a direct instance of
retrieval similarity rising without downstream utility: the new top-1 is more
"similar" but not more useful to the frozen generator.
For the D track the same replacement is mildly positive (+0.26 / +0.86), so the
decoupling is clearer for raw retrieval than after ROUGE reranking.

Artifacts: `reports/post_fix/candidate_churn_10k_vs_50k_100k.json`.
