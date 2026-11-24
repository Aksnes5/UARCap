# Locked follow-up statistical analysis plan

Lock after the v0.6 manuscript commit and before any follow-up inference. This plan is prospective for the new interventions on a previously analyzed benchmark. The historical choice of test benchmark and scale has precedent, so inference is conditional on this setting.

## Estimands and families

1. **A family (two tests).** A1−A0 and A7−A0 on `test2690`. The directional scientific expectation is positive effect of useful external evidence, with unknown difference at the inherited budget. Both tests are two-sided. A1−A7 can be reported as a descriptive reference without entering the hypothesis family.
2. **B family (three tests).** B7HQ−B3, B7LQ−B3, and B7HQ−B7LQ on exactly the frozen 2,618-query eligible set. Test whether either extension alters utility and whether the fixed-count relevance-band contrast distinguishes the two additions. A common deficit and a quality-dependent deficit are both possible. Ineligible and pilot cases do not enter this family. Label subgroup and prompt-length investigations exploratory.

Primary outcome is CIDEr points, using the same frozen COCO implementation and Stanford PTB tokenizer as the independent audit. Fit CIDEr document statistics on the reference set of the scored cohort, shared across its arms. Reference text never enters matching. Give paired source-weighted percentile 95% intervals and 10,000 source-cluster sign flips with seed `20261125`. Resample source groups with all constituent clips, weight by query count as in the audit, and use Holm adjustment within each family at .05. Intervals are pointwise. If reporting an umbrella assertion across both experiments on the same cohorts, additionally recompute Holm across all five declared tests. Source IDs come from the same hashed public metadata. Optional secondary measures and IDF re-estimation are sensitivity checks, with no role in primary outcomes.

## Precision and practical effect

Use **1.0 CIDEr point** as the smallest effect of practical interest for this study. It is close to the archived 1.13-point difference between two small external settings on MSR-VTT and below the 1.80-point D3−D7 contrast. It is a study-specific engineering benchmark consistent with the observed response, not a universal metric threshold. A claim of a practically clear improvement requires an adjusted discovery and a positive interval entirely beyond +1.0. Use the symmetric rule for harm. An interval within [−1,1] is a precision statement under the fixed-reference analysis, not a formal equivalence test.

The existing D1−D3 and D3−D7 effects give planning source-cluster standard errors .410 and .477 points on all 2,690 queries (2,004 sources), and .408 and .483 on the eligible 2,618 (1,969 sources). The corresponding approximate pointwise 95% half-widths are .80–.94 for A and .80–.95 for B. These use the old differences' dispersion and source groups, not new responses. The matched k=0 and controlled text differences could have higher dispersion. Holm-adjusted detection of a one-point effect therefore remains uncertain. Keep all eligible observations without adding cases to chase an effect.

## Completeness and stops

One valid result for every required query and arm. No interim significance checks. Validate data receipts, unchanged hashes, cohort/order, reference equality, schema, visual parity and exact trace counts before opening inference. A failure of invariants stops scoring and prompts engineering diagnosis inside the fixed design, with all attempts retained. If coverage fell below 1,000 eligible queries or 50% under the frozen construction, treat B as infeasible and return the causes. CPU preflight passes this gate. After a compliant run, stop with the locked family irrespective of direction or significance. Samples, margins, arm definitions, families and outcome remain fixed.
