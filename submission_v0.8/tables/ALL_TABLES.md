# Exported Manuscript and Supplementary Tables (UARCap v0.8)

## Table 1

**Table 1.** Repaired MSR-VTT validation CIDEr (official PTB points; *n* = 497, 361 sources) across six nested VideoUFO clip scales, two candidate orderings (CF raw similarity, D full-caption consensus), and four paired text–visual budgets.

| Corpus scale (clips) | CF1 | CF3 | CF5 | CF7 | D1 | D3 | D5 | D7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10K (10,000) | 70.43 | 70.13 | 69.30 | 68.73 | 69.21 | 67.80 | 67.67 | 67.33 |
| 50K (50,000) | 69.98 | 68.47 | 68.08 | 67.09 | 68.89 | 67.84 | 66.80 | 66.12 |
| 100K (100,000) | 69.89 | 70.10 | 69.32 | 67.57 | 69.61 | 68.68 | 66.69 | 66.15 |
| 250K (250,000) | 70.50 | 69.09 | 67.80 | 68.77 | 70.29 | 69.29 | 68.75 | 66.99 |
| 500K (500,000) | 70.50 | 68.87 | 68.89 | 67.23 | 70.59 | 70.08 | 68.25 | 67.62 |
| FULL (1,091,712) | 71.17 | 68.88 | 68.84 | 67.95 | 69.97 | 68.63 | 67.31 | 67.22 |

## Table 2

**Table 2.** Frozen test-set PTB CIDEr means on pilot-excluded MSR-VTT (*n* = 2,690) and MSVD (*n* = 670) using the FULL external VideoUFO memory. The MSR-VTT in-domain comparator uses training captions; separately attributed MSVD in-domain references (published 158.1; archival reproduction 158.32) are documented in Supplementary Materials S3.

| Benchmark | D1 | D3 | D5 | D7 | Adaptive | In-domain |
| --- | --- | --- | --- | --- | --- | --- |
| MSR-VTT (`test2690`) | 71.21 | 70.08 | 69.48 | 68.27 | 70.67 | 73.09 |
| MSVD (`test670`) | 156.39 | 151.45 | 144.39 | 141.59 | 151.94 | — |

## Table 3

**Table 3.** Selected paired test-set contrasts (PTB CIDEr points) with pointwise 95% source-cluster bootstrap CIs, two-sided sign-flip *p*-values, and within-benchmark BH-adjusted *q*-values (7 contrasts in MSR-VTT; 6 in MSVD; complete tables in Supplementary Table S2).

| Benchmark | Contrast | Effect | 95% source-cluster CI | Source *p* | Source BH *q* |
| --- | --- | --- | --- | --- | --- |
| MSR-VTT | D1 − D3 | +1.13 | [+0.35, +1.94] | .0059 | .0083 |
| MSR-VTT | D3 − D7 | +1.80 | [+0.86, +2.76] | .0002 | .0007 |
| MSR-VTT | D1 − D7 | +2.93 | [+1.83, +4.06] | .0001 | .0007 |
| MSR-VTT | Adaptive − D1 | −0.54 | [−1.20, +0.10] | .1006 | .1006 |
| MSVD | D1 − D3 | +4.94 | [+1.50, +8.39] | .0046 | .0069 |
| MSVD | D3 − D7 | +9.86 | [+5.49, +15.05] | .0001 | .0003 |
| MSVD | D1 − D7 | +14.80 | [+9.92, +19.97] | .0001 | .0003 |
| MSVD | Adaptive − D1 | −4.46 | [−7.68, −1.47] | .0030 | .0060 |

## Table 4

**Table 4.** Cohort-specific PTB CIDEr means and observed wrapped prompt lengths (including visual token positions) for Experiments A and B on MSR-VTT (FULL external memory). Because reference IDF is fitted separately within each scored cohort, absolute means should be compared strictly within Experiment A or within Experiment B.

| Experiment | Arm | Text / retrieved visuals | Scored queries (sources) | Cohort CIDEr | Mean prompt length (median) |
| --- | --- | --- | --- | --- | --- |
| A | A0 | 0 / 0 | 2,690 (2,004) | 69.81 | — |
| A | A1 | 1 / 1 | 2,690 (2,004) | 71.21 | — |
| A | A7 | 7 / 7 | 2,690 (2,004) | 68.27 | — |
| B | B3 | 3 / 3 | 2,618 (1,969) | 69.88 | 150.69 (148) |
| B | B7HQ | 7 / 3 | 2,618 (1,969) | 68.62 | 226.86 (224) |
| B | B7LQ | 7 / 3 | 2,618 (1,969) | 68.75 | 226.38 (223) |

## Table 5

**Table 5.** All five prespecified paired contrasts from Experiments A and B (PTB CIDEr points), reporting pointwise 95% source-cluster bootstrap CIs, two-sided source-level sign-flip *p*-values, within-family Holm *p*-values, five-test Holm *p*-values, and whether each contrast meets the prespecified ±1.0-point practical clarity rule.

| Family | Contrast | Queries (sources) | Effect | 95% source-cluster CI | Raw *p* | Family Holm *p* | Five-test Holm *p* | Clear practical effect (±1 pt) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | A1 − A0 | 2,690 (2,004) | +1.40 | [+0.45, +2.36] | .0033 | **.0066** | **.0165** | No |
| A | A7 − A0 | 2,690 (2,004) | −1.54 | [−2.94, −0.18] | .0271 | **.0271** | .0813 | No |
| B | B7HQ − B3 | 2,618 (1,969) | −1.27 | [−2.34, −0.20] | .0191 | .0573 | .0764 | No |
| B | B7LQ − B3 | 2,618 (1,969) | −1.13 | [−2.15, −0.14] | .0306 | .0612 | .0813 | No |
| B | B7HQ − B7LQ | 2,618 (1,969) | −0.14 | [−1.05, +0.80] | .7698 | .7698 | .7698 | No |

## Table 6

**Table 6.** Reference-guided Oracle contrasts on repaired MSR-VTT validation (*n* = 497; 361 sources) with pointwise 95% source-cluster CIs and BH adjustment within the four-contrast Oracle family.

| Contrast | Effect (CIDEr pts) | 95% source-cluster CI | Source *p* | Source BH *q* |
| --- | --- | --- | --- | --- |
| Oracle1 − D1 | +1.67 | [−0.50, +3.78] | .1282 | .1531 |
| Oracle3 − D3 | +3.04 | [+0.85, +5.27] | .0073 | .0146 |
| Oracle7 − D7 | +1.72 | [−0.62, +4.08] | .1531 | .1531 |
| Oracle7 − Oracle3 | −2.73 | [−4.72, −0.80] | .0068 | .0146 |

## Table S1

**Table S1.** Verified clip counts, unique source-video counts, nesting integrity, and source-ID overlap with MSR-VTT across the six nested VideoUFO manifests.

| Manifest scale | Unique clips | Unique source videos | Clips missing from larger level | Overlap with MSR-VTT train (6,513) | Overlap with MSR-VTT val (497) | Overlap with MSR-VTT test (2,990) |
| --- | --- | --- | --- | --- | --- | --- |
| 10K | 10,000 | 10,000 | 0 | 0 | 0 | 0 |
| 50K | 50,000 | 50,000 | 0 | 0 | 0 | 0 |
| 100K | 100,000 | 100,000 | 0 | 0 | 0 | 0 |
| 250K | 250,000 | 241,211 | 0 | 0 | 0 | 0 |
| 500K | 500,000 | 338,877 | 0 | 0 | 0 | 0 |
| FULL | 1,091,712 | 345,437 | — | 0 | 0 | 0 |

## Table S2

**Table S2.** Complete within-benchmark paired contrast families (official PTB CIDEr points, source-cluster inference, BH adjustment within each benchmark family).

| Benchmark | Contrast | Effect | 95% source-cluster CI | Source *p* | Source BH *q* |
| --- | --- | --- | --- | --- | --- |
| MSR-VTT (`test2690`) | D1 − D3 | +1.13 | [+0.35, +1.94] | .0059 | .0083 |
| MSR-VTT (`test2690`) | D3 − D7 | +1.80 | [+0.86, +2.76] | .0002 | .0007 |
| MSR-VTT (`test2690`) | D1 − D7 | +2.93 | [+1.83, +4.06] | .0001 | .0007 |
| MSR-VTT (`test2690`) | D5 − D7 | +1.21 | [+0.44, +1.97] | .0018 | .0031 |
| MSR-VTT (`test2690`) | Adaptive − D1 | −0.54 | [−1.20, +0.10] | .1006 | .1006 |
| MSR-VTT (`test2690`) | Adaptive − D3 | +0.59 | [+0.02, +1.18] | .0399 | .0465 |
| MSR-VTT (`test2690`) | In-domain − D1 | +1.88 | [+0.77, +2.96] | .0008 | .0019 |
| MSVD (`test670`) | D1 − D3 | +4.94 | [+1.50, +8.39] | .0046 | .0069 |
| MSVD (`test670`) | D3 − D7 | +9.86 | [+5.49, +15.05] | .0001 | .0003 |
| MSVD (`test670`) | D1 − D7 | +14.80 | [+9.92, +19.97] | .0001 | .0003 |
| MSVD (`test670`) | D5 − D7 | +2.80 | [−0.37, +5.97] | .0934 | .1121 |
| MSVD (`test670`) | Adaptive − D1 | −4.46 | [−7.68, −1.47] | .0030 | .0060 |
| MSVD (`test670`) | Adaptive − D3 | +0.49 | [−2.73, +3.55] | .7661 | .7661 |

## Table S3

**Table S3.** Complete repaired validation diagnostic contrasts (official PTB CIDEr points, pointwise 95% source-cluster CIs, two-sided sign-flip *p*-values, and within-family BH *q*-values; 18 mechanism contrasts in `val497`, 8 LOO contrasts in `val150`).

| Family | Contrast | Effect | 95% source-cluster CI | Source *p* | Source BH *q* |
| --- | --- | --- | --- | --- | --- |
| Mechanism (`val497`) | Text effect at V = 3 (T7V3 − T3V3) | −1.45 | [−4.22, +1.22] | .2970 | .7349 |
| Mechanism (`val497`) | Text effect at V = 7 (T7V7 − T3V7) | −1.29 | [−4.02, +1.35] | .3439 | .7349 |
| Mechanism (`val497`) | Visual effect at T = 3 (T3V7 − T3V3) | −0.12 | [−0.78, +0.53] | .7404 | .9031 |
| Mechanism (`val497`) | Visual effect at T = 7 (T7V7 − T7V3) | +0.04 | [−0.35, +0.50] | .8448 | .9031 |
| Mechanism (`val497`) | Text count main effect | −1.37 | [−4.10, +1.26] | .3154 | .7349 |
| Mechanism (`val497`) | Visual count main effect | −0.04 | [−0.42, +0.35] | .8529 | .9031 |
| Mechanism (`val497`) | Text × visual interaction | +0.16 | [−0.60, +0.94] | .7003 | .9031 |
| Mechanism (`val497`) | External FULL D3 − D7 | +1.41 | [−1.21, +4.17] | .3042 | .7349 |
| Mechanism (`val497`) | Matched in-domain D3 − D7 | −1.34 | [−3.59, +0.92] | .2553 | .7349 |
| Mechanism (`val497`) | Source × budget interaction | +2.74 | [−0.76, +6.29] | .1322 | .7349 |
| Mechanism (`val497`) | Reverse prompt order vs. D7 | −0.24 | [−2.08, +1.67] | .8061 | .9031 |
| Mechanism (`val497`) | Top-three-last order vs. D7 | +0.02 | [−1.76, +1.92] | .9813 | .9813 |
| Mechanism (`val497`) | MidTail7 (ranks 8–11) vs. D7 | +1.20 | [−0.80, +3.36] | .2574 | .7349 |
| Mechanism (`val497`) | LowTail7 (ranks 17–20) vs. D7 | −0.64 | [−2.77, +1.36] | .5513 | .9031 |
| Mechanism (`val497`) | GoodTail vs. D7 | −0.56 | [−3.23, +2.04] | .6827 | .9031 |
| Mechanism (`val497`) | HardWrong vs. D7 | −1.45 | [−3.55, +0.66] | .1754 | .7349 |
| Mechanism (`val497`) | EasyWrong vs. D7 | −0.37 | [−2.35, +1.64] | .7259 | .9031 |
| Mechanism (`val497`) | HardWrong vs. EasyWrong | −1.08 | [−3.44, +1.22] | .3675 | .7349 |
| LOO (`val150`) | Remove rank 1 vs. D7 | +1.97 | [−1.04, +5.11] | .2201 | .8803 |
| LOO (`val150`) | Remove rank 2 vs. D7 | −0.56 | [−4.64, +3.22] | .7854 | .9265 |
| LOO (`val150`) | Remove rank 3 vs. D7 | +0.39 | [−3.24, +3.84] | .8357 | .9265 |
| LOO (`val150`) | Remove rank 4 vs. D7 | +0.58 | [−2.98, +4.01] | .7588 | .9265 |
| LOO (`val150`) | Remove rank 5 vs. D7 | −0.75 | [−4.31, +2.47] | .6764 | .9265 |
| LOO (`val150`) | Remove rank 6 vs. D7 | −0.17 | [−3.68, +3.09] | .9265 | .9265 |
| LOO (`val150`) | Remove rank 7 vs. D7 | −1.51 | [−5.28, +1.97] | .4236 | .9265 |
| LOO (`val150`) | LOO head (1–3) − tail (4–7) | +1.06 | [−0.18, +2.40] | .1127 | .8803 |

## Table S4

**Table S4.** Experiment B matching balance and observed wrapped prompt lengths across the 2,618 eligible queries.

| Metric | Mean | Median | Min | Max |
| --- | --- | --- | --- | --- |
| Mean raw X-CLIP similarity gap (HQ − LQ) | 0.00835 | — | 0.00500 | 0.02606 |
| Absolute quartet word-count imbalance | 0.35 | — | 0 | 7 |
| Absolute head-overlap Jaccard imbalance | 0.0085 | — | 0.000004 | 0.0798 |
| Absolute internal-overlap Jaccard imbalance | 0.0074 | — | 0.000003 | 0.0781 |
| Wrapped prompt length: B3 (positions) | 150.69 | 148 | 125 | 219 |
| Wrapped prompt length: B7HQ (positions) | 226.86 | 224 | 172 | 346 |
| Wrapped prompt length: B7LQ (positions) | 226.38 | 223 | 176 | 347 |
| Prompt-length difference (B7HQ − B7LQ) | +0.48 | 3 (abs. median) | — | — |

## Table S5

**Table S5.** Full query-level and source-cluster inference for all five prespecified contrasts in Experiments A and B (PTB CIDEr points). Primary inference uses source-cluster Holm adjustment.

| Contrast | Effect | Paired $d_z$ | Query 95% CI | Query *p* | Source 95% CI | Source *p* | Family Holm *p* | Five-test Holm *p* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 − A0 | +1.40 | +0.057 | [+0.50, +2.33] | .0018 | [+0.45, +2.36] | .0033 | .0066 | .0165 |
| A7 − A0 | −1.54 | −0.045 | [−2.84, −0.25] | .0174 | [−2.94, −0.18] | .0271 | .0271 | .0813 |
| B7HQ − B3 | −1.27 | −0.046 | [−2.33, −0.21] | .0203 | [−2.34, −0.20] | .0191 | .0573 | .0764 |
| B7LQ − B3 | −1.13 | −0.043 | [−2.11, −0.14] | .0251 | [−2.15, −0.14] | .0306 | .0612 | .0813 |
| B7HQ − B7LQ | −0.14 | −0.006 | [−1.03, +0.79] | .7623 | [−1.05, +0.80] | .7698 | .7698 | .7698 |

## Table S6

**Table S6.** Summary of verified runtime invariants across Experiments A and B (2,990 queries per arm).

| Verification domain | Verified condition | Outcome |
| --- | --- | --- |
| Experiment A zero-evidence arm (A0) | Zero captions, zero retrieved features loaded, no visual softmax; $1 \times 32 \times 768$ external aggregate and attention K/V equal to exact zero tensor | PASS (2,990 / 2,990) |
| Experiment A cross-arm parity | Identical query frames, Q-Former inputs, query visual projection, masks, module states, and CPU/CUDA encoder RNG across A0, A1, A7 | PASS (2,990 / 2,990) |
| Experiment A endpoint reproducibility | Parsed outputs of A1 and A7 match archived D1 and D7 records identically across all 2,990 test queries | PASS (2,990 / 2,990) |
| Experiment B visual-tensor parity | Identical query frames, representative clip IDs, raw similarity weights, ordered feature-tensor hashes, fused aggregate, attention K/V, gate inputs, visual projections, and encoder RNG across B3, B7HQ, B7LQ | PASS (2,990 / 2,990) |
| Experiment B ineligible-query parity | For all 372 non-eligible queries (300 pilots + 72 unmatched), B7HQ and B7LQ match B3 predictions and traces identically | PASS (372 / 372) |
| Reference & source metadata alignment | Exact query reference list/order match; zero fallback singleton clusters (2,004 clusters in A; 1,969 in B) | PASS |

## Table S7

**Table S7.** Descriptive output word counts and trigram match rates across D-ordered joint budgets.

| Split | Budget $k$ | Mean words | Fixed top-1 trigram match | Supplied-union trigram match |
| --- | --- | --- | --- | --- |
| MSR-VTT `test2690` (*n* = 2,690) | 1 | 7.92 | 0.142 | 0.142 |
| MSR-VTT `test2690` (*n* = 2,690) | 3 | 8.10 | 0.164 | 0.223 |
| MSR-VTT `test2690` (*n* = 2,690) | 5 | 8.36 | 0.179 | 0.284 |
| MSR-VTT `test2690` (*n* = 2,690) | 7 | 8.75 | 0.190 | 0.334 |
| MSR-VTT repaired `val497` (*n* = 497) | 1 | 8.01 | 0.147 | 0.147 |
| MSR-VTT repaired `val497` (*n* = 497) | 3 | 8.18 | 0.174 | 0.223 |
| MSR-VTT repaired `val497` (*n* = 497) | 5 | 8.40 | 0.186 | 0.288 |
| MSR-VTT repaired `val497` (*n* = 497) | 7 | 8.65 | 0.191 | 0.331 |
| MSVD `test670` (*n* = 670) | 1 | 6.77 | — | — |
| MSVD `test670` (*n* = 670) | 3 | 6.97 | — | — |
| MSVD `test670` (*n* = 670) | 5 | 7.35 | — | — |
| MSVD `test670` (*n* = 670) | 7 | 7.61 | — | — |
