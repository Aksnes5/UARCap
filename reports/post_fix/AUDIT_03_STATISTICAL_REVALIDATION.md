# AUDIT 03 — Statistical revalidation

Independent recomputation from sample-level predictions; no values copied from
previous reports.

## Method
- Unit: paired per-query CIDEr (Cider, pycocoevalcap, same references).
- Bootstrap: 2,000 replicates for the 72-test main grid, seed 20261124;
  10,000 for mechanism tests, seed 42; percentile 95% CI.
- Permutation: sign-flip, 2,000/10,000 replicates.
- Multiplicity: Benjamini–Hochberg over the 72 main-grid tests.
- Source clusters: MSR-VTT val497 maps to **361 unique YouTube sources**
  (train_val metadata), used for Oracle robustness.

## Main grid (6 scales × CF/D; k1−k3, k1−k5, k1−k7, k3−k7; CF−D at each k)
72 paired tests:
- 16 have raw p<0.05;
- **0 survive BH FDR 0.05** (smallest q=0.30+).
Examples: 10K CF k1−k7 +1.76, CI [−0.42,+3.99], p=0.115, q=0.32;
10K D k1−k3 +1.41, CI [−0.40,+3.30], p=0.13, q=0.34.

Consequence: "k=7 was never optimal in 12 settings" is a **descriptive count**,
not twelve independent significant tests. k7 penalties (1.7–3.5) are point
estimates; per-setting significance is not established after correction.

## Oracle
| comparison | Δ | query CI | source-cluster CI | p |
|---|---|---|---|---|
| Oracle3−D3 | +3.04 | [+0.74,+5.37] | **[+0.87,+5.39]** | 0.0096 |
| Oracle7−Oracle3 | −2.78 | [−4.92,−0.77] | **[−4.83,−0.84]** | 0.0076 |
| Oracle1−D1 | +1.67 | [−0.55,+3.85] | — | 0.13 |
| Oracle7−D7 | +1.70 | [−0.73,+4.13] | — | 0.17 |

BH over the four Oracle tests: the two significant tests survive (q<0.05).
Source-cluster bootstrap preserves both conclusions.

## In-domain / Text×Visual
- External D3−D7 +1.44, CI [−1.20,+4.13], p=0.29.
- In-domain D3−D7 −1.33, CI [−3.57,+0.85], p=0.24.
- DiD +2.77, CI [−0.61,+6.39], p=0.13.
- Text main −1.40, CI [−4.05,+1.25], p=0.31; visual −0.04, p=0.86;
  interaction +0.16, p=0.69.
None significant. Source-cluster bootstrap was not run for these; query-level
CIs are reported and should be treated as the weaker evidence.

## Consistency with prior report
Values match `POST_FIX_MECHANISM_STATS.json` and
`POST_FIX_PRIORITY1_STATS.json` to rounding. Differences: (i) the main grid was
never FDR-corrected before — this audit adds it and finds no surviving tests;
(ii) source-cluster CIs were added for Oracle and confirm robustness.

## Multiple-comparison caveat
With 72 exploratory tests, raw p-values cannot be used to claim specific
scale×budget effects. Confirmatory claims should be limited to the pre-specified
Oracle contrasts (and even those are two of four planned tests).
