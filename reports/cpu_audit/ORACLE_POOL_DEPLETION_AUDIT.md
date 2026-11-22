# Oracle candidate-pool depletion audit

**Status: PARTIAL EVIDENCE.** CPU-only, read-only descriptive audit. No Oracle
caption was regenerated, no frozen generation statistic was replaced. The existing
frozen Oracle1/3/7 outputs are used only as a paired reference.

## 1. Question and scope

> Is `Oracle7 < Oracle3` plausibly explained mainly by depletion of candidate
> quality in ranks 4–7 of the reference-guided ranking?

The audit does **not** regenerate captions. It recomputes candidate-level
reference compatibility on the existing top-20 pool with the current frozen
unigram fixed-reference TF-IDF scorer, exactly as specified in
`reports/oracle_fixeddf_audit.md` and `scripts/oracle_fixeddf.py`:

- IDF estimated once from the 497-query × 20-reference val corpus (9,940 documents,
  5,924 distinct tokens), `idf = log(N / df)`, TF-IDF cosine, mean over the 20
  references.
- Candidate pool: `external/external_val_sims.json` (497 queries × 20 captions).
- Ranking: descending compatibility, ties by original index.
- Reference: val497 GT references.

This is a **new descriptive audit** of candidate quality; it is not a re-run or a
replacement of the frozen Oracle generation experiment.

## 2. Compatibility by reference-guided rank

| rank | mean | median | Q25 | Q75 |
|---:|---:|---:|---:|---:|
| 1 | 0.1559 | 0.1414 | 0.0877 | 0.2084 |
| 2 | 0.1350 | 0.1232 | 0.0674 | 0.1870 |
| 3 | 0.1231 | 0.1093 | 0.0581 | 0.1669 |
| 4 | 0.1139 | 0.1011 | 0.0532 | 0.1551 |
| 5 | 0.1060 | 0.0899 | 0.0483 | 0.1427 |
| 6 | 0.0993 | 0.0837 | 0.0437 | 0.1356 |
| 7 | 0.0934 | 0.0767 | 0.0376 | 0.1284 |
| 8 | 0.0879 | 0.0693 | 0.0349 | 0.1215 |
| 9 | 0.0825 | 0.0655 | 0.0325 | 0.1110 |
| 10 | 0.0782 | 0.0598 | 0.0296 | 0.1068 |
| 11 | 0.0735 | 0.0542 | 0.0263 | 0.1019 |
| 12 | 0.0685 | 0.0502 | 0.0228 | 0.0951 |
| 13 | 0.0640 | 0.0459 | 0.0193 | 0.0906 |
| 14 | 0.0593 | 0.0421 | 0.0166 | 0.0852 |
| 15 | 0.0544 | 0.0374 | 0.0149 | 0.0758 |
| 16 | 0.0497 | 0.0310 | 0.0125 | 0.0717 |
| 17 | 0.0438 | 0.0262 | 0.0102 | 0.0620 |
| 18 | 0.0376 | 0.0201 | 0.0080 | 0.0512 |
| 19 | 0.0308 | 0.0158 | 0.0063 | 0.0383 |
| 20 | 0.0213 | 0.0094 | 0.0038 | 0.0202 |

### Band summaries

| band | mean | median | SD | Q10 | Q90 |
|---|---:|---:|---:|---:|---:|
| ranks 1–3 | 0.1380 | 0.1247 | 0.0804 | 0.0523 | 0.2459 |
| ranks 4–7 | 0.1031 | 0.0882 | 0.0714 | 0.0267 | 0.1992 |
| ranks 8–20 | 0.0578 | 0.0415 | 0.0516 | 0.0106 | 0.1249 |

- Relative depletion, ranks 1–3 → ranks 4–7: **25.3%**.
- Mean rank-3 compatibility is 77.2% of rank-1; rank-4 is 70.6%; rank-7 is 56.7%;
  rank-20 is 12.1%.

### Paired per-query difference (ranks 1–3 minus ranks 4–7)

| statistic | value |
|---|---:|
| n | 497 |
| mean | 0.0349 |
| median | 0.0293 |
| SD | 0.0230 |
| 95% bootstrap CI | [0.0329, 0.0369] |
| sign-flip permutation p | < 1e-4 (20k permutations) |
| fraction of queries with positive difference | 1.000 |

Candidate quality at ranks 4–7 is therefore systematically below ranks 1–3 in every
query.

## 3. Is there a 3→4 discontinuity?

| adjacent drop | mean drop | fraction positive |
|---|---:|---:|
| 1→2 | 0.0209 | 1.000 |
| 2→3 | 0.0119 | 0.992 |
| **3→4** | **0.0092** | **0.992** |
| 4→5 | 0.0080 | 0.988 |
| 5→6 | 0.0067 | 0.992 |
| 6→7 | 0.0059 | 0.998 |
| 7→8 | 0.0055 | 0.994 |

The 3→4 gap is mean 0.0092, 95% CI [0.0082, 0.0104], sign-flip p < 1e-4; rank 4 is
below rank 3 in 99.2% of queries. **But it is not a cliff**: the 3→4 drop is smaller
than the 1→2 drop and only slightly larger than the 4→5 drop. The decay is smooth
and monotone, with the steepest loss at 1→2.

## 4. Does per-query depletion track the per-query Oracle3−Oracle7 CIDEr loss?

The frozen Oracle3 and Oracle7 generated captions were re-scored per query with the
same PTB/CIDEr implementation (metric re-evaluation only, no generation):

| statistic | value |
|---|---:|
| n | 497 |
| mean Oracle3 − Oracle7 CIDEr | 1.95 points |
| frozen paired estimate (for reference) | 2.03 points (CI [−3.95,−0.23], p = 0.033) |
| Spearman(compatibility depletion, Oracle3−Oracle7 CIDEr) | **−0.040** |
| mean CIDEr gain, high-depletion half | 1.71 points |
| mean CIDEr gain, low-depletion half | 2.19 points |

The audit value is consistent with the frozen paired statistic (0.5518 − 0.5323 =
0.0195) and does not replace it. The key descriptive result is the **absence of a
query-level relationship**: queries whose ranks 4–7 are most depleted do not show a
larger Oracle3−Oracle7 CIDEr loss (the high-depletion half is if anything slightly
smaller).

## 5. Verdict

**PARTIAL EVIDENCE.**

- *For depletion:* ranks 4–7 are systematically and substantially below ranks 1–3
  (mean 0.1031 vs 0.1380, a 25.3% relative drop; per-query difference positive in
  497/497 queries; 3→4 drop significant). Adding four lower-compatibility candidates
  is a real, measurable change in the candidate set, so candidate-quality depletion
  is a plausible contributor to `Oracle7 < Oracle3`.
- *Against depletion as the main explanation:* there is no 3→4 discontinuity, and the
  magnitude of per-query depletion does not predict the per-query CIDEr loss
  (Spearman −0.04; high- vs low-depletion halves 1.71 vs 2.19 points). A pure
  candidate-quality account would predict a positive relationship; it is not present.
- **Integration overload cannot be excluded.** The data are consistent with a mixture
  of lower average candidate quality and a set-level cost of adding more contexts.
  This audit does not identify which dominates, and does not license a claim that
  depletion "explains" Oracle7 < Oracle3.

## 6. Caveats

- val497 is the split affected by the static-first-frame generator bug documented in
  `VAL_TEST_CIDER_GAP_AUDIT.md`. The Oracle3/Oracle7 comparison is a paired contrast
  under the same (degraded) visual input, so it is internally valid, but the absolute
  magnitudes may not transfer to a properly framed val re-run.
- The compatibility scorer is a unigram TF-IDF surrogate, not CIDEr; it is the frozen
  selection criterion, which is exactly why it is the right object for this audit.
- No frozen Oracle generation output or reported statistic was modified.

## 7. Files

- `reports/cpu_audit/oracle_pool_depletion.json` — all aggregates
- `reports/cpu_audit/oracle_rank_compatibility.csv` — 497 × 20 candidate compatibility scores
- `scripts/cpu_audit/oracle_depletion.py`
