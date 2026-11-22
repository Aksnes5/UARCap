# Matched external × in-domain source × budget interaction

**Status: RESOLVED (secondary analysis).** CPU-only, read-only. Uses only frozen
val497 generation outputs. Does not modify, replace, or re-test the two original
separate external and in-domain comparisons.

## 1. Data and pairing

All four conditions exist as frozen generated-caption files and cover the **same
497 val497 queries** (image-name intersection = 497/497):

| condition | artifact |
|---|---|
| external D3 | `eval_val497_D3` |
| external D7 | `eval_val497_D7` |
| in-domain D3 | `eval_val497_indomain_D3` |
| in-domain D7 | `eval_val497_indomain_D7` |

Per-query CIDEr was recomputed from the frozen predictions and references with the
same PTB tokenizer + `pycocoevalcap` CIDEr implementation used by
`evaluation/cocoeval.py`. The recomputed condition means match the frozen values:
external D3 53.14 / D7 50.05; in-domain D3 55.14 / D7 55.43.

For each query:

```
Δ_external   = CIDEr_ext(D3)   − CIDEr_ext(D7)
Δ_indomain   = CIDEr_in(D3)    − CIDEr_in(D7)
interaction  = Δ_external      − Δ_indomain
```

Positive interaction means the external corpus loses more from D3→D7 than the
matched in-domain corpus does.

Source clusters: video ID → YouTube source from
`datasets/MSRVTT-v2/train_val_videodatainfo.json`. All 497 val videos map to a URL;
there are **361 unique sources**.

## 2. Results (CIDEr points ×100)

| quantity | mean | 95% CI | sign-flip p | median | positive / negative / tied |
|---|---:|---|---:|---:|---:|
| Δ_external (D3 − D7) | **+3.08** | [+0.70, +5.68] | 0.0138 | 0.00 | 112 / 88 / 297 |
| Δ_indomain (D3 − D7) | **−0.29** | [−2.31, +1.69] | 0.7790 | 0.00 | 91 / 87 / 319 |
| **interaction** | **+3.37** | [+0.30, +6.69] | **0.0365** | 0.00 | 143 / 126 / 228 |

Source-cluster bootstrap (resampling the 361 sources, 10k replicates) for the
interaction: **95% CI [+0.27, +6.88]**.

n = 497 queries; n_sources = 361.

The large tie counts are expected: for many queries, D3 and D7 yield the same
prediction (or the same CIDEr), so the budget manipulation has no effect. The tests
are on the paired mean, not on the win rate.

## 3. Interpretation

- The external corpus shows the known D3→D7 drop (+3.08 points, CI excludes 0),
  while the matched in-domain corpus is flat (−0.29, CI includes 0). The difference
  between these two budget effects is positive and its 95% CI excludes 0, both under
  query resampling and under source clustering.
- This is consistent with the existing claim that the large-k penalty is specific to
  the external evidence source in this matched control, and it does not depend on
  re-running either of the original tests.
- It is a **secondary analysis of frozen outputs**, not a replacement for the
  original external D3−D7 test (+3.11, p=0.012) or in-domain D3−D7 test
  (−0.27, p=0.80). Those frozen statistics stand unchanged.

## 4. Caveats

- **Static-first-frame caveat.** All four conditions were generated on val497 with
  the YAML affected by the dataloader bug documented in
  `VAL_TEST_CIDER_GAP_AUDIT.md` (the generator saw a static first frame). Because
  all four conditions share the same visual input, the interaction is internally
  valid as a paired contrast, but its absolute magnitude may differ under a
  properly framed val re-run. The direction is supported by the unaffected
  test2690 external replication (D3−D7 +1.81) and MSVD external replication
  (D3−D7 +9.86); no matched in-domain D3/D7 exists on those splits.
- The in-domain control confounds source and corpus size (MSR-VTT train, ~6.5k
  videos, vs VideoUFO FULL, ~1.09M clips), as already noted in the paper's
  limitations. The interaction is therefore "external vs matched in-domain
  retrieval as implemented", not a clean source-only contrast.
- Source clustering assumes independent sources; 497 clips map to 361 sources.
- No frozen output or frozen statistic was modified.

## 5. Files

- `reports/cpu_audit/source_budget_interaction.json` — aggregates, CIs, p-values
- `reports/cpu_audit/source_budget_interaction_per_query.csv` — 497 rows with all four
  per-query CIDEr values, deltas, interaction, and source cluster
- `scripts/cpu_audit/source_budget_interaction.py`
