# AUDIT 05 — Confounders, alternatives and data risks

## Data and contamination
- VideoUFO: 1,091,712 clips, 345,437 YouTube sources (verified counts in repo).
- Dedup: full manifest naming uses 812,932 unique brief-caption groups; the CSV
  has 812,014 distinct brief-caption strings. The two counts use different
  grouping paths and must not be conflated; retrieval indexes captions per
  manifest group.
- Nested manifests 10K→FULL verified (`manifest_stats.json`).
- Source overlap with MSR-VTT: **0** YouTube-source overlap across all splits;
  candidate same-source hits 0. Caption exact overlap 0.08–0.17% (generic
  phrases). Visual dHash flags were graphics/static false positives.
- Residual risk: no guarantee of zero *semantic* near-duplication; only exact
  source and sampled visual checks were performed.
- Caption provenance: the brief captions are machine-generated metadata; a
  systematic human-quality audit was not performed. This is a live confounder
  for "external captions are lower quality" explanations.

## Training/test budget mismatch
- MSR-VTT checkpoint trained/served with k̃=7; MSVD with k̃=3 (predecessor).
- POST_FIX external prefers k=1 on MSR-VTT; this is a **distribution shift /
  budget mismatch**, not evidence that k=1 is universally better.
- MSVD k=7 is a stress test outside the training budget; k=1 vs k=3 is the
  relevant contrast there.

## Adaptive-k
- The selector rule is parameter-free, but the deployment decision, k mapping and
  comparison to D3 were made on PRE_FIX val497 (frame bug).
- POST_FIX val497 prefers k=1, so the adaptive selection premise and the choice of
  D3 as reference are invalidated. Adaptive must be exploratory only.

## k=0
No matched no-retrieval baseline exists in POST_FIX. Status
`BLOCKED_REQUIRES_GENERATION`. The study cannot claim external retrieval beats
no retrieval; old 70.8/64.18 numbers are not substitutes.

## Copying / length
Existing audits show length and copying changes with k, but correlation with
CIDEr does not establish that copying causes the drop. Alternative explanations
(prompt format, caption quality, training mismatch) are not excluded.

## Negative results: what they do and do not establish
- Position ≈ 0: strong evidence against a simple order/primacy artifact.
- Tail quality (Mid/Low/Good): null; **not** evidence that quality is irrelevant,
  only that these manipulations did not produce a detectable effect at n=497.
- HardWrong vs EasyWrong: hard lower by 1.05, ns; cannot claim the plausible-
  distractor mechanism.
- LOO: medians 0 at every rank; supports "no single toxic item", but with 150
  queries the power to detect small rank effects is low.
