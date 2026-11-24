# Paper revision roadmap

## Title/abstract
Frame the paper as: "Context-budget behavior of frozen caption generators under
million-scale external retrieval: selection headroom and integration overload",
not "more retrieval hurts".

## Abstract must state
- k=7 never optimal across 12 settings; k1/k3 optimal.
- Oracle3 > D3 (+3.04, p=.0096); Oracle7 < Oracle3 (−2.78, p=.0076).
- In-domain vs external budget response reverses (directional).
- Effects that are directional must be labeled as such.

## Method section
- Document the frame-loader fix and its scope (val only).
- Explicitly state the reranker deviation from the published noun-phrase
  pseudo-label definition.
- Remove or demote adaptive-k to an exploratory section.

## Experiments
- Main table: six-scale × CF/D × k grid (POST_FIX).
- Mechanism table: Text×Visual, position, tail/distractor, LOO.
- Oracle table: Oracle1/3/7 vs D1/3/7.
- Cross-dataset: MSR-VTT test2690 and MSVD, with training-budget caveat
  (k=7 vs k=3).
- Add a limitations section: single val split, missing k=0, reranker mismatch,
  no deployable selector.

## Required additional experiments (future, not this round)
- Matched k=0 baseline.
- A faithful reimplementation of the published pseudo-label reranker.
- At least one independent validation split or larger n for text/external effects.
- Optional: a second generator to test generality.
