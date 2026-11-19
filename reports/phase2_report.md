# Phase 2 report — external retrieval into EVCap (dev300)

## Baseline (300 queries)

| group | B1 | B4 | METEOR | ROUGE-L | CIDEr | SPICE |
|---|---:|---:|---:|---:|---:|---:|
| A No Retrieval | 85.23 | 46.69 | 31.68 | 64.38 | 64.18 | 8.28 |
| B In-domain Full | 84.32 | 48.10 | 31.63 | 64.23 | 66.51 | 8.10 |
| CF External FULL + fusion no rerank | 85.15 | 45.42 | 31.80 | 64.19 | 64.38 | 8.77 |
| D External FULL + rerank | 84.33 | 44.42 | 31.59 | 63.98 | 63.25 | 8.55 |

D-CF: CIDEr -1.12, B4 -0.99. Rerank improves retrieval top1 CIDEr
(0.101 -> 0.130) but hurts generation.

## Oracle diagnostic

(oracle@20 high, actual top1 low -> ranking/reranking bottleneck.)

## Taxonomy (50 worst D-B)

scene/action 18%, object/relation 12%, domain 8%, generic 26%,
overly specific 4%, useful external evidence 32%; same-source 12%;
near-duplicate top-7 pairs 62%.

## Scale (D10K done; 100K/500K running)

D10K: 86.63 / 47.34 / 31.84 / 64.22 / 63.99 / 8.81.
D10K vs FULL: wins 88, losses 65. D10K > FULL CIDEr.
