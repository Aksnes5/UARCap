# Phase 2 final scale table (dev300, diagnostic)

| scale | top1 sim | top20 sim | X-CLIP top1 cap CIDEr | rerank top1 cap CIDEr | oracle@20 | oracle rank mean | hit@7 | unique src /20 | same-src ratio | unique topics /20 | B1 | B4 | METEOR | ROUGE-L | CIDEr | SPICE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10K | 0.2568 | 0.2364 | 0.0629 | 0.0664 | 0.1904 | 8.61 | 44.3% | 20.00 | 0.000 | 19.50 | 86.63 | 47.34 | 31.84 | 64.22 | 63.99 | 8.81 |
| 100K | 0.2702 | 0.2542 | 0.0651 | 0.0997 | 0.2141 | 9.05 | 33.3% | 20.00 | 0.000 | 18.70 | 86.18 | 45.57 | 32.36 | 64.78 | 64.98 | 8.90 |
| 500K | 0.2789 | 0.2644 | 0.0725 | 0.1057 | 0.2128 | 9.51 | 34.0% | 19.82 | 0.167 | 17.91 | 85.43 | 45.37 | 32.07 | 64.35 | 63.31 | 8.77 |
| FULL | 0.2820 | 0.2686 | 0.0762 | 0.0981 | 0.2189 | 10.21 | 30.7% | 19.33 | 0.420 | 17.51 | 84.33 | 44.42 | 31.59 | 63.98 | 63.25 | 8.55 |
| B in-domain | - | - | - | - | - | - | - | - | - | - | 84.32 | 48.10 | 31.63 | 64.23 | 66.51 | 8.10 |

Answers:
1. Generation best external scale = 100K (CIDEr 64.98); 10K best B1/B4; FULL worst on most.
2. top1/top20 similarity increase monotonically with scale.
3. oracle@20 rises 10K->100K then plateaus (0.214/0.213/0.219).
4. generation does not follow: peaks 100K, then drops.
5. clear non-monotonic scaling: 63.99 -> 64.98 -> 63.31 -> 63.25.

Bottleneck decomposition:
- Candidate recall: top-20 oracle high (0.19-0.22) -> recall is not the main limit.
- Candidate ranking: hit@1 3.3-8.3%, oracle rank 8.6-10.2 -> severe ranking bottleneck.
- Generation utility: rerank raises caption CIDEr 0.101->0.130 but D < CF by 1.12 CIDEr -> relevance != utility.
