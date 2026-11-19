# Scale retrieval stats (dev300, exact X-CLIP, faithful dedup)

| scale | top1 sim | top20 sim | unique src /20 | same-src query ratio | unique topics /20 | top20 unique clips | top7 unique clips | naive top1 CIDEr | rerank top1 CIDEr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10K | 0.2568 | 0.2364 | 20.00 | 0.000 | 19.50 | 2,964 | 1,228 | 0.084 | 0.089 |
| 100K | 0.2702 | 0.2542 | 20.00 | 0.000 | 18.70 | 5,473 | 1,784 | 0.087 | 0.133 |
| 500K | 0.2789 | 0.2644 | 19.82 | 0.167 | 17.91 | 6,843 | 1,954 | 0.096 | 0.140 |
| FULL | 0.2820 | 0.2686 | 19.33 | 0.420 | 17.51 | 7,459 | 1,989 | 0.101 | 0.130 |

Similarity rises monotonically; retrieval-only caption quality peaks at 500K;
the quality rerank helps most at 100K/500K, least at 10K/FULL.
