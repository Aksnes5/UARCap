# Phase 2 baseline table (dev300)

| group | B1 | B4 | METEOR | ROUGE-L | CIDEr | SPICE |
|---|---:|---:|---:|---:|---:|---:|
| A No Retrieval (fusion branch, empty retrieval) | 85.23 | 46.69 | 31.68 | 64.38 | 64.18 | 8.28 |
| B In-domain Full | 84.32 | 48.10 | 31.63 | 64.23 | 66.51 | 8.10 |
| CF External FULL + fusion, no rerank | pending corrected run | | | | | |
| D External FULL + rerank + fusion | 84.33 | 44.42 | 31.59 | 63.98 | 63.25 | 8.55 |

First CF attempt was stopped: 798 `failed to load ...pt` warnings because the
naive top-7 first-pick clips differ from the D reranked top-7 features
(1,158 of 1,973 clips missing). Metrics from that run are invalid and discarded.
The missing features are being extracted; CF will be rerun with zero missing.

Bootstrap over 300 queries: A 63.82, B 66.40, D 63.29.
D-B mean -0.0324, 95% CI [-0.0696, +0.0019].
Paired tags: B>A but D<A: 15 queries; D>B by >0.05: 60; D<B by >0.05: 71.
