# CF scale-grid cache preflight (CPU only)

Candidate union = first clip of each raw top-7 caption per val497 query.

| scale | unique candidates | cache hit | missing | hit rate |
|---|---|---|---|---|
| 10K | 1,796 | 1,796 | 0 | 100% |
| 100K | 2,682 | 2,682 | 0 | 100% |
| 500K | 3,008 | 3,008 | 0 | 100% |
| FULL | 3,086 | 3,086 | 0 | 100% |

All CF grids can run generation immediately once the GPU queue reaches them;
no feature extraction is needed. (50K/250K CF unions depend on their yet-to-run
retrieval and will be audited then.)
