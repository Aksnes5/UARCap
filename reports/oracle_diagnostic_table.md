# Top-20 Oracle Retrieval Diagnostic (diagnostic only, not inference)

| scale | X-CLIP top1 | rerank top1 | oracle@20 | top7 oracle (X-CLIP) | top7 oracle (rerank) | oracle rank mean/median | hit@1 | hit@5 | hit@7 | hit@20 | rerank moves oracle in | out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10K | 0.0629 | 0.0664 | 0.1904 | 0.1513 | 0.1322 | 8.61 / 8 | 8.3% | 31.7% | 44.3% | 100% | 19.0% | 25.0% |
| 100K | 0.0651 | 0.0997 | 0.2141 | 0.1617 | 0.1686 | 9.05 / 9 | 5.7% | 25.3% | 33.3% | 100% | 24.7% | 14.7% |
| 500K | 0.0725 | 0.1057 | 0.2128 | 0.1592 | 0.1694 | 9.51 / 10 | 4.7% | 23.3% | 34.0% | 100% | 28.3% | 17.3% |
| FULL | 0.0762 | 0.0981 | 0.2189 | 0.1548 | 0.1648 | 10.21 / 11 | 3.3% | 22.0% | 30.7% | 100% | 26.0% | 19.0% |

- oracle@20 is 2.6-2.9x the actual top-1 CIDEr at every scale.
- oracle candidates are in top-20 100% of the time, but hit@1 is only 3.3-8.3% and mean rank is 8.6-10.2.
- oracle@20 rises 10K->100K then plateaus; hit@7 declines with scale.
- rerank moves oracle into top-7 19-28% and out 15-25%; net positive at 100K/500K, negative at 10K, slightly positive at FULL.
