# Source-cluster bootstrap (S7)

Cluster unit = original YouTube source. MSR-VTT test2690: 2,004 sources among
2,690 queries; MSVD670: 522 sources among 670 clips. Resampling sources with
replacement (10k replicates) gives:

| comparison | Δ CIDEr | cluster 95% CI |
|---|---|---|
| test2690 D3−D7 | +0.0181 | [+0.0088, +0.0275] |
| test2690 D1−D3 | +0.0114 | [+0.0035, +0.0196] |
| test2690 D3−D5 | +0.0060 | [−0.0020, +0.0141] |
| test2690 Adaptive−D3 | +0.0058 | [+0.0001, +0.0116] |
| test2690 Adaptive−D7 | +0.0240 | [+0.0147, +0.0335] |
| MSVD D3−D7 | +0.0986 | [+0.0538, +0.1484] |
| MSVD D1−D3 | +0.0494 | [+0.0164, +0.0842] |
| MSVD D3−D5 | +0.0706 | [+0.0282, +0.1144] |
| MSVD Adaptive−D3 | +0.0049 | [−0.0268, +0.0354] |
| MSVD Adaptive−D7 | +0.1035 | [+0.0630, +0.1449] |

Source-clustered CIs are essentially identical to query-level CIs; significance
conclusions are unchanged, including the borderline Adaptive−D3 on test2690.
