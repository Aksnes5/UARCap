# POST_FIX scale × budget results (val497, frame-loader fixed)

CIDEr points. All 48 configurations are VALID_FINAL with manifests in
`reports/post_fix/manifests/`.

## D track
| scale | k1 | k3 | k5 | k7 | best | k7 penalty |
|---|---|---|---|---|---|---|
| 10K | 69.21 | 67.80 | 67.67 | 67.33 | 1 | 1.88 |
| 50K | 68.89 | 67.84 | 66.80 | 66.12 | 1 | 2.77 |
| 100K | 69.61 | 68.68 | 66.69 | 66.15 | 1 | 3.46 |
| 250K | 70.29 | 69.29 | 68.75 | 66.99 | 1 | 3.30 |
| 500K | 70.59 | 70.08 | 68.25 | 67.62 | 1 | 2.97 |
| FULL | 69.97 | 68.63 | 67.31 | 67.22 | 1 | 2.75 |

## CF track
| scale | k1 | k3 | k5 | k7 | best | k7 penalty |
|---|---|---|---|---|---|---|
| 10K | 70.43 | 70.13 | 69.30 | 68.73 | 1 | 1.70 |
| 50K | 69.98 | 68.47 | 68.08 | 67.09 | 1 | 2.89 |
| 100K | 69.89 | 70.10 | 69.32 | 67.57 | 3 | 2.53 |
| 250K | 70.50 | 69.09 | 67.80 | 68.77 | 1 | 1.73 |
| 500K | 70.50 | 68.87 | 68.89 | 67.23 | 1 | 3.27 |
| FULL | 71.17 | 68.88 | 68.84 | 67.95 | 1 | 3.22 |

- Best k: k1 11/12, k3 1/12, k5 0/12, k7 0/12.
- k7 penalty: D mean 2.86 / median 2.87; CF mean 2.56 / median 2.71; combined 2.71.
- Global maximum: FULL CF1 **71.17**; D-track maximum 500K D1 **70.59**.
- Scale vs k1 is non-monotone and small (CF1 69.89–71.17; D1 68.89–70.59).
