# Phase 3 UAR H5/H7 (dev300 diagnostic)

| variant | B1 | B4 | METEOR | ROUGE-L | CIDEr | SPICE |
|---|---:|---:|---:|---:|---:|---:|
| H7 k=7 | 85.25 | 46.14 | 32.16 | 64.27 | 64.01 | 8.97 |
| H5 k=5 | 85.60 | 46.76 | 32.10 | 64.59 | 65.41 | 8.74 |
| D FULL old | 84.33 | 44.42 | 31.59 | 63.98 | 63.25 | 8.55 |
| CF FULL | 85.15 | 45.42 | 31.80 | 64.19 | 64.38 | 8.77 |
| D 100K old | 86.18 | 45.57 | 32.36 | 64.78 | 64.98 | 8.90 |
| B in-domain | 84.32 | 48.10 | 31.63 | 64.23 | 66.51 | 8.10 |

Paired bootstrap (10k, seed 42), CIDEr units:
- H5-D: +0.0214, 95% CI [-0.0041,+0.0476], permutation p=0.103
- H5-CF: +0.0103, CI [-0.0124,+0.0329], p=0.370
- H5-100K: +0.0043, CI [-0.0224,+0.0313], p=0.761
- H5-B: -0.0110, CI [-0.0397,+0.0172], p=0.446

So H5 numerically exceeds 100K old but the paired CI crosses zero; not statistically significant at n=300.
