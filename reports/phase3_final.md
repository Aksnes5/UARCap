# Phase 3 UAR — current evidence

## Fixed-k (H5/H7 done; H1/H3 running)
H5 k=5 CIDEr 65.41, H7 k=7 64.01.

## Paired bootstrap (10k/50k, seed 42), CIDEr units
H5-D   +0.0214  CI [-0.0041,+0.0476]
H5-CF  +0.0103  CI [-0.0124,+0.0325]
H5-100K +0.0043 CI [-0.0231,+0.0319]
H5-B   -0.0110  CI [-0.0396,+0.0167]
=> all CIs cross zero; H5 > 100K is numerical only.

## Utility table (retrieval vs generation)
D: top1 0.1303, oracle7 0.1648, near-dup 0.561, CIDEr 63.25, SPICE 8.55
CF: top1 0.101, oracle7 ~0.155, near-dup ~0.56, CIDEr 64.38, SPICE 8.77
H5: top1 0.0746, oracle5 0.1479, near-dup 0.193, CIDEr 65.41, SPICE 8.74
H7: top1 0.0746, oracle7 0.1610, near-dup 0.181, CIDEr 64.01, SPICE 8.97

## Per-query H5 vs D100K bins
>+1: 3; +0.2..1: 25; -0.2..+0.2: 248; -1..-0.2: 23; <-1: 1.
Not outlier-driven; broad small shifts.

## Validation protocol
MSR-VTT official val: 497 videos, video6513-7009; train 0-6512, test 7010-9999.
Use val (or fixed 300 subset) for k/weights/λ/μ/thresholds; frames on-the-fly 16 or pre-extract;
freeze, then test once. Alternative: train-heldout (generator saw it; retrieval-side only).
