# E2-Metric-Bridge (post-hoc exploratory)

- Generated: 2026-11-26T05:52:58.771078+00:00
- Candidates: 600; predictions scored: 7200
- R-Short valid human substring rate: 1.0 (600/600)
- Word length R-Full mean/median: 228.09 / 218.0; R-Short: 14.53 / 14.0
- SPICE: SKIPPED

Traditional metrics are computed on the same 600 videos and same 3,600 captions. CaReBench is single-reference; absolute scores are NOT comparable to multi-reference MSR-VTT/MSVD.

## Corpus summary

| model | ref | arm | n | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| formal_videollama | R-Full | ARM_A0 | 600 | 0.0000 | 0.0000 | 0.0109 | 0.0374 | 0.0000 | None |
| formal_videollama | R-Full | ARM_A1 | 600 | 0.0000 | 0.0000 | 0.0113 | 0.0385 | 0.0000 | None |
| formal_videollama | R-Full | ARM_A7 | 600 | 0.0000 | 0.0000 | 0.0121 | 0.0406 | 0.0000 | None |
| formal_videollama | R-Short | ARM_A0 | 600 | 0.1565 | 0.0246 | 0.0999 | 0.2412 | 0.3103 | None |
| formal_videollama | R-Short | ARM_A1 | 600 | 0.1664 | 0.0290 | 0.1056 | 0.2517 | 0.3534 | None |
| formal_videollama | R-Short | ARM_A7 | 600 | 0.1813 | 0.0318 | 0.1091 | 0.2539 | 0.3664 | None |
| formal_qwen | R-Full | ARM_A0 | 600 | 0.0000 | 0.0000 | 0.0209 | 0.0473 | 0.0000 | None |
| formal_qwen | R-Full | ARM_A1 | 600 | 0.0000 | 0.0000 | 0.0212 | 0.0486 | 0.0000 | None |
| formal_qwen | R-Full | ARM_A7 | 600 | 0.0000 | 0.0000 | 0.0216 | 0.0495 | 0.0000 | None |
| formal_qwen | R-Short | ARM_A0 | 600 | 0.2297 | 0.0229 | 0.1287 | 0.1818 | 0.3393 | None |
| formal_qwen | R-Short | ARM_A1 | 600 | 0.2356 | 0.0241 | 0.1304 | 0.1844 | 0.3423 | None |
| formal_qwen | R-Short | ARM_A7 | 600 | 0.2398 | 0.0250 | 0.1320 | 0.1849 | 0.3419 | None |

## Paired change relations (video-level, Spearman)

| model | ref | transition | n | CIDEr~support | CIDEr~conflict | CIDEr~core | CIDEr~length | CIDEr↓ & support↑ | CIDEr↑ & conflict↑ | judge direction agreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| formal_videollama | R-Full | ARM_A0->ARM_A1 | 600 | 0.0123 | -0.0776 | 0.0781 | -0.0141 | 0.0267 | 0.0067 | 0.9333 |
| formal_videollama | R-Full | ARM_A1->ARM_A7 | 600 | 0.0284 | -0.0503 | 0.0363 | -0.0212 | 0.0133 | 0.01 | 0.9367 |
| formal_videollama | R-Full | ARM_A0->ARM_A7 | 600 | 0.0017 | 0.0243 | 0.0662 | 0.0088 | 0.04 | 0.02 | 0.915 |
| formal_videollama | R-Short | ARM_A0->ARM_A1 | 600 | -0.0258 | -0.0052 | -0.0109 | -0.045 | 0.0317 | 0.0117 | 0.9333 |
| formal_videollama | R-Short | ARM_A1->ARM_A7 | 600 | -0.0124 | -0.0498 | 0.0327 | -0.0466 | 0.025 | 0.0133 | 0.9367 |
| formal_videollama | R-Short | ARM_A0->ARM_A7 | 600 | -0.0045 | -0.0435 | 0.0001 | -0.051 | 0.055 | 0.02 | 0.915 |
| formal_qwen | R-Full | ARM_A0->ARM_A1 | 600 | -0.0771 | 0.0558 | -0.1105 | -0.0858 | 0.075 | 0.035 | 0.8483 |
| formal_qwen | R-Full | ARM_A1->ARM_A7 | 600 | -0.0047 | -0.0039 | -0.0199 | -0.0001 | 0.0267 | 0.0267 | 0.8817 |
| formal_qwen | R-Full | ARM_A0->ARM_A7 | 600 | 0.0242 | 0.0226 | -0.016 | -0.055 | 0.0717 | 0.0517 | 0.8317 |
| formal_qwen | R-Short | ARM_A0->ARM_A1 | 600 | -0.0032 | -0.0701 | 0.0445 | 0.001 | 0.0817 | 0.03 | 0.8483 |
| formal_qwen | R-Short | ARM_A1->ARM_A7 | 600 | 0.034 | -0.0681 | 0.04 | -0.0371 | 0.0567 | 0.0217 | 0.8817 |
| formal_qwen | R-Short | ARM_A0->ARM_A7 | 600 | 0.051 | -0.0988 | 0.0003 | -0.0383 | 0.095 | 0.0517 | 0.8317 |

These are post-hoc descriptive associations; semantic rewrites, length and judge errors are plausible explanations. No causal claim; fact-level conclusions remain subject to human calibration.
