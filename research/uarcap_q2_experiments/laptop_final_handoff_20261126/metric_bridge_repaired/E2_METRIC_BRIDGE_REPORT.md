# E2-Metric-Bridge (post-hoc exploratory)

- Generated: 2026-11-26T08:08:46.983135+00:00
- Candidates: 600; predictions scored: 7200
- R-Short valid human substring rate: 1.0 (600/600)
- Word length R-Full mean/median: 228.09 / 218.0; R-Short: 14.53 / 14.0
- SPICE: SKIPPED

Traditional metrics use one fixed reference cohort per mode across all arms. CaReBench is single-reference; absolute scores are NOT comparable to multi-reference MSR-VTT/MSVD.

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
| formal_videollama | R-Full | ARM_A0->ARM_A1 | 600 | 0.2872 | -0.0083 | 0.0826 | 0.7084 | 0.035 | 0.025 | 0.9333 |
| formal_videollama | R-Full | ARM_A0->ARM_A7 | 600 | 0.2599 | 0.0463 | 0.0018 | 0.689 | 0.0367 | 0.0433 | 0.915 |
| formal_videollama | R-Full | ARM_A1->ARM_A7 | 600 | 0.2994 | 0.0862 | 0.0398 | 0.7097 | 0.0167 | 0.035 | 0.9367 |
| formal_videollama | R-Short | ARM_A0->ARM_A1 | 600 | 0.2412 | -0.1003 | 0.3159 | 0.2352 | 0.045 | 0.0283 | 0.9333 |
| formal_videollama | R-Short | ARM_A0->ARM_A7 | 600 | 0.2149 | -0.1301 | 0.2978 | 0.1707 | 0.0583 | 0.035 | 0.915 |
| formal_videollama | R-Short | ARM_A1->ARM_A7 | 600 | 0.2191 | -0.1245 | 0.1874 | 0.1327 | 0.0433 | 0.0217 | 0.9367 |
| formal_qwen | R-Full | ARM_A0->ARM_A1 | 600 | 0.1459 | -0.0569 | 0.0426 | 0.7208 | 0.07 | 0.0433 | 0.8481 |
| formal_qwen | R-Full | ARM_A0->ARM_A7 | 600 | 0.1031 | 0.0662 | -0.0158 | 0.7023 | 0.0767 | 0.065 | 0.8331 |
| formal_qwen | R-Full | ARM_A1->ARM_A7 | 600 | 0.1199 | 0.0992 | 0.017 | 0.6864 | 0.0517 | 0.0567 | 0.8815 |
| formal_qwen | R-Short | ARM_A0->ARM_A1 | 600 | 0.0193 | -0.0705 | 0.1686 | -0.0837 | 0.115 | 0.0517 | 0.8481 |
| formal_qwen | R-Short | ARM_A0->ARM_A7 | 600 | -0.053 | -0.1037 | 0.1954 | -0.0638 | 0.1583 | 0.06 | 0.8331 |
| formal_qwen | R-Short | ARM_A1->ARM_A7 | 600 | -0.0393 | -0.0654 | -0.0158 | -0.1118 | 0.1167 | 0.0467 | 0.8815 |

These are post-hoc descriptive associations; semantic rewrites, length and judge errors are plausible explanations. No causal claim; fact-level conclusions remain subject to human calibration.
