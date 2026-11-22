# Style / length / copying audit for the SPICE↑–CIDEr↓ pattern

**Status: RESOLVED (descriptive).** CPU-only, read-only audit of frozen generation
text and the retrieved captions that were actually supplied to the generator. No
generation, no CUDA, no frozen artifact or frozen metric was modified.

## 1. Data actually available

| requested condition | generated text | actually supplied evidence | status |
|---|---|---|---|
| MSR-VTT test2690 D1/D3/D5/D7 | `eval_test2690_D{k}/*_generated_captions.json` | `external/external_test_d7_cap.json` (D-order, first k) | complete |
| MSR-VTT val497 D1/D3/D5/D7 | `eval_val497_D{k}/*_generated_captions.json` | `external/external_val_d7_cap.json` (D-order, first k) | complete, but **caveated** (static-first-frame bug; see `VAL_TEST_CIDER_GAP_AUDIT.md`) |
| MSVD670 D1/D3/D5/D7 | `eval_msvd_D{k}/*_generated_captions.json` | `external/external_msvd_d7_cap.json` (D-order, first k) | complete |

The `captions` field inside each generated JSON is the **ground truth**, not the
evidence. The evidence was recovered from the `ext_*_d7_cap.json` files and the
generation logs (`--topn k`); `prompt_wrap` consumes `cap[image_id][0][:k]` in that
order. Both the generated text and the actual supplied evidence are therefore
available; nothing is missing.

The val497 split is used for completeness but its generator saw a static first
frame. The main conclusions below are carried by test2690 and MSVD, which use
pre-extracted frame TSVs and are unaffected.

## 2. Definitions

Word tokens are lowercased `[a-z0-9']+` matches (no LLM tokenizer used).
`P` = prediction tokens; `E1` = top-1 supplied evidence; `Eunion` = union of the k
supplied evidence captions, with n-grams not crossing caption boundaries.

- `unigram overlap` = |P ∩ E| / |P|; `Jaccard` = |P ∩ E| / |P ∪ E|.
- `ROUGE-L` = LCS F1 on word tokens.
- `copyN_tok` = fraction of generated N-gram **occurrences** that occur in `Eunion`.
- `has_copy3/4` = fraction of captions containing at least one copied 3-/4-gram.
- `exact_copy` = fraction of captions identical (case/punctuation-insensitive) to a
  supplied evidence caption.

Machine-readable row-per-condition results: `reports/cpu_audit/style_length_copying_summary.csv`.
Full percentiles/correlations: `reports/cpu_audit/style_length_copying_full.json`.

## 3. Output length

| dataset | k | mean WC | median WC | SD | IQR | Q10–Q90 |
|---|---:|---:|---:|---:|---:|---:|
| MSR-VTT test2690 | 1 | 7.91 | 7 | 2.51 | 3 | 6–11 |
| MSR-VTT test2690 | 3 | 8.09 | 7 | 2.76 | 3 | 6–12 |
| MSR-VTT test2690 | 5 | 8.35 | 7 | 3.05 | 4 | 6–12 |
| MSR-VTT test2690 | 7 | 8.74 | 8 | 3.39 | 4 | 6–13 |
| MSVD670 | 1 | 6.77 | 6 | 1.75 | 2 | 4–9 |
| MSVD670 | 3 | 6.97 | 7 | 1.82 | 2 | 5–9 |
| MSVD670 | 5 | 7.35 | 7 | 2.03 | 3 | 5–10 |
| MSVD670 | 7 | 7.61 | 7 | 2.16 | 3 | 6–10 |
| MSR-VTT val497 (caveat) | 1 | 8.38 | 8 | 2.51 | 2 | 6–12 |
| MSR-VTT val497 (caveat) | 3 | 8.60 | 8 | 2.70 | 3 | 6–12 |
| MSR-VTT val497 (caveat) | 5 | 8.70 | 8 | 2.94 | 3 | 6–13 |
| MSR-VTT val497 (caveat) | 7 | 9.14 | 8 | 3.43 | 3 | 6–14 |

**Answer A: yes, outputs get systematically longer as k increases.** The mean rises
by +0.83 words (test), +0.85 (MSVD), +0.76 (val) from k=1 to k=7. The median shifts
by at most one word (test: 7→8 at k=7; MSVD: 6→7 at k=3), so this is mainly a
right-tail / variance expansion (SD and IQR grow), not a uniform shift of every
caption.

## 4. Lexical overlap and copying from the supplied evidence

### MSR-VTT test2690 (unaffected)

| k | ov(E1) | Jacc(E1) | ov(Eunion) | Jacc(Eunion) | ROUGE-L(E1) | ROUGE-L max | copy1 | copy2 | copy3 | copy4 | has_copy3 | has_copy4 | exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.465 | 0.236 | 0.465 | 0.236 | 0.339 | 0.339 | 0.505 | 0.228 | 0.142 | 0.093 | 0.342 | 0.181 | 0.011 |
| 3 | 0.483 | 0.256 | 0.555 | 0.218 | 0.358 | 0.404 | 0.589 | 0.326 | 0.223 | 0.161 | 0.455 | 0.271 | 0.026 |
| 5 | 0.494 | 0.272 | 0.620 | 0.202 | 0.373 | 0.449 | 0.648 | 0.396 | 0.284 | 0.215 | 0.528 | 0.347 | 0.045 |
| 7 | 0.502 | 0.287 | 0.669 | 0.186 | 0.390 | 0.487 | 0.695 | 0.453 | 0.334 | 0.261 | 0.589 | 0.405 | 0.060 |

### MSVD670 (unaffected)

| k | ov(E1) | Jacc(E1) | ov(Eunion) | Jacc(Eunion) | ROUGE-L(E1) | ROUGE-L max | copy1 | copy2 | copy3 | copy4 | has_copy3 | has_copy4 | exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.516 | 0.243 | 0.516 | 0.243 | 0.367 | 0.367 | 0.561 | 0.234 | 0.138 | 0.085 | 0.330 | 0.185 | 0.000 |
| 3 | 0.554 | 0.276 | 0.623 | 0.225 | 0.401 | 0.448 | 0.658 | 0.357 | 0.242 | 0.168 | 0.493 | 0.313 | 0.009 |
| 5 | 0.583 | 0.308 | 0.698 | 0.210 | 0.435 | 0.512 | 0.730 | 0.455 | 0.327 | 0.239 | 0.610 | 0.409 | 0.031 |
| 7 | 0.595 | 0.327 | 0.741 | 0.190 | 0.454 | 0.547 | 0.768 | 0.519 | 0.387 | 0.290 | 0.690 | 0.479 | 0.039 |

### MSR-VTT val497 (static-first-frame caveat)

| k | ov(E1) | Jacc(E1) | ov(Eunion) | Jacc(Eunion) | ROUGE-L(E1) | ROUGE-L max | copy1 | copy2 | copy3 | copy4 | has_copy3 | has_copy4 | exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.416 | 0.216 | 0.416 | 0.216 | 0.315 | 0.315 | 0.458 | 0.188 | 0.114 | 0.073 | 0.292 | 0.155 | 0.014 |
| 3 | 0.432 | 0.234 | 0.500 | 0.199 | 0.337 | 0.375 | 0.538 | 0.270 | 0.175 | 0.124 | 0.380 | 0.229 | 0.030 |
| 5 | 0.449 | 0.249 | 0.569 | 0.184 | 0.352 | 0.419 | 0.600 | 0.337 | 0.233 | 0.176 | 0.453 | 0.286 | 0.056 |
| 7 | 0.453 | 0.263 | 0.612 | 0.172 | 0.365 | 0.452 | 0.643 | 0.383 | 0.268 | 0.208 | 0.493 | 0.326 | 0.077 |

**Answer B: yes, outputs copy the retrieved captions much more strongly as k grows.**
From k=1 to k=7, the 3-gram copy rate rises 0.142→0.334 (test, ×2.35),
0.138→0.387 (MSVD, ×2.80), 0.114→0.268 (val, ×2.35). The 4-gram rate rises
0.093→0.261 / 0.085→0.290 / 0.073→0.208. Captions containing at least one copied
4-gram rise from 18.1% to 40.5% (test) and 18.5% to 47.9% (MSVD). Verbatim copies of
a supplied caption rise from 1.1% to 6.0% (test) and 0% to 3.9% (MSVD). Overlap with
the union of supplied evidence rises 0.465→0.669 / 0.516→0.741, while Jaccard falls
(the evidence union grows faster than the output), so overlap and Jaccard must be
read together.

## 5. Co-occurrence with SPICE↑ and CIDEr↓

| dataset | k=1 | k=3 | k=5 | k=7 | direction |
|---|---:|---:|---:|---:|---|
| MSR-VTT test2690 CIDEr | 71.21 | 70.08 | 69.48 | 68.27 | ↓ 2.94 |
| MSR-VTT test2690 SPICE | 8.82 | 8.88 | 9.00 | 9.12 | ↑ 0.30 |
| MSVD670 CIDEr | 156.39 | 151.45 | 144.39 | 141.59 | ↓ 14.80 |
| MSVD670 SPICE | 9.08 | 9.15 | 9.15 | 9.25 | ↑ 0.17 |
| MSR-VTT val497 CIDEr (caveat) | 51.77 | 53.14 | 50.51 | 50.05 | non-monotone |
| MSR-VTT val497 SPICE (caveat) | 7.55 | 7.89 | 7.66 | 7.88 | non-monotone |

**Answer C: yes, the length/copying gradient co-occurs with SPICE↑ and CIDEr↓ on the
two splits whose visual input is intact.** On test2690 and MSVD, k increases output
length and evidence copying, and in the same sweep SPICE rises while CIDEr falls. On
val497 the copying/length trend is identical, but the CIDEr path is non-monotone
(and val497 is the split affected by the static-first-frame bug), so val497 cannot be
used to support C.

Condition-level co-occurrence is descriptive, not causal. Two further observations
argue against a simple "copying causes the loss" reading:

- Within every condition, longer outputs score **lower** CIDEr (Spearman
  −0.199…−0.291 on test; −0.186…−0.292 on MSVD), and outputs with more evidence
  overlap score **higher** CIDEr (Spearman +0.236 at D1 → +0.087 at D7 on test;
  +0.257 → +0.132 on MSVD). The within-condition sign is therefore not the same as
  the between-k sign.
- The per-query correlation between 3-gram copying and CIDEr is positive at k=1
  (+0.102 test, +0.127 MSVD) and decays toward zero at k=7 (+0.026 / +0.051). Thus
  copying is not uniformly harmful; at large k the added copied material simply does
  not align with the reference-relevant content.

A plausible descriptive reading — not a causal claim — is that raising k adds
increasingly redundant, generic evidence that lengthens and styles the caption while
diluting precision against the 20 references. The audit cannot separate this from
other k-dependent effects.

## 6. What was not computed

- Per-query SPICE was not computed; SPICE is available only at the frozen
  condition level from each `metrics.json`. Condition-level co-occurrence is
  sufficient for question C, but query-level SPICE/CIDEr correlations are not
  available in this audit.
- No LLM tokenizer was loaded; token counts are lightweight regex/PTB-style word
  counts, not Vicuna tokens.

## 7. Manuscript implication

- Safe: a descriptive Results/Supplement paragraph stating that larger k produces
  longer, more evidence-copying captions, and that this co-occurs with higher SPICE
  and lower CIDEr on MSR-VTT test2690 and MSVD.
- Not safe: causal wording ("copying causes the CIDEr drop"), or using val497 as the
  supporting split for this claim.
- The val497 version of the analysis is internally valid as a paired description but
  should be labelled with the static-first-frame caveat.
