---
title: "Supplementary Materials: When Does External Retrieval Help Video Captioning? Evidence Budgets in a Million-Clip Memory"
lang: en
---

# Supplementary Materials

**Manuscript:** *When Does External Retrieval Help Video Captioning? Evidence Budgets in a Million-Clip Memory*  
**Authors:** *[Author List — TO BE COMPLETED BY AUTHORS]*

---

## S1. Nested External Corpus Construction and Source-Overlap Audit

Six strictly nested external clip manifests were constructed from VideoUFO (Wang and Yang 2025; VideoUFO 2025). Every smaller manifest is an exact subset of every larger manifest (zero missing clip IDs across adjacent levels). Up to 100,000 clips, each admitted clip originates from a distinct YouTube source video; from 250,000 to FULL (1,091,712 clips), additional clips may be drawn from already represented source videos. Source-ID audits verify zero overlap between any manifest level and the MSR-VTT training, validation, or test splits (Table S1).

**Table S1.** Verified clip counts, unique source-video counts, nesting integrity, and source-ID overlap with MSR-VTT across the six nested VideoUFO manifests.

| Manifest scale | Unique clips | Unique source videos | Clips missing from larger level | Overlap with MSR-VTT train (6,513) | Overlap with MSR-VTT val (497) | Overlap with MSR-VTT test (2,990) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10K | 10,000 | 10,000 | 0 | 0 | 0 | 0 |
| 50K | 50,000 | 50,000 | 0 | 0 | 0 | 0 |
| 100K | 100,000 | 100,000 | 0 | 0 | 0 | 0 |
| 250K | 250,000 | 241,211 | 0 | 0 | 0 | 0 |
| 500K | 500,000 | 338,877 | 0 | 0 | 0 | 0 |
| FULL | 1,091,712 | 345,437 | — | 0 | 0 | 0 |

**Clip count versus caption-group cardinality:** Retrieval searches deduplicated brief-caption groups formed by exact string equality after stripping leading and trailing whitespace while preserving case and punctuation, mapping each group deterministically to its first sorted clip ID for visual fusion. Historical project records list two caption-group/string totals (812,932 and 812,014) whose complete raw-CSV derivations were not re-executed during the final audit. Throughout the main manuscript, corpus scale is therefore reported using the verified clip-manifest denominator (10,000 to 1,091,712 clips), noting explicitly that clip count is distinct from unique searchable caption count. Exact source-ID checks do not rule out visual near-duplicates or upstream pretraining exposure.

---

## S2. Model, Feature-Extraction, and Decoding Configuration

All experiments hold the inherited captioning architecture from Fang et al. (2026) frozen without retraining or fine-tuning:

- **Retriever:** Video-conditioned X-CLIP (`k600_16_16.pth`, Ni et al. 2022), scoring top-20 brief-caption groups via exact cosine similarity. Query videos for X-CLIP use 16 centre-of-bin frames, shorter-side resize to 256 (nearest-neighbour), centre-crop to 224, and ImageNet normalization.
- **Visual encoder and fusion:** Frozen EVA-CLIP ViT-g + BLIP-2 Q-Former (Li et al. 2023) + Video-LLaMA 2-layer Video Q-Former (Zhang et al. 2023), producing 32 visual tokens of dimension 768 per clip. Retrieved clip features are extracted from 16 uniformly spaced frames including temporal endpoints at 224 resolution. Cross-gating fusion ($F_\theta$) aggregates retrieved video tensors via softmax weights over raw X-CLIP similarities, applies 8-head cross-attention against the query video representation, combines features through learned GRU-style update and reset gates, and projects into the LLM embedding space via a separate linear layer.
- **Language decoder and decoding:** Frozen Vicuna-7B-v1.3 (LMSYS 2023), beam search width 5, random seed 42, deterministic caption-to-clip association, and fixed benchmark prompt template.
- **Checkpoints and module states:** MSR-VTT uses the inherited 7-context checkpoint (SHA-256 `a8b5af79d55a2c7e684771475d645be6373208bdaae5a4211bee556581eb8002`); MSVD uses the inherited 3-context checkpoint. Across all six follow-up arms, the inherited module states (parent, Video Q-Former, and cross-attention in training mode; image Q-Former and Vicuna-7B in evaluation mode) and encoder RNG states were preserved identically per query to guarantee strict cross-arm parity.
- **Predecessor reranking fidelity note:** Fang et al. (2026, Section 3.4) describe quality-aware reranking using noun-phrase pseudo-labels extracted from the remaining candidates. Our frozen **D** ordering instead computes mean pairwise full-caption ROUGE-L over complete candidate strings, $q_i = \frac{1}{19}\sum_{j\ne i}\mathrm{ROUGE\text{-}L}(c_i, c_j)$, and is therefore documented explicitly as a full-caption consensus approximation.

---

## S3. Complete Historical Benchmark Contrast Families

Table S2 reports all seven contrasts in the pilot-excluded MSR-VTT benchmark family (*n* = 2,690; 2,004 source clusters) and all six contrasts in the MSVD benchmark family (*n* = 670; 522 source clusters) under official PTB CIDEr and source-cluster inference (10,000 bootstrap resamples and 10,000 two-sided sign flips).

**Table S2.** Complete within-benchmark paired contrast families (official PTB CIDEr points, source-cluster inference, BH adjustment within each benchmark family).

| Benchmark | Contrast | Effect | 95% source-cluster CI | Source *p* | Source BH *q* |
| --- | --- | ---: | --- | ---: | ---: |
| MSR-VTT (`test2690`) | D1 − D3 | +1.13 | [+0.35, +1.94] | .0059 | .0083 |
| MSR-VTT (`test2690`) | D3 − D7 | +1.80 | [+0.86, +2.76] | .0002 | .0007 |
| MSR-VTT (`test2690`) | D1 − D7 | +2.93 | [+1.83, +4.06] | .0001 | .0007 |
| MSR-VTT (`test2690`) | D5 − D7 | +1.21 | [+0.44, +1.97] | .0018 | .0031 |
| MSR-VTT (`test2690`) | Adaptive − D1 | −0.54 | [−1.20, +0.10] | .1006 | .1006 |
| MSR-VTT (`test2690`) | Adaptive − D3 | +0.59 | [+0.02, +1.18] | .0399 | .0465 |
| MSR-VTT (`test2690`) | In-domain − D1 | +1.88 | [+0.77, +2.96] | .0008 | .0019 |
| MSVD (`test670`) | D1 − D3 | +4.94 | [+1.50, +8.39] | .0046 | .0069 |
| MSVD (`test670`) | D3 − D7 | +9.86 | [+5.49, +15.05] | .0001 | .0003 |
| MSVD (`test670`) | D1 − D7 | +14.80 | [+9.92, +19.97] | .0001 | .0003 |
| MSVD (`test670`) | D5 − D7 | +2.80 | [−0.37, +5.97] | .0934 | .1121 |
| MSVD (`test670`) | Adaptive − D1 | −4.46 | [−7.68, −1.47] | .0030 | .0060 |
| MSVD (`test670`) | Adaptive − D3 | +0.49 | [−2.73, +3.55] | .7661 | .7661 |

**MSVD in-domain reference provenance:** Fang et al. (2026, Table 1) report 158.1 CIDEr (BLEU-4 69.5) on MSVD test; a separately saved archival engineering reproduction records 158.32 CIDEr. Because the original reproduction run environment and full output files are outside the verified external-evaluation archive, both values are noted separately as context and omitted from paired hypothesis testing in Table S2.

---

## S4. Reference-Guided Oracle Analysis, Pool Depletion, and Refitted-IDF Sensitivity

Independent replay verified exact agreement of candidate captions, clip associations, and stored X-CLIP similarity weights across all 497 MSR-VTT validation queries for Oracle1, Oracle3, and Oracle7 (strict prefixes of the same reference-guided top-20 list). Official PTB means on repaired validation are **71.64** (Oracle1), **71.67** (Oracle3), and **68.94** (Oracle7), compared with **69.97** (D1), **68.63** (D3), and **67.22** (D7).

- **Multiplicity sensitivity:** Within the 4-contrast Oracle family, both Oracle3 − D3 (+3.04, source CI [+0.85, +5.27]) and Oracle7 − Oracle3 (−2.73, [−4.72, −0.80]) pass BH adjustment (source *q* = .0146; query *q* = .0176). When combined with the 72-contrast validation grid into a 76-test sensitivity family, query-level BH yields *q* = .1368 for both.
- **Refitted-IDF bootstrap sensitivity:** Re-estimating reference document frequencies inside each of 999 source-cluster bootstrap draws (361 sources; maximum discrepancy against the frozen evaluator $2.27 \times 10^{-13}$) produces 95% intervals of **[+0.78, +5.30]** for Oracle3 − D3 (fixed-IDF same draws: [+0.81, +5.35]) and **[−4.87, −0.94]** for Oracle7 − Oracle3 (fixed-IDF same draws: [−4.75, −0.80]), confirming robustness to IDF estimation variability.
- **Candidate-pool compatibility decline:** Unigram fixed-reference TF-IDF compatibility averages **0.1380** across ranks 1–3 versus **0.1031** across ranks 4–7 (smooth decline across ranks). Across the 497 queries, the rank correlation between head-to-tail compatibility drop and Oracle3 − Oracle7 CIDEr difference is **$\rho = .046$** (95% source-cluster CI [−.044, +.131]), with roughly 60% of per-query CIDEr effects tied at zero. Because extending the Oracle prefix changes candidate count, candidate compatibility, and visual features jointly, Oracle7 − Oracle3 does not isolate pure textual overload.

---

## S5. Complete Repaired Validation Mechanism and Leave-One-Out Controls

Table S3 reports all 18 repaired validation mechanism contrasts (`val497`, 361 source clusters) and all 8 leave-one-out removal contrasts (`val150`, 137 source clusters; removing one pair from D7 to leave six without refill). None of the 26 contrasts pass BH adjustment within their respective families.

**Table S3.** Complete repaired validation diagnostic contrasts (official PTB CIDEr points, pointwise 95% source-cluster CIs, two-sided sign-flip *p*-values, and within-family BH *q*-values; 18 mechanism contrasts in `val497`, 8 LOO contrasts in `val150`).

| Family | Contrast | Effect | 95% source-cluster CI | Source *p* | Source BH *q* |
| --- | --- | ---: | --- | ---: | ---: |
| Mechanism (`val497`) | Text effect at V = 3 (T7V3 − T3V3) | −1.45 | [−4.22, +1.22] | .2970 | .7349 |
| Mechanism (`val497`) | Text effect at V = 7 (T7V7 − T3V7) | −1.29 | [−4.02, +1.35] | .3439 | .7349 |
| Mechanism (`val497`) | Visual effect at T = 3 (T3V7 − T3V3) | −0.12 | [−0.78, +0.53] | .7404 | .9031 |
| Mechanism (`val497`) | Visual effect at T = 7 (T7V7 − T7V3) | +0.04 | [−0.35, +0.50] | .8448 | .9031 |
| Mechanism (`val497`) | Text count main effect | −1.37 | [−4.10, +1.26] | .3154 | .7349 |
| Mechanism (`val497`) | Visual count main effect | −0.04 | [−0.42, +0.35] | .8529 | .9031 |
| Mechanism (`val497`) | Text × visual interaction | +0.16 | [−0.60, +0.94] | .7003 | .9031 |
| Mechanism (`val497`) | External FULL D3 − D7 | +1.41 | [−1.21, +4.17] | .3042 | .7349 |
| Mechanism (`val497`) | Matched in-domain D3 − D7 | −1.34 | [−3.59, +0.92] | .2553 | .7349 |
| Mechanism (`val497`) | Source × budget interaction | +2.74 | [−0.76, +6.29] | .1322 | .7349 |
| Mechanism (`val497`) | Reverse prompt order vs. D7 | −0.24 | [−2.08, +1.67] | .8061 | .9031 |
| Mechanism (`val497`) | Top-three-last order vs. D7 | +0.02 | [−1.76, +1.92] | .9813 | .9813 |
| Mechanism (`val497`) | MidTail7 (ranks 8–11) vs. D7 | +1.20 | [−0.80, +3.36] | .2574 | .7349 |
| Mechanism (`val497`) | LowTail7 (ranks 17–20) vs. D7 | −0.64 | [−2.77, +1.36] | .5513 | .9031 |
| Mechanism (`val497`) | GoodTail vs. D7 | −0.56 | [−3.23, +2.04] | .6827 | .9031 |
| Mechanism (`val497`) | HardWrong vs. D7 | −1.45 | [−3.55, +0.66] | .1754 | .7349 |
| Mechanism (`val497`) | EasyWrong vs. D7 | −0.37 | [−2.35, +1.64] | .7259 | .9031 |
| Mechanism (`val497`) | HardWrong vs. EasyWrong | −1.08 | [−3.44, +1.22] | .3675 | .7349 |
| LOO (`val150`) | Remove rank 1 vs. D7 | +1.97 | [−1.04, +5.11] | .2201 | .8803 |
| LOO (`val150`) | Remove rank 2 vs. D7 | −0.56 | [−4.64, +3.22] | .7854 | .9265 |
| LOO (`val150`) | Remove rank 3 vs. D7 | +0.39 | [−3.24, +3.84] | .8357 | .9265 |
| LOO (`val150`) | Remove rank 4 vs. D7 | +0.58 | [−2.98, +4.01] | .7588 | .9265 |
| LOO (`val150`) | Remove rank 5 vs. D7 | −0.75 | [−4.31, +2.47] | .6764 | .9265 |
| LOO (`val150`) | Remove rank 6 vs. D7 | −0.17 | [−3.68, +3.09] | .9265 | .9265 |
| LOO (`val150`) | Remove rank 7 vs. D7 | −1.51 | [−5.28, +1.97] | .4236 | .9265 |
| LOO (`val150`) | LOO head (1–3) − tail (4–7) | +1.06 | [−0.18, +2.40] | .1127 | .8803 |

---

## S6. Experiment B Eligible Cohort Construction and Matching Diagnostics

For each of the 2,690 pilot-excluded MSR-VTT test queries, Experiment B retained the D-ordered head of three caption–clip pairs (`B3`) and searched the remaining 17 candidates in the FULL top-20 pool for a matched pair of four-caption text-only extensions: an **HQ** quartet drawn from the eight highest raw X-CLIP similarity remaining candidates and an **LQ** quartet drawn from the eight lowest remaining candidates. Eligibility required: (i) four distinct representative video sources within each quartet, disjoint from the three head sources; (ii) a minimum mean raw X-CLIP similarity gap of 0.005 between HQ and LQ; and (iii) selection of the pair minimizing normalized imbalances in total word count, head Jaccard overlap, and internal quartet Jaccard overlap (breaking ties by similarity gap and candidate index, without accessing references or predictions). Exactly **2,618** queries (1,969 unique sources) satisfied all criteria; 72 unmatched queries and the 300 excluded pilot queries received the B3 head identically across all three arms. Table S4 summarizes achieved balance.

**Table S4.** Experiment B matching balance and observed wrapped prompt lengths across the 2,618 eligible queries.

| Metric | Mean | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| Mean raw X-CLIP similarity gap (HQ − LQ) | 0.00835 | — | 0.00500 | 0.02606 |
| Absolute quartet word-count imbalance | 0.35 | — | 0 | 7 |
| Absolute head-overlap Jaccard imbalance | 0.0085 | — | 0.000004 | 0.0798 |
| Absolute internal-overlap Jaccard imbalance | 0.0074 | — | 0.000003 | 0.0781 |
| Wrapped prompt length: B3 (positions) | 150.69 | 148 | 125 | 219 |
| Wrapped prompt length: B7HQ (positions) | 226.86 | 224 | 172 | 346 |
| Wrapped prompt length: B7LQ (positions) | 226.38 | 223 | 176 | 347 |
| Prompt-length difference (B7HQ − B7LQ) | +0.48 | 3 (abs. median) | — | — |

---

## S7. Complete Statistical Details for Experiments A and B

Both follow-up experiments used official PTB CIDEr with cohort-specific reference IDF (2,690 queries / 2,004 clusters in A; 2,618 queries / 1,969 clusters in B; maximum cluster size 6; zero fallback singletons), 10,000 bootstrap draws, and 10,000 two-sided sign flips with seed `20261125`. Table S5 lists both query-level and primary source-cluster inference for full transparency alongside paired standardized effect sizes ($d_z$).

**Table S5.** Full query-level and source-cluster inference for all five prespecified contrasts in Experiments A and B (PTB CIDEr points). Primary inference uses source-cluster Holm adjustment.

| Contrast | Effect | Paired $d_z$ | Query 95% CI | Query *p* | Source 95% CI | Source *p* | Family Holm *p* | Five-test Holm *p* |
| --- | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: |
| A1 − A0 | +1.40 | +0.057 | [+0.50, +2.33] | .0018 | [+0.45, +2.36] | .0033 | .0066 | .0165 |
| A7 − A0 | −1.54 | −0.045 | [−2.84, −0.25] | .0174 | [−2.94, −0.18] | .0271 | .0271 | .0813 |
| B7HQ − B3 | −1.27 | −0.046 | [−2.33, −0.21] | .0203 | [−2.34, −0.20] | .0191 | .0573 | .0764 |
| B7LQ − B3 | −1.13 | −0.043 | [−2.11, −0.14] | .0251 | [−2.15, −0.14] | .0306 | .0612 | .0813 |
| B7HQ − B7LQ | −0.14 | −0.006 | [−1.03, +0.79] | .7623 | [−1.05, +0.80] | .7698 | .7698 | .7698 |

---

## S8. Runtime Integrity, Endpoint Parity, and Visual-Tensor Verification

The six formal arms of Experiments A and B produced 17,940 predictions and 17,940 per-query runtime trace records (2,990 queries × 6 arms; 11.2245 total GPU hours; zero failed runs or retries), preceded by 48 smoke-test generations excluded from formal statistics. Independent offline CPU validation verified all 45 artifact SHA-256 digests, completion seals, and per-query trace invariants prior to scoring (Table S6).

**Table S6.** Summary of verified runtime invariants across Experiments A and B (2,990 queries per arm).

| Verification domain | Verified condition | Outcome |
| --- | --- | --- |
| Experiment A zero-evidence arm (A0) | Zero captions, zero retrieved features loaded, no visual softmax; $1 \times 32 \times 768$ external aggregate and attention K/V equal to exact zero tensor | PASS (2,990 / 2,990) |
| Experiment A cross-arm parity | Identical query frames, Q-Former inputs, query visual projection, masks, module states, and CPU/CUDA encoder RNG across A0, A1, A7 | PASS (2,990 / 2,990) |
| Experiment A endpoint reproducibility | Parsed outputs of A1 and A7 match archived D1 and D7 records identically across all 2,990 test queries | PASS (2,990 / 2,990) |
| Experiment B visual-tensor parity | Identical query frames, representative clip IDs, raw similarity weights, ordered feature-tensor hashes, fused aggregate, attention K/V, gate inputs, visual projections, and encoder RNG across B3, B7HQ, B7LQ | PASS (2,990 / 2,990) |
| Experiment B ineligible-query parity | For all 372 non-eligible queries (300 pilots + 72 unmatched), B7HQ and B7LQ match B3 predictions and traces identically | PASS (372 / 372) |
| Reference & source metadata alignment | Exact query reference list/order match; zero fallback singleton clusters (2,004 clusters in A; 1,969 in B) | PASS |

---

## S9. Validation Video-Decoder Repair and Pre-Fix Exclusion

During independent auditing of earlier development runs, a raw-video dataloader defect was identified specifically in the MSR-VTT validation evaluation path (`val497` and `val150`). Because the validation configuration referenced raw video paths rather than pre-extracted frame TSVs, the evaluation branch passed `start = 0, end = 0` to the on-the-fly PyAV decoder, truncating decoding to 1–2 early frames and replicating frame 0 across 15 of 16 input slots. Test evaluations (`test2690`, full `test2990`, and MSVD `test670`) loaded pre-extracted 16-frame TSVs that bypassed `start/end` and delivered 16 distinct frames; X-CLIP retrieval embeddings were likewise pre-extracted with whole-video sampling and unaffected.

Restoring whole-video decoding for `val497` and `val150` brought validation CIDEr back to the expected ~67–74 range and required re-running all 48 scale-grid conditions and all mechanism diagnostics. Earlier pre-fix validation results—under which collapsed visual input amplified apparent textual-count and source-interaction contrasts—are permanently excluded from scientific claims and retained solely as historical provenance.

---

## S10. Output Length, Trigram Reuse, and Metric-Reconciliation Notes

Table S7 summarizes descriptive output word count and trigram matching against retrieved captions across budgets $k \in \{1, 3, 5, 7\}$. On MSR-VTT `test2690`, mean length rises from 7.92 to 8.75 words, supplied-union trigram match rises from 0.142 to 0.334, and fixed-top-1 trigram match rises from 0.142 to 0.190; repaired `val497` displays a closely parallel gradient (8.01 to 8.65 words; 0.147 to 0.331 union match; 0.147 to 0.191 fixed-top-1 match). On MSVD, mean length rises from 6.77 to 7.61 words (MSVD supplied-caption trigram matching was not re-evaluated in the local independent audit archive because the MSVD candidate artifact was external to that replay bundle).

**Table S7.** Descriptive output word counts and trigram match rates across D-ordered joint budgets.

| Split | Budget $k$ | Mean words | Fixed top-1 trigram match | Supplied-union trigram match |
| --- | :---: | ---: | ---: | ---: |
| MSR-VTT `test2690` (*n* = 2,690) | 1 | 7.92 | 0.142 | 0.142 |
| MSR-VTT `test2690` (*n* = 2,690) | 3 | 8.10 | 0.164 | 0.223 |
| MSR-VTT `test2690` (*n* = 2,690) | 5 | 8.36 | 0.179 | 0.284 |
| MSR-VTT `test2690` (*n* = 2,690) | 7 | 8.75 | 0.190 | 0.334 |
| MSR-VTT repaired `val497` (*n* = 497) | 1 | 8.01 | 0.147 | 0.147 |
| MSR-VTT repaired `val497` (*n* = 497) | 3 | 8.18 | 0.174 | 0.223 |
| MSR-VTT repaired `val497` (*n* = 497) | 5 | 8.40 | 0.186 | 0.288 |
| MSR-VTT repaired `val497` (*n* = 497) | 7 | 8.65 | 0.191 | 0.331 |
| MSVD `test670` (*n* = 670) | 1 | 6.77 | — | — |
| MSVD `test670` (*n* = 670) | 3 | 6.97 | — | — |
| MSVD `test670` (*n* = 670) | 5 | 7.35 | — | — |
| MSVD `test670` (*n* = 670) | 7 | 7.61 | — | — |

**Evaluation tokenization reconciliation:** All main-text absolute means and paired effects use official Stanford PTB tokenization (`stanford-corenlp-3.4.1.jar`, SHA-256 `2fcb91bb7a111f93d71e264f4ee0e3afd19ba0dde6d21b38605088df9e940399`) with the frozen COCO CIDEr scorer. Earlier exploratory project summaries invoked `Cider.compute_score` directly on whitespace-tokenized captions, producing slight numerical differences (e.g., Oracle7 − Oracle3 of −2.78 whitespace vs. −2.73 PTB) with identical substantive conclusions.
