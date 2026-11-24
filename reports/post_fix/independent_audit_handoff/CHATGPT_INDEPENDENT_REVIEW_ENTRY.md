# UARCap POST_FIX — independent review entry

## 1. Research question
Does increasing a frozen video caption generator's retrieved textual evidence
(from an external million-scale corpus) improve captions, and if not, why?
We study context budget k∈{1,3,5,7}, corpus scale 10K→1.09M, two orderings
(raw CF, ROUGE-L quality rerank D), and diagnostic controls.

## 2. Predecessor vs this work
Predecessor: Displays 93 (2026) 103420 — in-domain X-CLIP retrieval (k=20),
noun-phrase pseudo-label quality reranking, select k̃=7 (MSR-VTT)/3 (MSVD),
cross-gating fusion. It already contains a k̃ ablation (k̃=7 best in-domain).
This work adds: external VideoUFO corpus, six-scale budget study, untouched
test protocol, MSVD external validation, Oracle/2×2/position/tail/LOO
diagnostics, and an in-domain-vs-external budget-response comparison.
The reranker here is mean pairwise ROUGE-L over full captions, NOT the
predecessor's noun-phrase pseudo-label — this is an implementation deviation.

## 3. Data/model/retrieval/eval
- VideoUFO: 1,091,712 clips, 345,437 YouTube sources; frozen nested manifests
  10K/50K/100K/250K/500K/FULL.
- MSR-VTT: train 6,513, val 497, test 2,990 (pilot300 + untouched2690).
- MSVD: 1,200/100/670.
- Frozen EVCap checkpoint (MSR-VTT) sha256 a8b5af79…; MSVD checkpoint 93e3665b…;
  Vicuna-7B-v1.3, beam 5, seed 42, deterministic.
- Retrieval: X-CLIP ViT-B/16 video→caption, top-20; D = ROUGE-L rerank; CF = raw.
- Evaluation: pycocoevalcap COCOEvalCap; paired per-query bootstrap.

## 4. Important validity event
val497/val150 evaluation had a frame-loader bug: start,end=0,0 made PyAV keep
1–2 frames; 16-frame input was 15×frame0+1×frame1. Fixed at
`vision_language_tsv.py:338` (commit 2b3da9d) and all val experiments rerun
(POST_FIX). test2690/MSVD use pre-extracted frame TSVs and were unaffected.
Retrieval-side features were computed with decord over whole videos and are
unaffected (see AUDIT_01).

## 5. Completed experiments
48 main grid configs (6 scales × CF/D × k=1/3/5/7), 4 in-domain, Text×Visual
(T7V3/T3V7; T3V3=D3, T7V7=D7), Oracle1/3/7, position (reverse/top3-last),
tail/distractor (Mid/Low/Good/Hard/Easy), LOO 150×7. 71 formal configs verified,
plus one 20-query smoke (excluded).

## 6. POST_FIX key numbers (CIDEr points)
- k7 never optimal (0/12); k1 11/12; k7 penalty mean D 2.86 / CF 2.56.
- FULL CF1 71.17; 500K D1 70.59; scale vs k1 non-monotone (69.89–71.17 CF1).
- In-domain D1/3/5/7 = 71.93/72.42/71.07/73.76; external 69.97/68.63/67.31/67.22.
- Text×Visual: T3V3 68.63 / T7V3 67.18 / T3V7 68.51 / T7V7 67.22.
- Oracle: D1/D3/D7 69.97/68.63/67.22; Oracle1/3/7 71.64/71.67/68.94.

## 7. Significance summary
- Oracle3−D3 +3.04 (p=.0096; source-cluster CI [+0.87,+5.39]).
- Oracle7−Oracle3 −2.78 (p=.0076; source-cluster CI [−4.83,−0.84]).
- Text main −1.40 (p=.31); visual −0.04; interaction +0.16 (all ns).
- External D3−D7 +1.44 (p=.29); in-domain D3−D7 −1.33 (p=.24); DiD +2.77 (p=.13).
- Main grid: 72 paired tests, 0 survive BH FDR; k7-never-optimal is descriptive.

## 8. Mechanism disputes
Selection headroom is directly supported; "two independent bottlenecks" is only
partially identified. Oracle7's extra four candidates have lower compatibility
(head .138 vs tail .103), but per-query tail-quality decay does not correlate
with Oracle7 loss (ρ=.039, p=.38). Integration vs evidence-composition remains
unseparated.

## 9. Biggest novelty risk
The predecessor already ablated k̃ and found 7 best in-domain. This work must
lean on the in-domain/external reversal and the Oracle evidence, not on
"smaller k is better".

## 10. What the independent reviewer should decide
- Are the two Oracle conclusions correctly interpreted as selection + integration?
- Is novelty sufficient beyond the predecessor's k ablation?
- Are non-significant results over-interpreted anywhere?
- Is the no-leakage/statistics case adequate?
- Which 1–2 additional experiments are indispensable before submission?
- What is the strongest defensible contribution statement?

## 11. Read first
1. `05_RED_TEAM_REPORTS/AUDIT_02_ORACLE_CAUSAL_VALIDITY.md`
2. `05_RED_TEAM_REPORTS/AUDIT_03_STATISTICAL_REVALIDATION.md`
3. `05_RED_TEAM_REPORTS/AUDIT_04_NOVELTY_AND_PREDECESSOR.md`
4. `05_RED_TEAM_REPORTS/POST_FIX_FINAL_COMPREHENSIVE_RESEARCH_REPORT.md`
5. `02_COMPLETE_RESULTS/POST_FIX_SCALE_BUDGET_RESULTS.csv`
6. `03_SAMPLE_LEVEL_EVIDENCE/` (predictions, top-20 candidates, oracle caps)

## 12. Not included / unverified
- The Elsevier PDF is not redistributed; cite DOI 10.1016/j.displa.2026.103420.
- Original frame videos and model weights are excluded (size/licence).
- No matched k=0 baseline exists (`BLOCKED_REQUIRES_GENERATION`).
- VideoUFO caption provenance/human quality audit was not performed.
