# UARCap — final research status

Last updated: 2026-11-20 (Phase 4; untouched-test results pending).

## 1. Research-question evolution

1. Can a million-scale independent external corpus replace EVCap's small
   in-domain retrieval corpus for video captioning?
2. Scale study showed retrieval similarity and candidate coverage improve with
   corpus size while generation CIDEr does not → retrieval relevance ≠
   downstream utility.
3. Oracle diagnostics and paired analysis localized the problem to evidence
   **selection**, not candidate recall → context overload.
4. Val497 k-grid confirmed an inverted-U: coverage ↑ with k, generation utility
   peaks at k=3 (D) and declines at k=5/7.
5. Phase 4 asks: how should a frozen generator's evidence budget be decided?
   Oracle headroom is large (+6.7 CIDEr), simple predictability is weak;
   a parameter-free similarity-gap rule passes the pre-registered moderate gate
   and is frozen for confirmatory testing.

## 2. Dataset engineering

- VideoUFO FULL: 1,091,712 clips, 200 tar shards (804.77 GiB), 93 parquet
  metadata files (42.67 GiB), 345,437 YouTube sources, CC BY 4.0; verified
  complete (200/200 shards, 1:1 metadata↔mp4, decode checks).
- Nested frozen manifests 10K / 50K / 100K / 250K / 500K / FULL with
  source-hash ordering, round-robin topic allocation, strict nesting
  (`manifests/`, `manifest_stats.json`, `SHA256SUMS`).
- Caption dedup: 812,932 / 812,014 unique brief captions (full / CSV count as
  used by retrieval).
- X-CLIP caption embeddings: 1,091,712 × 512 fp16; FAISS flat + IVF4096.

## 3. Original-paper reproduction

- Checkpoint `train_evcap_top7_bs1_..._cross_fusion_gate_..._review.pt`,
  single-epoch training, no val-based selection (audit in
  `reports/audit_checkpoint_usage.md`).
- Reproduced paper Table 1 exactly: MSR-VTT 86.69 / 50.11 / 32.47 / 66.64 / 72.20
  and MSVD 91.53 / 69.67 / 47.04 / 84.04 / 158.32 (B@1/B@4/M/R/CIDEr).

## 4. Scale study (dev300, PRE_FIX features)

| corpus | 10K | 100K | 500K | FULL |
|---|---|---|---|---|
| CIDEr | 63.99 | 64.98 | 63.31 | 63.25 |

Retrieval similarity rises monotonically with scale; generation does not.
B in-domain 66.51, A no-retrieval 64.18, CF7 64.38, D7 63.25.

## 5. Oracle diagnostics

oracle@20 ≈ 0.219 (FULL) vs actual top-1 ≈ 0.076: useful captions are usually
already in the top-20, so recall is not the bottleneck; ranking/selection is.
Source redundancy rises with scale (same-source ratio 0 → 0.42).

## 6. Failure taxonomy

`reports/taxonomy.md`, `reports/D_vs_B_cases.md`, 50-case analysis: action
mismatch, object/relation mismatch, genericity, redundancy, source duplication,
metric under-reward.

## 7. UAR negative result (kept honestly)

First-version Utility-Aware Reranking (relevance + action + grounding +
diversity): PRE_FIX H3 67.63 looked strong, but after faithful preprocessing and
same-k baselines: CF3v2 65.57, D3v2 67.09, H3v2 66.32; H3−D3 = −0.77
(CI [−2.7, +1.1], p=0.43). **Independent UAR ranking contribution not
established**; archived as exploratory.

## 8. FAITHFUL_V2 correction

- Query X-CLIP resize INTER_LINEAR → INTER_NEAREST.
- Fusion features center-of-bin → `linspace(0,N-1,16)` endpoints.
- Pre/post retrieval: top1 changed 50.7%, top20 overlap 62.2%, quality metrics
  flat (±0.005) → fix is fidelity-neutral.
- All pre-fix results labelled `PRE_FIX_DIAGNOSTIC`.

## 9. val497 context-budget result

| k | CF CIDEr | D CIDEr | CF oracle | D oracle |
|---|---|---|---|---|
| 1 | **53.20** | 51.77 | 0.0800 | 0.1176 |
| 3 | 52.42 | **53.14** | 0.1314 | 0.1481 |
| 5 | 51.29 | 50.51 | 0.1535 | 0.1720 |
| 7 | 51.13 | 50.05 | 0.1698 | 0.1835 |

D3−D7 = +3.11 (CI [+0.72,+5.70], p=0.012); D3−D5 = +2.65 (p=0.012);
CF1−CF7 = +2.11 (p=0.098). Context overload replicates on the independent split.
55–60% of outputs are k-insensitive; on changed queries D3−D7 = +7.64.

## 10. Adaptive oracle headroom

k* distribution 336/80/43/38 for k=1/3/5/7; oracle adaptive 59.84 vs fixed D3
53.12 → **+6.71 CIDEr, CI [+5.17,+8.47]** (Gate G3).

## 11. Adaptive OOF / frozen decision

5-fold OOF (deterministic folds by video id):

| rule | Δ vs D3 | 95% CI | mean k | folds > 0 |
|---|---|---|---|---|
| similarity gap | +1.03 | [−0.30,+2.52] | 2.30 | 5/5 |
| redundancy stop | −1.19 | — | 1.32 | 0/5 |
| quality tail | −0.87 | — | 2.50 | 2/5 |
| saturation | −1.13 | — | 1.62 | 1/5 |

Strong gate not met; moderate gate met (gain ≥0.5, −67% evidence, 5/5 folds).
**Frozen method: similarity-gap adaptive budget** (`frozen_method_config.json`),
documented as a generic baseline with a non-significant +1.03 edge over D3 and
significant edges over D1/D5/D7.

## 12. Checkpoint / contamination audits

- No val-based checkpoint selection; no training leakage (audit report).
- 0 YouTube-source overlap between all 10,000 MSR-VTT videos and 345,437
  VideoUFO sources; 0 same-source+time-overlap among retrieved val candidates.
- Caption exact overlap 0.08% (generic templates); dHash flags explained by
  graphics/static frames. No severe contamination.

## 13. Untouched test2690

**Completed.** B 73.09 > D1 71.21 > Adaptive 70.67 > D3 70.08 > D5 69.48 > D7 68.27
CIDEr. D3−D7 = +1.81 (CI [0.87,2.75], p=0.0002); D1−D3 = +1.14 (p=0.005);
Adaptive−D3 = +0.58 (p=0.046); Adaptive−D1 = −0.56 (n.s.);
D3−B = −3.02 (p<1e-4). Context overload confirmed; adaptive transfers but
fixed D1 is stronger and in-domain B remains ahead.
Full report: `reports/msrtt_test2690_confirmatory.md`.

## 14. Full test2990 standard reporting

D1 70.31 > Adaptive 69.76 > D3 69.41 > D5 68.78 > D7 67.46 CIDEr;
B in-domain 72.22. Pilot300 flagged exploratory, excluded from confirmatory
claims; full ranking matches the untouched subset.

## 15. Statistical significance

Untouched2690 paired CIDEr: D3−D7 +1.81 (p=0.0002), D1−D3 +1.14 (p=0.005),
Adaptive−D3 +0.58 (p=0.046), Adaptive−D7 +2.40 (p=1e-4),
Adaptive−B −2.44, D3−B −3.02 (both p<1e-4). D3−D5 +0.60 (p=0.15) is not
significant. Details: `reports/test_confirmatory.json`.

## 16. Ablations and scale × budget interaction

Scale × budget grid on val497 (full details in `reports/scale_budget.md`):

| corpus | D1 | D3 | D5 | D7 | best k |
|---|---|---|---|---|---|
| 10K | 50.49 | 51.46 | **52.40** | 50.07 | 5 |
| 100K | **52.90** | 52.60 | 50.89 | 51.10 | 1 |
| 500K | 52.83 | **52.96** | 51.49 | 51.52 | 3 |
| FULL | 51.77 | **53.14** | 50.51 | 50.05 | 3 |

Coverage saturates above 100K (oracle@7 0.171 → 0.183 → 0.184) while the
optimal budget shrinks from 5 (10K) to 3 (500K/FULL). Supported claim: at
million scale the best fixed budget is 3; only the smallest corpus benefits
from k=5. The 100K k=1 optimum is within noise of k=3.

Signal ablations of the adaptive selector are run only if it beats D3 on
untouched test (pending).

## 17. Paper-ready claims

**SUPPORTED**
- Million-scale external retrieval of video captions is feasible with frozen
  EVCap; candidate coverage grows with corpus scale.
- Larger retrieval corpora do not monotonically improve downstream captioning.
- Candidate coverage and generation utility decouple with increasing k; the
  effect is significant for quality-reranked evidence (D3 > D5/D7, p=0.012).
- A parameter-free similarity-gap budget keeps most of D3's quality while using
  67% fewer contexts; it beats D1/D5/D7 significantly and D3 non-significantly.

**PARTIALLY SUPPORTED**
- Query-level adaptivity is worth pursuing: oracle headroom is large, but simple
  inference-time signals explain little of it.
- External retrieval can approach in-domain quality at k=3 on dev300, not yet
  confirmed on untouched test.

**NOT SUPPORTED**
- Utility-Aware Reranking (first version) independent gain.
- "More retrieval data always helps."
- Any claim that the adaptive selector is a novel complex method.

## 18. Cross-dataset generalization (MSVD test670)

External FULL VideoUFO, frozen method: D1 156.39 > Adaptive 151.94 ≈ D3 151.45 >
D5 144.39 > D7 141.59 CIDEr. Oracle rises 0.170→0.278 with k; generation falls
monotonically. D3−D7 = +9.86 (p=1e-4), D1−D3 = +4.94 (p=0.003),
Adaptive−D3 = +0.49 (n.s.), Adaptive−D1 ≈ −4.45.

Verdict: **CONTEXT_OVERLOAD_GENERALIZES; ADAPTIVE_QUALITY_GAIN_NOT_CONFIRMED**.
External D1 is within 1.9 CIDEr of the in-domain reference (158.32).
Report: `reports/msvd_external_frozen_validation.md`.

## Protocol statement

All method choices (k, rule, thresholds, feature definitions) were made on
val497 with 5-fold OOF and frozen before test. Untouched test2690 is evaluated
once. Pilot300 was used during exploration and is reported separately from
confirmatory results.
