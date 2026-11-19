# Adaptive context budgeting — oracle headroom, OOF gates, freeze decision

Date: 2026-11-20. Selection split: official val497 (FAITHFUL_V2, FULL corpus).
No test data was read in this analysis.

## 1. Oracle headroom (Gate G1–G3)

Per-query CIDEr over k ∈ {1,3,5,7} (fixed D ordering):

- k* distribution: k=1 336 (67.6%), k=3 80 (16.1%), k=5 43 (8.7%), k=7 38 (7.6%)
- 195/497 queries have identical output at all k; 302 are context-sensitive
- fixed best: D3 = 0.5312
- oracle adaptive: 0.5984
- **headroom = +6.71 CIDEr, 95% CI [+5.17, +8.47]**

Gate G3 (headroom ≥ +3.0) → adaptive budgeting is worth pursuing.

## 2. Predictability (diagnostic)

Best single features vs Δ(D3→D7): |Spearman| ≤ 0.06; AUC for "k=7 harm" ≤ 0.66
(`ground_coverage`, `repeat_sources7`). No inference-time feature reliably
predicts context harm, which caps what any simple selector can recover from the
oracle headroom.

## 3. 5-fold OOF (deterministic folds by video id)

| rule | OOF CIDEr | Δ vs D3 | 95% CI | mean k | folds > 0 |
|---|---|---|---|---|---|
| similarity gap | **54.15** | **+1.03** | [−0.30, +2.52] | 2.30 | 5/5 |
| redundancy stop | 51.93 | −1.19 | [−3.36, +0.92] | 1.32 | 0/5 |
| quality-tail stop | 52.25 | −0.87 | [−2.85, +1.00] | 2.50 | 2/5 |
| saturation score | 51.99 | −1.13 | [−3.30, +0.86] | 1.62 | 1/5 |

Six metrics for the OOF gap rule: B1 79.88, B4 39.14, M 28.58, R-L 59.80,
CIDEr 54.21 (dataset-level), SPICE 7.80; mean k 2.30 (evidence −67% vs k=7).

## 4. Paired statistics of the frozen OOF selector

| Comparison | Δ CIDEr | 95% CI | p | W/L/T |
|---|---|---|---|---|
| gap − D3 | +1.03 | [−0.30, +2.52] | 0.163 | 63/60/374 |
| gap − D1 | +2.41 | [+0.57, +4.49] | 0.016 | 55/42/400 |
| gap − D5 | +3.68 | [+1.71, +5.81] | 0.0002 | 103/73/321 |
| gap − D7 | +4.14 | [+1.74, +6.73] | 0.0008 | 123/97/277 |

## 5. Gate decision

- Strong gate (Δ ≥ +1.0 and CI excluding 0): **fails** (CI includes 0).
- Moderate gate (Δ ≥ +0.5 and ≥25% context-cost reduction, same direction in
  ≥4/5 folds): **passes** (+1.03; mean k 2.30 vs fixed 3/7; evidence −67%;
  5/5 folds positive).
- All purpose-built signals (redundancy, quality tail, saturation) **fail**.

**Frozen final method: similarity-gap adaptive context budget.**

Rule (no fitted parameters, inference-time only):
1. Score top-7 candidates with X-CLIP; take similarities s1..s7.
2. Compute adjacent gaps g_i = s_i − s_{i+1}, i = 1..6.
3. Budget = position of the largest gap mapped to {1,3,5,7}:
   i ≤ 1 → 1; i ∈ {2,3} → 3; i ∈ {4,5} → 5; i = 6 → 7.
4. Candidate order is the fixed ROUGE-L D ranking; only k changes.

`frozen_method_config.json` pins the rule, preprocessing and artifacts.

Caveat recorded for the paper: the selector is a generic baseline, not a novel
module; its val gain over D3 is not individually significant. The confirmatory
untouched-test comparison (frozen rule, no further tuning) decides whether it is
reported as the method or as a secondary finding next to fixed D3.
