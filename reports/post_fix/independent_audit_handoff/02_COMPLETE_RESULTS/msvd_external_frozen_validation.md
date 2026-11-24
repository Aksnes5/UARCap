# MSVD test670 — frozen cross-dataset validation (external VideoUFO)

Date: 2026-11-20. Method frozen on MSR-VTT val497; no MSVD tuning.
Data: `reports/msvd_existing_data_audit.md`; contamination:
`reports/msvd_videoufo_contamination_audit.md` (0 source overlap).

## 1. Dataset / pipeline validation

- 670/670 captions, 670 unique ids, all `.avi` present.
- FAITHFUL_V2 query features: 670×512 + 670×196×512, finite, id sets equal to
  retrieval keys; recompute of 3 videos bit-identical (max abs diff **0.0**).
- Video health: 670/670 readable, frozen frame plan
  (`reports/msvd_test670_frame_plan.json`).
- Candidate features: 6,425 clips in D-grid union, 4,617 newly extracted,
  0 missing. Checkpoint `train_evcap_top3_..._msvd.pt`, same prompt/beam/seed.

## 2. Retrieval-side coverage (D ordering)

| k | oracle CIDEr | unique sources | near-dup ratio |
|---|---|---|---|
| 1 | 0.1699 | 1.0 | 0.000 |
| 3 | 0.2217 | 3.0 | 0.364 |
| 5 | 0.2499 | 5.0 | 0.239 |
| 7 | 0.2780 | 7.0 | 0.169 |

## 3. Generation grid (×100)

| | B1 | B4 | MET | R-L | CIDEr | SPICE |
|---|---|---|---|---|---|---|
| D1 | 91.62 | 69.75 | 46.73 | 83.85 | **156.39** | 9.08 |
| D3 | 90.54 | 67.43 | 46.22 | 82.99 | 151.45 | 9.15 |
| D5 | 88.79 | 64.24 | 44.98 | 81.43 | 144.39 | 9.16 |
| D7 | 87.69 | 62.03 | 44.38 | 80.64 | 141.59 | 9.25 |
| Adaptive (frozen) | 90.61 | 67.67 | 46.21 | 82.93 | 151.94 | 9.13 |
| In-domain reference (original) | 91.5 | 69.7 | 47.0 | 84.0 | 158.32 | 8.8 |

## 4. Paired statistics

| comparison | Δ CIDEr | 95% CI | p | W/L/T |
|---|---|---|---|---|
| D3 − D7 | +9.86 | [+6.03, +13.94] | 0.0001 | 179/92/399 |
| D3 − D5 | +7.06 | [+3.43, +10.98] | 0.0002 | 121/76/473 |
| D1 − D3 | +4.94 | [+1.65, +8.36] | 0.0026 | 131/80/459 |
| Adaptive − D3 | +0.49 | [−2.39, +3.30] | 0.739 | 79/64/527 |
| Adaptive − D7 | +10.35 | [+6.37, +14.71] | 0.0001 | 193/100/377 |

Adaptive budget: mean k **2.42** (322/237/94/17), 65.4% fewer contexts than k=7.

## 5. Generalization verdict

`CONTEXT_OVERLOAD_GENERALIZES; ADAPTIVE_QUALITY_GAIN_NOT_CONFIRMED`

- Coverage rises with k (oracle 0.170 → 0.278) while generation falls
  monotonically (156.39 → 141.59); overload is stronger on MSVD than MSR-VTT.
- The frozen similarity-gap selector transfers and lands on D3-equivalent
  quality (151.94 vs 151.45, n.s.), far above D7, but does **not** reach the best
  fixed budget D1 (156.39).
- External D1 is within 1.9 CIDEr of the in-domain reference (156.39 vs 158.32),
  the closest external/in-domain gap observed so far.
- No method choice was tuned on MSVD; all thresholds/rules are the val497 frozen
  versions.
