# UARCap Q2 overnight execution + remaining-code delivery

**Status: `ALL_REMAINING_RESEARCH_CODE_IMPLEMENTED`** (engineering modules implemented and
tested; several real GPU experiments remain queued/running — listed below).
**Queue start:** 2026-11-26 02:09 local · **Report time:** 2026-11-26 13:35 local
**Fixed code commit:** `9945f3a` (branch `antigravity/uarcap-q2-bn-carebench-pilot-a01`)

## 1. Runtime / compute

| Item | Value |
| --- | --- |
| Wall time | ~11.4 h (02:09 → 13:35) |
| Sum of GPU-task elapsed | ~7.5 h |
| Approx. GPU-busy fraction | ~66% of wall (rest = blocked/serial CPU+VLM setup gaps) |
| Network | VidHalluc 16.75 GB downloaded with multiple signed-URL/range retries; all local tasks unaffected |
| GPU lock | cooperative lock honoured; no foreign process killed |

## 2. Real results completed

### A2 / A3 machine-assisted
- A2 cross-judge: 119/119 unique claims; Gemma vs Llama agreement 85.7%, Cohen κ=0.671, 17 disagreements.
- A2 Qwen 16-frame visual verification: 41/41 flagged claims; 28 entailed / 11 contradicted / 2 unverifiable; 39/41 fully visible.
- A3 Qwen3.8: 90/90 generations; Gemma eval 90 rows (68 unique captions); Llama cross 260 claims, agreement 80.4%, κ=0.604; Qwen visual 120/120.
- A3 Qwen3.5 backup: 90/90 generations and evaluation completed.

### Formal CaReBench (600 pre-frozen candidates, 60 validation)
- Video-LLaMA: **1800/1800** A0/A1/A7 predictions, 0 failures, 6525 s.
- Qwen3.8: **1800/1800**, 0 failures, 1910 s.
- Gemma evaluation: 1800 rows old (1093 unique), 1800 rows Qwen (1360 unique), 0 parse failures.
- Llama cross: 2660 old claims (agreement 87.74%, κ=0.703); 5098/5099 Qwen claims (agreement 87.23%, κ=0.710; one claim pending retry).
- **Integrity audit 20/20**: counts, trace linkage, A0 zero evidence, A1⊂A7 prefix, 16-frame hash, constant checkpoint, retrieval top20/rerank7, features, decode, pilot/validation isolation.
- Isolation: 913 parents after dropping 87 same-parent siblings; 30 pilot parents excluded; 883 eligible; near-dup scan at Jaccard 0.92 ran (0 flagged); 600 selected with 60 independent validation IDs, disjoint main/validation, parent isolation verified. **Residual risk:** CaReBench parent IDs are anonymized; YouTube-level source independence cannot be proven.
- Descriptions (machine, uncalibrated): old model support rates A0/A1/A7 = .793/.826/.807, conflict .106/.089/.083, core .617/.672/.677; Qwen support .746/.759/.761, conflict .096/.094/.098, core .727/.753/.757. Llama-vs-Gemma differential is systematic (≈ −0.04 entailed, +0.04–0.06 contradicted) and stable across A0/A1/A7 — a judge bias, not a budget-dependent shift.

### VidHalluc E3 pilot
- 16.75 GB downloaded and SHA-256 verified; extracted; 3897 raw records → 2 VideoUFO overlaps, 1979 same-source duplicates, 331 unknown-source excluded; 1585 unique sources; 1555 after pilot exclusion.
- 30 pilot candidates; text screen 30/30; 16-frame visual screen 30/30; **20 approved**.
- Video-LLaMA **80/80** and Qwen3.8 **80/80** Base/Sup/Con/Neu generations.
- Dual-target evaluation: 320 rows / 160 pairs; Gemma–Llama agreement 91.25% (true target) and 98.75% (conflict target). Qwen expresses true facts more often than the frozen Video-LLaMA in this pilot; exploratory only, no significance claim.
- Formal E3: 180 candidates prepared, **159 machine-approved (≥150 gate)**; plan = 159 × 4 arms × 2 models = **1272 planned generations**; Video-LLaMA generation is running now; Qwen/eval queued.

## 3. Remaining code delivered (commit 9945f3a)

`uarcap_q2/final_analysis/`:
- `core.py` — per-arm descriptive stats, semantic fact transitions, parent-cluster bootstrap, paired permutation, Holm, effect sizes, missing-value sensitivity, Gemma/Llama confusion/kappa/budget differential, random agreed audit, two-stage human correction.
- `reporting.py` — PDF/SVG/PNG figures and CSV/JSON/MD tables; missing inputs marked `PENDING`.
- `pipelines.py` — formal E3 plan/gate, E3-Bridge candidate/approved builders, historical MSR-VTT/MSVD metric loader, 60-video blind validation package.
- `scripts/run_final_analysis.py` — unified modes: `--describe-only`, `--machine-assisted`, `--human-calibrated`, `--confirmatory`, `--integrity-check`, `--generate-figures`, `--export-paper-tables`, `--prepare-validation-package`, `--e3-plan`, `--e3-bridge`.
- Overnight tasks: `e3_formal_generate.py`, `e3_formal_evaluate.py`, `e3_formal_visual.py`, `e3_formal_text_screen.py`, `e3_bridge_prepare.py`, `vidhalluc_formal_pool.py`.
- Tests: **73/73 pass**, including confirmatory-gate refusal, E3 150-candidate refusal, statistics, figures, partial/missing inputs.

Commands:
```bash
cd /home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments
python scripts/run_final_analysis.py --machine-assisted --generate-figures --export-paper-tables
python scripts/run_final_analysis.py --integrity-check
python scripts/run_final_analysis.py --prepare-validation-package
python scripts/run_final_analysis.py --e3-plan
python scripts/run_final_analysis.py --confirmatory   # exits 3 until real human gates pass
```

## 4. Still pending (real data / human gates, not code)

- Formal Qwen cross-judge: 1 missing claim; retry task queued (`formal_cross_qwen_v2`).
- Formal E3: 1272 generations + dual-target evaluation queued/running (machine-assisted only).
- E3-Bridge: candidate/approved/generation/eval code deployed with a 20-candidate gate; not enabled until candidate count and scientific review permit.
- A2 remains `PILOT_A2_HUMAN_VALIDATION_PENDING`; A3/E3/formal outputs are exploratory/unvalidated. No human signatures/agreement/calibration were fabricated.
- Minimal human review: 117 items in `final_audit/USER_REVIEW_TODO_FINAL.json` (17 A2 visual, 50 formal definite disagreements, 20 random judge-agreed audit, 30 E3 candidates); 60-video blind validation package prepared (key withheld).

## 5. Artifacts

- Overnight runtime: `/home/bn/uarcap_q2_runtime/overnight/20261126/`
- Results: `outputs/formal_candidate_unvalidated/`, `outputs/pilot_a3_exploratory/`, `outputs/pilot_a3_machine_*`, `outputs/pilot_e3_exploratory/`, `outputs/pilot_a2_machine*`, `outputs/final_analysis/`
- Repo mirror: `bn_carebench_pilot_a01/overnight_20261126/` (scripts/queue/state) and `..._artifacts/` (predictions, traces, eval rows, summaries, figures, tables, audits)
- Excluded: raw videos/frames, weights/embeddings/features, blind keys, incomplete human answers, work TSVs, raw judge-call logs.
