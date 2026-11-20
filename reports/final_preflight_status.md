# Final preflight status — Phase 4 evening queue

Date: 2026-11-20 (D5 running). Legend: **READY** (done/verified),
**NEEDS_GPU** (prepared, waiting), **BLOCKED** (needs action/decision).

| item | status | evidence |
|---|---|---|
| MSVD data canonical / 1970 avi / splits 1200-100-670 | READY | `reports/msvd_existing_data_audit.md` |
| MSVD ↔ VideoUFO source leakage | READY (NONE) | `reports/msvd_videoufo_contamination_audit.md` (0 overlap, all splits) |
| MSVD caption overlap | READY (0.10–0.17% generic) | `reports/msvd_caption_overlap.json` |
| MSVD checkpoint/split audit | READY (no leakage found) | `reports/msvd_checkpoint_split_audit.md` |
| MSVD test670 video health | READY (670/670 ok) | `reports/msvd_test670_frame_plan.json` |
| MSVD analysis plan frozen | READY | `configs/msvd_confirmatory_analysis.json` |
| MSVD query features / retrieval / grid / generation | NEEDS_GPU | `logs/queue_state`, `scripts/msvd_chain.sh` |
| untouched2690 manifest | READY (2690, SHA pinned) | `manifests/msrvtt_untouched2690_stats.json` |
| adaptive routing rule frozen | READY | `frozen_method_config.json` |
| adaptive routing dry-run | READY (2,638 checked, 100% equal) | this report |
| test/full metrics pipeline | READY (await D5) | `scripts/test_confirmatory_chain.sh` |
| 50K/250K preflight | READY for CPU parts; retrieval NEEDS_GPU | `reports/scale_50k_250k_preflight.md` |
| 50K/250K subset row indexes | READY (counts/nesting verified) | `reports/scale_subset_index_preflight.json`, `indexes/scale_*_rows.npy` |
| CF cache audits (10K/100K/500K/FULL) | READY (100% hit) | `reports/cf_scale_cache_preflight.md` |
| GPU lock / no concurrent generation | READY (flock on all runners) | `scripts/run_*.sh` |
| process survival | READY (nohup, persistent logs, distinct sessions) | `ps` audit |
| disk / inodes | READY | 1.7 TB free, 241 M inodes free |
| experiment registry | READY (30 rows) | `reports/experiment_registry.csv` |
| D5 full2990 generation | NEEDS_GPU (running) | ~2,400/2,990 at audit time |

## Queue (auto, resume-safe)

D5 integrity → MSVD670 (frozen) → 50K D grid → 250K D grid → CF 10K/100K/500K
→ CF 50K/250K if GPU remains. Stage markers in `logs/queue_state/`.

## Safety notes

- No additional GPU process may start outside the queue: runners take
  `gpu.lock` (flock -w 7200).
- D5 is never restarted by the orchestrator; it only validates the finished file.
- No destructive disk operations; old outputs untouched.
- Network outage only delays `git push`; experiments continue locally.
