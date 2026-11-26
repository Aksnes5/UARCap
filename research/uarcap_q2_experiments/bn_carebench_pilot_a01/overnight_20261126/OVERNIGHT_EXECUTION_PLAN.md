# Overnight UARCap Q2 execution plan (2026-11-26)

Persistent, Claude-independent queue launched 2026-11-26 02:09 local from
`/home/bn/uarcap_q2_runtime/overnight/20261126/`.

## Components

| File | Role |
| --- | --- |
| `run_overnight.py` / `run_overnight.sh` | main queue: hot-reload `task_queue.json`, atomic `task_state.json`, per-task logs/progress, STOP_REQUESTED |
| `overnight_supervisor.py` / `run_supervisor.sh` | independent 60 s heartbeat + 30 min checks; restarts a dead orchestrator; terminates only our own task after 60 min zero progress + idle GPU |
| `watch_download_and_queue.py` | appends the 9-stage VidHalluc E3 chain when the download completes |
| `report_writer.py` | every 30 min writes `OVERNIGHT_UARCAP_Q2_EXECUTION_REPORT.md` + `USER_REVIEW_TODO.md` |
| `tasks/` | all stage scripts (A2 cross/visual, A3 Qwen, Gemma eval, VidHalluc E3, formal manifest/retrieval/features/frames/generation/audit) |
| `task_queue.json` / `vidhalluc_task_template.json` | deployed queue and automatic E3 template |

## Queue order

1. A2 Llama cross-judge (119 unique claims) → A2 Qwen visual verification.
2. A3 CaReBench: Qwen3.8 90 generations → Gemma eval → Llama cross → Qwen visual; Qwen3.5 backup.
3. VidHalluc (auto after download): SHA/extract → VideoUFO/source filter + freeze → text screen → 16-frame visual screen → approve → Video-LLaMA + Qwen Base/Sup/Con/Neu → dual-target eval → report.
4. Formal CaReBench candidates (600 pre-frozen, 60 validation): X-CLIP retrieval → faithful_v2 features → 16-frame decode → Video-LLaMA 1800 → Qwen 1800 → Gemma eval → Llama cross.
5. CPU: formal integrity audit, descriptive judge-bias analysis, minimal human-review list, morning report.

## Recovery

- `python3 run_overnight.py --base <base>` resumes from `task_state.json`; completed tasks are not repeated.
- `touch STOP_REQUESTED` stops after the current task.
- Tasks acquire `/home/bn/videoufo_research/gpu.lock`; foreign GPU processes are never killed.
- Model incompatibility fallback: Qwen3.8 → Qwen3.5; Mistral FP8 → Gemma 3 27B (documented in Pilot A2 results).

## Scientific gates

A2 remains `PILOT_A2_HUMAN_VALIDATION_PENDING`; A3/E3 are exploratory; formal outputs are
`FORMAL_CANDIDATE_GENERATION_UNVALIDATED`; no human signatures and no confirmatory statistics.
