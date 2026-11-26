# OVERNIGHT UARCap Q2 Execution Report

- Generated: 2026-11-26T06:24:32.101232+00:00
- Queue started: 2026-11-25T18:09:21.512942+00:00; elapsed: **12.25 h**
- Orchestrator: True (pid None); current task: `None`
- Task counts: {'RUNNING': 1, 'COMPLETED': 38, 'BLOCKED': 12, 'PENDING': 9}
- GPU: {'utilization_pct': 0, 'memory_used_mib': 2, 'memory_total_mib': 49140}
- Supervisor checks: 25; recovery actions: 1; pending repairs: 11

## Network / download
- VidHalluc download: status=COMPLETED bytes=16754119398 percent=100.0 speed_kBps=102.4
- Download errors: Range download failed [4160749568-4294967295] after 5 attempts: <urlopen error _ssl.c:983: The handshake operation timed out>

## A2 machine-assisted calibration
- Cross-judge: 119/119 unique claims; agreement=0.8571; kappa=0.6713; disagreements=17
- Visual verification: verified=41 of 41; video_truth={'contradicted': 11, 'entailed': 28, 'unverifiable': 2}; still uncertain=2
- Human review items: 17

## A3 CaReBench cross-model (exploratory)
- Qwen3.8 generation: 90 predictions, status=EXPLORATORY_COMPLETED_UNCALIBRATED, peak=42364 MiB
- Gemma eval: rows=90, parse_failures=0
- Llama cross-judge: evaluated=260, agreement=0.8038, kappa=0.6038

## Formal candidate generation (UNVALIDATED)
- Frozen manifest: 600 candidates (60 validation); hash=ebc02155e73dbce4c848a9e96b5b9a57d40c85058d3998de1368660c2e96a3b3
- Video-LLaMA: 1800 predictions; per_arm={'ARM_A0': 600, 'ARM_A1': 600, 'ARM_A7': 600}; failures=0
- Qwen3.8: 1800 predictions; per_arm={'ARM_A0': 600, 'ARM_A1': 600, 'ARM_A7': 600}; failures=0
- Gemma eval old/new: 1800 / 1800

## E3 exploratory (VidHalluc)
- Approved candidates: 20 (machine-assisted); review candidates: None
- Cross-judge: pairs=None; true_agree=None; conflict_agree=None

## Integrity / gates
- A2 remains `PILOT_A2_HUMAN_VALIDATION_PENDING`; no human signatures or agreement were fabricated.
- A3/E3/formal outputs are exploratory or `FORMAL_CANDIDATE_GENERATION_UNVALIDATED`; no confirmatory statistics.

## Recovery
```bash
cd /home/bn/uarcap_q2_runtime/overnight/20261126 && bash run_overnight.sh   # resumes from task_state.json
touch STOP_REQUESTED                 # graceful stop after current task
```

## Artifacts
- Overnight base: `/home/bn/uarcap_q2_runtime/overnight/20261126`
- Task state/logs: `/home/bn/uarcap_q2_runtime/overnight/20261126/task_state.json`, `/home/bn/uarcap_q2_runtime/overnight/20261126/logs/`
- A2 machine: `/home/bn/uarcap_q2_runtime/outputs/pilot_a2_machine*`
- A3 exploratory: `/home/bn/uarcap_q2_runtime/outputs/pilot_a3_exploratory`, `/home/bn/uarcap_q2_runtime/outputs/pilot_a3_machine*`
- VidHalluc/E3: `/home/bn/uarcap_q2_runtime/outputs/pilot_e3_exploratory`
- Formal candidate: `/home/bn/uarcap_q2_runtime/outputs/formal_candidate_unvalidated`
- Branch/commit: `antigravity/uarcap-q2-bn-carebench-pilot-a01` (latest commit at session end)
