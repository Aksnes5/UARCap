# BN runtime interface (what the laptop must target)

Everything below is the actual interface observed on the server at handoff. The laptop
has no GPU, videos, weights or large features; use the sample files in
`handoff_20261126/samples/` and the CPU tests.

## 1. Repository

- Path on BN: `/home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments`
- Remote: `ssh://git@ssh.github.com:443/Aksnes5/UARCap.git`
- Handoff branch: `antigravity/uarcap-q2-bn-to-laptop-handoff-20261126`
- Python package: `uarcap_q2/`
- Entrypoints: `scripts/`, `tests/`
- CPU test command:
  ```bash
  cd /home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments
  /home/bn/anaconda3/envs/videollama2/bin/python -m pytest -q
  ```
  Current result: 73 passed.

## 2. Python environments (server only)

| Purpose | Path | Notes |
| --- | --- | --- |
| CPU analysis/tests | `/home/bn/anaconda3/envs/videollama2/bin/python` | Python 3.10.8; has av, cv2, decord; used for the 73 tests |
| Shared vLLM 0.29.0 | `/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/.venv_rag_vllm/bin/python` | Qwen/Gemma/Llama vLLM; cv2 5.0, no decord |
| Frozen EVCap Video-LLaMA | `/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python` | Do not upgrade Transformers 4.28 |
| Orchestrator | `/usr/bin/python3` | Runs `run_overnight.py` |

The `configs/bn_paths.local.yaml` on BN is intentionally gitignored (it may contain local
paths). The committed example is `configs/bn_paths.example.yaml`.

## 3. Models and frozen assets

| Asset | Path / ID |
| --- | --- |
| Frozen Video-LLaMA EVCap checkpoint | `/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/EVCapCode/outputs/train_evcap_msrvtt_quality_text/train_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review.pt` |
| Checkpoint SHA-256 | `a8b5af79d55a2c7e684771475d645be6373208bdaae5a4211bee556581eb8002` |
| Vicuna 7B | `/media/.../repro_evcap/models/vicuna-7b-v1.3` |
| Video-LLaMA feature cache | `/home/bn/videoufo_research/features/videollama_faithful_v2` |
| Qwen3.8 FP8 (primary) | `/media/.../Jnh_rag/models/Qwen3.8-27B-FP8`, ID `Qwen/Qwen3.8-27B-FP8` |
| Qwen3.5 27B (backup) | `/media/.../Jnh_rag/models/Qwen3.5-27B`, ID `Qwen/Qwen3.5-27B` |
| Gemma 3 27B judge | `/media/.../Jnh_rag/models_campaign/M04_gemma3_27b`, ID `RedHatAI/gemma-3-27b-it-FP8-dynamic` |
| Llama 3.3 70B W4A16 judge | `/media/.../Jnh_rag/models_campaign/M08_llama33_70b`, ID `meta-llama/Llama-3.3-70B-Instruct-W4A16` |
| X-CLIP repo / ckpt | `/media/.../EvcapProject/X-CLIP`, `.../modelpth/k600_16_16.pth` |
| VideoUFO root | `/home/bn/datasets/VideoUFO`; research `/home/bn/videoufo_research` |

The config lists Mistral-Small as the nominal primary judge, but Mistral FP8 was blocked by
a vLLM 0.29/transformers 5.17 Pixtral incompatibility. Actual judges are Gemma and Llama.
Do not “fix” this by claiming Mistral ran.

## 4. Data and frame locations

| Data | Path |
| --- | --- |
| CaReBench videos | `/home/bn/uarcap_q2_runtime/data/carebench/videos` (e.g. `v_00001358_22.mp4`) |
| Formal 16-frame indices | `/home/bn/uarcap_q2_runtime/outputs/formal_candidate_unvalidated/formal_frame_indices.jsonl` |
| Formal frames | `.../formal_candidate_unvalidated/formal_frames/<query_id>/frame_00.jpg ... frame_15.jpg` |
| VidHalluc videos | `/home/bn/uarcap_q2_runtime/data/vidhalluc/videos/ACH/<source_id>_clip_<n>.mp4` |
| E3 formal frames | `/home/bn/uarcap_q2_runtime/outputs/pilot_e3_exploratory/E3_FORMAL_FRAMES/<query_id>/frame_*.jpg` |
| VidHalluc zip | `/home/bn/uarcap_q2_runtime/data/vidhalluc/data/ACH_videos.zip` (16,754,119,398 bytes; SHA-256 `8f16...af5`) |

Never upload videos, frames, weights, feature caches, or the blind key.

## 5. GPU lock

- Path: `/home/bn/videoufo_research/gpu.lock`
- API: `uarcap_q2.common.gpu_lock.GPULockConfig` +
  `acquire_cooperative_gpu_lock`.
- GPU UUID: `GPU-66d83fcc-2695-33a7-33aa-af30b7e3d26d`
- Rules: one task at a time; lock is re-entrant in-process but excludes other processes;
  never kill a foreign PID (notably PID 1705522); never start a second scheduler.

## 6. Task queue / scheduler interface

- Base: `/home/bn/uarcap_q2_runtime/overnight/20261126/`
- Queue: `task_queue.json`
  ```json
  {"schema":"uarcap_overnight_queue_v1","updated_utc":"...","tasks":[
    {"id":"...","priority":10,"cwd":"...","cmd":"...","timeout_sec":3600,
     "max_retries":1,"requires_gpu":true,"depends_on":[]}
  ]}
  ```
- State: `task_state.json` (`tasks.<id>.status`: PENDING/RUNNING/COMPLETED/BLOCKED/
  RETRYABLE_FAILURE/SKIPPED/AWAITING_HUMAN)
- Heartbeat: `heartbeat.json` every 60 s; progress per task:
  `progress/<task_id>.json`; logs: `logs/<task_id>.log`
- STOP: create `STOP_REQUESTED`; current task finishes, rest are
  `SKIPPED_STOPPED`.
- The orchestrator reloads the queue every loop. Add tasks by editing the JSON atomically;
  never run two orchestrators.
- Task processes receive:
  `OVERNIGHT_BASE`, `OVERNIGHT_TASK_ID`, `PYTHONPATH=<repo>`,
  `VLLM_USE_FLASHINFER_SAMPLER=0`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.

## 7. Resume model

- Generation scripts are append-only JSONL and skip keys already present:
  - formal: `(query_id, arm)`
  - E3 formal: `(query_id, arm)`
  - formal E3 eval: `(query_id, arm, generator, judge)`
  - supplementary: `(query_id, arm)` per output dir
- Re-running the same script resumes missing rows; it does not overwrite valid rows.
- Do not delete partial output dirs unless intentionally restarting that condition.

## 8. Input/output formats (small real samples in `samples/`)

- Prediction row (formal/supp): `query_id, arm, model_id, caption, input_trace, ...`
- Input trace: `query_id, arm, stage, model_id, checkpoint_sha256, visual_input_sha256,
  frame_indices, text_evidence, visual_evidence_ids, prompt_rendered_sha256, ...`
- Eval row: `query_id, arm, model_id, caption, claims[] ({claim_id, claim, label}),
  n_claims, n_entail_ref, n_contra_ref, n_unment_ref, r_core_ref, parse_status`
- E3 eval row: `query_id, arm, generator, caption, y_true, y_conflict,
  true_detail, conflict_detail, judge, parse_ok`
- Human import schema: `uarcap_q2.human_primary_endpoints.v1` — see
  `samples/HUMAN_ENDPOINT_IMPORT_TEMPLATE.json` and
  `pipelines.verify_human_primary_endpoints`.
- E3-Bridge candidate/job/approved: see `samples/E3_BRIDGE_*_SAMPLE.json`.

## 9. Output roots on BN

- `/home/bn/uarcap_q2_runtime/outputs/formal_candidate_unvalidated/` (269 MB incl. 166 MB frames)
- `/home/bn/uarcap_q2_runtime/outputs/pilot_e3_exploratory/` (105 MB incl. frames/work)
- `/home/bn/uarcap_q2_runtime/outputs/supplementary/`
- `/home/bn/uarcap_q2_runtime/outputs/final_analysis/`
- `/home/bn/uarcap_q2_runtime/outputs/final_audit/`

Only curated subsets are committed under
`bn_carebench_pilot_a01/overnight_20261126_artifacts/`.

## 10. What the laptop cannot do

- Run vLLM/Video-LLaMA (no GPU, no weights).
- Decode CaReBench/VidHalluc videos (not present).
- Recompute X-CLIP features or full 600-video metrics.
- Open the blind key (never leaves the server).
