# Returning laptop-developed code to the BN server

Anti-Gravity develops on Windows with no GPU. BN executes real inference. Follow this
order; never edit a running task's file in place.

## 1. Laptop-side development contract

- Branch from the handoff branch:
  `antigravity/uarcap-q2-bn-to-laptop-handoff-20261126`.
- Keep the CPU test suite green:
  ```bash
  python -m pytest -q
  ```
  The suite must pass without GPU, videos, weights or network.
- Add/extend tests for every new pure function. Use `handoff_20261126/samples/` as fixtures.
- Do not add model weights, videos, feature caches, `.env`, credentials or blind keys.
- Do not invent human labels or inclusion probabilities.
- Preserve these invariants:
  - A0 has strictly empty external evidence.
  - A1 ⊂ A3 ⊂ A5 ⊂ A7 prefix consistency on the same reranked list.
  - E3 arms share identical visual input within a candidate.
  - parse failures are `None`, never coerced to 0.
  - gates read real receipts; missing receipts mean false.
- Open a PR to the handoff branch (or a new `antigravity/...` branch). Do not force-push
  over BN's branch.

## 2. What to run on BN after `git fetch`

```bash
cd /home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments
git fetch origin
git switch <laptop-branch>
git pull --ff-only
/home/bn/anaconda3/envs/videollama2/bin/python -m pytest -q
```

If tests fail, do not deploy GPU jobs.

Then run a CPU smoke:

```bash
/home/bn/anaconda3/envs/videollama2/bin/python scripts/run_final_analysis.py --describe-only
/home/bn/anaconda3/envs/videollama2/bin/python scripts/run_supplementary.py --e3-bridge-candidates
```

Do not regenerate existing formal predictions.

## 3. Deploying a new GPU task

1. Put the script in
   `/home/bn/uarcap_q2_runtime/overnight/20261126/tasks/<name>.py`.
2. Make it resumable: append JSONL, skip existing `(query_id, arm)` or equivalent keys,
   write atomic receipts, honour the shared GPU lock.
3. Add one task object to `task_queue.json` (atomic write; preserve existing tasks):
   ```json
   {"id":"supp_budget_a5_videollama","priority":30,
    "cwd":"/home/bn/uarcap_q2_runtime/overnight/20261126",
    "cmd":"OVERNIGHT_BASE=... OVERNIGHT_TASK_ID=... PYTHONPATH=<repo> <python> -u tasks/<name>.py ...",
    "timeout_sec":14400,"max_retries":1,"requires_gpu":true,"depends_on":[]}
   ```
4. The running orchestrator reloads the queue. Do not start another orchestrator.
5. Verify with `progress/<id>.json`, `logs/<id>.log`, and `task_state.json`.
6. Use `touch STOP_REQUESTED` to stop after the current task; remove it before resuming.

## 4. Required checks before accepting GPU output

- Prediction count equals planned count; failures accounted for.
- Every row has a trace with `visual_input_sha256`, `frame_indices`, `text_evidence`,
  `prompt_rendered_sha256`.
- A0 traces have empty text/visual evidence.
- A1/A3/A5/A7 are exact prefixes.
- Fixed checkpoint/model revision and decoding config.
- E3 four arms share the same visual hash per candidate.
- Parse failures are `None` and excluded.
- Receipts mark `UNVALIDATED`/`MACHINE_ASSISTED`; no human gate is implied.

## 5. Evaluation and analysis

- Gemma evaluation for new predictions:
  ```bash
  <vllm-python> tasks/eval_captions_gemma.py --predictions <pred.jsonl> \
    --core-lock <core_lock.json> --out-dir <eval_dir>
  ```
- Llama cross-judge task must consume the same eval dir and core lock.
- Run `scripts/run_final_analysis.py --machine-assisted` for descriptive stats.
- Human-calibrated mode requires the signed human import; it will refuse otherwise.

## 6. Artifact return policy

Commit only:
- code and tests,
- small JSON/JSONL/MD/CSV receipts, predictions, traces, eval rows,
- figures/tables.

Do not commit: videos, frames, features, weights, blind keys/tasks, incomplete human
answers, raw API logs with secrets.

Update `COMPLETED_EXPERIMENTS_INDEX.json` and `CURRENT_PROJECT_STATUS.md` when results land.

## 7. Rollback

- Code: `git revert <sha>` or switch back to the handoff branch commit.
- Queue: restore the previous `task_queue.json` (state is separate and resumable).
- Never delete completed output directories; write a new condition directory instead.
- If a task holds the GPU lock and is unhealthy, inspect the log first; only then stop the
  task process, never PID 1705522 or unrelated processes.
