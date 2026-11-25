"""Isolated local open-weight text judge worker (runs inside `uarcap_q2_evaluator` conda env).

Processes a batch of {request_id, system_prompt, user_prompt} JSONL rows in a single model load,
writes {request_id, response} JSONL, and cleanly releases GPU memory on exit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

EXP_ROOT = Path(__file__).resolve().parents[2]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from uarcap_q2.common.gpu_runtime import release_gpu_memory
from uarcap_q2.common.io_utils import atomic_write_jsonl, read_json, read_jsonl
from uarcap_q2.evaluators.backends import JudgeBackendConfig, LocalTransformersJudgeBackend, LocalVLLMJudgeBackend


def run_judge_batch_job(config_json: Path | str, requests_jsonl: Path | str, responses_jsonl: Path | str) -> list[dict[str, Any]]:
    cfg = JudgeBackendConfig(**read_json(config_json))
    reqs = read_jsonl(requests_jsonl)
    out_rows: list[dict[str, Any]] = []
    try:
        if cfg.backend_type == "vllm_offline":
            backend = LocalVLLMJudgeBackend(cfg)
            responses = backend.complete_batch([(str(r["system_prompt"]), str(r["user_prompt"])) for r in reqs])
            for r, txt in zip(reqs, responses):
                out_rows.append({"request_id": r["request_id"], "response": txt})
        else:
            backend_tf = LocalTransformersJudgeBackend(cfg)
            for r in reqs:
                txt = backend_tf.complete(str(r["system_prompt"]), str(r["user_prompt"]))
                out_rows.append({"request_id": r["request_id"], "response": txt})
        atomic_write_jsonl(responses_jsonl, out_rows)
        return out_rows
    finally:
        release_gpu_memory()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Isolated local Transformers text judge worker")
    ap.add_argument("--config-json", type=Path, required=True)
    ap.add_argument("--requests-jsonl", type=Path, required=True)
    ap.add_argument("--responses-jsonl", type=Path, required=True)
    args = ap.parse_args(argv)
    run_judge_batch_job(args.config_json, args.requests_jsonl, args.responses_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
