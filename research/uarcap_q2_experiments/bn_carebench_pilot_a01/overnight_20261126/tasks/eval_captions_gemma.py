#!/usr/bin/env python3
"""Machine-assisted fact evaluation of arbitrary generated captions with the existing
Gemma 3 27B FP8 judge. Produces A2-compatible eval rows + a core-event lock copy so the
Llama cross-judge task can consume the same directory. Uncalibrated; never human Gold."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

BASE = Path(os.environ.get("OVERNIGHT_BASE", "/home/bn/uarcap_q2_runtime/overnight/20261126"))
TASK_ID = os.environ.get("OVERNIGHT_TASK_ID", "eval_captions_gemma")
PROGRESS = BASE / "progress" / f"{TASK_ID}.json"
REPO = "/home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments"
sys.path.insert(0, REPO)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def atomic_write(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def write_jsonl(p: Path, rows: list[dict]) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def progress(phase: str, done: int, total: int, **extra) -> None:
    atomic_write(PROGRESS, {"task": TASK_ID, "phase": phase, "model": "Gemma-3-27B-FP8",
                            "completed": done, "total": total, "updated_utc": utcnow(), **extra})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--core-lock", type=Path, default=Path("/home/bn/uarcap_q2_runtime/outputs/pilot_a2/pilot_a2_carebench_eval_v1/a2_core_event_lock.json"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model-path", default="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/models_campaign/M04_gemma3_27b")
    ap.add_argument("--model-id", default="RedHatAI/gemma-3-27b-it-FP8-dynamic")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--dataset-label", default="CaReBench")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from uarcap_q2.common.gpu_lock import GPULockConfig, acquire_cooperative_gpu_lock
    from uarcap_q2.evaluators.backends import JudgeBackendConfig, LocalVLLMJudgeBackend
    from uarcap_q2.evaluators.batched_reference_fact_eval import evaluate_captions_batched

    preds = read_jsonl(args.predictions)
    lock = json.loads(args.core_lock.read_text(encoding="utf-8"))
    ref_by_qid = {}
    core_by_qid = {}
    for e in lock["entries"]:
        parts = []
        if e.get("reference_caption"):
            parts.append("Caption: " + e["reference_caption"])
        if e.get("spatial_caption"):
            parts.append("Spatial description: " + e["spatial_caption"])
        if e.get("temporal_caption"):
            parts.append("Temporal description: " + e["temporal_caption"])
        ref_by_qid[e["query_id"]] = "\n".join(parts)
        core_by_qid[e["query_id"]] = [e["core_event_candidate"]] if e.get("core_event_candidate") else []

    items = []
    for p in preds:
        qid = str(p["query_id"])
        if qid not in ref_by_qid:
            continue
        items.append({"query_id": qid, "arm": str(p["arm"]), "model_id": str(p["model_id"]),
                      "caption": str(p.get("caption", "")), "reference_description": ref_by_qid[qid],
                      "reference_fields": ["caption", "spatial_caption", "temporal_caption"],
                      "core_events": core_by_qid[qid]})
    progress("prepare", 0, len(items))

    backend = LocalVLLMJudgeBackend(JudgeBackendConfig(
        backend_type="vllm_offline", model_id=args.model_id, model_family="gemma",
        local_model_path=args.model_path, temperature=0.0, max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem, language_model_only=True,
    ))
    lock_cfg = GPULockConfig(lock_path=Path("/home/bn/videoufo_research/gpu.lock"), run_id=TASK_ID,
                             gpu_uuid="GPU-66d83fcc-2695-33a7-33aa-af30b7e3d26d",
                             require_nvidia_smi_idle=True, sidecar_dir=args.out_dir)
    t0 = time.time()
    with acquire_cooperative_gpu_lock(lock_cfg):
        res = evaluate_captions_batched(backend, items, max_retries=args.max_retries, chunk_size=args.chunk)
    elapsed = round(time.time() - t0, 1)
    write_jsonl(args.out_dir / "a2_ref_fact_eval.jsonl", res["rows"])
    with (args.out_dir / "a2_judge_raw_calls.jsonl").open("w", encoding="utf-8") as f:
        for r in res["raw_calls"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copy2(args.core_lock, args.out_dir / "a2_core_event_lock.json")
    receipt = {
        "schema": "uarcap_q2.machine_caption_eval.v1", "status": "MACHINE_ASSISTED_UNCALIBRATED",
        "timestamp_utc": utcnow(), "judge_model_id": args.model_id, "judge_model_path": args.model_path,
        "predictions_path": str(args.predictions), "n_items": len(items), "n_rows": len(res["rows"]),
        "n_unique_captions": res["unique_captions"], "n_duplicate_reuses": res["n_duplicate_reuses"],
        "n_retries": res["n_retries"], "n_parse_failures": res["parse_failures"],
        "n_raw_calls": len(res["raw_calls"]), "elapsed_sec": elapsed,
        "core_event_lock_sha256": lock.get("lock_sha256"),
    }
    atomic_write(args.out_dir / "a2_judge_run_receipt.json", receipt)
    progress("completed", len(res["rows"]), len(items), parse_failures=res["parse_failures"])
    print(json.dumps({k: receipt[k] for k in ("status", "n_rows", "n_unique_captions", "n_parse_failures", "elapsed_sec")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
