#!/usr/bin/env python3
"""Exploratory A3 CaReBench cross-model generation with Qwen3.8-27B-FP8 (16 real frames).

This is an EXPLORATORY cross-model pilot, NOT a gate-passing Pilot A3 and NOT Formal B.
It reuses the frozen 30-video sample, the original A1 retrieval evidence and the exact
16 frames used by the original model. A0/A1/A7 differ only in injected retrieval text.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

BASE = Path(os.environ.get("OVERNIGHT_BASE", "/home/bn/uarcap_q2_runtime/overnight/20261126"))
TASK_ID = os.environ.get("OVERNIGHT_TASK_ID", "a3_carebench_qwen")
PROGRESS = BASE / "progress" / f"{TASK_ID}.json"
REPO = "/home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments"
sys.path.insert(0, REPO)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def append_jsonl(p: Path, row: dict) -> None:
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_write(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def progress(phase: str, done: int, total: int, **extra) -> None:
    atomic_write(PROGRESS, {"task": TASK_ID, "phase": phase, "model": "Qwen3.8-27B-FP8",
                            "completed": done, "total": total, "updated_utc": utcnow(), **extra})


def gpu_sampler(stop: threading.Event, out: dict) -> None:
    while not stop.wait(2):
        try:
            v = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, timeout=5
            ).strip().splitlines()[0]
            out["peak_mib"] = max(out.get("peak_mib", 0), int(v))
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1-dir", type=Path, default=Path("/home/bn/uarcap_q2_runtime/outputs/pilot_a1/pilot_a1_carebench_v1"))
    ap.add_argument("--frames-dir", type=Path, default=Path("/home/bn/uarcap_q2_runtime/outputs/pilot_a2/pilot_a2_carebench_eval_v1/blind_annotation_package/frames"))
    ap.add_argument("--out-dir", type=Path, default=Path("/home/bn/uarcap_q2_runtime/outputs/pilot_a3_exploratory/carebench_qwen38"))
    ap.add_argument("--model-path", default="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/models/Qwen3.8-27B-FP8")
    ap.add_argument("--model-id", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--multimodal-key", choices=["video", "image"], default="video")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--run-id", default="a3_exploratory_qwen38_carebench_v1")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    import yaml
    from uarcap_q2.common.gpu_lock import GPULockConfig, acquire_cooperative_gpu_lock
    from uarcap_q2.common.schemas import PilotStage
    from uarcap_q2.datasets.carebench import load_carebench_records
    from uarcap_q2.generators.base import BatchItem
    from uarcap_q2.generators.modern_vlm_adapter import hash_pil_frames
    from uarcap_q2.generators.qwen3_vllm_adapter import Qwen3VLLMConfig, Qwen3VLLMMultimodalAdapter
    from uarcap_q2.retrieval.arm_builder import build_natural_retrieval_arm
    from uarcap_q2.retrieval.reranker import RetrievalCandidate

    paths = yaml.safe_load(Path(f"{REPO}/configs/bn_paths.local.yaml").read_text(encoding="utf-8"))
    data_root = Path(paths["q2_writable_workspace"]["data_root"])
    recs = load_carebench_records(data_root / "carebench/json/metadata.json", data_root / "carebench/videos", require_video_exists=False)
    rec_by_qid = {r.query_id: r for r in recs}
    frozen = json.loads((args.a1_dir / "FROZEN_SAMPLING_MANIFEST.json").read_text(encoding="utf-8"))
    qids = [e["query_id"] for e in frozen["selected"]]

    retrieval = {r["query_id"]: [RetrievalCandidate(**c) for c in r["reranked_d"]] for r in read_jsonl(args.a1_dir / "retrieval_candidates.jsonl")}
    traces = {r["query_id"]: r["frame_indices"] for r in read_jsonl(args.a1_dir / "input_traces.jsonl") if r.get("arm") == "ARM_A0"}

    pred_path = args.out_dir / "predictions.jsonl"
    trace_path = args.out_dir / "input_traces.jsonl"
    fail_path = args.out_dir / "failures.jsonl"
    done = set()
    if pred_path.is_file():
        for r in read_jsonl(pred_path):
            done.add((r["query_id"], r["arm"]))
    pairs = [(q, a) for a in ("ARM_A0", "ARM_A1", "ARM_A7") for q in qids if (q, a) not in done]
    if args.smoke_only:
        pairs = [(qids[0], "ARM_A7")]
    if args.limit:
        pairs = pairs[: args.limit]
    total = len(pairs)
    progress("prepare", 0, total, n_pending=total, n_done=len(done))

    cfg = Qwen3VLLMConfig(
        model_id=args.model_id, local_model_path=args.model_path, backend_mode="vllm_offline",
        pinned_revision="LOCAL_BN_QWEN38_FP8_66SHARDS", num_frames=16,
        multimodal_input_key=args.multimodal_key, prompt_regime="short_aligned",
        max_new_tokens=48, temperature=0.0, top_p=1.0, enable_thinking=False,
        quantization="fp8", gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
        enforce_eager=False, random_seed=42, timeout_sec=3600,
    )
    adapter = Qwen3VLLMMultimodalAdapter(cfg)
    lock_cfg = GPULockConfig(lock_path=Path("/home/bn/videoufo_research/gpu.lock"), run_id=args.run_id,
                             gpu_uuid="GPU-66d83fcc-2695-33a7-33aa-af30b7e3d26d",
                             require_nvidia_smi_idle=True, sidecar_dir=args.out_dir)
    peak = {"peak_mib": 0}
    stop = threading.Event()
    sampler = threading.Thread(target=gpu_sampler, args=(stop, peak), daemon=True)
    t0 = time.time()
    with acquire_cooperative_gpu_lock(lock_cfg):
        sampler.start()
        try:
            for start in range(0, len(pairs), args.chunk):
                batch_pairs = pairs[start:start + args.chunk]
                items = []
                for qid, arm in batch_pairs:
                    fdir = args.frames_dir / qid
                    fpaths = sorted(fdir.glob("frame_*.jpg"))
                    if len(fpaths) != 16:
                        append_jsonl(fail_path, {"query_id": qid, "arm": arm, "error": f"frames={len(fpaths)}"})
                        continue
                    frames = [Image.open(p).convert("RGB") for p in fpaths]
                    q = rec_by_qid[qid]
                    packet = build_natural_retrieval_arm(qid, arm, retrieval[qid])
                    items.append(BatchItem(query=q, packet=packet, visual_input_sha256=hash_pil_frames(frames),
                                           frame_indices=list(traces.get(qid, list(range(16)))), frame_images=frames))
                if not items:
                    continue
                try:
                    outs = adapter.generate_batch(items, PilotStage.PILOT_A3, work_dir=args.out_dir / "work", run_id=args.run_id)
                except Exception as exc:
                    for it in items:
                        append_jsonl(fail_path, {"query_id": it.query.query_id, "arm": it.packet.arm,
                                                 "error": f"{type(exc).__name__}: {exc}"[:800]})
                    progress("generation_error", len(done), total, last_error=str(exc)[:300])
                    continue
                for o in outs:
                    row = o.to_dict()
                    row["execution_mode"] = "EXPLORATORY_A3_CAREBENCH_CROSS_MODEL_UNCALIBRATED"
                    row["judge_independent_of_generator"] = True
                    append_jsonl(pred_path, row)
                    append_jsonl(trace_path, {**row["input_trace"], "query_id": row["query_id"], "arm": row["arm"]})
                    done.add((row["query_id"], row["arm"]))
                progress("generation", len(done), total, last_batch=len(outs), peak_mib=peak["peak_mib"])
        finally:
            stop.set()
            sampler.join(timeout=3)
    elapsed = round(time.time() - t0, 1)
    rows = read_jsonl(pred_path)
    counts = Counter(r["arm"] for r in rows)
    receipt = {
        "schema": "uarcap_q2.a3_carebench_exploratory.v1",
        "status": "EXPLORATORY_COMPLETED_UNCALIBRATED" if len(rows) == 90 else "EXPLORATORY_PARTIAL",
        "run_id": args.run_id, "timestamp_utc": utcnow(), "execution_mode": "EXPLORATORY_A3_CAREBENCH_CROSS_MODEL",
        "model_id": args.model_id, "model_path": args.model_path, "multimodal_input_key": args.multimodal_key,
        "num_frames": 16, "max_new_tokens": 48, "temperature": 0.0, "prompt_regime": "short_aligned",
        "frames_source": str(args.frames_dir), "retrieval_source": str(args.a1_dir / "retrieval_candidates.jsonl"),
        "n_predictions": len(rows), "per_arm": dict(counts), "elapsed_sec": elapsed,
        "peak_vram_mib": peak["peak_mib"], "n_failed_rows_in_failures": len(read_jsonl(fail_path)) if fail_path.is_file() else 0,
        "note": "Exploratory cross-model generation only. Old model used visual+text evidence; Qwen uses retrieval text only. Not a formal gate pass.",
    }
    atomic_write(args.out_dir / "run_receipt.json", receipt)
    progress("completed", len(rows), 90, peak_mib=peak["peak_mib"], elapsed_sec=elapsed)
    print(json.dumps({k: receipt[k] for k in ("status", "n_predictions", "per_arm", "elapsed_sec", "peak_vram_mib")}, indent=2))
    if args.smoke_only:
        first = rows[0] if rows else None
        print(json.dumps({"smoke_prediction": first.get("caption") if first else None,
                          "trace_telemetry": (first or {}).get("input_trace", {}).get("actual_processor_telemetry")}, ensure_ascii=False)[:1200])
    return 0 if (args.smoke_only and rows) or len(rows) == 90 else 2


if __name__ == "__main__":
    raise SystemExit(main())
