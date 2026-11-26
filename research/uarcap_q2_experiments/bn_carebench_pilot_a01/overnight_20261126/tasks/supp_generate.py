#!/usr/bin/env python3
"""Supplementary CaReBench generation: budget extension (A3/A5), length-control (Qwen
detailed prompt), and visual-sampling sensitivity (32/64 frames).

Real GPU work only. Resumable by (query_id, arm[, setting]). Never modifies the frozen
3,600-generation formal run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time

BASE = Path(os.environ.get("OVERNIGHT_BASE", "/home/bn/uarcap_q2_runtime/overnight/20261126"))
TASK_ID = os.environ.get("OVERNIGHT_TASK_ID", "supp_generate")
PROGRESS = BASE / "progress" / f"{TASK_ID}.json"
REPO = "/home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments"
__import__("sys").path.insert(0, REPO)
FORMAL = Path("/home/bn/uarcap_q2_runtime/outputs/formal_candidate_unvalidated")


def utcnow(): return datetime.now(timezone.utc).isoformat()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []


def append_jsonl(p: Path, row: dict) -> None:
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_write(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def progress(phase, done, total, **extra):
    atomic_write(PROGRESS, {"task": TASK_ID, "phase": phase, "completed": done, "total": total,
                            "updated_utc": utcnow(), **extra})


def sample_frames(video: Path, n: int):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release(); raise RuntimeError("cannot read frame count")
    idx = [int(round(i * (total - 1) / max(1, n - 1))) for i in range(n)]
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            cap.release(); raise RuntimeError(f"decode failed at frame {i}")
        out.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return out, idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", choices=["videollama", "qwen"], required=True)
    ap.add_argument("--arms", default="ARM_A0,ARM_A1,ARM_A7")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--qids-file", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt-regime", default="short_aligned", choices=["short_aligned", "detailed_sensitivity"])
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--condition", default="supplementary")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not arms:
        raise SystemExit("no arms")
    from PIL import Image
    from uarcap_q2.common.gpu_lock import GPULockConfig, acquire_cooperative_gpu_lock
    from uarcap_q2.common.schemas import PilotStage, QueryRecord
    from uarcap_q2.generators.base import BatchItem
    from uarcap_q2.generators.modern_vlm_adapter import hash_pil_frames
    from uarcap_q2.retrieval.arm_builder import build_natural_retrieval_arm
    from uarcap_q2.retrieval.reranker import RetrievalCandidate

    cands = {c["query_id"]: c for c in read_jsonl(FORMAL / "formal_candidates.jsonl")}
    retr = {r["query_id"]: [RetrievalCandidate(**x) for x in r["reranked_d"]] for r in read_jsonl(FORMAL / "formal_retrieval.jsonl")}
    frames_idx = {r["query_id"]: r for r in read_jsonl(FORMAL / "formal_frame_indices.jsonl")}
    qids = sorted(q for q in cands if q in retr)
    if args.qids_file:
        qids = [q for q in read_jsonl(args.qids_file) if q in cands] if args.qids_file.suffix == ".jsonl" else json.loads(args.qids_file.read_text()).get("query_ids", [])
        qids = [q for q in qids if q in cands]
    if args.limit:
        qids = qids[: args.limit]
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pred = out / "predictions.jsonl"; trace = out / "input_traces.jsonl"; fail = out / "failures.jsonl"
    done = {(r["query_id"], r["arm"]) for r in read_jsonl(pred)}
    pairs = [(q, a) for a in arms for q in qids if (q, a) not in done]
    progress("prepare", len(done), len(pairs), generator=args.generator, condition=args.condition)

    if args.generator == "videollama":
        if args.num_frames != 16:
            raise SystemExit("Video-LLaMA frozen feature contract only supports 16 frames")
        from uarcap_q2.generators.videollama_adapter import FrozenVideoLLaMAAdapter, FrozenVideoLLaMAConfig
        adapter = FrozenVideoLLaMAAdapter(FrozenVideoLLaMAConfig(
            python_executable="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python",
            evcap_work_root="/home/bn/uar_v06_runtime/evcap_work",
            checkpoint_path="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/EVCapCode/outputs/train_evcap_msrvtt_quality_text/train_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review.pt",
            llm_model_dir="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/models/vicuna-7b-v1.3",
            video_feature_dir="/home/bn/videoufo_research/features/videollama_faithful_v2",
            torch_home="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/torch_cache",
            hf_hub_cache="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/hf_cache",
            expected_checkpoint_sha256="a8b5af79d55a2c7e684771475d645be6373208bdaae5a4211bee556581eb8002"))
        stage = PilotStage.PILOT_A1
    else:
        from uarcap_q2.generators.qwen3_vllm_adapter import Qwen3VLLMConfig, Qwen3VLLMMultimodalAdapter
        mnt = args.max_new_tokens or (128 if args.prompt_regime == "detailed_sensitivity" else 48)
        adapter = Qwen3VLLMMultimodalAdapter(Qwen3VLLMConfig(
            model_id="Qwen/Qwen3.8-27B-FP8", local_model_path="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/models/Qwen3.8-27B-FP8",
            backend_mode="vllm_offline", pinned_revision="LOCAL_BN_QWEN38_FP8_66SHARDS", num_frames=args.num_frames,
            multimodal_input_key="image", prompt_regime=args.prompt_regime, max_new_tokens=mnt, temperature=0.0,
            quantization="fp8", gpu_memory_utilization=0.88, max_model_len=8192, max_num_seqs=64))
        stage = PilotStage.PILOT_A3

    lock = GPULockConfig(Path("/home/bn/videoufo_research/gpu.lock"), TASK_ID,
                         "GPU-66d83fcc-2695-33a7-33aa-af30b7e3d26d", True, out)
    peak = {"peak": 0}; stop = threading.Event()

    def sampler():
        import subprocess
        while not stop.wait(2):
            try:
                v = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, timeout=5).strip().splitlines()[0]
                peak["peak"] = max(peak.get("peak", 0), int(v))
            except Exception:
                pass
    th = threading.Thread(target=sampler, daemon=True)
    t0 = time.time()
    with acquire_cooperative_gpu_lock(lock):
        th.start()
        try:
            step = args.chunk if args.generator == "videollama" else 5
            for start in range(0, len(pairs), step):
                batch = pairs[start:start + step]
                items = []
                for qid, arm in batch:
                    c = cands[qid]; fi = frames_idx.get(qid, {})
                    try:
                        if args.num_frames == 16 and fi.get("frames"):
                            frames = [Image.open(p).convert("RGB") for p in fi["frames"]]
                            fidx = list(fi.get("frame_indices", range(16)))
                        else:
                            frames, fidx = sample_frames(Path(c["video_path"]), args.num_frames)
                    except Exception as e:
                        append_jsonl(fail, {"query_id": qid, "arm": arm, "error": f"frames:{e}"}); continue
                    q = QueryRecord(query_id=qid, dataset_id="CaReBench", source_id=None, source_id_status="UNKNOWN_OPAQUE",
                                    video_path=c["video_path"], video_sha256=None, duration=None, domain=c["category"],
                                    reference_description=c["reference_caption"], reference_events=[],
                                    core_events=[c["core_event_candidate"]] if c["core_event_candidate"] else [],
                                    split=f"supplementary_{args.condition}", eligibility_status="CANDIDATE_PENDING_VERIFICATION",
                                    metadata={"condition": args.condition, "prompt_regime": args.prompt_regime,
                                              "num_frames": args.num_frames, "frame_sampling": "uniform_decode"})
                    packet = build_natural_retrieval_arm(qid, arm, retr[qid])
                    items.append(BatchItem(query=q, packet=packet, visual_input_sha256=hash_pil_frames(frames),
                                           frame_indices=list(fidx), frame_images=frames))
                if not items:
                    continue
                try:
                    outs = adapter.generate_batch(items, stage, work_dir=out / "work", run_id=args.condition)
                    for o in outs:
                        row = o.to_dict()
                        row["execution_mode"] = "SUPPLEMENTARY_GENERATION_UNVALIDATED"
                        row["condition"] = args.condition
                        row["prompt_regime"] = args.prompt_regime
                        row["num_frames"] = args.num_frames
                        append_jsonl(pred, row); append_jsonl(trace, row["input_trace"])
                except Exception as e:
                    for it in items:
                        append_jsonl(fail, {"query_id": it.query.query_id, "arm": it.packet.arm, "error": str(e)[:800]})
                progress("generation", len(read_jsonl(pred)), len(pairs), generator=args.generator,
                         condition=args.condition, peak_mib=peak["peak"])
        finally:
            stop.set(); th.join(timeout=3)
    rows = read_jsonl(pred)
    receipt = {"schema": "uarcap_q2.supplementary_generation.v1", "status": "SUPPLEMENTARY_GENERATION_UNVALIDATED",
               "condition": args.condition, "generator": args.generator, "model_id": adapter.model_id,
               "utc": utcnow(), "n_predictions": len(rows), "per_arm": dict(Counter(r["arm"] for r in rows)),
               "n_failures": len(read_jsonl(fail)), "elapsed_sec": round(time.time() - t0, 1),
               "peak_vram_mib": peak["peak"], "prompt_regime": args.prompt_regime, "num_frames": args.num_frames,
               "qids": len(qids), "note": "Machine generation only; no human validation; formal run untouched."}
    atomic_write(out / "run_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
