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
REPO = str(Path(__file__).resolve().parents[3])
__import__("sys").path.insert(0, REPO)
FORMAL = Path(os.environ.get("UARCAP_FORMAL","/home/bn/uarcap_q2_runtime/outputs/formal_candidate_unvalidated"))


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


def verify_supplementary_row(row,packet,anchor,n_frames,prompt_regime,generator,factorial=False):
    t=row["input_trace"]
    evcap=generator=="videollama"
    if (row.get("ok") is not True or not row.get("caption","").strip()
            or any(row[k]!=t[k] for k in ("query_id","model_id","arm"))
            or t["checkpoint_sha256"]!=anchor["checkpoint_sha256"]
            or t["text_evidence"]!=packet.text_evidence
            or t["visual_evidence_ids"]!=(packet.visual_clip_ids if evcap or factorial else [])
            or t["visual_evidence_scores"]!=(packet.visual_scores if evcap or factorial else [])
            or len(t["frame_indices"])!=n_frames):
        raise ValueError("Supplementary trace differs from model or evidence contract")
    if n_frames==16:
        for k in ("visual_input_sha256","frame_indices"):
            if t[k]!=anchor[k]:
                raise ValueError("Original visual frames differ")
    if evcap or (n_frames==16 and prompt_regime=="short_aligned"):
        if t["decoding_config"]!=anchor["decoding_config"]:
            raise ValueError("Incompatible decode config")
    if evcap:
        observed=t["actual_processor_telemetry"]
        ref=anchor["actual_processor_telemetry"]
        if observed["image_tensor_sha256"]!=ref["image_tensor_sha256"]:
            raise ValueError("EVCap query tensor changed")
        if not packet.visual_clip_ids and observed["fused_nonzero"]!=0:
            raise ValueError("EVCap zero-visual control is nonzero")
        if packet.visual_clip_ids==anchor["visual_evidence_ids"] and observed["fused_aggregate_sha256"]!=ref["fused_aggregate_sha256"]:
            raise ValueError("EVCap A7 visual aggregate changed")
    if not evcap:
        telemetry=t["actual_processor_telemetry"]
        if (telemetry["actual_frames_received"]!=n_frames or
                telemetry["local_model_manifest_sha256"]!=anchor["actual_processor_telemetry"]["local_model_manifest_sha256"]):
            raise ValueError("Qwen processor or model manifest mismatch")
        if factorial and (telemetry["processor_image_blocks"] != n_frames + telemetry["external_frame_count"]
                          or not telemetry["processor_prompt_token_ids_sha256"]):
            raise ValueError("Qwen factorial consumption could not be verified")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", choices=["videollama", "qwen"], required=True)
    ap.add_argument("--arms", default="ARM_A0,ARM_A1,ARM_A7")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--qids-file", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt-regime", default="short_aligned", choices=["short_aligned", "detailed_sensitivity", "uniform_length"])
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--condition", default="supplementary")
    ap.add_argument("--config-sha256", required=True)
    ap.add_argument("--smoke-queries",type=int,default=0)
    ap.add_argument("--decoded-frames",type=Path)
    ap.add_argument("--modality-ablation",action="store_true")
    ap.add_argument("--factorial-frames", type=Path)
    ap.add_argument("--length-design", type=Path)
    args = ap.parse_args()
    from uarcap_q2.common.io_utils import primitive_hash, sha256_file, append_jsonl_durable
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    limits = {"ARM_A0":0,"ARM_A1":1,"ARM_A3":3,"ARM_A5":5,"ARM_A7":7}
    from uarcap_q2.retrieval.arm_builder import MODALITY_ARMS, build_evcap_modality_arm
    factorial_arms = {"T0F0": (0,0), "T1F0": (1,0), "T0F1": (0,1), "T1F1": (1,1)}
    allowed=factorial_arms if args.factorial_frames else MODALITY_ARMS if args.modality_ablation else limits
    if args.factorial_frames and (args.generator != "qwen" or args.num_frames != 16 or args.modality_ablation):
        raise ValueError("Retrieved-frame controls require Qwen and all 16 frozen query frames")
    if bool(args.length_design) != (args.prompt_regime == "uniform_length"):
        raise ValueError("Need the approved length revision for uniform length")
    if args.modality_ablation and args.generator!="videollama":
        raise ValueError("Modality arms use the inherited EVCap fusion interface")
    if not arms or len(set(arms)) != len(arms) or set(arms)-allowed.keys():
        raise SystemExit("Invalid natural arms")
    if args.generator=="videollama" and (args.prompt_regime!="short_aligned" or args.num_frames!=16 or args.max_new_tokens):
        raise SystemExit("Frozen Video-LLaMA prompt/16-frame contract")
    PROGRESS.parent.mkdir(parents=True,exist_ok=True)
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
        qids = ([row["query_id"] for row in read_jsonl(args.qids_file)] if args.qids_file.suffix == ".jsonl"
                else json.loads(args.qids_file.read_text(encoding="utf-8"))["query_ids"])
    if not qids or len(set(qids))!=len(qids) or set(qids)-set(cands).intersection(retr,frames_idx):
        raise ValueError("Invalid supplement cohort")
    if args.limit:
        qids = qids[: args.limit]
    decoded=json.loads(args.decoded_frames.read_text(encoding="utf-8")) if args.decoded_frames else {}
    if args.num_frames != 16 and (not set(qids)<=set(decoded.get("queries",{})) or decoded.get("num_frames")!=args.num_frames):
        raise ValueError("32/64-frame runs need the checked CPU-decoded manifest covering this cohort")
    length = json.loads(args.length_design.read_text(encoding="utf-8")) if args.length_design else None
    if length and (length.get("revision") != "DETAILED_TO_UNIFORM_LENGTH" or length["query_ids"] != qids):
        raise ValueError("Length revision or cohort is unbound")
    external = json.loads(args.factorial_frames.read_text(encoding="utf-8")) if args.factorial_frames else None
    if external and set(external["queries"]) != set(qids):
        raise ValueError("Frame factorial must cover the fixed query cohort")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pred = out / "predictions.jsonl"; trace = out / "input_traces.jsonl"; fail = out / "failures.jsonl"
    from uarcap_q2.final_analysis.validation_identity import index
    from uarcap_q2.retrieval.arm_builder import verify_natural_prefix_consistency
    anchor = index(read_jsonl(FORMAL / args.generator / "predictions.jsonl"))
    by_query = {k[1]:r["input_trace"] for k,r in anchor.items() if k[2]=="ARM_A7"}
    expected = {(q,a) for q in qids for a in arms}
    binding = primitive_hash({"config_sha256":args.config_sha256, "arms":arms,"qids":qids, "generator":args.generator,
        "prompt_regime":args.prompt_regime,"num_frames":args.num_frames,"max_tokens":args.max_new_tokens,
        "decode_receipt":sha256_file(args.decoded_frames) if args.decoded_frames else None,
        **({"length_receipt":sha256_file(args.length_design)} if args.length_design else {}),
        **({"factorial_frames":sha256_file(args.factorial_frames)} if args.factorial_frames else {}),
        "sources":[sha256_file(FORMAL/name) for name in ("formal_candidates.jsonl","formal_frame_indices.jsonl","formal_retrieval.jsonl",
                    args.generator+"/predictions.jsonl")]})
    existing=read_jsonl(pred)
    if any(r.get("binding_sha256")!=binding or (r["query_id"],r["arm"]) not in expected for r in existing):
        raise ValueError("Mismatched supplemental resume binding")
    done = {(r["query_id"], r["arm"]) for r in existing if r.get("ok") is True}
    if len(done)!=len(existing):
        raise ValueError("Duplicate or invalid sealed predictions")
    natural={q:{a:build_natural_retrieval_arm(q,a,retr[q]) for a in limits} for q in qids}
    for q,packets in natural.items():
        verify_natural_prefix_consistency(packets)
        if args.modality_ablation:
            packets.update({a:build_evcap_modality_arm(q,a,retr[q]) for a in arms})
        if external:
            from dataclasses import replace
            for a in arms:
                t,f = factorial_arms[a]
                base = packets["ARM_A1"]
                packets[a] = replace(base, arm=a, text_topn=t, visual_topn=f, text_evidence=base.text_evidence[:t],
                                     visual_clip_ids=base.visual_clip_ids[:f], visual_scores=base.visual_scores[:f],
                                     base_visual_mode="QWEN_RETRIEVED_FRAME_FACTORIAL")
    active_qids=qids[:args.smoke_queries] if args.smoke_queries else qids
    active={(q,a) for q in active_qids for a in arms}
    pairs = [(q, a) for a in arms for q in active_qids if (q,a) not in done]
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
            word_range=tuple(length["word_range"]) if length else None, external_frames=external["num_frames"] if external else 0,
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
                            info=decoded["queries"][qid]
                            frames=[Image.open(p).convert("RGB") for p in info["paths"]]
                            fidx=info["indices"]
                            if info["video_path"]!=c["video_path"] or info["sha256"]!=hash_pil_frames(frames):
                                raise ValueError("Decoded visual inputs changed")
                    except Exception as e:
                        append_jsonl(fail, {"query_id": qid, "arm": arm, "error": f"frames:{e}"}); continue
                    q = QueryRecord(query_id=qid, dataset_id="CaReBench", source_id=None, source_id_status="UNKNOWN_OPAQUE",
                                    video_path=c["video_path"], video_sha256=None, duration=None, domain=c["category"],
                                    reference_description=c["reference_caption"], reference_events=[],
                                    core_events=[c["core_event_candidate"]] if c["core_event_candidate"] else [],
                                    split=f"supplementary_{args.condition}", eligibility_status="CANDIDATE_PENDING_VERIFICATION",
                                    metadata={"condition": args.condition, "prompt_regime": args.prompt_regime,
                                              "num_frames": args.num_frames, "frame_sampling": "uniform_decode"})
                    packet = natural[qid][arm]
                    extras = {}
                    if external and packet.visual_clip_ids:
                        info = external["queries"][qid]
                        if info.get("clip_id") != packet.visual_clip_ids[0] or "paths" not in info:
                            raise ValueError("Unverified Top-1 retrieved video")
                        rframes = [Image.open(p).convert("RGB") for p in info["paths"]]
                        if hash_pil_frames(rframes) != info["sha256"]:
                            raise ValueError("Retrieved clip frames differ from bound manifest")
                        extras = {"external_frame_images": rframes, "external_frame_indices": info["indices"],
                                  "external_video_sha256": info["source_video_sha256"]}
                    items.append(BatchItem(query=q, packet=packet, visual_input_sha256=hash_pil_frames(frames),
                                           frame_indices=list(fidx), frame_images=frames, **extras))
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
                        t=row["input_trace"]
                        expected_trace=by_query[row["query_id"]]
                        packet=natural[row["query_id"]][row["arm"]]
                        verify_supplementary_row(row,packet,expected_trace,args.num_frames,args.prompt_regime,args.generator,bool(external))
                        row["binding_sha256"]=binding
                        append_jsonl_durable(pred,row)
                except Exception as e:
                    for it in items:
                        append_jsonl(fail, {"query_id": it.query.query_id, "arm": it.packet.arm, "error": str(e)[:800]})
                progress("generation", len(read_jsonl(pred)), len(pairs), generator=args.generator,
                         condition=args.condition, peak_mib=peak["peak"])
        finally:
            stop.set(); th.join(timeout=3)
    rows = read_jsonl(pred)
    from uarcap_q2.common.io_utils import atomic_write_jsonl
    atomic_write_jsonl(trace,(r["input_trace"] for r in rows))
    keys = {(r["query_id"],r["arm"]) for r in rows}
    receipt = {"schema": "uarcap_q2.supplementary_generation.v1", "status": "SUPPLEMENTARY_GENERATION_UNVALIDATED",
               "condition": args.condition, "generator": args.generator, "model_id": adapter.model_id,
               "utc": utcnow(), "n_predictions": len(rows), "per_arm": dict(Counter(r["arm"] for r in rows)),
               "n_failures": len(read_jsonl(fail)), "elapsed_sec": round(time.time() - t0, 1),
               "peak_vram_mib": peak["peak"], "prompt_regime": args.prompt_regime, "num_frames": args.num_frames,
               "qids": len(qids), "n_expected":len(expected), "binding_sha256":binding, "predictions_sha256":sha256_file(pred) if pred.is_file() else None, "note": "Machine generation only; no human validation; formal run untouched."}
    receipt["accepted_for_request"]=active<=keys
    atomic_write(out / ("smoke_receipt.json" if args.smoke_queries else "run_receipt.json"), receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if active<=keys<=expected and len(keys)==len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
