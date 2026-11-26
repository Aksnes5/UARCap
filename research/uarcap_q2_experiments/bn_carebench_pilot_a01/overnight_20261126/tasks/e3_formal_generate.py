#!/usr/bin/env python3
"""Formal E3 generation (150-180 machine-approved candidates; human-unvalidated).

Refuses to run below the pre-declared 150-candidate floor. 4 arms × approved candidates.
"""

from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

BASE=Path(os.environ.get("OVERNIGHT_BASE","/home/bn/uarcap_q2_runtime/overnight/20261126"))
REPO="/home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments"; sys.path.insert(0,REPO)
W=Path("/home/bn/uarcap_q2_runtime/outputs/pilot_e3_exploratory")
APPROVED=W/"E3_FORMAL_APPROVED.json"
FRAMES=W/"E3_FORMAL_FRAMES"
MIN=150

def utcnow(): return datetime.now(timezone.utc).isoformat()
def read(p): return [json.loads(l) for l in open(p)] if Path(p).is_file() else []

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--generator",choices=["videollama","qwen"],required=True); ap.add_argument("--chunk",type=int,default=45)
    ap.add_argument("--approved",type=Path,default=APPROVED); ap.add_argument("--out-prefix",default="formal")
    ap.add_argument("--frames-dir",type=Path,default=FRAMES); ap.add_argument("--min-approved",type=int,default=MIN); a=ap.parse_args()
    if not a.approved.is_file(): print(f"HARD_BLOCK: {a.approved} missing"); return 3
    ap_data=json.loads(a.approved.read_text()); approved=ap_data.get("approved",[])
    if len(approved)<a.min_approved:
        print(json.dumps({"status":"AWAITING_SUFFICIENT_CANDIDATES","n_approved":len(approved),"min":a.min_approved})); return 3
    out=W/(a.out_prefix+"_"+a.generator); out.mkdir(parents=True,exist_ok=True)
    pred=out/"predictions.jsonl"; trace=out/"input_traces.jsonl"; fail=out/"failures.jsonl"
    done={(r["query_id"],r["arm"]) for r in read(pred)}
    from uarcap_q2.common.gpu_lock import GPULockConfig, acquire_cooperative_gpu_lock
    from uarcap_q2.common.schemas import PilotStage, QueryRecord
    from uarcap_q2.generators.base import BatchItem
    from uarcap_q2.generators.modern_vlm_adapter import hash_pil_frames
    from uarcap_q2.retrieval.arm_builder import build_e3_intervention_arms
    lock=GPULockConfig(Path("/home/bn/videoufo_research/gpu.lock"),"e3_formal_"+a.generator,"GPU-66d83fcc-2695-33a7-33aa-af30b7e3d26d",True,out)
    if a.generator=="videollama":
        from uarcap_q2.generators.videollama_adapter import FrozenVideoLLaMAAdapter, FrozenVideoLLaMAConfig
        adapter=FrozenVideoLLaMAAdapter(FrozenVideoLLaMAConfig(
            python_executable="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python",
            evcap_work_root="/home/bn/uar_v06_runtime/evcap_work",
            checkpoint_path="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/EVCapCode/outputs/train_evcap_msrvtt_quality_text/train_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review.pt",
            llm_model_dir="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/models/vicuna-7b-v1.3",
            video_feature_dir="/home/bn/videoufo_research/features/videollama_faithful_v2",
            torch_home="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/torch_cache",
            hf_hub_cache="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/hf_cache",
            expected_checkpoint_sha256="a8b5af79d55a2c7e684771475d645be6373208bdaae5a4211bee556581eb8002"))
        stage=PilotStage.PILOT_A1
    else:
        from uarcap_q2.generators.qwen3_vllm_adapter import Qwen3VLLMConfig, Qwen3VLLMMultimodalAdapter
        adapter=Qwen3VLLMMultimodalAdapter(Qwen3VLLMConfig(model_id="Qwen/Qwen3.8-27B-FP8",
            local_model_path="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/models/Qwen3.8-27B-FP8",
            backend_mode="vllm_offline",pinned_revision="LOCAL_BN_QWEN38_FP8_66SHARDS",num_frames=16,
            multimodal_input_key="image",prompt_regime="short_aligned",max_new_tokens=48,temperature=0.0,
            quantization="fp8",gpu_memory_utilization=0.88,max_model_len=8192,max_num_seqs=64))
        stage=PilotStage.PILOT_A3
    with acquire_cooperative_gpu_lock(lock):
        for arm in ("E3_BASE","E3_SUP","E3_CON","E3_NEU"):
            pend=[c for c in approved if (c["query_id"],arm) not in done]
            step=a.chunk if a.generator=="videollama" else 5
            for start in range(0,len(pend),step):
                batch=pend[start:start+step]; items=[]
                for c in batch:
                    fps=sorted((a.frames_dir/c["query_id"]).glob("frame_*.jpg"))
                    if len(fps)!=16: continue
                    if not str(c.get("neutral_text") or "").strip():
                        with fail.open("a") as f: f.write(json.dumps({"query_id":c["query_id"],"arm":arm,"error":"MISSING_NEUTRAL_TEXT"})+"\n")
                        continue
                    frames=[Image.open(p).convert("RGB") for p in fps]
                    sid=str(c.get("source_id") or "")
                    sid_status="VERIFIED_EXACT" if len(sid)==11 and all(ch.isalnum() or ch in "-_" for ch in sid) else "UNKNOWN_OPAQUE"
                    q=QueryRecord(query_id=c["query_id"],dataset_id="VidHalluc-ACH",source_id=(sid or None),source_id_status=sid_status,
                        video_path=c["video_path"],video_sha256=None,duration=None,domain="ACH",reference_description=c["true_action"],
                        reference_events=[c["true_action"]],core_events=[c["true_action"]],target_true_fact=c["true_action"],
                        candidate_conflict_fact=c["conflict_action"],split="formal_e3_candidate_unvalidated",
                        eligibility_status="CANDIDATE_PENDING_VERIFICATION",
                        metadata={"frame_sampling":"16 uniformly sampled frames; not full video","neutral_text_provenance":c.get("neutral_provenance","UNVERIFIED_POOL")})
                    pk=build_e3_intervention_arms(c["query_id"],c["true_action"],c["conflict_action"],c["neutral_text"],base_visual_mode="BASE_A0")
                    items.append(BatchItem(query=q,packet=pk[arm],visual_input_sha256=hash_pil_frames(frames),frame_indices=list(range(16)),frame_images=frames))
                if not items: continue
                try:
                    outs=adapter.generate_batch(items,stage,work_dir=out/"work",run_id="formal_e3")
                    for o in outs:
                        row=o.to_dict(); row["execution_mode"]="FORMAL_E3_MACHINE_ASSISTED_UNVALIDATED"
                        with pred.open("a") as f: f.write(json.dumps(row,ensure_ascii=False)+"\n")
                        with trace.open("a") as f: f.write(json.dumps(row["input_trace"],ensure_ascii=False)+"\n")
                except Exception as e:
                    for it in items:
                        with fail.open("a") as f: f.write(json.dumps({"query_id":it.query.query_id,"arm":arm,"error":str(e)[:600]})+"\n")
    rows=read(pred); receipt={"schema":"uarcap_q2.formal_e3_generation.v1","status":"FORMAL_E3_MACHINE_ASSISTED_UNVALIDATED",
        "generator":a.generator,"n_approved":len(approved),"n_predictions":len(rows),"n_failures":len(read(fail)),
        "utc":utcnow(),"note":"No human verification; no confirmatory statistics."}
    (out/"run_receipt.json").write_text(json.dumps(receipt,indent=2)); print(json.dumps(receipt,indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
