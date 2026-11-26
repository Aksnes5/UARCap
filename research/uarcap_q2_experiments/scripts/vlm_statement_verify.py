"""Role-blind Qwen verification against generator frames and separate denser sampling."""

import argparse
from base64 import b64encode
from io import BytesIO
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e3_repair import load_frames
from uarcap_q2.common.io_utils import append_jsonl_durable, atomic_write_json, primitive_hash, read_json, read_jsonl
from uarcap_q2.interventions.neutral_repair import VISUAL, parse_visual
from uarcap_q2.generators.modern_vlm_adapter import hash_pil_frames


def messages(statement, images):
    prompt = ("Assess the single statement below from the images in temporal order. "
              "Use contradicted only when evidence shows an incompatible event or property. "
              "Mere absence from the supplied frames is unverifiable. Report concrete frame evidence. "
              "Return only JSON with verdict (entailed, contradicted, or unverifiable), confidence "
              "(low, medium, or high), evidence_frames (zero-based integer list), and rationale.\n"
              f"Statement: {statement}")
    media=[]
    for image in images:
        data=BytesIO()
        image.save(data,format="PNG")
        url="data:image/png;base64,"+b64encode(data.getvalue()).decode("ascii")
        media.append({"type":"image_url","image_url":{"url":url}})
    return [{"role":"user","content":[*media,{"type":"text","text":prompt}]}]


def dense_frames(video, root, n=64):
    from uarcap_q2.datasets.video_decoder import decode_and_sample_video
    frames, report=decode_and_sample_video(video,num_frames=n)
    if not report.ok or len(frames)!=n or len(set(report.sampled_indices))!=n:
        raise ValueError(f"Need {n} distinct decoded frame indices: {report.status}")
    root.mkdir(parents=True,exist_ok=True)
    paths=[]
    for offset,image in enumerate(frames):
        path=root/f"frame_{offset:02d}.jpg"
        image.save(path,"JPEG",quality=92)
        paths.append(str(path))
    return {"paths":paths,"indices":report.sampled_indices,"timestamps_seconds":report.sampled_pts_sec,
            "source_video_sha256":report.video_sha256,"sha256":hash_pil_frames(load_frames(paths))}


def prepare(jobs_path, work, out):
    jobs=read_json(jobs_path)["jobs"]
    queries={}
    for job in jobs:
        q=job["query_id"]
        if q in queries:
            if queries[q]["video_path"]!=job["video_path"]:
                raise ValueError(f"{q}: video binding conflict")
            continue
        paths=sorted(Path(job["frames_dir"]).glob("frame_*.jpg"))
        if len(paths)!=16 or hash_pil_frames(load_frames(paths))!=job["expected_16_frames_sha256"]:
            raise ValueError(f"{q}: original input frame mismatch")
        record={"video_path":job["video_path"],"original_paths":[str(p) for p in paths]}
        try:
            record["dense"]=dense_frames(job["video_path"],work/q)
        except ValueError as error:
            record["decode_error"]=str(error)
        queries[q]=record
    atomic_write_json(out,{"schema":"uarcap_q2.e3_video_decode.v1","jobs_sha256":primitive_hash(jobs),
                           "queries":queries})


def verify(args):
    from vllm import SamplingParams
    from uarcap_q2.common.gpu_lock import GPULockConfig, acquire_cooperative_gpu_lock
    from uarcap_q2.generators.qwen3_vllm_adapter import Qwen3VLLMConfig, Qwen3VLLMMultimodalAdapter
    jobs=read_json(args.jobs)["jobs"]
    inventory=read_json(args.frames)
    if inventory["jobs_sha256"]!=primitive_hash(jobs):
        raise ValueError("Frame manifest belongs to other jobs")
    if len({j["job_id"] for j in jobs})!=len(jobs):
        raise ValueError("Duplicate verification job")
    model=Qwen3VLLMMultimodalAdapter(Qwen3VLLMConfig(model_id="Qwen/Qwen3.8-27B-FP8",
            local_model_path=str(args.qwen),backend_mode="vllm_offline",pinned_revision="LOCAL_BN_QWEN38_FP8_66SHARDS",
            num_frames=64,multimodal_input_key="image",quantization="fp8",gpu_memory_utilization=.88,
            max_model_len=32768,max_num_seqs=2))
    binding=primitive_hash([primitive_hash(inventory),str(args.qwen),model._ensure_manifest()])
    done={}
    for row in read_jsonl(args.out):
        if row["binding"]!=binding:
            raise ValueError("Existing visual results belong to other inputs")
        done[row["job_id"]]=row
    pending=[j for j in jobs if j["job_id"] not in done]
    if not pending:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.touch(exist_ok=True)
        return
    sampling=SamplingParams(temperature=0,max_tokens=512,seed=42)
    lock=GPULockConfig(args.lock,"q2_blind_visual",args.gpu_uuid,True,args.out.parent)
    with acquire_cooperative_gpu_lock(lock):
        engine=model._ensure_vllm_engine()
        for job in pending:
            info=inventory["queries"][job["query_id"]]
            row={"schema":VISUAL,"job_id":job["job_id"],"job_sha256":primitive_hash(job),
                 "query_id":job["query_id"],"bridge_id":job.get("bridge_id"),"binding":binding,
                 "judge_model_id":model.model_id, "parse_status":"MISSING","attempts":[]}
            dense=info.get("dense")
            if dense:
                imgs=[load_frames(info["original_paths"]),load_frames(dense["paths"])]
                digest=[hash_pil_frames(images) for images in imgs]
                if digest!=[job["expected_16_frames_sha256"],dense["sha256"]]:
                    raise ValueError("Visual frame payload changed after decode")
                prompts=[messages(job["statement"],images) for images in imgs]
                for i,prompt in enumerate(prompts):
                    for attempt in range(2):
                        replies=engine.chat([prompt],sampling_params=sampling,chat_template_kwargs={"enable_thinking":False})
                        raw=replies[0].outputs[0].text
                        evidence={"input":"generator16" if i==0 else "dense64","attempt":attempt+1,"raw":raw}
                        try:
                            evidence["parsed"]=parse_visual(raw,len(imgs[i]))
                        except ValueError as error:
                            evidence["error"]=str(error)
                        row["attempts"].append(evidence)
                        if "parsed" in evidence:
                            row[evidence["input"]]=evidence["parsed"]
                            break
                if all(k in row for k in ("generator16","dense64")):
                    sample, proxy=row["generator16"],row["dense64"]
                    row.update(parse_status="OK",parse_ok=True,n_frames=16,n_dense_frames=64,frames_sha256=digest[0],
                           dense_frames_sha256=digest[1],video_truth=proxy["verdict"],truth_scope="64_FRAME_PROXY",
                           visible_in_16_frames="yes" if sample["verdict"]=="entailed" else "uncertain",
                           confidence=proxy["confidence"],evidence=proxy["rationale"],
                           evidence_timestamps_seconds=[dense["timestamps_seconds"][k] for k in proxy["evidence_frames"]])
                else:
                    row["parse_status"]="INVALID"
            else:
                row["decode_error"]=info["decode_error"]
            append_jsonl_durable(args.out,row)
        model.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("decode")
    for k in ("jobs","work","out"):
        p.add_argument("--"+k,type=Path,required=True)
    p=sub.add_parser("verify")
    for k in ("jobs","frames","qwen","lock","out"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--gpu-uuid",required=True)
    args=parser.parse_args()
    if args.command=="decode":
        prepare(args.jobs,args.work,args.out)
    else:
        verify(args)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
