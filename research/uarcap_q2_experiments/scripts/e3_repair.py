"""Prepare and freeze a versioned outcome-blind E3 cohort on BN."""

import argparse
import csv
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uarcap_q2.common.io_utils import (append_jsonl_durable, atomic_write_json, primitive_hash,
                                        read_json, read_jsonl, sha256_file)
from uarcap_q2.interventions.neutral_repair import (
    freeze, parse_text_screen, proposals, statement_job, text_pass, SCREEN)
from uarcap_q2.generators.modern_vlm_adapter import hash_pil_frames

YT = re.compile(r"^[A-Za-z0-9_-]{11}$")
CLIP = re.compile(r"^([A-Za-z0-9_-]{11})(?:\.\d+|_clip_\d+)(?:\.mp4)?$")


def youtube_sources(tsv):
    sources = set()
    with tsv.open(encoding="utf-8-sig",newline="") as stream:
        table=csv.DictReader(stream,delimiter="\t")
        if not table.fieldnames or not {"clip_id","source_id"}<=set(table.fieldnames):
            raise ValueError("Expected official clip_id and source_id manifest columns")
        for n,row in enumerate(table,2):
            sid=row["source_id"]
            clip=row["clip_id"]
            if not sid or not YT.fullmatch(sid):
                raise ValueError(f"{tsv.name}:{n}: invalid official source ID")
            match=CLIP.fullmatch(clip or "")
            if not match or match[1]!=sid:
                raise ValueError(f"{tsv.name}:{n}: clip/source mapping mismatch")
            sources.add(sid)
    if not sources:
        raise ValueError("Empty VideoUFO source manifest")
    return sources


def load_frames(paths):
    from PIL import Image
    images=[]
    for path in paths:
        with Image.open(path) as im:
            images.append(im.convert("RGB"))
    return images


def tokenizers(vicuna, qwen):
    from transformers import AutoTokenizer
    paths = (vicuna,qwen)
    models = [AutoTokenizer.from_pretrained(str(path),local_files_only=True) for path in paths]
    lengths = tuple((lambda text, model=model: len(model.encode(text,add_special_tokens=False))) for model in models)
    receipt = [{p.name:sha256_file(p) for p in sorted(path.glob("tokenizer*")) if p.is_file()} for path in paths]
    if not all(receipt):
        raise ValueError("Tokenizer files are not pinned")
    return lengths, receipt


def prepare(args):
    original = read_json(args.candidates)
    candidates = original["candidates" if args.bridge else "approved"]
    lengths, tok_receipt = tokenizers(args.vicuna,args.qwen)
    config = {"tokenizers":tok_receipt, "proposal_rule":"neutral-pool-v2", "length_span":3,
              "dense_frames":64,"compatibility":"positive visual evidence for neutral",
              "judge":args.gemma_id}
    manifest_refs = {}
    blocked_sources = set()
    if args.bridge:
        man = read_json(args.formal_manifest)
        manifest_refs = man["units"]
        retrieval = {r["query_id"]:r["reranked_d"] for r in read_jsonl(args.retrieval)}
        config["sources"] = {p.name:sha256_file(p) for p in (args.formal_manifest,args.retrieval)}
    else:
        blocked_sources = youtube_sources(args.videoufo_clips)
        pilot = read_json(args.pilot_exclusions)
        blocked_sources.update(pilot["excluded_source_ids"])
        historical = {line.strip() for line in args.historical_sources.read_text(encoding="utf-8").splitlines() if line.strip()}
        if not historical or not all(YT.fullmatch(s) for s in historical):
            raise ValueError("Historical source manifest must contain verified YouTube source IDs")
        blocked_sources.update(historical)
        config["sources"] = {p.name:sha256_file(p) for p in (args.videoufo_clips,args.pilot_exclusions,args.historical_sources)}
    entries = []
    for c in sorted(candidates,key=lambda row:row["query_id"]):
        q = c["query_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+",q):
            raise ValueError("Unsafe query identity")
        frames_dir = args.frames / q
        paths = sorted(frames_dir.glob("frame_*.jpg"))
        frame_sha = hash_pil_frames(load_frames(paths)) if len(paths)==16 else None
        if args.bridge:
            sid = manifest_refs.get(q,{}).get("parent_video_id")
            witness = c["f_minus_provenance"]
            rank = witness["retrieved_rank"]
            ranked = retrieval.get(q,[])
            verified = (sid and type(rank) is int and 0<=rank<len(ranked)
                        and ranked[rank]["caption"]==witness["external_caption"]
                        and ranked[rank]["clip_ids"]==witness["clip_ids"]
                        and str(c["f_minus"]).strip()==witness["external_caption"].strip())
            text = [c["f_plus"],c["f_minus"]]
        else:
            sid = c.get("source_id") or ""
            match = CLIP.fullmatch(Path(c["video_path"]).name)
            verified = bool(YT.fullmatch(sid) and match and match[1]==sid and sid not in blocked_sources)
            text = [c["true_action"],c["conflict_action"]]
        row = {"candidate":c,"source_verified":bool(verified),"independence_group":sid,"targets":[],"proposals":[]}
        for statement in text:
            row["targets"].append(statement_job(q,statement,c["video_path"],frames_dir,frame_sha,bridge_id=c.get("bridge_id")))
        for option in proposals(c,lengths,bridge=args.bridge):
            job = statement_job(q,option["statement"],c["video_path"],frames_dir,frame_sha,bridge_id=c.get("bridge_id"))
            row["proposals"].append({**option,"job":job,"text_screen_pass":False})
        entries.append(row)
    payload = {"schema":"uarcap_q2.e3_neutral_proposals.v2", "original_sha256":primitive_hash(original),
               "original_file_sha256":sha256_file(args.candidates), "config":config,
               "config_sha256":primitive_hash(config), "candidates":entries}
    if args.out.exists() and read_json(args.out)!=payload:
        raise ValueError("Proposal inputs changed. Choose a new output version")
    atomic_write_json(args.out,payload)


def screen(args):
    from uarcap_q2.common.gpu_lock import GPULockConfig, acquire_cooperative_gpu_lock
    from uarcap_q2.evaluators.backends import JudgeBackendConfig, LocalVLLMJudgeBackend
    plan=read_json(args.proposals)
    if plan["config"]["judge"]!=args.gemma_id:
        raise ValueError("Text judge differs from proposed config")
    out=args.out
    staged=[]
    for entry in plan["candidates"]:
        if not entry["source_verified"] or not entry["targets"][0]["expected_16_frames_sha256"]:
            continue
        target = sorted(job["statement"] for job in entry["targets"])
        for proposal in entry["proposals"]:
            inputs = [*target,proposal["statement"]]
            key = primitive_hash([plan["config_sha256"],str(args.gemma),inputs])
            staged.append((entry,proposal,inputs,key))
    ledger = out.with_suffix(".text_attempts.jsonl")
    done = {row["key"]:row for row in read_jsonl(ledger) if row["parse_status"]=="OK"}
    pending={key:inputs for _,_,inputs,key in staged if key not in done}
    if pending:
        backend=LocalVLLMJudgeBackend(JudgeBackendConfig(backend_type="vllm_offline",model_id=args.gemma_id,
                    model_family="gemma",local_model_path=str(args.gemma),temperature=0,max_new_tokens=512,max_model_len=4096))
        system=("Assess three English caption phrases. Consider implication in either direction, denial, and pragmatic "
                "presupposition of either target by the candidate, including paraphrases. Check that the two targets differ "
                "in one specific action or property and can be distinguished. Return strict JSON with booleans natural, "
                "same_form, implies_target, negates_target, presupposes_target, pair_single_target and string rationale.")
        with acquire_cooperative_gpu_lock(GPULockConfig(args.lock,"q2_e3_text_screen",args.gpu_uuid,True,out.parent)):
            for attempt in range(1,3):
                queue=list(pending.items())
                for start in range(0,len(queue),12):
                    batch=queue[start:start+12]
                    prompts=[(system,f"Targets: {x[:2]}\nCandidate: {x[2]}") for _,x in batch]
                    for (key,_),raw in zip(batch,backend.complete_batch(prompts),strict=True):
                        try:
                            result={"parse_status":"OK","verification":parse_text_screen(raw)}
                        except (ValueError,TypeError) as exc:
                            result={"parse_status":"INVALID","error":str(exc)}
                        record={"schema":SCREEN,"key":key,"attempt":attempt,"raw":raw,**result}
                        append_jsonl_durable(ledger,record)
                        if result["parse_status"]=="OK":
                            done[key]=record
                            pending.pop(key)
    jobs={}
    for entry,proposal,_,key in staged:
        row=done.get(key)
        proposal["text_screen"]=row
        proposal["text_screen_pass"]=bool(row and text_pass(row["verification"]))
        if proposal["text_screen_pass"]:
            for job in [*entry["targets"],proposal["job"]]:
                jobs[job["job_id"]]=job
    payload={"schema":"uarcap_q2.blind_statement_jobs.v1",
             "jobs":[jobs[k] for k in sorted(jobs)]}
    atomic_write_json(args.jobs,payload)
    atomic_write_json(out,plan)
    print(f"Prepared {len(jobs)} visual verifications. Unparsed text screens: {len(pending)}")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    subs=ap.add_subparsers(dest="command",required=True)
    p=subs.add_parser("prepare")
    for k in ("candidates","frames","vicuna","qwen","out"):
        p.add_argument("--"+k,type=Path,required=True)
    for k in ("formal-manifest","retrieval","videoufo-clips","historical-sources","pilot-exclusions"):
        p.add_argument("--"+k,type=Path)
    p.add_argument("--gemma-id",required=True)
    p.add_argument("--bridge",action="store_true")
    p.set_defaults(func=prepare)
    s=subs.add_parser("screen")
    for k in ("proposals","gemma","out","jobs","lock"):
        s.add_argument("--"+k,type=Path,required=True)
    s.add_argument("--gpu-uuid",required=True)
    s.add_argument("--gemma-id",required=True)
    s.set_defaults(func=screen)
    f=subs.add_parser("freeze")
    for k in ("candidates","screened","verdicts","out"):
        f.add_argument("--"+k,type=Path,required=True)
    f.add_argument("--bridge",action="store_true")
    args=ap.parse_args()
    if args.command!="freeze":
        args.func(args)
        return 0
    original, plan = read_json(args.candidates),read_json(args.screened)
    result=freeze(original,plan,read_jsonl(args.verdicts),bridge=args.bridge)
    if args.out.resolve()==args.candidates.resolve() or (args.out.exists() and read_json(args.out)!=result):
        raise ValueError("Use a new freeze path. Existing freeze is immutable")
    atomic_write_json(args.out,result)
    if result["eligible"]:
        return 0
    print(f"HARD_BLOCK {result['status']}: {result['n_approved']} / {result['minimum']}")
    return 3


if __name__=="__main__":
    raise SystemExit(main())
