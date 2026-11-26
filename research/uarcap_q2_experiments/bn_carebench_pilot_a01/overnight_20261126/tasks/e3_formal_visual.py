#!/usr/bin/env python3
"""Formal E3 candidates: decode 16 frames, Qwen visual screening, machine approval.
All outputs remain MACHINE_ASSISTED_UNVALIDATED (no human verification)."""

from __future__ import annotations
import argparse, base64, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

BASE=Path(os.environ.get("OVERNIGHT_BASE","/home/bn/uarcap_q2_runtime/overnight/20261126"))
REPO="/home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments"; sys.path.insert(0,REPO)
W=Path("/home/bn/uarcap_q2_runtime/outputs/pilot_e3_exploratory")
QWEN="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/models/Qwen3.8-27B-FP8"

def utcnow(): return datetime.now(timezone.utc).isoformat()

def load(): return json.loads((W/"E3_FORMAL_CANDIDATES_180.json").read_text())["candidates"]

def stage_frames():
    from uarcap_q2.datasets.video_decoder import decode_and_sample_video
    cands=load(); done=0
    for i,c in enumerate(cands):
        d=W/"E3_FORMAL_FRAMES"/c["query_id"]
        if d.is_dir() and len(list(d.glob("frame_*.jpg")))==16: done+=1; continue
        try:
            frames,_=decode_and_sample_video(Path(c["video_path"]),num_frames=16)
            d.mkdir(parents=True,exist_ok=True)
            for j,im in enumerate(frames): im.save(d/f"frame_{j:02d}.jpg","JPEG",quality=92)
            done+=1
        except Exception as e:
            with (W/"E3_FORMAL_FRAME_FAILURES.jsonl").open("a") as f: f.write(json.dumps({"query_id":c["query_id"],"error":str(e)[:400]})+"\n")
    print(json.dumps({"decoded":done,"total":len(cands)})); return 0

def stage_visual():
    from vllm import LLM, SamplingParams
    from uarcap_q2.common.gpu_lock import GPULockConfig, acquire_cooperative_gpu_lock
    cands=load(); out=W/"E3_FORMAL_VISUAL_SCREEN.jsonl"; done=set()
    if out.is_file():
        for l in out.read_text().splitlines():
            if l.strip():
                r=json.loads(l)
                if r.get("parse_ok"): done.add(r["query_id"])
    pending=[c for c in cands if c["query_id"] not in done]
    sysp=("You see 16 real frames. TRUE is the dataset answer action; CONFLICT is a distractor. Decide independently: "
          'is TRUE visible? is CONFLICT absent? are frames sufficient? Output JSON {"true_present":"yes|no|partial",'
          '"conflict_present":"yes|no|partial","frames_sufficient":"yes|no|partial","evidence":"..."}')
    lock=GPULockConfig(Path("/home/bn/videoufo_research/gpu.lock"),"e3_formal_visual","GPU-66d83fcc-2695-33a7-33aa-af30b7e3d26d",True,W)
    with acquire_cooperative_gpu_lock(lock):
        llm=LLM(model=QWEN,gpu_memory_utilization=0.88,max_model_len=16384,max_num_seqs=64,
                limit_mm_per_prompt={"image":16},quantization="fp8",seed=42)
        sp=SamplingParams(temperature=0.0,max_tokens=256)
        for i in range(0,len(pending),2):
            batch=pending[i:i+2]; convs=[]; meta=[]
            for c in batch:
                fps=sorted((W/"E3_FORMAL_FRAMES"/c["query_id"]).glob("frame_*.jpg"))
                if len(fps)!=16: continue
                content=[{"type":"text","text":f"TRUE ACTION: {c['true_action']}\nCONFLICT ACTION: {c['conflict_action']}"}]
                for p in fps: content.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(p.read_bytes()).decode()}})
                content.append({"type":"text","text":"Return the JSON verdict."})
                convs.append([{"role":"system","content":sysp},{"role":"user","content":content}]); meta.append(c)
            if not convs: continue
            outs=llm.chat(convs,sampling_params=sp,chat_template_kwargs={"enable_thinking":False})
            for c,o in zip(meta,outs):
                raw=str(o.outputs[0].text); ok=False; p={}
                try:
                    s=raw.strip()
                    if s.startswith("```"): s=s.split("```")[1].strip(); s=s[4:] if s.startswith("json") else s
                    p=json.loads(s); ok=True
                except Exception as e: p={"parse_error":str(e)}
                with out.open("a") as f: f.write(json.dumps({"query_id":c["query_id"],"screen":p,"parse_ok":ok,"raw":raw[:3000],"model":"qwen3.8-27b-fp8"},ensure_ascii=False)+"\n")
    print(json.dumps({"n_total":len(cands),"n_screened":len(done)+len(pending)})); return 0

def stage_approve():
    cands=load(); text={}
    tp=W/"E3_FORMAL_TEXT_SCREEN.jsonl"
    if tp.is_file():
        for l in tp.read_text().splitlines():
            if l.strip():
                r=json.loads(l); text[r["query_id"]]=r.get("screen") or {}
    vis={}
    vp=W/"E3_FORMAL_VISUAL_SCREEN.jsonl"
    if vp.is_file():
        for l in vp.read_text().splitlines():
            if l.strip():
                r=json.loads(l); vis[r["query_id"]]=r.get("screen") or {}
    approved=[];review=[]
    for c in cands:
        t=text.get(c["query_id"],{}); v=vis.get(c["query_id"],{})
        yn=lambda x:str(x).strip().lower()
        ok=(t.get("mutually_exclusive") and t.get("natural") and t.get("length_ok") and t.get("conflict_plausibly_absent")
            and yn(v.get("true_present"))=="yes" and yn(v.get("conflict_present"))=="no" and yn(v.get("frames_sufficient"))=="yes")
        rec={**c,"text_screen":t,"visual_screen":v,"machine_approved":bool(ok)}
        (approved if ok else review).append(rec)
    payload={"schema":"uarcap_q2.e3_formal_approved.v1","status":"MACHINE_ASSISTED_UNVALIDATED","utc":utcnow(),
             "n_candidates":len(cands),"n_approved":len(approved),"n_review":len(review),"approved":approved}
    (W/"E3_FORMAL_APPROVED.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    (W/"E3_FORMAL_HUMAN_REVIEW.json").write_text(json.dumps({"utc":utcnow(),"items":review},ensure_ascii=False,indent=2))
    print(json.dumps({"n_approved":len(approved),"n_review":len(review)})); return 0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--stage",choices=["frames","visual","approve"],required=True); a=ap.parse_args()
    return {"frames":stage_frames,"visual":stage_visual,"approve":stage_approve}[a.stage]()

if __name__=="__main__": raise SystemExit(main())
