#!/usr/bin/env python3
"""Dual-target evaluation of formal E3 outputs with Gemma + Llama cross-family judges."""
import json,os,sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,"/home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments"); sys.path.insert(0,str(Path(__file__).parent))
from vidhalluc_pipeline import _gemma_backend,_llama_backend,_eval_prompt,_parse_json
from uarcap_q2.common.gpu_lock import GPULockConfig,acquire_cooperative_gpu_lock
import argparse
W=Path("/home/bn/uarcap_q2_runtime/outputs/pilot_e3_exploratory")
aparser=argparse.ArgumentParser(); aparser.add_argument("--approved",type=Path,default=W/"E3_FORMAL_APPROVED.json")
aparser.add_argument("--out-prefix",default="formal"); aargs=aparser.parse_args()
OUT=W/(aargs.out_prefix+"_evaluation"); OUT.mkdir(parents=True,exist_ok=True)
def utcnow(): return datetime.now(timezone.utc).isoformat()
def read(p): return [json.loads(l) for l in open(p)] if Path(p).is_file() else []
approved=json.loads(aargs.approved.read_text())["approved"] if aargs.approved.is_file() else []
if len(approved)<150: print(json.dumps({"status":"AWAITING_SUFFICIENT_CANDIDATES","n":len(approved)})); raise SystemExit(3)
ap={c["query_id"]:c for c in approved}
preds=[]
for gen in (aargs.out_prefix+"_videollama",aargs.out_prefix+"_qwen"):
    for r in read(W/gen/"predictions.jsonl"):
        if r["query_id"] in ap: preds.append({**r,"generator":gen})
rows_path=OUT/"E3_FORMAL_EVAL.jsonl"; done={(r["query_id"],r["arm"],r["generator"],r["judge"]) for r in read(rows_path)}
lock=GPULockConfig(Path("/home/bn/videoufo_research/gpu.lock"),"e3_formal_evaluate","GPU-66d83fcc-2695-33a7-33aa-af30b7e3d26d",True,OUT)
def run(backend,name):
    todo=[p for p in preds if (p["query_id"],p["arm"],p["generator"],name) not in done]
    for i in range(0,len(todo),16):
        batch=todo[i:i+16]; prompts=[]
        for p in batch:
            c=ap[p["query_id"]]; prompts.append(_eval_prompt(p["caption"],c["true_action"])); prompts.append(_eval_prompt(p["caption"],c["conflict_action"]))
        resps=backend.complete_batch(prompts)
        for j,p in enumerate(batch):
            details=[]; ok=True
            for k,resp in enumerate((resps[2*j],resps[2*j+1])):
                d=None
                for attempt in range(2):
                    try: d=_parse_json(resp); break
                    except Exception as e:
                        if attempt==0:
                            try: resp=backend.complete_batch([prompts[2*j+k]])[0]
                            except Exception: break
                        d={"parse_error":str(e)}
                if d is None or "parse_error" in d: ok=False
                details.append(d)
            yt,yc=details
            # Never coerce a parse failure to 0: expressed is None and excluded downstream.
            with rows_path.open("a") as f: f.write(json.dumps({"query_id":p["query_id"],"arm":p["arm"],"generator":p["generator"],"caption":p["caption"],"y_true":(int(yt.get("expressed",0) or 0) if ok else None),"y_conflict":(int(yc.get("expressed",0) or 0) if ok else None),"true_detail":yt,"conflict_detail":yc,"judge":name,"parse_ok":ok},ensure_ascii=False)+"\n")
with acquire_cooperative_gpu_lock(lock):
    gb=_gemma_backend(); run(gb,"gemma-3-27b-fp8"); gb._engine=None; del gb
    import time; time.sleep(8)
    lb=_llama_backend(); run(lb,"llama-3.3-70b-w4a16"); lb._engine=None; del lb
rows=read(rows_path); by=defaultdict(dict)
for r in rows: by[(r["query_id"],r["arm"],r["generator"])][r["judge"]]=r  # later row (retry) wins
pairs=[v for v in by.values() if len(v)==2 and all(x.get("parse_ok") for x in v.values())]
summ={"schema":"uarcap_q2.formal_e3_eval.v1","status":"MACHINE_ASSISTED_UNVALIDATED","utc":utcnow(),"n_rows":len(rows),"n_pairs":len(pairs),
 "n_parse_failures":sum(1 for r in rows if not r.get("parse_ok")),"parse_failure_policy":"expressed=None and excluded; never coerced to 0",
 "true_agreement":round(sum(v["gemma-3-27b-fp8"]["y_true"]==v["llama-3.3-70b-w4a16"]["y_true"] for v in pairs)/max(1,len(pairs)),4),
 "conflict_agreement":round(sum(v["gemma-3-27b-fp8"]["y_conflict"]==v["llama-3.3-70b-w4a16"]["y_conflict"] for v in pairs)/max(1,len(pairs)),4)}
(OUT/f"E3_{aargs.out_prefix.upper()}_CROSS_JUDGE.json").write_text(json.dumps(summ,indent=2)); print(json.dumps(summ,indent=2)); raise SystemExit(0)
