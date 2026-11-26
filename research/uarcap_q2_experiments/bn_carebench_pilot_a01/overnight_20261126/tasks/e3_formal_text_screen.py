#!/usr/bin/env python3
"""Machine-assisted text screening of the 180 formal E3 candidates (unvalidated)."""
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
BASE=Path(os.environ.get("OVERNIGHT_BASE","/home/bn/uarcap_q2_runtime/overnight/20261126"))
sys.path.insert(0,"/home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments")
sys.path.insert(0,str(Path(__file__).parent))
from vidhalluc_pipeline import _gemma_backend, _parse_json
from uarcap_q2.common.gpu_lock import GPULockConfig, acquire_cooperative_gpu_lock
W=Path("/home/bn/uarcap_q2_runtime/outputs/pilot_e3_exploratory")
rows=json.loads((W/"E3_FORMAL_CANDIDATES_180.json").read_text())["candidates"]
out=W/"E3_FORMAL_TEXT_SCREEN.jsonl"
done=set()
if out.is_file():
    for l in out.read_text().splitlines():
        if l.strip(): done.add(json.loads(l)["query_id"])
pending=[c for c in rows if c["query_id"] not in done]
sysp=("Check whether a correct and a conflict action pair is suitable for a controlled intervention. "
      "They must be mutually exclusive, natural English, similar length, and the conflict must be a clear different action "
      '(not a synonym/substep). Output JSON {"mutually_exclusive":true/false,"natural":true/false,"length_ok":true/false,'
      '"conflict_plausibly_absent":true/false,"ambiguity":"..."}')
lock=GPULockConfig(Path("/home/bn/videoufo_research/gpu.lock"),"e3_formal_text_screen","GPU-66d83fcc-2695-33a7-33aa-af30b7e3d26d",True,W)
with acquire_cooperative_gpu_lock(lock):
    b=_gemma_backend()
    for i in range(0,len(pending),12):
        batch=pending[i:i+12]
        resps=b.complete_batch([(sysp,f"TRUE ACTION: {c['true_action']}\nCONFLICT ACTION: {c['conflict_action']}\nQUESTION: {c.get('question','')}") for c in batch])
        for c,r in zip(batch,resps):
            try: p=_parse_json(r); ok=True
            except Exception as e: p={"parse_error":str(e)}; ok=False
            with out.open("a") as f: f.write(json.dumps({"query_id":c["query_id"],"screen":p,"parse_ok":ok,"raw":r[:3000],"model":"gemma-3-27b-fp8"},ensure_ascii=False)+"\n")
    try: b._engine=None
    except Exception: pass
print(json.dumps({"n_total":len(rows),"n_screened":len(done)+len(pending),"out":str(out)},indent=2))
