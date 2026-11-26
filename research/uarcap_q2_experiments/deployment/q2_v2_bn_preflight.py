"""Report exact BN V2 paths and interpreter capabilities without model loading."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from q2_laptop_queue import BASE, CPU, DATA, E3, FORMAL, GEMMA, GPU, LLAMA, METRIC, QWEN, ROOT, VLLM
from uarcap_q2.common.io_utils import atomic_write_json


def check():
    groups = {"common":[Path(BASE)/"task_queue.json",Path(BASE)/"task_state.json",Path(GPU)],
              "formal":[Path(FORMAL)/name for name in ("formal_candidates.jsonl","formal_retrieval.jsonl",
                   "formal_core_lock.json","formal_frame_indices.jsonl","FORMAL_CAREBENCH_ANALYSIS_MANIFEST.json",
                   "videollama/predictions.jsonl","qwen/predictions.jsonl")],
              "e3":[Path(E3)/"E3_FORMAL_APPROVED.json",Path(E3)/"E3_FORMAL_FRAMES"],
              "e3_optional_reserve":[Path(DATA)/"data/vidhalluc/ach_mcq.json",
                   Path(DATA)/"data/source_witness/vidhalluc_debug_source_ids.txt",
                   Path(DATA)/"deploy/laptop-v2-e3-reserve-approval.json"],
              "visual_clip_location":[Path(DATA)/"data/source_witness/videoufo_clip_locations.jsonl"],
              "historical_source_witness":[Path(DATA)/"data/source_witness/MSRVTT_data.json"],
              "model_paths":[Path(path)/"config.json" for path in (GEMMA,LLAMA,QWEN)]}
    assets = {k:{str(p):p.exists() for p in files} for k,files in groups.items()}
    modules = {CPU:["numpy","PIL","yaml","av","scipy","matplotlib"],VLLM:["transformers","vllm","PIL"],
               METRIC:["pycocoevalcap","scipy","numpy"]}
    runtime = {}
    expr = "import importlib.util,json,sys;print(json.dumps({'version':sys.version.split()[0], 'modules':{m:importlib.util.find_spec(m) is not None for m in sys.argv[1:]}}))"
    for path, required in modules.items():
        if Path(path).is_file():
            p = subprocess.run([path,"-c",expr,*required],capture_output=True,text=True,timeout=30)
            runtime[path] = json.loads(p.stdout) if p.returncode==0 else {"error":p.stderr[-600:]}
        else:
            runtime[path] = {"available":False}
    return {"schema":"uarcap_q2.bn_v2_metadata_preflight.v1","assets":assets,"runtime":runtime,
            "commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "preflight_only":True,"gpu_smoke_executed":False,"model_loading_executed":False}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path)
    a=parser.parse_args()
    result=check()
    if a.out:
        atomic_write_json(a.out,result)
    print(json.dumps(result,indent=2))
    if not all(result["assets"]["common"].values()):
        raise SystemExit(3)
