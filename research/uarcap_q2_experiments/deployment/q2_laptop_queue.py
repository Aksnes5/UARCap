"""Compile and install the laptop tasks into the existing BN overnight scheduler."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from uarcap_q2.common.io_utils import atomic_write_json, primitive_hash, read_json, sha256_file

BASE="/home/bn/uarcap_q2_runtime/overnight/20261126"
DATA="/home/bn/uarcap_q2_runtime"
RELEASE=f"{DATA}/deploy/laptop-20261126-v1/research/uarcap_q2_experiments"
OUT=f"{DATA}/outputs/laptop-20261126-v1"
DISPLAY="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display"
JNH="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag"
CPU="/home/bn/anaconda3/envs/videollama2/bin/python"
VLLM=f"{JNH}/.venv_rag_vllm/bin/python"
METRIC=f"{DISPLAY}/repro_evcap/venv/bin/python"
GPU="/home/bn/videoufo_research/gpu.lock"
UUID="GPU-66d83fcc-2695-33a7-33aa-af30b7e3d26d"
QWEN=f"{JNH}/models/Qwen3.8-27B-FP8"
GEMMA=f"{JNH}/models_campaign/M04_gemma3_27b"
GEMMA_ID="RedHatAI/gemma-3-27b-it-FP8-dynamic"
LLAMA=f"{JNH}/models_campaign/M08_llama33_70b"
LLAMA_ID="meta-llama/Llama-3.3-70B-Instruct-W4A16"
FORMAL=f"{DATA}/outputs/formal_candidate_unvalidated"
E3=f"{DATA}/outputs/pilot_e3_exploratory"
SUPP=f"{DATA}/outputs/supplementary"
ARCH="bn_carebench_pilot_a01/overnight_20261126_artifacts"
TASKS="bn_carebench_pilot_a01/overnight_20261126/tasks"
FROZEN="2027e4267dce105d47a60c8f9eedcef5c9099bda"
SCHEMA="uarcap_overnight_queue_v1"


def build():
    tasks=[]
    def task(name, env, script, argv, inputs, artifact, expected, depends=(), gpu=False, priority=50, timeout=14400, gate=""):
        tid="q2l_v1_"+name
        cmd=[env,"-u",f"{RELEASE}/{script}",*argv]
        contract={"argv":cmd,"inputs":inputs,"output":artifact,"expected":expected,"gate":gate,
                  "script_sha256":sha256_file(ROOT/script)}
        config=primitive_hash(contract)
        if script.endswith("/supp_generate.py"):
            scope=list(cmd)
            if "--smoke-queries" in scope:
                i=scope.index("--smoke-queries")
                del scope[i:i+2]
            generation_sha=primitive_hash([scope,contract["script_sha256"]])
            cmd+=["--config-sha256",generation_sha]
        entry={"id":tid,"priority":priority,"cwd":RELEASE,"python":env,"script":script,
               "timeout_sec":timeout,"max_retries":2,"requires_gpu":gpu,"depends_on":["q2l_v1_"+d for d in depends],
               "config_sha256":config,"inputs":inputs,"output":artifact,"acceptance":expected,
               "science_gate":gate,"resume_key":"declared artifact key and input/config fingerprint",
               "failure_policy":"Retain validated records. Retry incomplete work. Isolate scientific blocks."}
        launcher=[CPU,"-u",f"{RELEASE}/deployment/q2_task_entry.py","--contract",
                  f"{RELEASE}/laptop_final_handoff_20261126/REMAINING_GPU_TASK_QUEUE.json","--task-id",tid]
        envvars=["env",f"OVERNIGHT_BASE={BASE}",f"OVERNIGHT_TASK_ID={tid}",f"PYTHONPATH={RELEASE}",
                 "VLLM_USE_FLASHINFER_SAMPLER=0","HF_HUB_OFFLINE=1","TRANSFORMERS_OFFLINE=1"]
        entry.update(argv=cmd,argv_sha256=primitive_hash(cmd),script_sha256=contract["script_sha256"],
                     cmd=shlex.join([*envvars,*launcher]))
        tasks.append(entry)
        return name

    shared=["--lock",GPU,"--gpu-uuid",UUID]
    for bridge in (False,True):
        slug="bridge" if bridge else "e3"
        prefix=f"{OUT}/{slug}"
        original=(f"{SUPP}/E3_BRIDGE_CANDIDATES_V2.json" if bridge else f"{E3}/E3_FORMAL_APPROVED.json")
        frames=(f"{FORMAL}/formal_frames" if bridge else f"{E3}/E3_FORMAL_FRAMES")
        approved=f"{prefix}/"+("E3_BRIDGE_APPROVED_V2.json" if bridge else "E3_FORMAL_APPROVED_V2.json")
        extra=(["--formal-manifest",f"{FORMAL}/FORMAL_CAREBENCH_ANALYSIS_MANIFEST.json","--retrieval",f"{FORMAL}/formal_retrieval.jsonl","--bridge"]
               if bridge else ["--videoufo-clips","/home/bn/videoufo_research/manifests/clips_full.tsv",
                                "--historical-sources",f"{RELEASE}/laptop_final_handoff_20261126/msrvtt_mapped_source_ids.txt",
                                "--pilot-exclusions",f"{RELEASE}/{ARCH}/vidhalluc/E3_PILOT_EXCLUSION.json"])
        proposal=f"{prefix}/proposals.json"
        screened=f"{prefix}/screened.json"
        jobs=f"{prefix}/jobs.json"
        decoded=f"{prefix}/decoded.json"
        verdicts=f"{prefix}/visual_verdicts.jsonl"
        priority=70 if bridge else 10
        prepared=task(slug+"_propose",VLLM,"scripts/e3_repair.py",
            ["prepare","--candidates",original,"--frames",frames,"--vicuna",f"{DISPLAY}/repro_evcap/models/vicuna-7b-v1.3",
             "--qwen",QWEN,"--gemma-id",GEMMA_ID,"--out",proposal,*extra],
            [original,frames,*[x for x in extra if x.startswith("/")]],proposal,
            {"kind":"json","schema":"uarcap_q2.e3_neutral_proposals.v2"},priority=priority,
            gate="Frozen cohort, official source evidence and local tokenizers")
        screening=task(slug+"_text",VLLM,"scripts/e3_repair.py",
            ["screen","--proposals",proposal,"--gemma",GEMMA,"--gemma-id",GEMMA_ID,"--out",screened,"--jobs",jobs,*shared],
            [proposal,GEMMA],jobs,{"kind":"json","schema":"uarcap_q2.blind_statement_jobs.v1"},[prepared],
            gpu=True,priority=priority+1)
        decoding=task(slug+"_decode",CPU,"scripts/vlm_statement_verify.py",
            ["decode","--jobs",jobs,"--work",f"{prefix}/dense","--out",decoded],[jobs],decoded,
            {"kind":"json","schema":"uarcap_q2.e3_video_decode.v1"},[screening],priority=priority+2)
        verification=task(slug+"_vision",VLLM,"scripts/vlm_statement_verify.py",
            ["verify","--jobs",jobs,"--frames",decoded,"--qwen",QWEN,"--out",verdicts,*shared],
            [jobs,decoded,QWEN],verdicts,{"kind":"comparison_ledger","jobs":jobs},[decoding],gpu=True,priority=priority+3,timeout=43200)
        freezing=task(slug+"_freeze",CPU,"scripts/e3_repair.py",
            ["freeze","--candidates",original,"--screened",screened,"--verdicts",verdicts,"--out",approved,
             *(["--bridge"] if bridge else [])],[original,screened,verdicts],approved,
            {"kind":"json","schema":"uarcap_q2.e3_neutral_approved.v2","eligible":True},[verification],priority=priority+4,
            gate=f"At least {30 if bridge else 150} fully screened distinct candidates")
        generation=[]
        for model,env in (("videollama",CPU),("qwen",VLLM)):
            folder=f"{prefix}/{model}"
            reuse=["--reuse-predictions",f"{E3}/formal_videollama/predictions.jsonl"] if not bridge and model=="videollama" else []
            generation.append(task(f"{slug}_{model}",env,f"{TASKS}/e3_formal_generate.py",
                ["--generator",model,"--approved",approved,"--frames-dir",frames,"--out-dir",folder,"--min-approved",
                 "30" if bridge else "150",*reuse],[approved,frames],f"{folder}/run_receipt.json",
                {"kind":"generation","dynamic_cohort":approved,"arms":4},[freezing],gpu=True,priority=priority+5,
                gate="V2 freeze and exact generation evidence traces"))
        evals=[]
        evfolder=f"{prefix}/evaluation"
        pred=[f"{prefix}/videollama/predictions.jsonl",f"{prefix}/qwen/predictions.jsonl"]
        common=["--approved",approved,"--old",pred[0],"--qwen",pred[1],"--out",evfolder]
        for judge,weight in (("gemma",GEMMA),("llama",LLAMA)):
            evals.append(task(f"{slug}_judge_{judge}",VLLM,"scripts/evaluate_e3_v2.py",
                [*common,"--judge",judge,"--weights",weight,*shared],[*pred,approved,weight],
                f"{evfolder}/{judge}.jsonl",{"kind":"e3_evaluation","dynamic_cohort":approved},
                generation,gpu=True,priority=priority+6))
        task(f"{slug}_analyze",CPU,"scripts/evaluate_e3_v2.py",[*common,"--judge","summarize"],[*pred,approved],
            f"{evfolder}/E3_ANALYSIS.json",{"kind":"json","schema":"uarcap_q2.e3_analysis.v2"},evals,priority=priority+7)

    seed=f"{RELEASE}/{ARCH}/formal/cross_qwen/A2_CROSS_JUDGE_LABELS.jsonl"
    cross_args=["--model-path",LLAMA,"--model-id",LLAMA_ID,"--model-family","llama","--gpu-mem","0.95","--chunk","8"]
    qwenfix=task("qwen_cross_single",VLLM,f"{TASKS}/a2_cross_judge.py",
        ["--a2-dir",f"{FORMAL}/qwen_eval","--out-dir",f"{OUT}/formal_qwen_cross",*cross_args,
         "--live-labels",f"{FORMAL}/qwen_machine_cross/A2_CROSS_JUDGE_LABELS.jsonl",
         "--seed-labels",seed,"--seed-sha256","cb035005f7d8408e3b8b57bf9200838aaa260e6159911b2938a9a8512ee5e756"],
        [f"{FORMAL}/qwen_eval/a2_ref_fact_eval.jsonl",seed],f"{OUT}/formal_qwen_cross/A2_CROSS_JUDGE_LABELS.jsonl",
        {"kind":"rows","count":5099,"keys":["query_id","norm_key"]},gpu=True,priority=30,timeout=1800,
        gate="Reuse 5098 hash-verified labels. Score only the missing key if still absent")

    budget={}
    sensitivities=[]
    def supplement(slug,model,arms,n,extra=(),dependencies=(),smoke=False,priority=40):
        env=CPU if model=="videollama" else VLLM
        path=f"{OUT}/{slug}/{model}"
        gen=[ "--generator",model,"--arms",",".join(arms),"--condition",slug,"--out-dir",path,*extra]
        inputs=[f"{FORMAL}/{model}/predictions.jsonl",f"{FORMAL}/formal_retrieval.jsonl",
                *[extra[i+1] for i,key in enumerate(extra) if key in ("--qids-file","--decoded-frames")]]
        expected={"kind":"generation","count":n*len(arms),"arms":len(arms)}
        if smoke:
            s=task(f"{slug}_{model}_smoke",env,f"{TASKS}/supp_generate.py",[*gen,"--smoke-queries","2"],
                   inputs,f"{path}/smoke_receipt.json",{"kind":"generation","count":2*len(arms),"smoke":True},
                   dependencies,gpu=True,priority=priority,timeout=3600)
            dependencies=[*dependencies,s]
        g=task(f"{slug}_{model}",env,f"{TASKS}/supp_generate.py",gen,inputs,f"{path}/run_receipt.json",expected,
               dependencies,gpu=True,priority=priority+1,gate="Real query frame and natural prefix parity")
        e=task(f"{slug}_{model}_gemma",VLLM,f"{TASKS}/eval_captions_gemma.py",
               ["--predictions",f"{path}/predictions.jsonl","--core-lock",f"{FORMAL}/formal_core_lock.json",
                "--out-dir",f"{path}/eval","--model-path",GEMMA],
               [f"{path}/predictions.jsonl",f"{FORMAL}/formal_core_lock.json",GEMMA],f"{path}/eval/a2_ref_fact_eval.jsonl",
               {"kind":"rows","count":expected["count"],"keys":["model_id","query_id","arm"],"parse_status":"OK"},
               [g],gpu=True,priority=priority+2)
        x=task(f"{slug}_{model}_llama",VLLM,f"{TASKS}/a2_cross_judge.py",
               ["--a2-dir",f"{path}/eval","--out-dir",f"{path}/cross",*cross_args],
               [f"{path}/eval/a2_ref_fact_eval.jsonl",LLAMA],f"{path}/cross/A2_CROSS_JUDGE_AGREEMENT.json",
               {"kind":"cross","eval":f"{path}/eval/a2_ref_fact_eval.jsonl"},[e],gpu=True,priority=priority+3)
        return (e,x,path)

    for model in ("videollama","qwen"):
        budget[model]=supplement("budget",model,["ARM_A3","ARM_A5"],600,smoke=True)

    metrics=task("budget_metrics",METRIC,"scripts/e2_metric_bridge.py",
         ["--formal",FORMAL,"--out-dir",f"{OUT}/metrics","--extension-root",f"{OUT}/budget","--skip-spice"],
         [f"{FORMAL}/formal_candidates.jsonl"],f"{OUT}/metrics/E2_METRIC_BRIDGE_RECEIPT.json",
         {"kind":"metric","count":12000},[v[1] for v in budget.values()],priority=47)
    src_pred=[]
    src_eval=[]
    for model,(_,_,path) in budget.items():
        src_pred+=["--predictions",f"{FORMAL}/{model}/predictions.jsonl","--predictions",f"{path}/predictions.jsonl"]
        src_eval+=["--eval",f"{FORMAL}/{model}_eval/a2_ref_fact_eval.jsonl","--eval",f"{path}/eval/a2_ref_fact_eval.jsonl"]
    task("fact_provenance",VLLM,"scripts/fact_provenance.py",
          [*src_pred,*src_eval,"--retrieval",f"{FORMAL}/formal_retrieval.jsonl","--weights",GEMMA,"--out",f"{OUT}/provenance",*shared],
          [src_pred[1],src_pred[3],src_pred[5],src_pred[7],src_eval[1],src_eval[3],src_eval[5],src_eval[7]],
          f"{OUT}/provenance/FACT_PROVENANCE.json",{"kind":"json","schema":"uarcap_q2.fact_provenance.v1"},
          [v[0] for v in budget.values()],gpu=True,priority=60,timeout=43200)

    frozen_plans=f"{RELEASE}/{ARCH}/supplementary"
    sensitivities.append(supplement("length","qwen",["ARM_A0","ARM_A1","ARM_A7"],120,
               ["--qids-file",f"{frozen_plans}/LENGTH_CONTROL_PLAN.json","--prompt-regime","detailed_sensitivity"],priority=100))
    supplement("modality","videollama",["T1V0","T7V0","T7V1","T1V7"],40,
               ["--qids-file",f"{frozen_plans}/LENGTH_CONTROL_PLAN.json","--modality-ablation","--limit","40"],smoke=True,priority=110)
    for n in (32,64):
        plan=f"{frozen_plans}/VISUAL_SAMPLING_PLAN.json"
        dec=f"{OUT}/visual{n}/decoded.json"
        d=task(f"visual{n}_decode",CPU,"scripts/prepare_q2_frames.py",
               ["--plan",plan,"--candidates",f"{FORMAL}/formal_candidates.jsonl","--frame-root",f"{OUT}/visual{n}/frames",
                "--out",dec,"--num-frames",str(n)],[plan,f"{FORMAL}/formal_candidates.jsonl"],dec,
               {"kind":"json","schema":"uarcap_q2.qwen_sampling_decode.v1"},priority=100)
        sensitivities.append(supplement(f"visual{n}","qwen",["ARM_A7"],150,
                   ["--qids-file",plan,"--num-frames",str(n),"--decoded-frames",dec],[d],priority=101))

    def reporting(name, extensions, output_metric, deps):
        pp=[]
        for model,labels in (("videollama",f"{FORMAL}/videollama_machine_cross"),("qwen",f"{OUT}/formal_qwen_cross")):
            pp.extend(["--predictions",f"{FORMAL}/{model}/predictions.jsonl","--eval",f"{FORMAL}/{model}_eval/a2_ref_fact_eval.jsonl",
                       "--cross",f"{labels}/A2_CROSS_JUDGE_LABELS.jsonl"])
        for _,_,path in extensions:
            pp.extend(["--predictions",f"{path}/predictions.jsonl","--eval",f"{path}/eval/a2_ref_fact_eval.jsonl",
                       "--cross",f"{path}/cross/A2_CROSS_JUDGE_LABELS.jsonl"])
        task(name,CPU,"scripts/q2_budget_report.py",[*pp,"--metric",output_metric,"--out",f"{OUT}/{name}"],
             [pp[k] for k in range(1,len(pp),2)]+[output_metric],f"{OUT}/{name}/Q2_BUDGET_REPORT.json",
             {"kind":"json","schema":"uarcap_q2.budget_report.v1"},deps,priority=125)
    reporting("budget_report",list(budget.values()),f"{OUT}/metrics/E2_METRIC_BRIDGE_PER_VIDEO.jsonl",[metrics,qwenfix])
    reporting("sensitivity_report",[*budget.values(),*sensitivities],f"{OUT}/metrics/E2_METRIC_BRIDGE_PER_VIDEO.jsonl",
              [metrics,qwenfix,*[t[1] for t in sensitivities]])
    history=f"{OUT}/historical"
    meta=f"{DATA}/data/source_witness/MSRVTT_data.json"
    ziproot=str(PurePosixPath(RELEASE).parents[1]/"research/uarcap_v06_followup_deployment/bn_v06_full_ab/raw_zips")
    archived_ids=str(PurePosixPath(RELEASE).parents[1]/"manifests/msrvtt_untouched2690.tsv")
    hist=task("historical_stage",CPU,"scripts/historical_fact_bridge.py",
              ["--metadata",meta,"--test-ids",archived_ids,"--raw-zips",ziproot,"--out",history],
              [meta,archived_ids,ziproot],f"{history}/HISTORICAL_BRIDGE_MANIFEST.json",
              {"kind":"json","schema":"uarcap_q2.historical_bridge_manifest.v1"},priority=115,
              gate="Archived A-arm checksums and public source metadata")
    hist_eval=task("historical_gemma",VLLM,f"{TASKS}/eval_captions_gemma.py",
              ["--predictions",f"{history}/predictions.jsonl","--core-lock",f"{history}/historical_core_lock.json",
               "--out-dir",f"{history}/eval","--model-path",GEMMA,"--dataset-label","MSR-VTT"],
              [f"{history}/predictions.jsonl",f"{history}/historical_core_lock.json",GEMMA],f"{history}/eval/a2_ref_fact_eval.jsonl",
              {"kind":"rows","count":180,"keys":["model_id","query_id","arm"],"parse_status":"OK"},[hist],gpu=True,priority=116)
    task("historical_llama",VLLM,f"{TASKS}/a2_cross_judge.py",
         ["--a2-dir",f"{history}/eval","--out-dir",f"{history}/cross",*cross_args],[f"{history}/eval/a2_ref_fact_eval.jsonl",LLAMA],
         f"{history}/cross/A2_CROSS_JUDGE_AGREEMENT.json",{"kind":"cross","eval":f"{history}/eval/a2_ref_fact_eval.jsonl"},
         [hist_eval],gpu=True,priority=117)
    sig=f"{DATA}/data/source_witness"
    task("visual_risk",CPU,"scripts/source_and_visual_risk.py",
        ["visual","--query-signatures",f"{sig}/carebench_ordered_dhash.jsonl","--clip-signatures",f"{sig}/videoufo_top7_ordered_dhash.jsonl",
         "--retrieval",f"{FORMAL}/formal_retrieval.jsonl","--out",f"{OUT}/visual_risk.json"],
        [f"{sig}/carebench_ordered_dhash.jsonl",f"{sig}/videoufo_top7_ordered_dhash.jsonl",f"{FORMAL}/formal_retrieval.jsonl"],
        f"{OUT}/visual_risk.json",{"kind":"json","schema":"uarcap_q2.visual_risk.v1"},priority=119,
        gate="Existing aligned perceptual signatures; missing inputs remain unresolved")

    e=f"{OUT}/retrieval_relevance"
    ep=f"{e}/evidence.jsonl"
    ee=f"{e}/eval/a2_ref_fact_eval.jsonl"
    evidence_stage=task("retrieval_relevance_stage",CPU,"scripts/retrieved_fact_relevance.py",
                      ["stage","--retrieval",f"{FORMAL}/formal_retrieval.jsonl","--out",ep],
                      [f"{FORMAL}/formal_retrieval.jsonl"],ep,{"kind":"rows","count":4200,"keys":["model_id","query_id","arm"]},priority=127)
    evidence_judge=task("retrieval_relevance_gemma",VLLM,f"{TASKS}/eval_captions_gemma.py",
                      ["--predictions",ep,"--core-lock",f"{FORMAL}/formal_core_lock.json","--out-dir",f"{e}/eval","--model-path",GEMMA],
                      [ep,f"{FORMAL}/formal_core_lock.json",GEMMA],ee,
                      {"kind":"rows","count":4200,"keys":["model_id","query_id","arm"],"parse_status":"OK"},
                      [evidence_stage],gpu=True,priority=128,timeout=43200)
    task("retrieval_relevance_report",CPU,"scripts/retrieved_fact_relevance.py",
         ["summarize","--predictions",ep,"--eval",ee,"--out",f"{e}/SEMANTIC_RELEVANCE.json"],
         [ep,ee],f"{e}/SEMANTIC_RELEVANCE.json",{"kind":"json","schema":"uarcap_q2.retrieved_fact_relevance.v1"},
         [evidence_judge],priority=129)
    return {"schema":SCHEMA,"version":"laptop-20261126-v1","base_commit":FROZEN,"tasks":tasks}


def validate(tasks):
    seen={}
    for t in tasks:
        tid=t["id"]
        if tid in seen or not tid or type(t["requires_gpu"]) is not bool or t["timeout_sec"]<=0:
            raise ValueError(f"Invalid or duplicate task: {tid}")
        seen[tid]=t
    def visit(tid,active,done):
        if tid in active or tid not in seen:
            raise ValueError(f"Unknown or cyclic dependency: {tid}")
        if tid in done:
            return
        for dep in seen[tid].get("depends_on",[]):
            visit(dep,active|{tid},done)
        done.add(tid)
    done=set()
    for tid in seen:
        visit(tid,set(),done)


def merge(existing, new):
    validate(existing)
    by={r["id"]:r for r in existing}
    for task in new:
        if task["id"] in by and by[task["id"]]!=task:
            raise ValueError(f"Existing task ID differs: {task['id']}")
        by.setdefault(task["id"],task)
    merged=list(by.values())
    validate(merged)
    return merged


SUPERSEDED={"e3_formal_generate_videollama","e3_formal_generate_qwen","e3_formal_evaluate",
            "e3_bridge_prepare","e3_bridge_generate_videollama","e3_bridge_generate_qwen","e3_bridge_evaluate",
            "formal_cross_qwen","formal_cross_qwen_v2"}


def preserve_superseded(old,state):
    tasks=old["tasks"]
    keys={t["id"] for t in tasks}&SUPERSEDED
    dependent={dep for t in tasks if t["id"] not in keys for dep in t.get("depends_on",[])}
    retired=keys-dependent
    if any(state["tasks"].get(k,{}).get("status")=="RUNNING" for k in retired):
        raise ValueError("Wait for existing E3 or cross-judge task to finish before superseding")
    return [t for t in tasks if t["id"] not in retired],[t for t in tasks if t["id"] in retired]


def process_ids(base,script):
    found=[]
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                cmd=(p/"cmdline").read_bytes().decode(errors="replace").split("\x00")
                cwd=os.readlink(p/"cwd")
            except OSError:
                continue
            match=[v for v in cmd[1:3] if v.endswith(".py") and Path(v).name==script]
            if match and (str(base) in cmd or cwd==str(base)):
                found.append(int(p.name))
    return found


def deployment(args):
    if os.name!="posix" or ROOT.resolve()!=Path(RELEASE).resolve():
        raise ValueError("Install from the isolated BN release checkout")
    import fcntl
    base=Path(BASE)
    with (base/"q2_laptop_install.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (base/"STOP_REQUESTED").exists():
            raise ValueError("STOP_REQUESTED is active")
        head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        if head!=args.commit:
            raise ValueError("Deployment checkout does not match the requested full commit")
        if subprocess.check_output(["git","diff","HEAD","--name-only"],cwd=ROOT,text=True).strip():
            raise ValueError("Deployment checkout contains tracked edits")
        queue=base/"task_queue.json"
        old=read_json(queue)
        if old["schema"]!=SCHEMA:
            raise ValueError("Unexpected scheduler queue schema")
        live={name:process_ids(base,name) for name in ("run_overnight.py","overnight_supervisor.py","report_writer.py")}
        if any(len(pids)>1 for pids in live.values()):
            raise ValueError("Duplicate scheduler processes detected")
        generated=build()
        deployed=read_json(ROOT/"laptop_final_handoff_20261126/REMAINING_GPU_TASK_QUEUE.json")
        if generated!=deployed:
            raise ValueError("Versioned queue does not match release code")
        state=read_json(base/"task_state.json")
        active,retired=preserve_superseded(old,state)
        new=merge(active,deployed["tasks"])
        print(json.dumps({"running":live,"existing":len(old["tasks"]),"after_merge":len(new),"superseded":[t["id"] for t in retired],
                          "queue_sha256":sha256_file(queue)},indent=2))
        if args.dry_run:
            return
        if not (args.install or args.rollback):
            raise ValueError("An explicit BN install action is required")
        if not args.gpu_approval or not args.gpu_approval.is_file():
            raise ValueError("BN operator authorization receipt required")
        auth=read_json(args.gpu_approval)
        if auth.get("authorized") is not True or auth.get("commit")!=head or not auth.get("approved_by"):
            raise ValueError("Authorization does not bind this release")
        if not live["run_overnight.py"] and any(v.get("status")=="RUNNING" for k,v in state["tasks"].items()
                                                if k!="__orchestrator__"):
            raise ValueError("Stale or detached RUNNING task. Inspect before restarting scheduler")
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        backup=base/f"task_queue.before_q2l_{stamp}.json"
        from uarcap_q2.common.io_utils import atomic_write_bytes
        atomic_write_bytes(backup,queue.read_bytes())
        installed={**old,"updated_utc":datetime.now(timezone.utc).isoformat(),"tasks":new,
                   "superseded_task_records":[*old.get("superseded_task_records",[]),*retired]}
        if args.rollback:
            if any(v.get("status")=="RUNNING" for k,v in state["tasks"].items() if k.startswith("q2l_v1_")):
                raise ValueError("Wait for the current release task to finish before withdrawal")
            installed["tasks"]=[t for t in old["tasks"] if not t["id"].startswith("q2l_v1_")]
            installed["withdrawn_task_records"]=[t for t in old["tasks"] if t["id"].startswith("q2l_v1_")]
            installed["superseded_task_records"]=old.get("superseded_task_records",[])
            validate(installed["tasks"])
        atomic_write_json(queue,installed)
        if args.rollback:
            return
        for name,pids in live.items():
            if pids:
                continue
            suffix=["--base",str(base)] if name!="report_writer.py" else []
            if name=="overnight_supervisor.py":
                suffix+=["--check-interval","1800","--heartbeat","60","--duration-hours","48"]
            with (base/("q2l_v1_"+name+".log")).open("ab") as log:
                subprocess.Popen(["/usr/bin/python3","-u",str(base/name),*suffix],cwd=base,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emit",type=Path)
    p.add_argument("--commit")
    mode=p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run",action="store_true")
    mode.add_argument("--install",action="store_true")
    mode.add_argument("--rollback",action="store_true")
    p.add_argument("--gpu-approval",type=Path)
    a=p.parse_args()
    if a.emit:
        data=build()
        validate(data["tasks"])
        atomic_write_json(a.emit,data)
    else:
        deployment(a)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
