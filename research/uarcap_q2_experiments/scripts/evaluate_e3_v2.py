"""Score two independent E3 targets per caption with a version-bound append-only ledger."""

import argparse
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from uarcap_q2.common.io_utils import (append_jsonl_durable, atomic_write_json, primitive_hash,
                                        read_json, read_jsonl, sha256_file)
from uarcap_q2.interventions.neutral_repair import V2, parse_expressed
from uarcap_q2.interventions.e3_output import ARMS, collect_generation, target_text
from uarcap_q2.final_analysis.core import cluster_bootstrap

MODELS=("videollama_evcap_frozen_v08","Qwen/Qwen3.8-27B-FP8")
JUDGES=("RedHatAI/gemma-3-27b-it-FP8-dynamic","meta-llama/Llama-3.3-70B-Instruct-W4A16")
TARGETS=("f_plus","f_minus")
SYSTEM=("Decide whether the caption expresses the complete target proposition. Use AFFIRMED only for an unhedged "
        "affirmative assertion with the same concrete action, entities, polarity and relevant attributes. A generic "
        "action or related object is VAGUE. Use NEGATED, ABSENT or UNVERIFIABLE where appropriate. "
        "Return strict JSON with status, expressed (1 for AFFIRMED, null for UNVERIFIABLE, 0 otherwise), "
        "and evidence copied from the caption for AFFIRMED, NEGATED and VAGUE.")


def panel(rows, expected, judge):
    latest={}
    for row in rows:
        if row["judge"]!=judge:
            continue
        key=tuple(row[k] for k in ("model_id","query_id","arm","target_id"))
        if key not in expected:
            raise ValueError("Out-of-cohort evaluation")
        if row["parse_status"]=="OK":
            if key in latest:
                raise ValueError("Duplicate accepted target result")
            latest[key]=row
    return latest


def describe_panels(evaluations):
    output={}
    for judge, rows in evaluations.items():
        for model in MODELS:
            qids=sorted({k[1] for k in rows if k[0]==model})
            for target in TARGETS:
                values=defaultdict(dict)
                for q in qids:
                    for arm in ARMS:
                        row=rows.get((model,q,arm,target),{})
                        value=row.get("expressed")
                        if value is not None:
                            values[arm][q]=value
                complete=set.intersection(*(set(values[a]) for a in ARMS))
                result={"n_target_rows":len(qids)*4,"complete_four_arm_queries":len(complete),
                        "n_unusable_or_unavailable":len(qids)-len(complete),
                        "arms":{a:{"n":len(values[a]),"n_affirmed":sum(values[a].values())} for a in ARMS},
                        "paired_exploratory":{}}
                for low,high in (("E3_NEU","E3_CON"),("E3_BASE","E3_CON"),("E3_NEU","E3_SUP")):
                    diffs={q:values[high][q]-values[low][q] for q in complete}
                    result["paired_exploratory"][f"{high}-{low}"]=cluster_bootstrap(diffs)
                output[f"{judge}|{model}|{target}"]=result
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("approved","old","qwen","out"):
        ap.add_argument("--"+k,type=Path,required=True)
    for k in ("lock","weights"):
        ap.add_argument("--"+k,type=Path)
    ap.add_argument("--gpu-uuid")
    ap.add_argument("--judge",choices=("gemma","llama","summarize"),required=True)
    args=ap.parse_args()
    frozen=read_json(args.approved)
    if frozen.get("schema")!=V2 or frozen.get("eligible") is not True:
        ap.exit(3,"HARD_BLOCK missing eligible V2 freeze\n")
    bridge=frozen["scope"]=="CAREB_BRIDGE"
    approved=frozen["approved"]
    if len(approved)<(30 if bridge else 150):
        ap.exit(3,"HARD_BLOCK insufficient approved candidates\n")
    candidates={r["query_id"]:r for r in approved}
    pred={}
    paths=[args.old,args.qwen]
    for path, model in zip(paths,MODELS,strict=True):
        receipt=read_json(path.parent/"run_receipt.json")
        if (not receipt.get("complete") or receipt["cohort_sha256"]!=primitive_hash(frozen)):
            raise ValueError("E3 generation receipt does not match freeze")
        rows=read_jsonl(path)
        if any(row.get("binding_sha256")!=receipt["binding_sha256"] for row in rows):
            raise ValueError("E3 row receipt mismatch")
        pred.update(collect_generation(rows,approved,model,bridge))
    binding=primitive_hash([primitive_hash(frozen),[sha256_file(p) for p in paths],SYSTEM])
    jobs={}
    for (model,q,arm),r in pred.items():
        for target,text in zip(TARGETS,target_text(candidates[q],bridge),strict=True):
            jobs[(model,q,arm,target)]=(r["caption"],text)
    output=args.out
    hist={j:output/(name+".jsonl") for j,name in zip(JUDGES,("gemma","llama"),strict=True)}
    def verified_history(judge):
        rows=read_jsonl(hist[judge])
        if any(row["binding_sha256"]!=binding or row["judge"]!=judge for row in rows):
            raise ValueError("E3 evaluation provenance mismatch")
        return panel(rows,jobs,judge)
    if args.judge=="summarize":
        panels={j:verified_history(j) for j in JUDGES}
        out={"schema":"uarcap_q2.e3_analysis.v2","binding_sha256":binding,
             "scope":frozen["scope"],"status":"MACHINE_ASSISTED_UNVALIDATED","n_candidates":len(candidates),
             "n_expected_per_judge":len(jobs), "parsed_by_judge":{j:len(p) for j,p in panels.items()},
             "results":describe_panels(panels)}
        atomic_write_json(output/"E3_ANALYSIS.json",out)
        return 0 if all(len(p)==len(jobs) for p in panels.values()) else 2

    from uarcap_q2.common.gpu_lock import GPULockConfig, acquire_cooperative_gpu_lock
    from uarcap_q2.evaluators.backends import JudgeBackendConfig, LocalVLLMJudgeBackend
    judge=JUDGES[0 if args.judge=="gemma" else 1]
    done=verified_history(judge)
    engine=LocalVLLMJudgeBackend(JudgeBackendConfig(backend_type="vllm_offline",model_id=judge,
            model_family=args.judge,local_model_path=str(args.weights),temperature=0,max_new_tokens=256,
            max_model_len=2048,gpu_memory_utilization=.85 if args.judge=="gemma" else .95,
            language_model_only=args.judge=="gemma"))
    pending=sorted(set(jobs)-done.keys())
    if not pending:
        return 0
    with acquire_cooperative_gpu_lock(GPULockConfig(args.lock,"q2_e3_dual_target",args.gpu_uuid,True,output)):
        for attempt in range(1,3):
            for i in range(0,len(pending),16):
                batch=pending[i:i+16]
                prompts=[(SYSTEM,f"Caption: {jobs[k][0]}\nTarget: {jobs[k][1]}") for k in batch]
                for key,raw in zip(batch,engine.complete_batch(prompts),strict=True):
                    row=dict(zip(("model_id","query_id","arm","target_id"),key,strict=True))
                    row.update(binding_sha256=binding,judge=judge,attempt=attempt,response=raw,expressed=None)
                    try:
                        row.update(parse_expressed(raw),parse_status="OK")
                    except ValueError as error:
                        row.update(parse_status="INVALID",error=str(error))
                    append_jsonl_durable(hist[judge],row)
                    if row["parse_status"]=="OK":
                        done[key]=row
            pending=sorted(set(jobs)-done.keys())
    return 0 if not pending else 2


if __name__=="__main__":
    raise SystemExit(main())
