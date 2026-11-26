"""Link generated factual changes to actual external captions with explicit semantic judgments."""

import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from uarcap_q2.common.io_utils import append_jsonl_durable, atomic_write_json, primitive_hash, read_jsonl, sha256_file
from uarcap_q2.final_analysis.validation_identity import index, verify_caption_pair
from uarcap_q2.interventions.neutral_repair import parse_json

PROMPT=("Audit two captions' already extracted atomic facts and the retrieved evidence. The arrays use zero-based indices. "
        "Match equivalent facts one-to-one. Preserve action specificity, entity, quantity, attributes and polarity. "
        "Do not match a related activity or a shared object as equivalent. Also report unambiguous "
        "replacement pairs where aligned alternatives describe a different value for the same event or property. "
        "Use uncertain_pairs when an equivalence or replacement cannot be resolved. For every new-caption fact classify its "
        "relationship to EACH retrieved caption, where the retrieved caption is the premise. entailed means the precise fact follows, "
        "contradicted means explicit incompatible content, unrelated includes non-entailment without conflict, and uncertain means ambiguity. "
        "Return JSON {equivalent_pairs:[[old,new]], replacement_pairs:[[old,new]], uncertain_pairs:[[old,new]], "
        "evidence_labels:[[label per evidence caption] per new fact]}. Return empty arrays where appropriate.")
LABELS={"entailed","contradicted","unrelated","uncertain"}


def parse_relations(raw, before, after, evidence_count):
    value=parse_json(raw)
    linked_before,linked_after=set(),set()
    for kind in ("equivalent_pairs","replacement_pairs","uncertain_pairs"):
        pairs=value.get(kind)
        if not isinstance(pairs,list):
            raise ValueError(f"Missing {kind}")
        for pair in pairs:
            if not isinstance(pair,list) or len(pair)!=2 or any(type(i) is not int for i in pair):
                raise ValueError("Invalid fact indices")
            i,j=pair
            if not 0<=i<len(before) or not 0<=j<len(after) or i in linked_before or j in linked_after:
                raise ValueError("Fact reused or missing from the source set")
            linked_before.add(i)
            linked_after.add(j)
    pairs=value["equivalent_pairs"]
    for i,j in pairs:
        # Guard explicit polarity changes even if the judge calls them equivalent.
        from uarcap_q2.interventions.neutral_repair import NEGATION
        if bool(NEGATION.search(before[i]))!=bool(NEGATION.search(after[j])):
            raise ValueError("Equivalence crosses explicit polarity")
    labels=value.get("evidence_labels")
    if (not isinstance(labels,list) or len(labels)!=len(after) or any(not isinstance(row,list)
         or len(row)!=evidence_count or any(x not in LABELS for x in row) for row in labels)):
        raise ValueError("Malformed fact-to-evidence matrix")
    value["removed_indices"]=sorted(set(range(len(before)))-linked_before)
    value["added_indices"]=sorted(set(range(len(after)))-linked_after)
    if len(linked_before)+len(value["removed_indices"])!=len(before) or len(linked_after)+len(value["added_indices"])!=len(after):
        raise ValueError("Non-conserving fact transition")
    return value


def job_records(pred_paths, eval_paths, retrieval):
    reranked={r["query_id"]:r["reranked_d"] for r in retrieval}
    evaluated,predicted={},{}
    for pr,er in zip(pred_paths,eval_paths,strict=True):
        for key,p in index(read_jsonl(pr)).items():
            if key in predicted:
                raise ValueError("Duplicate caption across files")
            predicted[key]=p
        for key,r in index(read_jsonl(er)).items():
            if key in evaluated:
                raise ValueError("Duplicate fact evaluation across files")
            evaluated[key]=r
    if set(evaluated)!=set(predicted):
        raise ValueError("Prediction/E2 evaluation keys differ")
    for key,row in sorted(predicted.items()):
        model,q,arm=key
        if arm=="ARM_A0":
            continue
        base=(model,q,"ARM_A0")
        verify_caption_pair(predicted[base],evaluated[base])
        verify_caption_pair(row,evaluated[key])
        trace=row["input_trace"]
        retrieved=reranked[q][:int(arm[-1])]
        if trace["text_evidence"]!=[item["caption"] for item in retrieved]:
            raise ValueError(f"{key}: retrieval trace mismatch")
        yield {"model_id":model,"query_id":q,"arm":arm,
               "before":evaluated[base]["claims"],"after":evaluated[key]["claims"],
               "caption_sha256":[evaluated[base]["caption_sha256"],evaluated[key]["caption_sha256"]],
               "reference_sha256":evaluated[key]["reference_description_sha256"],
               "evidence":[{"rank":i,"text":r["caption"],"clip_ids":r["clip_ids"],
                            "score":r["score"],"rerank_score":r.get("rerank_score")}
                           for i,r in enumerate(retrieved,1)]}


def main():
    import json
    from uarcap_q2.common.gpu_lock import GPULockConfig, acquire_cooperative_gpu_lock
    from uarcap_q2.evaluators.backends import JudgeBackendConfig, LocalVLLMJudgeBackend
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("predictions","eval"):
        parser.add_argument("--"+name,type=Path,action="append",required=True)
    for name in ("retrieval","weights","out","lock"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--gpu-uuid",required=True)
    args=parser.parse_args()
    jobs={primitive_hash(j):j for j in job_records(args.predictions,args.eval,read_jsonl(args.retrieval))}
    if not jobs:
        raise ValueError("No factuality comparisons")
    binding=primitive_hash([sorted(jobs),PROMPT,str(args.weights),
                           {p.name:sha256_file(p) for p in args.weights.glob("*config*.json")}])
    ledger=args.out/"FACT_PROVENANCE_ATTEMPTS.jsonl"
    accepted={}
    for row in read_jsonl(ledger):
        if row["binding"]!=binding or row["job_id"] not in jobs:
            raise ValueError("Fact ledger changed")
        if row["parse_status"]=="OK":
            if row["job_id"] in accepted:
                raise ValueError("Duplicate accepted fact map")
            accepted[row["job_id"]]=row["relations"]
    pending=sorted(jobs.keys()-accepted.keys())
    if pending:
        backend=LocalVLLMJudgeBackend(JudgeBackendConfig(backend_type="vllm_offline",
                model_id="RedHatAI/gemma-3-27b-it-FP8-dynamic",model_family="gemma",local_model_path=str(args.weights),
                temperature=0,max_new_tokens=1024,max_model_len=8192,gpu_memory_utilization=.85,language_model_only=True))
        with acquire_cooperative_gpu_lock(GPULockConfig(args.lock,"q2_fact_provenance",args.gpu_uuid,True,args.out)):
            for attempt in range(1,3):
                for start in range(0,len(pending),12):
                    keys=pending[start:start+12]
                    batch=[]
                    for key in keys:
                        j=jobs[key]
                        batch.append(tuple([ [r["claim"] for r in j[name]] for name in ("before","after") ]
                                           + [[r["text"] for r in j["evidence"]]]))
                    prompts=[(PROMPT,json.dumps(dict(zip(("old_facts","new_facts","evidence"),item,strict=True)),
                                                       ensure_ascii=False)) for item in batch]
                    replies=backend.complete_batch(prompts)
                    for key,raw,parts in zip(keys,replies,batch,strict=True):
                        row={"job_id":key,"binding":binding,"raw":raw,"attempt":attempt}
                        try:
                            value=parse_relations(raw,parts[0],parts[1],len(parts[2]))
                        except (ValueError,TypeError) as error:
                            row.update(parse_status="INVALID",error=str(error))
                        else:
                            row.update(parse_status="OK",relations=value)
                            accepted[key]=value
                        append_jsonl_durable(ledger,row)
                pending=sorted(jobs.keys()-accepted.keys())
    details=[]
    counts={}
    for key,j in jobs.items():
        stat=counts.setdefault(f"{j['model_id']}|{j['arm']}",Counter())
        stat["attempted_queries"]+=1
        result=accepted.get(key)
        if result is None:
            stat["missing_queries"]+=1
            continue
        stat["parsed_queries"]+=1
        stat["equivalent"]+=len(result["equivalent_pairs"])
        stat["rephrased"]+=sum(j["before"][i]["claim"]!=j["after"][k]["claim"] for i,k in result["equivalent_pairs"])
        stat["removed"]+=len(result["removed_indices"])
        stat["replaced"]+=len(result["replacement_pairs"])
        stat["uncertain"]+=len(result["uncertain_pairs"])
        for i in result["added_indices"]:
            stat["added"]+=1
            claim=j["after"][i]
            labels=result["evidence_labels"][i]
            stat[f"added_reference_{claim['label']}"]+=1
            stat["added_supported_by_top1"]+=bool(labels and labels[0]=="entailed")
            stat["added_supported_by_additional_evidence"]+=any(x=="entailed" for x in labels[1:])
            stat["added_conflicts_with_retrieved_text"]+=any(x=="contradicted" for x in labels)
        details.append({"job_id":key,**j,"relations":result,"visual_provenance":"UNRESOLVED"})
    atomic_write_json(args.out/"FACT_PROVENANCE.json",{"schema":"uarcap_q2.fact_provenance.v1","binding_sha256":binding,
                      "status":"MACHINE_ASSISTED_UNVALIDATED","counts":counts,"rows":details})
    return 0 if not pending else 2


if __name__=="__main__":
    raise SystemExit(main())
