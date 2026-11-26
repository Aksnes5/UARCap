"""Describe measured budget tradeoffs and exploratory paired Qwen sensitivities."""

import argparse
from collections import Counter,defaultdict
from pathlib import Path
import re
from statistics import mean
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from uarcap_q2.common.io_utils import atomic_write_json,read_json,read_jsonl
from uarcap_q2.final_analysis.core import cluster_bootstrap
from uarcap_q2.final_analysis.reporting import export_table
from uarcap_q2.final_analysis.validation_identity import index,verify_caption_pair

CONDITIONS={"length":("ARM_A0","ARM_A1","ARM_A7"),"visual32":("ARM_A7",),"visual64":("ARM_A7",)}


def norm(text):
    return " ".join(w for w in re.sub(r"[^a-z0-9'\s]"," ",text.lower()).split() if w not in ("a","an","the"))


def report_rows(pred_paths,eval_paths,cross_paths,metric_paths):
    results=[]
    for pred_path,eval_path,cross_path in zip(pred_paths,eval_paths,cross_paths,strict=True):
        pred,evaluations=index(read_jsonl(pred_path)),index(read_jsonl(eval_path))
        cross={(r["query_id"],r["norm_key"]):r["llama_label"] for r in read_jsonl(cross_path)}
        if set(pred)!=set(evaluations):
            raise ValueError("Prediction/evaluation grid differs")
        subset=[]
        for k in sorted(pred):
            p,r=pred[k],evaluations[k]
            claims=verify_caption_pair(p,r)
            labels=[cross.get((k[1],norm(c["claim"]))) for c in claims]
            tokens=p["input_trace"]["actual_processor_telemetry"].get("prompt_token_count")
            words=len(r["caption"].split())
            metric={field:r.get(field) for field in ("n_entail_ref","n_contra_ref","n_unment_ref","r_core_ref")}
            subset.append({"model_id":k[0],"query_id":k[1],"arm":k[2],"condition":p.get("condition","formal"),
                           "words":words,"sentences":len([seg for seg in re.split(r"[.!?]+",r["caption"]) if seg.strip()]),"claims_per_10_words":10*len(claims)/words if words else None,
                           "support_rate":metric["n_entail_ref"]/len(claims),**metric,
                           "llama_n_contra":labels.count("contradicted") if None not in labels else None,
                           "llama_n_entail":labels.count("entailed") if None not in labels else None,
                           "prompt_tokens":tokens,"generation_seconds":p.get("wall_seconds"),
                           "peak_vram_bytes":p.get("peak_vram_bytes") or None,
                           "actual_frame_count":len(p["input_trace"]["frame_indices"]),
                           "injection_interface":p["input_trace"]["injection_interface"]})
        results.extend(subset)
    # Join only the same comparison grid and reference mode. R-Full remains separate.
    scores={}
    for path in metric_paths:
        for r in read_jsonl(path):
            key=(r["model"],r["query_id"],r["arm"],r["ref_mode"])
            if key in scores:
                raise ValueError("Mixed or duplicate metric artifacts")
            scores[key]=r
    for r in results:
        model="formal_qwen" if r["model_id"].startswith("Qwen/") else "formal_videollama"
        if r["condition"] in ("formal","budget"):
            for mode in ("R-Short","R-Full"):
                score=scores.get((model,r["query_id"],r["arm"],mode),{})
                r[mode+"_CIDEr_points"]=score.get("CIDEr_points_x100")
    return results


def means(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[r["model_id"],r["condition"],r["arm"]].append(r)
    keys=("words","claims_per_10_words","support_rate","n_entail_ref","n_contra_ref","n_unment_ref","r_core_ref",
          "llama_n_contra","llama_n_entail","prompt_tokens","generation_seconds","peak_vram_bytes",
          "R-Short_CIDEr_points","R-Full_CIDEr_points")
    out=[]
    for (m,c,a),sample in sorted(groups.items()):
        if len({r["query_id"] for r in sample})!=len(sample):
            raise ValueError("Repeated query within arm/setting")
        stats={}
        denominators={}
        for key in keys:
            vals=[r[key] for r in sample if r.get(key) is not None]
            stats[key]=mean(vals) if vals else None
            denominators[key]=len(vals)
        out.append({"model_id":m,"condition":c,"arm":a,"n":len(sample),"denominators":denominators,**stats})
    return out


def paired(rows):
    by={(r["model_id"],r["condition"],r["arm"],r["query_id"]):r for r in rows}
    metrics=("words","n_entail_ref","n_contra_ref","support_rate","claims_per_10_words","r_core_ref","prompt_tokens")
    results={}
    for condition,arms in CONDITIONS.items():
        for arm in arms:
            diffs=defaultdict(dict)
            similar=defaultdict(dict)
            plan=0
            adherence=0
            for (m,c,a,q),r in by.items():
                if condition!=c or arm!=a:
                    continue
                base=by[m,"formal",a,q]
                plan+=1
                adherence+=2<=r["sentences"]<=4
                for metric in metrics:
                    if r[metric] is not None and base[metric] is not None:
                        diff=r[metric]-base[metric]
                        diffs[metric][q]=diff
                        if abs(r["words"]-base["words"])<=2:
                            similar[metric][q]=diff
            if plan:
                results[f"{condition}|{arm}"]={"planned_pairs":plan,
                   "paired":{k:cluster_bootstrap(v) for k,v in diffs.items()},
                   "length_matched_descriptive":{k:cluster_bootstrap(v) for k,v in similar.items()},
                   "longer_count":sum(delta>0 for delta in diffs["words"].values()),
                   "length_2_to_4_sentences_proxy_count":adherence if condition=="length" else None}
    # Judge bias on matched claim sets and query-arm pairs, rather than on micro agreement.
    arm_bias={}
    models={r["model_id"] for r in rows}
    for m in sorted(models):
        changes={}
        for a in ("ARM_A1","ARM_A3","ARM_A5","ARM_A7"):
            for q in {r["query_id"] for r in rows if r["model_id"]==m and r["arm"]==a}:
                prefix="formal" if a in ("ARM_A1","ARM_A7") else "budget"
                r,b=by.get((m,prefix,a,q)),by.get((m,"formal","ARM_A0",q))
                if not r or not b or r["llama_n_contra"] is None or b["llama_n_contra"] is None:
                    continue
                changes.setdefault(a,{})[q]=(r["llama_n_contra"]-r["n_contra_ref"])-(b["llama_n_contra"]-b["n_contra_ref"])
        arm_bias[m]={a:cluster_bootstrap(d) for a,d in changes.items()}
    return results,arm_bias


def render(rows,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3.25),layout="constrained")
    groups=defaultdict(list)
    for r in rows:
        if r["condition"] in ("formal","budget"):
            groups[r["model_id"]].append(r)
    for m,group in sorted(groups.items()):
        group=sorted(group,key=lambda r:r["arm"])
        bins={int(r["arm"][-1]):r for r in group}
        x=[0,1,3,5,7]
        name="Qwen" if m.startswith("Qwen/") else "Video-LLaMA"
        for ax,metric,label in zip(axes,("R-Short_CIDEr_points","n_contra_ref","r_core_ref"),
                                  ("R-Short CIDEr points","Reference conflicts per caption","Reference core-event coverage"),strict=True):
            ax.plot(x,[bins.get(k,{}).get(metric) for k in x],"o-",label=name)
            ax.set(xlabel="External evidence budget",ylabel=label,xticks=(0,1,3,5,7))
            ax.grid(axis="y",alpha=.3)
    axes[0].legend(frameon=False)
    pending=[str(k) for k in (3,5) if not all(any(int(r["arm"][-1])==k for r in g) for g in groups.values())]
    if pending:
        fig.supxlabel("Pending budgets: "+", ".join(pending),fontsize=9)
    fig.suptitle("CaReBench budget response · Machine-assisted factuality")
    for ext in ("pdf","svg","png"):
        path=out/f"budget_response.{ext}"
        fig.savefig(path,dpi=250)
        if ext=="svg":
            path.write_bytes(b"\n".join(line.rstrip() for line in path.read_bytes().splitlines())+b"\n")
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for action in ("predictions","eval","cross","metric"):
        p.add_argument("--"+action,action="append",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    rows=report_rows(a.predictions,a.eval,a.cross,a.metric)
    aggregate=means(rows)
    paired_sensitivity, bias=paired(rows)
    frontier=[]
    for r in aggregate:
        fields=("generation_seconds","n_contra_ref","R-Short_CIDEr_points","r_core_ref")
        metrics=[r.get(k) for k in fields]
        if any(x is None for x in metrics):
            continue
        def vector(v):
            return [v[fields[0]],v[fields[1]],-v[fields[2]],-v[fields[3]]]
        x=vector(r)
        comparable=[v for v in aggregate if v is not r and r["model_id"]==v["model_id"] and all(v.get(k) is not None for k in fields)]
        if not any(all(a<=b for a,b in zip(vector(v),x,strict=True)) and vector(v)!=x for v in comparable):
            frontier.append({"model_id":r["model_id"],"arm":r["arm"]})
    a.out.mkdir(parents=True,exist_ok=True)
    atomic_write_json(a.out/"Q2_BUDGET_REPORT.json",{"schema":"uarcap_q2.budget_report.v1","n_rows":len(rows),
        "status":"POST_HOC_MACHINE_ASSISTED","arm_means":aggregate,"paired_sensitivity":paired_sensitivity,
        "judge_conflict_delta_difference":bias,"within_model_frontier":frontier,
        "cost_scope":"Observed per-caption generation fields. Retrieval, load and feature costs require separate receipts.",
        "pending":{"full_budget":not all(any(r.get("arm")==arm for r in aggregate) for arm in ("ARM_A3","ARM_A5")),
                   "visual_budget_interaction":"NOT_IDENTIFIED_BY_A7_ONLY_DESIGN",
                   "calibrated_factuality":"AWAITING_HUMAN"}})
    export_table([{k:v for k,v in r.items() if k!="denominators"} for r in aggregate],a.out,"budget_arm_means")
    render(aggregate,a.out)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
