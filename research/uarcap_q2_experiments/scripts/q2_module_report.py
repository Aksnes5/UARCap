"""Produce source-bound exploratory module summaries and reliability contrasts."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uarcap_q2.common.io_utils import atomic_write_json, atomic_write_jsonl, primitive_hash, read_json, read_jsonl, sha256_file
from uarcap_q2.final_analysis.core import cluster_bootstrap
from uarcap_q2.final_analysis.validation_identity import index, verify_caption_pair


def plot(series, out, ylabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    finite = [(k,v) for k,v in series.items() if v.get("ci95") is not None]
    if not finite:
        return
    fig, ax = plt.subplots(figsize=(8, max(2.3,.31*len(finite))), layout="constrained")
    for i,(key,value) in enumerate(finite):
        m, (lo,hi) = value["mean"],value["ci95"]
        ax.hlines(i,lo,hi)
        ax.plot(m,i,"o")
    ax.set_yticks(range(len(finite)), [k for k,_ in finite], fontsize=8)
    ax.set_xlabel(ylabel)
    ax.axvline(0,color=".7",lw=.8)
    for suffix in ("svg","pdf","png"):
        fig.savefig(out.with_suffix("."+suffix),dpi=220)
    plt.close(fig)


def report_history(folder, include_cross):
    lock = read_json(folder/"HISTORICAL_BRIDGE_MANIFEST.json")
    pred, ev = (index(read_jsonl(folder/f)) for f in ("predictions.jsonl","eval/a2_ref_fact_eval.jsonl"))
    if len(pred)!=180 or pred.keys()!=ev.keys() or len({r["source_id"] for r in pred.values()})!=60 or set(lock["query_ids"])!={r["query_id"] for r in pred.values()}:
        raise ValueError("Historical source, caption or evaluator grid changed")
    from q2_budget_report import norm
    labels = {(r["query_id"],r["norm_key"]): r["llama_label"] for r in read_jsonl(folder/"cross/A2_CROSS_JUDGE_LABELS.jsonl")} if include_cross else {}
    cells={}
    for key,p in pred.items():
        claims = verify_caption_pair(p, ev[key])
        ls = [labels.get((key[1],norm(c["claim"]))) for c in claims]
        cells[key[1], key[2]] = [ev[key]["n_contra_ref"],
                                  ls.count("contradicted") if include_cross and None not in ls else None]
    effects = {}
    for arm in ("ARM_A1","ARM_A7"):
        for j,name in enumerate(("Gemma","Llama same claim set")):
            d = {q:cells[q,arm][j]-cells[q,"ARM_A0"][j] for q in lock["query_ids"]
                 if cells[q,arm][j] is not None and cells[q,"ARM_A0"][j] is not None}
            effects[f"{name} · {arm} − A0"] = cluster_bootstrap(d)
    return {"n_sources":60,"n_archived_captions":180,"estimates":effects,
            "scope":"MSR-VTT written reference conflict; core-event coverage not defined"}


def report_provenance(primary, second):
    by_model = defaultdict(dict)
    primary_rows = {r["job_id"]:r for r in primary["rows"]}
    if len(primary_rows)!=len(primary["rows"]):
        raise ValueError("Repeated fact-provenance jobs")
    for row in primary_rows.values():
        rel = row["relations"]
        labels = rel["evidence_labels"]
        added = rel["added_indices"]
        name = f"{row['model_id']} {row['arm']}"
        by_model[name + " · added"][row["query_id"]] = len(added)
        by_model[name + " · added reference conflicts"][row["query_id"]] = sum(row["after"][i]["label"]=="contradicted" for i in added)
        by_model[name + " · added supported by evidence"][row["query_id"]] = sum("entailed" in labels[i] for i in added)
    agree = defaultdict(list)
    if second:
        for r in second["rows"]:
            if r["job_id"] not in primary_rows:
                continue
            first = primary_rows[r["job_id"]]
            if any(first[k] != r[k] for k in ("before","after","evidence","caption_sha256")):
                raise ValueError("Cross-judge provenance binding differs")
            pairs = [(a,b) for x,y in zip(first["relations"]["evidence_labels"],r["relations"]["evidence_labels"]) for a,b in zip(x,y)]
            if pairs:
                agree[r["query_id"]].append(mean(a==b for a,b in pairs))
    return {"estimates":{k:cluster_bootstrap(v) for k,v in by_model.items()},
            "primary_jobs_planned":primary.get("expected_jobs"),"primary_jobs_parsed":len(primary_rows),
            "second_jobs_parsed":len(second["rows"]) if second else 0,
            "same_claim_and_evidence_agreement":cluster_bootstrap({q:mean(a) for q,a in agree.items()}),
            "scope":"Relation to retrieved text; visual provenance and causal origin unresolved"}


def sample_relevance(source, lock, out):
    rows = read_jsonl(source)
    if len(index(rows)) != 4200:
        raise ValueError("Require frozen 600 x 7 natural evidence evaluation")
    qids = sorted({r["query_id"] for r in rows},key=lambda q:primitive_hash([20261126,q]))[:60]
    subset = [r for r in rows if r["query_id"] in set(qids)]
    if len(subset)!=420 or len(qids)!=60:
        raise ValueError("Unexpected reliability sample grid")
    atomic_write_jsonl(out/"a2_ref_fact_eval.jsonl",subset)
    atomic_write_json(out/"a2_core_event_lock.json",read_json(lock))
    atomic_write_json(out/"RELEVANCE_SAMPLE.json",{"schema":"uarcap_q2.relevance_reliability_sample.v2",
        "query_ids":qids,"n_captions":len(subset),"sample_rule":"lowest 60 SHA256(seed, query_id)",
        "input_sha256":{str(p):sha256_file(p) for p in (source,lock)}})


def report_relevance(source, sample, cross):
    main = read_json(source)
    by_rank = defaultdict(dict)
    for r in main["rows"]:
        for metric in ("n_support","n_conflict","n_unmentioned"):
            by_rank[f"Rank {r['rank']} · {metric}"][r["query_id"]] = r[metric]
    agreement = {}
    if cross:
        from q2_budget_report import norm
        labels = {(r["query_id"],r["norm_key"]):r["llama_label"] for r in read_jsonl(cross)}
        by_q = defaultdict(list)
        for row in read_jsonl(sample):
            for c in row["claims"]:
                k = (row["query_id"],norm(c["claim"]))
                if k in labels:
                    by_q[k[0]].append(labels[k] == c["label"])
        agreement = cluster_bootstrap({q:mean(v) for q,v in by_q.items()})
    return {"estimates":{k:cluster_bootstrap(v) for k,v in by_rank.items()}, "n_queries":main["n_queries"],
            "n_captions":main["n_captions"],"reliability":agreement,"scope":main["estimand"]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind",choices=("history","provenance","relevance","relevance-sample"),required=True)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--second",type=Path)
    p.add_argument("--sample",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    paths=[a.input]
    if a.kind=="relevance-sample":
        sample_relevance(a.input,a.second,a.out)
        return
    if a.kind=="history":
        result=report_history(a.input,bool(a.second))
        paths=[a.input/f for f in ("HISTORICAL_BRIDGE_MANIFEST.json","predictions.jsonl","eval/a2_ref_fact_eval.jsonl")]
    elif a.kind=="provenance":
        result=report_provenance(read_json(a.input),read_json(a.second) if a.second else None)
    else:
        result=report_relevance(a.input,a.sample,a.second)
        if a.sample:
            paths.append(a.sample)
    if a.second:
        paths.append(a.second)
    atomic_write_json(a.out, {"schema":"uarcap_q2.module_report.v2","kind":a.kind,"status":"EXPLORATORY_MACHINE_ASSISTED",
                             "judge_scope":"TWO_JUDGES" if a.second else "GEMMA_ONLY",**result,
                             "input_sha256":{str(p):sha256_file(p) for p in paths}})
    plot(result["estimates"], a.out,"Mean per query (95% bootstrap interval)")


if __name__ == "__main__":
    main()
