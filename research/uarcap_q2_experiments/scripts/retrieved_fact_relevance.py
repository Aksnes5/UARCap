"""Evaluate admitted Top-7 external captions against their query references."""

import argparse
from collections import Counter,defaultdict
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from uarcap_q2.common.io_utils import atomic_write_json,atomic_write_jsonl,read_jsonl,sha256_file
from uarcap_q2.final_analysis.core import cluster_bootstrap
from uarcap_q2.final_analysis.validation_identity import index,verify_caption_pair


def inputs(retrieval):
    seen=set()
    for row in retrieval:
        q=row["query_id"]
        ranked=row["reranked_d"][:7]
        if q in seen or len(ranked)!=7:
            raise ValueError("Expected a unique Top-7 prefix per query")
        seen.add(q)
        for rank,c in enumerate(ranked,1):
            if not c["caption"].strip() or not c["clip_ids"]:
                raise ValueError("Empty retrieved example")
            yield {"query_id":q,"arm":f"RANK_{rank}","model_id":"VideoUFO_brief_caption_bank","ok":True,
                   "caption":c["caption"],"clip_ids":c["clip_ids"],
                   "raw_similarity":c["score"],"D_score":c.get("rerank_score")}


def analyze(prediction,evaluated):
    pred,ev=index(prediction),index(evaluated)
    if set(pred)!=set(ev):
        raise ValueError("Evidence label coverage differs from declared Top-7")
    ancestry=defaultdict(set)
    by_rank=defaultdict(list)
    by_q=defaultdict(list)
    rows=[]
    for key,p in sorted(pred.items()):
        q=key[1]
        claims=verify_caption_pair(p,ev[key])
        words={c["claim"] for c in claims}
        labels=Counter(c["label"] for c in claims)
        supported_new={c["claim"] for c in claims if c["label"]=="entailed"}-ancestry[q]
        row={"query_id":q,"rank":int(key[2].split("_")[1]),"raw_similarity":p["raw_similarity"],
             "D_score":p["D_score"],"n_claims":len(claims),"n_support":labels["entailed"],
             "n_conflict":labels["contradicted"],"n_unmentioned":labels["unmentioned"],
             "exact_repeat_claims":len(words&ancestry[q]),"new_exact_supported_claims":len(supported_new)}
        ancestry[q].update(words)
        by_rank[row["rank"]].append(row)
        by_q[q].append(row)
        rows.append(row)
    def assoc(field):
        import numpy as np
        from scipy.stats import rankdata
        rho={}
        for q,ranked in by_q.items():
            x=[r["raw_similarity"] for r in ranked]
            y=[r[field] for r in ranked]
            if len(set(x))>1 and len(set(y))>1:
                rho[q]=float(np.corrcoef(rankdata(x),rankdata(y))[0,1])
        return cluster_bootstrap(rho)
    fields=("raw_similarity","n_support","n_conflict","n_unmentioned","exact_repeat_claims","new_exact_supported_claims")
    summary={rank:{"n":len(g),**{f:mean(r[f] for r in g) for f in fields}} for rank,g in by_rank.items()}
    return {"schema":"uarcap_q2.retrieved_fact_relevance.v1","status":"MACHINE_ASSISTED_POST_HOC",
            "rank_summary":summary,"n_queries":len(by_q),"n_captions":len(rows),
            "within_query_rank_association":{field:assoc(field) for field in ("n_support","n_conflict")},
            "estimand":"Within-query Spearman conditional on variable ranks and endpoint counts. Exact claim repeats are lexical.",
            "rows":rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest="command",required=True)
    a=s.add_parser("stage")
    a.add_argument("--retrieval",type=Path,required=True)
    a.add_argument("--out",type=Path,required=True)
    b=s.add_parser("summarize")
    for k in ("predictions","eval","out"):
        b.add_argument("--"+k,type=Path,required=True)
    args=p.parse_args()
    if args.command=="stage":
        rows=list(inputs(read_jsonl(args.retrieval)))
        if len(rows)!=4200:
            raise ValueError("Expected 600 × 7 external captions")
        atomic_write_jsonl(args.out,rows)
    else:
        result=analyze(read_jsonl(args.predictions),read_jsonl(args.eval))
        result.update(prediction_sha256=sha256_file(args.predictions),evaluation_sha256=sha256_file(args.eval))
        atomic_write_json(args.out,result)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
