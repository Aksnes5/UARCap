"""Keep parent-paired estimands fixed while excluding prespecified visual-risk queries."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uarcap_q2.common.io_utils import atomic_write_json, read_json, read_jsonl, sha256_file
from uarcap_q2.final_analysis.core import cluster_bootstrap
from uarcap_q2.final_analysis.validation_identity import index, verify_caption_pair

RISK = {"CONFIRMED_DUPLICATE","HIGH_RISK_NEAR_DUPLICATE","NO_OBVIOUS_SIGNATURE_RISK","UNVERIFIED"}


def summarize(risk, units, pairs):
    by_q = defaultdict(list)
    for r in risk["rows"]:
        if r["status"] not in RISK:
            raise ValueError("Unknown visual risk status")
        by_q[r["query_id"]].append(r)
    if set(by_q) != set(units) or any({r["rank"] for r in rows} != set(range(1,8)) or len(rows) != 7 for rows in by_q.values()):
        raise ValueError("Visual risk rows must cover the full frozen Top-7")
    confirmed = {q for q,rows in by_q.items() if any(r["status"] == "CONFIRMED_DUPLICATE" for r in rows)}
    high = {q for q,rows in by_q.items() if any(r["status"] == "HIGH_RISK_NEAR_DUPLICATE" for r in rows)}
    unknown = {q for q,rows in by_q.items() if any(r["status"] == "UNVERIFIED" for r in rows)}
    results = {}
    for name, values in pairs.items():
        result = {}
        for tag, excluded in (("frozen_parent", set()), ("exclude_confirmed", confirmed),
                              ("exclude_confirmed_and_high", confirmed | high)):
            clusters = defaultdict(list)
            remaining = set(values) - excluded
            for q in sorted(remaining):
                clusters[units[q]["parent_video_id"]].append(values[q])
            parent_means = {parent: mean(diffs) for parent,diffs in clusters.items()}
            result[tag] = {"n_queries":len(remaining), "n_parents":len(clusters),
                           "n_remaining_with_unverified_pair":len(remaining & unknown),
                           "estimate":cluster_bootstrap(parent_means) if parent_means else None}
        results[name] = result
    return {"schema":"uarcap_q2.visual_risk_sensitivity.v2", "results":results,
            "risk_pair_counts":Counter(r["status"] for r in risk["rows"]), "query_count":len(units),
            "n_queries_with_unverified_pair":len(unknown), "n_queries_excluded_confirmed":len(confirmed),
            "n_queries_excluded_confirmed_and_high":len(confirmed | high),
            "scope":"Paired exploratory diagnostics; missing/anonymous source identity is unresolved"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("risk","manifest","predictions","eval","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a = p.parse_args()
    units = read_json(a.manifest)["units"]
    pred,ev = (index(read_jsonl(x)) for x in (a.predictions,a.eval))
    if set(pred) != set(ev):
        raise ValueError("Caption and evaluation ledgers differ")
    for key in pred:
        verify_caption_pair(pred[key],ev[key])
    differences = {}
    for model in sorted({k[0] for k in pred}):
        for arm in ("ARM_A1","ARM_A7"):
            for metric in ("n_contra_ref","n_entail_ref","r_core_ref"):
                values = {}
                for q in units:
                    x,y = (ev[model,q,b][metric] for b in ("ARM_A0",arm))
                    if x is not None and y is not None:
                        values[q] = y-x
                differences[f"{model}|{arm}-A0|{metric}"] = values
    payload = summarize(read_json(a.risk), units, differences)
    payload["input_sha256"] = {str(x):sha256_file(x) for x in (a.risk,a.manifest,a.predictions,a.eval)}
    atomic_write_json(a.out,payload)


if __name__ == "__main__":
    main()
