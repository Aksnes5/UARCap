"""Validate release tables and descriptive controls without running a model or statistical test."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
from collections import Counter
from itertools import combinations

from replay import BASE, HERE, ROOT, blob, digest, read, store, verify
from prepare_inputs import words, jaccard


def check_quartets(work, eligible):
    prepared = work/"inputs/prepared"
    h, l = [read(prepared/f"B_text7_{label}_cap.json") for label in ("hq","lq")]
    result = read(HERE/"FOLLOWUP_B_FORMAL_RESULTS.json")["matching"]
    observed = {name:[] for name in result}
    def features(cap):
        strings, scores, _ = cap
        tokens = [words(s) for s in strings]
        head, tail = [set(t) for t in tokens[:3]], [set(t) for t in tokens[3:]]
        return (sum(scores[3:])/4, sum(len(t) for t in tokens[3:]),
                sum(jaccard(x,y) for x in tail for y in head)/12,
                sum(jaccard(x,y) for x,y in combinations(tail,2))/6)
    keys = ("relevance_gap","absolute_word_delta","absolute_head_jaccard_delta","absolute_internal_jaccard_delta")
    for q in eligible:
        verify(h[q][0][:3] == l[q][0][:3], "B head differs")
        for name, hi, lo in zip(keys,features(h[q]),features(l[q]),strict=True):
            observed[name].append(hi-lo if name=="relevance_gap" else abs(hi-lo))
    computed = {k:{"mean":sum(v)/len(v),"min":min(v),"max":max(v)} for k,v in observed.items()}
    verify(all(math.isclose(x,result[k][stat],rel_tol=1e-10,abs_tol=1e-12)
               for k,col in computed.items() for stat,x in col.items()),"Quartet balance disagrees with preflight")
    return computed


def check(work):
    paper = (ROOT/"paper/UARCap_paper_v0.7.md").read_text(encoding="utf-8")
    ordered = read(work/"inputs/prepared/B_eligible_queries.json")
    eligibility = set(ordered)
    counts = {}
    a_matches = {}
    excluded = {}
    models = {}
    arms = {}
    for family in "AB":
        results = read(HERE/f"FOLLOWUP_{family}_FORMAL_RESULTS.json")
        for arm, mean in results["CIDEr_points"].items():
            verify(re.search(rf"^\| {family} \| {arm} \| [^|]* \| {mean:.2f} \|", paper, re.M), "Arm table: "+arm)
            folder=work/"full"/arm
            seal=read(folder/"arm_completed.json")
            pred=folder/seal["predictions"]
            rows=read(pred)
            trace=[json.loads(s) for s in (folder/"input_trace.jsonl").read_bytes().splitlines()]
            counts[arm]={"predictions":len(rows),"traces":len(trace)}
            verify(all(v==2990 for v in counts[arm].values()),arm)
            mode=Counter(json.dumps([t["training"],t["module_training"]],sort_keys=True) for t in trace)
            verify(len(mode)==1,"Changing module state")
            models[arm]=json.loads(next(iter(mode)))
            if arm in ("A1","A7"):
                path=f"evcap_external/outputs/eval_test2990_D{arm[1]}/{pred.name}"
                old=json.loads(blob(BASE,path))
                matches=sum(x==y for x,y in zip(rows,old,strict=True))
                a_matches[arm]={"equal_full_records":matches,"historical_path":path}
                verify(matches==2990,"Historical endpoint disagreement")
            if family=="B":
                outside=[(p,t) for p,t in zip(rows,trace,strict=True) if t["query"] not in eligibility]
                if arm=="B3":
                    common=outside
                else:
                    excluded[arm]=sum(x==y for x,y in zip(common,outside,strict=True))
                    verify(excluded[arm]==372,"Excluded predictions/traces differ")
        arms.update(results["CIDEr_points"])
    with (HERE/"FOLLOWUP_FIVE_TEST_HOLM_RESULTS.csv").open(newline="") as f:
        contrasts=list(csv.DictReader(f))
    def signed(x):
        return f"{float(x):+.2f}".replace("-","−")
    for row in contrasts:
        fields=[row["contrast"].replace("-","−"),signed(row["effect"]),f'[{signed(row["ci_low"])}, {signed(row["ci_high"])}]']
        fields.extend(f"{float(row[key]):.4f}".lstrip("0") for key in ("p_raw","p_family_holm","p_five_holm"))
        verify("| "+" | ".join(fields)+" |" in paper, "Contrast table mismatch: "+row["contrast"])
    assets={}
    for ref in re.findall(r"!\[[^\]]*\]\(([^)]+)\)",paper):
        png=ROOT/"paper"/ref
        for ext in (".png",".svg",".pdf"):
            path=png.with_suffix(ext)
            assets[path.name]=digest(path)
    verify(len(assets)==15 and len(arms)==6 and len(contrasts)==5, "Incomplete release")
    store(HERE/"RELEASE_CHECK.json",{"arm_rows":counts,"archived_test2990_endpoint_parity":a_matches,
        "historical_revision":BASE, "identical_outside_B_eligibility_records_and_traces":excluded,
        "module_modes_parent_and_submodules":models,"exact_rounded_table_checks":{"arms":6,"contrasts":5},
        "matching_rechecked_from_consumed_caption_inputs":check_quartets(work,ordered),
        "figure_hashes":assets,"manuscript_sha256_lf":digest(ROOT/"paper/UARCap_paper_v0.7.md")})


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work",type=Path,required=True)
    check(ap.parse_args().work)
