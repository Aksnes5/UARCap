"""Bind independent blind reference reviews and adjudication to formal human endpoint imports."""

import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from uarcap_q2.annotation.human_endpoints import MODELS,METRICS,PINS,RECEIPT,SCHEMA,verify
from uarcap_q2.common.io_utils import primitive_hash,read_json,sha256_file,atomic_write_json
from uarcap_q2.final_analysis.validation_identity import ARMS,caption_key,require_cohort


def source_records(path,key_map,attestation):
    data=read_json(path)
    annotator=data["annotator_id"]
    trust=read_json(attestation)
    if (trust.get("annotator_id")!=annotator or trust.get("review_export_sha256")!=sha256_file(path)
        or trust.get("independent_blind_review_attested") is not True):
        raise ValueError("Blind human reference review must be explicitly attested")
    rows={}
    records=data.get("rows")
    if records is None:
        records=[{"blind_caption_id":bid,"endpoints":content["endpoints"]}
                 for record in data["records"] if record.get("completed") and record.get("reference_endpoints_reviewed")
                 for bid,content in record["captions"].items()]
    for r in records:
        meta=key_map[r["blind_caption_id"]]
        if meta["annotator_id"]!=annotator:
            raise ValueError("Review belongs to a different annotator")
        key=caption_key(meta)
        if key in rows:
            raise ValueError("Repeated review key")
        rows[key]={**{k:meta[k] for k in PINS},
                   **dict(zip(("model_id","query_id","arm"),key,strict=True)),
                   "endpoints":r["endpoints"]}
    return annotator,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("manifest","key","first","second","first-attestation","second-attestation","out","resolutions"):
        p.add_argument("--"+k,type=Path,required=k!="resolutions")
    p.add_argument("--adjudicator-id",required=True)
    a=p.parse_args()
    manifest=read_json(a.manifest)
    _,_,sample,_,_=require_cohort(manifest)
    key=read_json(a.key)
    mapping=key["mapping"]
    if key["package_sha256"]!=primitive_hash(mapping):
        raise ValueError("Blind key hash mismatch")
    first=source_records(a.first,mapping,a.first_attestation)
    second=source_records(a.second,mapping,a.second_attestation)
    expected={(m,q,u) for m in MODELS for q in sample for u in ARMS}
    if first[0]==second[0] or a.adjudicator_id in (first[0],second[0]) or any(set(v)!=expected for _,v in (first,second)):
        raise ValueError("Need independent and complete human captions, plus separate adjudicator")
    settings=read_json(a.resolutions) if a.resolutions else {"rows":[]}
    resolutions={caption_key(r):r for r in settings["rows"]}
    if len(settings["rows"])!=len(resolutions) or resolutions.keys()-expected:
        raise ValueError("Unexpected or duplicate resolution")
    a.out.mkdir(parents=True,exist_ok=True)
    resolved=[]
    differences=[]
    for k in sorted(expected):
        x,y=(v[k] for _,v in (first,second))
        if any(x[field]!=y[field] for field in PINS):
            raise ValueError("Review versions disagree")
        if x["endpoints"]!=y["endpoints"]:
            if k not in resolutions:
                differences.append({name:x[name] for name in ("model_id","query_id","arm")})
                continue
            row=resolutions[k]
            if not str(row.get("rationale","")).strip():
                raise ValueError("An adjudication reason is required")
            x={**x,"endpoints":row["endpoints"],"adjudication_rationale":row["rationale"]}
        resolved.append(x)
    if differences:
        atomic_write_json(a.out/"REFERENCE_DISAGREEMENTS_PENDING.json",{"rows":differences})
        p.exit(3,"AWAITING_HUMAN unresolved endpoint differences\n")

    def file_ref(path):
        return {"path":str(path.resolve()),"sha256":sha256_file(path)}
    reviews={}
    for (ann,rows),src,att,i in zip((first,second),(a.first,a.second),(a.first_attestation,a.second_attestation),(1,2),strict=True):
        path=a.out/f"review_{i}_bound.json"
        atomic_write_json(path,{"annotator_id":ann,"independent_blind_review_attested":True,
                                "original_export":file_ref(src),"attestation":file_ref(att),"rows":list(rows.values())})
        reviews[ann]=file_ref(path)
    ap=a.out/"adjudicated.json"
    atomic_write_json(ap,{"rows":resolved})
    rp=a.out/"adjudication_receipt.json"
    agreements=Counter({metric:sum(first[1][k]["endpoints"][metric]==second[1][k]["endpoints"][metric]
                                  for k in expected) for metric in METRICS})
    atomic_write_json(rp,{"schema":RECEIPT,"status":"ADJUDICATED_REFERENCE_ENDPOINTS",
                         "adjudicator_id":a.adjudicator_id,"annotators":reviews,"adjudicated":file_ref(ap),
                         "manifest_sha256_canonical":primitive_hash(manifest),"n_reviewed":len(expected),
                         "exact_endpoint_agreement":dict(agreements),"blind_key_sha256":sha256_file(a.key)})
    for im,model in enumerate(MODELS,1):
        for metric in METRICS:
            values={q:{} for q in sorted(sample)}
            for r in resolved:
                if r["model_id"]==model:
                    values[r["query_id"]][r["arm"]]=r["endpoints"][metric]
            path=a.out/f"model_{im}_{metric}.json"
            atomic_write_json(path,{"schema":SCHEMA,"dataset_id":"CaReBench","model_id":model,"metric":metric,
                 "n_annotators":2,"annotator_ids":list(reviews),"adjudication_receipt_path":str(rp.resolve()),
                 "adjudication_receipt_sha256":sha256_file(rp),"values":values,
                 "inclusion_by_qid":{q:manifest["units"][q]["val_inclusion_prob"] for q in values},
                 "stratum_by_qid":{q:manifest["units"][q]["stratum"] for q in values}})
            verify(path,manifest)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
