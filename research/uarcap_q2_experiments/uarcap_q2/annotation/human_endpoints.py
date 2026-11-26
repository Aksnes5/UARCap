"""Validate human reference-endpoint imports against separately retained review artifacts."""

from pathlib import Path
import math

from ..common.io_utils import primitive_hash, read_json, sha256_file
from ..final_analysis.validation_identity import ARMS, caption_key, require_cohort

MODELS = ("videollama_evcap_frozen_v08","Qwen/Qwen3.8-27B-FP8")
METRICS = ("n_contra_ref","r_core_ref")
SCHEMA = "uarcap_q2.human_primary_endpoints.v1"
RECEIPT = "uarcap_q2.human_reference_adjudication.v1"
PINS = ("caption_sha256","claim_split_sha256","reference_description_sha256","core_lock_sha256","evaluator_id")


def require_file(parent, witness):
    p=parent/witness["path"]
    if not p.is_file() or sha256_file(p)!=witness["sha256"]:
        raise ValueError(f"Human artifact missing or changed: {p.name}")
    return p


def verify(path, manifest):
    data=read_json(path)
    if data.get("schema")!=SCHEMA or data.get("_template_only") or data.get("dataset_id")!="CaReBench":
        raise ValueError("Expected a genuine CaReBench reference endpoint import")
    _,_,queries,_,_=require_cohort(manifest)
    model=data.get("model_id")
    metric=data.get("metric")
    if model not in MODELS or metric not in METRICS:
        raise ValueError("Explicit model-specific and endpoint-specific import required")
    units=manifest["units"]
    expected_inc={q:units[q]["val_inclusion_prob"] for q in queries}
    expected_strata={q:units[q]["stratum"] for q in queries}
    values=data.get("values") or {}
    inc=data.get("inclusion_by_qid") or {}
    if (set(values)!=queries or set(inc)!=queries or data.get("stratum_by_qid")!=expected_strata
        or any(type(inc[q]) not in (int,float) or not 0<inc[q]<=1 or abs(inc[q]-v)>1e-6
               for q,v in expected_inc.items())):
        raise ValueError("Human coverage, strata or probabilities differ from frozen design")
    if set(data.get("calibration_excluded_ids") or []) & queries:
        raise ValueError("Calibration IDs overlap validation IDs")
    for q, arms in values.items():
        if set(arms)!=set(ARMS):
            raise ValueError(f"{q}: incomplete arm pairing")
        for value in arms.values():
            if (type(value) not in (int,float) or not math.isfinite(value) or value<0
                or (metric=="r_core_ref" and value>1)
                or (metric=="n_contra_ref" and int(value)!=value)):
                raise ValueError(f"{q}: invalid adjudicated endpoint")

    receipt_path=require_file(path.parent,{"path":data.get("adjudication_receipt_path") or "__missing__",
                                           "sha256":data.get("adjudication_receipt_sha256")})
    r=read_json(receipt_path)
    reviewers=r.get("annotators") or {}
    if (r.get("schema")!=RECEIPT or r.get("status")!="ADJUDICATED_REFERENCE_ENDPOINTS"
        or len(reviewers)!=2 or set(reviewers)!=set(data.get("annotator_ids") or [])
        or data.get("n_annotators")!=2 or not r.get("adjudicator_id")
        or r.get("manifest_sha256_canonical")!=primitive_hash(manifest)):
        raise ValueError("Unbound or incomplete independent review receipt")
    review_files=[require_file(receipt_path.parent,v) for v in reviewers.values()]
    if len(set(review_files))!=2 or len({sha256_file(p) for p in review_files})!=2:
        raise ValueError("Independent review exports must be distinct")
    expected={(m,q,a) for m in MODELS for q in queries for a in ARMS}
    for reviewer,p in zip(reviewers,review_files,strict=True):
        review=read_json(p)
        if review.get("annotator_id")!=reviewer or review.get("independent_blind_review_attested") is not True:
            raise ValueError("Independent reviewer attestation missing")
        export=require_file(p.parent,review["original_export"])
        attestation=read_json(require_file(p.parent,review["attestation"]))
        if (attestation.get("annotator_id")!=reviewer or
                attestation.get("review_export_sha256")!=sha256_file(export) or
                attestation.get("independent_blind_review_attested") is not True):
            raise ValueError("External review attestation mismatch")
        observed=review.get("rows",[])
        if len(observed)!=len(expected) or {caption_key(x) for x in observed}!=expected:
            raise ValueError("Incomplete independent review")
        for row in observed:
            endpoints=row.get("endpoints") or {}
            if (type(endpoints.get("n_contra_ref")) is not int or endpoints["n_contra_ref"]<0
                or type(endpoints.get("r_core_ref")) not in (int,float) or not 0<=endpoints["r_core_ref"]<=1):
                raise ValueError("Missing individual reference endpoints")
    adjudicated=require_file(receipt_path.parent,r["adjudicated"])
    rows=read_json(adjudicated)["rows"]
    if len(rows)!=len(expected) or {caption_key(row) for row in rows}!=expected:
        raise ValueError("Incomplete adjudicated caption ledger")
    for row in rows:
        if any(not isinstance(row.get(k),str) or not row[k] for k in PINS):
            raise ValueError("Missing caption, fact, reference, evaluator or core lock binding")
        m,q,a=caption_key(row)
        if m==model and row.get("endpoints",{}).get(metric)!=values[q][a]:
            raise ValueError("Import disagrees with adjudicated endpoints")
    return {"payload":data,"receipt":{"import_sha256":sha256_file(path),"n_audited":len(queries),
                 "n_strata_covered":len(set(expected_strata.values())),"annotator_ids":sorted(reviewers),
                 "adjudication_receipt_sha256":sha256_file(receipt_path),"adjudicated_rows":rows}}
