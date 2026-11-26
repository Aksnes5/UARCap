"""Reconcile frozen sample roles, prediction/evaluation keys and source parents."""

from pathlib import Path
from collections import Counter
from ..common.io_utils import primitive_hash, read_json, read_jsonl, sha256_bytes, sha256_file

ARMS = ("ARM_A0","ARM_A1","ARM_A7")


def caption_key(row):
    key = tuple(row.get(name) for name in ("model_id","query_id","arm"))
    if any(not isinstance(v,str) or not v for v in key):
        raise ValueError("Incomplete (model_id, query_id, arm)")
    return key


def index(rows):
    data = {}
    for row in rows:
        key = caption_key(row)
        if key in data:
            raise ValueError(f"Duplicate caption key: {key}")
        data[key] = row
    return data


def require_cohort(manifest):
    def ids(values):
        result = set(values)
        if len(result) != len(values):
            raise ValueError("Duplicate cohort ID")
        return result
    formal = ids(manifest["selected_query_ids"])
    main = ids(manifest["formal_main_excluding_validation_ids"])
    hold = ids(manifest["e2_validation_set"]["validation_query_ids"])
    pilot = ids(manifest["pilot_excluded_query_ids"])
    if main & hold or main | hold != formal or formal & pilot:
        raise ValueError("Query overlap or invalid partition")
    units = manifest["units"]
    parents = {q: units[q]["parent_video_id"] for q in formal}
    if any(not p for p in parents.values()) or len(set(parents.values())) != len(formal):
        raise ValueError("Parent repeats or is missing in the formal sample")
    if set(parents.values()) & set(manifest["pilot_excluded_parent_video_ids"]):
        raise ValueError("Parent overlap with pilot")
    return formal, main, hold, pilot, parents


def verify_caption_pair(pred, evaluated):
    key = caption_key(pred)
    caption = pred.get("caption")
    if (caption_key(evaluated) != key or pred.get("ok") is not True or not isinstance(caption,str)
        or evaluated.get("parse_status") != "OK" or evaluated.get("caption") != caption
        or evaluated.get("caption_sha256") != sha256_bytes(caption.encode("utf-8"))):
        raise ValueError(f"Prediction/evaluation mismatch: {key}")
    claims = evaluated.get("claims")
    if not isinstance(claims,list) or not claims or any(not c.get("claim_id") or not c.get("claim") for c in claims):
        raise ValueError(f"Missing claim identity: {key}")
    if len({c["claim_id"] for c in claims}) != len(claims):
        raise ValueError(f"Duplicate claim identity: {key}")
    return claims


def audit(manifest_path, pred_paths, eval_paths, validation_receipt=None):
    manifest = read_json(manifest_path)
    formal, main, hold, pilot, parents = require_cohort(manifest)
    summary = {"schema":"uarcap_q2.validation_identity.v1","manifest_sha256":sha256_file(manifest_path),
               "manifest_timestamp_utc":manifest["timestamp_utc"],"sets":{}, "artifacts":{},
               "intersections":{"main_validation_queries":0, "main_validation_parents":0,
                                "pilot_formal_queries":0, "pilot_formal_parents":0}}
    for name, qids in (("formal",formal),("main",main),("validation",hold),("pilot",pilot)):
        parent_ids = [parents[q] for q in qids] if name!="pilot" else manifest["pilot_excluded_parent_video_ids"]
        summary["sets"][name] = {"queries":len(qids),"ids_sha256":primitive_hash(sorted(qids)),
                                 "parents":len(set(parent_ids)),"parent_ids_sha256":primitive_hash(sorted(set(parent_ids)))}
    for pred_path, eval_path in zip(pred_paths,eval_paths,strict=True):
        predicted, evaluated = index(read_jsonl(pred_path)), index(read_jsonl(eval_path))
        models = {k[0] for k in predicted}
        if len(models)!=1 or set(predicted)!=set(evaluated):
            raise ValueError("Incomplete paired prediction/evaluation assets")
        model = models.pop()
        if set(predicted) != {(model,q,a) for q in formal for a in ARMS}:
            raise ValueError("Prediction set differs from frozen cohort × arms")
        for key, pred in predicted.items():
            verify_caption_pair(pred,evaluated[key])
        count = Counter("validation" if k[1] in hold else "main" for k in evaluated)
        summary["artifacts"][model] = {"prediction_sha256":sha256_file(pred_path),"evaluation_sha256":sha256_file(eval_path),
                                        "total":len(predicted),**count}
    if validation_receipt:
        receipt = read_json(validation_receipt)
        actual = receipt["inclusion_by_qid"]
        if set(actual)!=hold or any(abs(actual[q]-manifest["units"][q]["val_inclusion_prob"])>1e-6 for q in hold):
            raise ValueError("Validation receipt disagrees with frozen sample")
        summary["package_receipt"] = {"timestamp":receipt["utc"],"sha256":sha256_file(validation_receipt)}
    summary["interpretation"] = "INTERNAL_PROBABILITY_SUBSAMPLE_ALREADY_GENERATED_AND_MACHINE_EVALUATED"
    return summary
