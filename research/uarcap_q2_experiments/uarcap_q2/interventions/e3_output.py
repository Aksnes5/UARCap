"""E3 output binding and archival reuse checks."""

from ..common.io_utils import primitive_hash
from ..retrieval.arm_builder import build_e3_intervention_arms

ARMS = ("E3_BASE","E3_SUP","E3_CON","E3_NEU")


def target_text(candidate, bridge):
    return ((candidate["f_plus"],candidate["f_minus"]) if bridge
            else (candidate["true_action"],candidate["conflict_action"]))


def verify_prediction(row, candidate, bridge=False):
    q=candidate["query_id"]
    t=row["input_trace"]
    pos,neg=target_text(candidate,bridge)
    expected=build_e3_intervention_arms(q,pos,neg,candidate["neutral_text"])[row["arm"]]
    if (row.get("ok") is not True or not row.get("caption","").strip()
        or row["query_id"]!=q or any(t[k]!=row[k] for k in ("model_id","query_id","arm"))
        or t.get("base_visual_mode")!="BASE_A0" or t.get("visual_evidence_ids")!=[]
        or t.get("visual_evidence_scores")!=[] or t.get("text_evidence")!=expected.text_evidence
        or t.get("visual_input_sha256")!=candidate["frame_sha256"]
        or len(t.get("frame_indices",[]))!=16 or not t.get("prompt_rendered_sha256")):
        raise ValueError(f"{q}/{row.get('arm')}: E3 evidence mismatch")
    return t


def can_reuse(row,candidate,decode,prompt_path):
    if row.get("arm") not in ARMS[:3] or row.get("model_id")!="videollama_evcap_frozen_v08":
        return False
    try:
        t=verify_prediction(row,candidate)
        telemetry=t["actual_processor_telemetry"]
        return (t["checkpoint_sha256"]=="a8b5af79d55a2c7e684771475d645be6373208bdaae5a4211bee556581eb8002"
                and t["decoding_config"]==decode and t["rng_seed"]==decode["random_seed"]
                and t["frame_indices"]==list(range(16)) and telemetry["fused_nonzero"]==0
                and bool(telemetry["image_tensor_sha256"]) and bool(telemetry["fused_aggregate_sha256"])
                and t["prompt_rendered_sha256"]==primitive_hash({"prompt_path":prompt_path,"text":t["text_evidence"]}))
    except (ValueError,KeyError,TypeError):
        return False


def collect_generation(rows, approved, model_id, bridge):
    by_id={c["query_id"]:c for c in approved}
    expected={(model_id,q,a) for q in by_id for a in ARMS}
    seen={}
    pins={}
    for row in rows:
        key=(row["model_id"],row["query_id"],row["arm"])
        if key not in expected or key in seen:
            raise ValueError(f"Invalid/duplicate E3 key: {key}")
        t=verify_prediction(row,by_id[row["query_id"]],bridge)
        tl=t["actual_processor_telemetry"]
        exact = tuple(tl.get(k) for k in ("image_tensor_sha256","actual_frames_sha256",
                                         "visual_payload_sha256","fused_aggregate_sha256"))
        if exact==(None,)*4:
            raise ValueError("No actual visual telemetry")
        pin=primitive_hash([t["decoding_config"],t["checkpoint_sha256"],t["frame_indices"],t["rng_seed"],exact])
        prev=pins.setdefault(row["query_id"],pin)
        if pin!=prev:
            raise ValueError("E3 actual visual/decode parity failure")
        seen[key]=row
    if seen.keys()!=expected:
        raise ValueError(f"Incomplete E3 pairing: {len(seen)}/{len(expected)}")
    return seen
