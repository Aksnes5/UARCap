"""Deterministic outcome-blind E3 proposal and machine-screening contracts."""

import json
import re
from collections import Counter

from ..common.io_utils import primitive_hash
from .vidhalluc_e3 import audit_intervention_surface_balance

V2 = "uarcap_q2.e3_neutral_approved.v2"
SCREEN = "uarcap_q2.e3_neutral_text_screen.v2"
VISUAL = "uarcap_q2.blind_statement_verdict.v1"
STATEMENT_FIELDS = ("f_plus", "f_minus", "neutral")
CHOICES = 4
FALLBACK = "reading a book in a library"

# Activity fragments match the grammatical form of the VidHalluc interventions.
ACTIONS = (
    "moving their hands", "moving their hands near a surface",
    "standing upright", "standing upright near a work surface",
    "standing with their arms bent", "standing beside a work surface",
    "sitting upright", "sitting in front of a work surface",
    "leaning slightly forward", "looking down at a surface",
    "looking to one side", "standing in a well-lit indoor area",
    "wearing a short-sleeved top", "wearing a long-sleeved top",
    "wearing dark pants", "standing on a flat surface",
    "facing a work surface", "turning their head slightly",
    "folding laundry", "folding clean laundry", "folding a stack of clean laundry",
    "tuning a guitar", "tuning an acoustic guitar", "tuning an acoustic guitar by hand",
    "watering plants", "watering potted plants", "watering a row of potted plants",
    "shoveling snow", "shoveling snow outdoors", "shoveling snow from a stone driveway",
    "kneading bread dough", "kneading bread dough by hand",
    "mending a shirt", "mending a torn cotton shirt", "mending a torn shirt with a needle",
    "polishing a shoe", "polishing a leather shoe by hand",
    "raking leaves", "raking fallen leaves in a garden",
    "sorting envelopes", "sorting colored envelopes on a desk",
    "painting a fence", "painting a wooden fence with a brush",
    "rolling a towel", "rolling a clean towel on a counter",
    "playing a piano", "playing a piano with both hands",
    "scrubbing a pan", "scrubbing a metal pan in a sink",
    "stacking plates", "stacking ceramic plates on a counter",
    "peeling an orange", "peeling an orange with bare hands",
    "combing hair", "combing long hair in front of a mirror",
)
NEGATION = re.compile(r"\b(?:not|no|never|without|instead|unable|neither|nor)\b|n['’]t\b", re.I)


def parse_json(raw):
    value = json.loads(re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw, flags=re.I))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def parse_text_screen(raw):
    value = parse_json(raw)
    required = ("natural", "same_form", "implies_target", "negates_target", "presupposes_target", "pair_single_target")
    if any(type(value.get(field)) is not bool for field in required):
        raise ValueError("Missing or invalid semantic screen fields")
    if not isinstance(value.get("rationale"), str) or not value["rationale"].strip():
        raise ValueError("Missing rationale")
    return value


def text_pass(value):
    return (value["natural"] and value["same_form"] and value["pair_single_target"]
            and not any(value[k] for k in ("implies_target", "negates_target", "presupposes_target")))


def proposals(candidate, tokenizers, bridge=False):
    positive = candidate["f_plus" if bridge else "true_action"].strip()
    negative = candidate["f_minus" if bridge else "conflict_action"].strip()
    if not positive or not negative or positive.casefold() == negative.casefold():
        return []
    qid = candidate["query_id"]
    pool = list(ACTIONS)
    if bridge:
        pool = [f"A person is {text}." for text in pool]
    ranked = []
    for text in pool:
        balance = audit_intervention_surface_balance(positive, negative, text, *tokenizers)
        if NEGATION.search(text) or not balance["surface_balanced"]:
            continue
        order = primitive_hash(["neutral-pool-v2", qid, text])
        ranked.append((order, text, balance))
    chosen = sorted(ranked)[:CHOICES]
    return [{"proposal_id": primitive_hash([qid, text]), "statement": text, "balance": balance}
            for _, text, balance in chosen]


def statement_job(query_id, statement, video_path, frames_dir, source_frame_hash, *, bridge_id=None):
    return {"job_id": primitive_hash([query_id, statement, source_frame_hash]),
            "query_id": query_id, "statement": statement, "video_path": video_path,
            "frames_dir": str(frames_dir), "expected_16_frames_sha256": source_frame_hash,
            "bridge_id": bridge_id}


def parse_visual(raw, n_frames):
    data = parse_json(raw)
    if data.get("verdict") not in ("entailed", "contradicted", "unverifiable"):
        raise ValueError("Unknown visual verdict")
    if data.get("confidence") not in ("low", "medium", "high"):
        raise ValueError("Missing confidence")
    evidence = data.get("evidence_frames")
    if not isinstance(evidence, list) or any(type(k) is not int or not 0 <= k < n_frames for k in evidence):
        raise ValueError("Invalid evidence frame")
    if data["verdict"] != "unverifiable" and not evidence:
        raise ValueError("Observed visual decisions need frame evidence")
    if not isinstance(data.get("rationale"), str) or not data["rationale"].strip():
        raise ValueError("Missing visual rationale")
    return data


def acceptable_visual(job, verdict):
    return (verdict and verdict.get("schema") == VISUAL and verdict.get("parse_status") == "OK"
            and verdict.get("job_sha256") == primitive_hash(job)
            and verdict.get("frames_sha256") == job["expected_16_frames_sha256"]
            and verdict.get("n_frames") == 16 and verdict.get("n_dense_frames") == 64
            and verdict.get("dense_frames_sha256")
            and verdict.get("video_truth") in ("entailed","contradicted","unverifiable"))


def freeze(original, plan, judgments, *, bridge=False):
    """Admit only fully linked screens, with a fixed 150/30 cohort floor."""
    if plan["original_sha256"] != primitive_hash(original):
        raise ValueError("Candidate source differs from the screening plan")
    original_rows=original.get("candidates" if bridge else "approved",[])
    input_by_id={r["query_id"]:r for r in original_rows}
    if len(input_by_id)!=len(original_rows) or len({e["candidate"]["query_id"] for e in plan["candidates"]})!=len(plan["candidates"]):
        raise ValueError("Repeated candidate identity")
    by_id = {}
    for row in judgments:
        key = row["job_id"]
        if key in by_id:
            raise ValueError("Duplicate visual judgment")
        by_id[key] = row
    approved, rejected = [], []
    reasons = Counter()
    sources = set()
    for entry in plan["candidates"]:
        c = entry["candidate"]
        if input_by_id.get(c["query_id"])!=c:
            raise ValueError("Candidate changed after freeze")
        jobs = entry["targets"]
        expected_statements=[c["f_plus"],c["f_minus"]] if bridge else [c["true_action"],c["conflict_action"]]
        if [j["statement"] for j in jobs]!=expected_statements:
            raise ValueError("Targets changed after screening")
        positive, negative = (by_id.get(job["job_id"]) for job in jobs)
        neutral = None
        reason = None
        if not entry["source_verified"]:
            reason = "SOURCE_NOT_VERIFIED"
        elif not all(acceptable_visual(j, v) for j,v in zip(jobs,(positive,negative),strict=True)):
            reason = "MISSING_OR_INVALID_TARGET_SCREEN"
        elif not (positive["video_truth"] == "entailed" and positive.get("visible_in_16_frames") == "yes"
                  and negative["video_truth"] == "contradicted"
                  and positive.get("confidence") != "low" and negative.get("confidence") != "low"):
            reason = "TARGET_PAIR_NOT_VISUALLY_ESTABLISHED"
        else:
            for proposal in entry["proposals"]:
                job = proposal["job"]
                verdict = by_id.get(job["job_id"])
                record=proposal.get("text_screen") or {}
                screened=record.get("schema")==SCREEN and record.get("parse_status")=="OK"
                if (screened and text_pass(parse_text_screen(record["raw"])) and proposal["text_screen_pass"]
                    and acceptable_visual(job, verdict)
                    and verdict["video_truth"] == "entailed" and verdict.get("confidence") in ("high","medium")):
                    neutral = {**proposal, "visual_screen": verdict}
                    break
            if neutral is None:
                reason = "NO_SEMANTICALLY_NEUTRAL_VIDEO_COMPATIBLE_TEXT"
        group = entry["independence_group"]
        if not reason and (not group or group in sources):
            reason = "DUPLICATE_OR_MISSING_INDEPENDENCE_GROUP"
        if reason:
            rejected.append({"query_id": c["query_id"], "reason": reason})
            reasons[reason] += 1
            continue
        sources.add(group)
        approved.append({**c, "original_candidate_sha256": primitive_hash(c),
                         "neutral_text": neutral["job"]["statement"], "neutral_provenance":"fixed pool, dual-tokenizer, semantic and visual machine screens",
                         "neutral_screen": neutral, "target_visual_screen": {"f_plus":positive,"f_minus":negative},
                         "frame_sha256":jobs[0]["expected_16_frames_sha256"],
                         "machine_approved":True, "human_verification":"PENDING"})
    minimum = 30 if bridge else 150
    return {"schema":V2, "scope":"CAREB_BRIDGE" if bridge else "VIDHALLUC_FORMAL",
            "eligible":len(approved)>=minimum, "n_approved":len(approved), "minimum":minimum,
            "status":"MACHINE_ASSISTED_UNVALIDATED" if len(approved)>=minimum else "BLOCKED_INSUFFICIENT_VERIFIED_CANDIDATES",
            "original_sha256":plan["original_sha256"], "screen_plan_sha256":primitive_hash(plan),
            "config_sha256":plan["config_sha256"], "approved":approved, "rejected":rejected,
            "rejection_reasons":dict(reasons)}


def parse_expressed(raw):
    value = parse_json(raw)
    label = value.get("status")
    if label not in ("AFFIRMED", "NEGATED", "VAGUE", "ABSENT", "UNVERIFIABLE"):
        raise ValueError("Unknown expression status")
    wanted = 1 if label == "AFFIRMED" else None if label == "UNVERIFIABLE" else 0
    observed = value.get("expressed")
    if "expressed" not in value or (observed is not None and type(observed) is not int) or observed != wanted:
        raise ValueError("Invalid expressed value")
    if not isinstance(value.get("evidence"), str) or (label in ("AFFIRMED","NEGATED","VAGUE") and not value["evidence"].strip()):
        raise ValueError("Missing expression evidence")
    return value
