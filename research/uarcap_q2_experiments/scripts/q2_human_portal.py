"""Build independent blinded scope bundles and import attested human judgments."""

import argparse
from collections import Counter
from hashlib import sha256
import hmac
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uarcap_q2.common.io_utils import atomic_write_json, primitive_hash, read_json, read_jsonl, sha256_file

SCOPES = {
    "FULL_VIDEO": ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE", "NON_FACTUAL"),
    "GENERATOR_FRAMES": ("VISIBLE_SUPPORT", "VISIBLE_CONTRADICTION", "NOT_VISIBLE", "UNVERIFIABLE"),
    "CAPTION_EXPRESSION": ("AFFIRMED", "NEGATED", "VAGUE", "ABSENT", "UNVERIFIABLE"),
    "WRITTEN_REFERENCE": ("ENTAILED", "CONTRADICTED", "UNMENTIONED", "NON_FACTUAL"),
}
MODULES = {
    "A2": {"FULL_VIDEO", "GENERATOR_FRAMES"}, "A3": {"CAPTION_EXPRESSION"},
    "E3_CANDIDATE": {"FULL_VIDEO", "GENERATOR_FRAMES"}, "E3_OUTPUT": {"CAPTION_EXPRESSION"},
    "E3_BRIDGE": {"FULL_VIDEO", "GENERATOR_FRAMES"},
}
SECONDS = {"FULL_VIDEO": 45, "GENERATOR_FRAMES": 25, "CAPTION_EXPRESSION": 18, "WRITTEN_REFERENCE": 25}


def bind(row):
    module, scope = row["module"], row["scope"]
    if scope not in MODULES[module] or not row["claim"].strip():
        raise ValueError("Illegal annotation scope or empty claim")
    q = row["query_id"]
    if not q or not row.get("video_sha256") or not re.fullmatch("[0-9a-f]{64}", row["video_sha256"]):
        raise ValueError("A source-bound video SHA-256 is required, including expression jobs")
    visible = {"task_type": "SCOPED_JUDGMENT", "scope": scope, "claim": row["claim"], "labels": SCOPES[scope]}
    fpaths = [Path(p) for p in row.get("frame_paths", [])]
    frame_hashes = []
    if scope == "FULL_VIDEO":
        path = Path(row["video_path"])
        if sha256_file(path) != row["video_sha256"]:
            raise ValueError("Full-video media changed")
        visible["video_path"] = str(path)
    if scope == "GENERATOR_FRAMES":
        if len(fpaths) != 16 or len(set(fpaths)) != 16:
            raise ValueError("Need the exact written 16 generator-frame assets")
        from e3_repair import load_frames
        from uarcap_q2.generators.modern_vlm_adapter import hash_pil_frames
        if hash_pil_frames(load_frames(fpaths)) != row["generator_frames_sha256"]:
            raise ValueError("Generator frame stack differs from prediction receipt")
        frame_hashes = [sha256_file(p) for p in fpaths]
        visible["sampled_frame_paths"] = [str(p) for p in fpaths]
    if row.get("caption") is not None:
        visible["caption"] = row["caption"]
    if scope == "CAPTION_EXPRESSION":
        if not visible.get("caption") or sha256(row["caption"].encode()).hexdigest() != row["caption_sha256"]:
            raise ValueError("Caption expression requires the original sealed text")
    identity = primitive_hash(["human-scopes-v2", q, row["video_sha256"], scope, row["claim"],
                               row.get("caption"), frame_hashes, row.get("reference_version")])
    return identity, visible


def build(inputs, annotators, secret, out, internal=(), pilot_a2=()):
    if len(annotators) != 2 or len(set(annotators)) != 2 or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,36}", a) for a in annotators) or len(secret) < 32:
        raise ValueError("Require two distinct safe annotator IDs and private random blinding key")
    if internal and len(internal) != 2:
        raise ValueError("Supply both existing independent internal-reference bundles")
    if pilot_a2 and len(pilot_a2) != 2:
        raise ValueError("Supply both existing independent Pilot A2 bundles")
    sources = {}
    tasks = {}
    module_counts = Counter()
    for p in inputs:
        envelope = read_json(p)
        if envelope.get("schema") != "uarcap_q2.human_requests.v2":
            raise ValueError("Invalid frozen annotation request")
        for r in envelope["rows"]:
            identity, visible = bind(r)
            tasks.setdefault(identity, visible)
            sources.setdefault(identity, []).append({k:r.get(k) for k in ("module","query_id","artifact_key","scope","target_id")})
            module_counts[r["module"]] += 1
    idmaps = {}
    for i, a in enumerate(annotators):
        blind = []
        idmaps[a] = {}
        for key, item in tasks.items():
            bid = hmac.new(secret, (a + ":" + key).encode(), sha256).hexdigest()[:32]
            idmaps[a][bid] = key
            blind.append({"blind_task_id": bid, **item})
        blind.sort(key=lambda t: (t.get("video_path", ""), tuple(t.get("sampled_frame_paths", [])), t["blind_task_id"]))
        scope_digest = primitive_hash(blind)
        old = read_json(internal[i]) if internal else {}
        a2 = read_json(pilot_a2[i]) if pilot_a2 else {}
        legacy = []
        for pack in (old,a2):
            if not pack:
                continue
            if pack["annotator_id"] != a:
                raise ValueError("Existing annotator identity does not match")
            legacy.extend(pack["tasks"])
        ids = [t.get("blind_video_id") or t.get("blind_task_id") for t in legacy+blind]
        if not all(ids) or len(set(ids)) != len(ids):
            raise ValueError("Existing task IDs collide across annotation packs")
        atomic_write_json(out / f"blind_{a}.json", {"annotator_id": a,
            "bundle_version": "q2_scoped_v2", "bundle_sha256": primitive_hash([scope_digest,ids]),
            "probability_bundle_sha256": old.get("bundle_sha256"), "pilot_a2_bundle_sha256": a2.get("bundle_sha256"),
            "tasks": legacy+blind})
    output = {"schema": "uarcap_q2.human_scope_key.v2", "idmaps": idmaps, "tasks": sources,
              "labels": {k:list(v["labels"]) for k,v in tasks.items()},
              "input_sha256": {str(p): sha256_file(p) for p in (*inputs,*internal,*pilot_a2)}}
    output["key_sha256"] = primitive_hash(output)
    atomic_write_json(out / "PRIVATE_PORTAL_KEY.json", output)
    workload = Counter(t["scope"] for t in tasks.values())
    atomic_write_json(out / "HUMAN_WORKLOAD.json", {"status": "AWAITING_INDEPENDENT_HUMAN_REVIEW",
        "source_slots_by_module": dict(module_counts), "unique_scoped_jobs_per_reviewer": dict(workload),
        "exact_reused_slots": sum(module_counts.values()) - len(tasks),
        "internal_probability_videos_per_reviewer": len(old.get("tasks", [])),
        "internal_probability_captions_per_reviewer": sum(len(r.get("captions", [])) for r in old.get("tasks", [])),
        "pilot_a2_videos_per_reviewer": len(a2.get("tasks", [])),
        "scoped_hours_estimate_two_reviewers": sum(2*SECONDS[k]*v for k,v in workload.items())/3600,
        "uninstantiated_modules": sorted(set(MODULES) - set(module_counts)),
        "external_human_reference_is_video_gold": False})


def stage_e3(module, freeze, frames, outputs, out):
    doc = read_json(freeze)
    bridge = doc.get("scope") == "CAREB_BRIDGE"
    cohort = doc.get("approved", [])
    if module == "A3":
        cohort = doc["items"]
    elif doc.get("schema") != "uarcap_q2.e3_neutral_approved.v2" or doc.get("eligible") is not True:
        raise ValueError("Require an eligible versioned V2 candidate freeze")
    if (module=="E3_BRIDGE" and not bridge) or (module=="E3_CANDIDATE" and bridge):
        raise ValueError("Candidate scope does not match the requested module")
    by_id = {r["query_id"]:r for r in cohort}
    if len(by_id) != len(cohort):
        raise ValueError("Repeated candidate in human cohort")
    vsha = {}
    rows = []
    def make(c, scope, role, target):
        q = c["query_id"]
        if q not in vsha:
            vsha[q] = sha256_file(Path(c["video_path"]))
        return {"module":module,"scope":scope,"query_id":q,"video_sha256":vsha[q],
                "target_id":role,"claim":target}

    from uarcap_q2.interventions.e3_output import target_text
    if module in ("E3_CANDIDATE","E3_BRIDGE"):
        for c in cohort:
            q = c["query_id"]
            frame_dir=Path(c["verified_reserve_frames_dir"]) if c.get("verified_reserve_frames_dir") else frames/q
            frame_paths=[str(p) for p in sorted(frame_dir.glob("frame_*.jpg"))]
            positive,negative = target_text(c,bridge)
            for role, target in (("f_plus",positive),("f_minus",negative),("neutral",c["neutral_text"])):
                rows.append({**make(c,"FULL_VIDEO",role,target),"video_path":c["video_path"],"artifact_key":q})
                rows.append({**make(c,"GENERATOR_FRAMES",role,target),"artifact_key":q,
                             "frame_paths":frame_paths,"generator_frames_sha256":c["frame_sha256"]})
    else:
        qids = (set(by_id) if module=="A3" else
                set(sorted(by_id,key=lambda q:primitive_hash(["e3-output-review-v2",20261126,q]))[:30]))
        if not outputs:
            raise ValueError("Need sealed caption outputs for expression review")
        for path in outputs:
            records = read_jsonl(path)
            if module != "A3":
                receipt = read_json(path.parent/"run_receipt.json")
                if (receipt.get("complete") is not True or receipt.get("cohort_sha256") != primitive_hash(doc)
                        or receipt.get("predictions_sha256") != sha256_file(path)):
                    raise ValueError("E3 output review must use exact V2 generation output")
            seen = set()
            for pred in records:
                q = pred["query_id"]
                if q not in qids or pred.get("ok") is not True:
                    continue
                if module=="A3" and pred["arm"]=="E3_NEU":
                    continue
                key = (q,pred["model_id"],pred["arm"])
                if key in seen:
                    raise ValueError("Repeated expression task caption")
                seen.add(key)
                c = by_id[q]
                if module=="A3":
                    pos,neg = c["true_action"],c["conflict_action"]
                else:
                    pos,neg = target_text(c,bridge)
                for role, target in (("f_plus",pos),("f_minus",neg)):
                    text = pred["caption"]
                    rows.append({**make(c,"CAPTION_EXPRESSION",role,target),"caption":text,
                                 "caption_sha256":sha256(text.encode()).hexdigest(),"artifact_key":list(key)})
    atomic_write_json(out, {"schema":"uarcap_q2.human_requests.v2","rows":rows,
        "scope":module,"input_sha256":{str(p):sha256_file(p) for p in (freeze,*outputs)}})


def import_reviews(key_path, paths, attestations, resolutions, out):
    if len(paths) != 2 or len(attestations) != 2:
        raise ValueError("Require two independent export and attestation pairs")
    key = read_json(key_path)
    digest = key.pop("key_sha256")
    if primitive_hash(key) != digest:
        raise ValueError("Private binding checksum changed")
    reviews = {}
    for path, attestation in zip(paths, attestations, strict=True):
        export, att = (read_json(p) for p in (path, attestation))
        who = export["annotator_id"]
        if (who in reviews or who not in key["idmaps"] or att.get("annotator_id") != who
                or att.get("review_export_sha256") != sha256_file(path) or att.get("independent_blind_review_attested") is not True):
            raise ValueError("Unattested or non-independent human export")
        labels = {}
        for rec in export["records"]:
            if not rec.get("completed") or rec["task_type"] != "SCOPED_JUDGMENT":
                continue
            identity = key["idmaps"][who][rec["task_key"]]
            label, evidence = rec.get("scope_label"), rec.get("scope_rationale")
            if identity in labels or label not in key["labels"][identity] or not evidence or not evidence.strip():
                raise ValueError("Duplicate or incomplete scoped judgment")
            labels[identity] = (label, evidence)
        reviews[who] = labels
    resolve = read_json(resolutions) if resolutions else {}
    rows, unresolved = {}, {}
    for identity in key["tasks"]:
        available = [v[identity] for v in reviews.values() if identity in v]
        reason = "MISSING_INDEPENDENT_REVIEWS"
        if len(available) == 2:
            if available[0][0] == available[1][0]:
                rows[identity] = {"label":available[0][0], "basis":"AGREEMENT", "reviews":available}
                continue
            reason = "AWAITING_ADJUDICATION"
            chosen = resolve.get("rows", {}).get(identity)
            if chosen:
                if (resolve.get("independent_adjudication_attested") is not True or not resolve.get("adjudicator_id")
                        or resolve["adjudicator_id"] in key["idmaps"] or resolve.get("key_sha256") != digest
                        or resolve.get("review_export_sha256") != [sha256_file(p) for p in paths]
                        or not chosen.get("rationale") or chosen.get("label") not in key["labels"][identity]):
                    raise ValueError("Adjudication missing independent attestation or bound rationale")
                rows[identity] = {"label":chosen["label"], "basis":"ADJUDICATION", "reviews":available, **chosen}
                continue
        unresolved[identity] = reason
    subcounts = {}
    candidate_labels = {}
    for identity, aliases in key["tasks"].items():
        for alias in aliases:
            c = subcounts.setdefault(alias["module"], Counter())
            c["planned"] += 1
            c["resolved"] += identity in rows
            if alias["module"] in ("E3_CANDIDATE","E3_BRIDGE") and identity in rows:
                key_tuple = (alias["module"],alias["query_id"])
                candidate_labels.setdefault(key_tuple,{})[alias["scope"],alias["target_id"]] = rows[identity]["label"]
    visual_gate = {}
    rules = {("FULL_VIDEO","f_plus"):"SUPPORTED",("FULL_VIDEO","f_minus"):"CONTRADICTED",
             ("FULL_VIDEO","neutral"):"SUPPORTED",("GENERATOR_FRAMES","f_plus"):"VISIBLE_SUPPORT"}
    for (module,q), labels in candidate_labels.items():
        required = {(scope,role) for scope in ("FULL_VIDEO","GENERATOR_FRAMES") for role in ("f_plus","f_minus","neutral")}
        state = "PENDING_INCOMPLETE" if labels.keys()!=required else ("SCREEN_COMPATIBLE" if all(labels.get(k)==v for k,v in rules.items()) else "HUMAN_APPLICABILITY_DISAGREES")
        visual_gate.setdefault(module,{})[q] = state
    atomic_write_json(out, {"schema":"uarcap_q2.adjudicated_scoped_human.v2", "key_sha256":digest,
        "status":"AWAITING_HUMAN" if unresolved else "SCOPED_REVIEW_COMPLETE",
        "rows": [{**row,"scope_id":i,"source_tasks":key["tasks"][i]} for i,row in rows.items()],
        "unresolved": unresolved, "modules": subcounts,
        "candidate_visual_review":visual_gate,
        "human_review_gate": {m: c["resolved"] == c["planned"] for m,c in subcounts.items()},
        "scientific_calibration":"REQUIRES_ENDPOINT_SPECIFIC_CHECKS",
        "input_sha256":{str(p):sha256_file(p) for p in (key_path,*paths,*attestations,*([resolutions] if resolutions else []))}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    build_p = sub.add_parser("build")
    build_p.add_argument("--input", type=Path, action="append", required=True)
    build_p.add_argument("--annotator", action="append", required=True)
    build_p.add_argument("--secret", type=Path, required=True)
    build_p.add_argument("--internal", type=Path, action="append", default=[])
    build_p.add_argument("--pilot-a2", type=Path, action="append", default=[])
    ingest = sub.add_parser("import")
    ingest.add_argument("--key", type=Path, required=True)
    ingest.add_argument("--review", type=Path, action="append", required=True)
    ingest.add_argument("--attestation", type=Path, action="append", required=True)
    ingest.add_argument("--resolutions", type=Path)
    stage = sub.add_parser("stage-e3")
    stage.add_argument("--module", choices=("A3","E3_CANDIDATE","E3_OUTPUT","E3_BRIDGE"), required=True)
    stage.add_argument("--freeze", type=Path, required=True)
    stage.add_argument("--frames", type=Path)
    stage.add_argument("--predictions", type=Path, action="append", default=[])
    for parser in (build_p, ingest, stage):
        parser.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.command == "build":
        build(a.input, a.annotator, a.secret.read_bytes(), a.out, a.internal, a.pilot_a2)
    elif a.command == "import":
        import_reviews(a.key, a.review, a.attestation, a.resolutions, a.out)
    else:
        stage_e3(a.module, a.freeze, a.frames, a.predictions, a.out)


if __name__ == "__main__":
    main()
