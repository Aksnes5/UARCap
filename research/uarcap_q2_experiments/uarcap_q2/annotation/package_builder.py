"""Blind annotation package builder with per-annotator randomization, Phase-1 core-event lock, and safe claim reuse.

Guarantees:
  - Blind tasks NEVER contain arm names, model names, retrieved captions, or reference descriptions.
  - Secret mapping (blind_caption_id -> {arm, model_id, query_id}) is written strictly to an isolated key file.
  - Normalized-claim reuse keys ONLY on (video_id, normalized_claim) without exposing which other arm/model shares it.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import random
import re
from typing import Any

from ..common.io_utils import atomic_write_json, primitive_hash, read_json

FORBIDDEN_BLIND_KEYS = {
    "arm",
    "model_id",
    "retrieved_captions",
    "text_evidence",
    "reference_description",
    "intervention_arm",
    "sup_text",
    "con_text",
    "neu_text",
}


def normalize_claim_key(claim: str) -> str:
    """Canonical normalization for safe within-video truth reuse."""
    s = re.sub(r"[^a-z0-9'\s]", " ", claim.lower())
    return re.sub(r"\s+", " ", s).strip()


def assert_blind_payload_clean(payload: Any, path: str = "root") -> None:
    """Recursively assert that NO unblinded metadata key ever leaks into an annotator task file."""
    if isinstance(payload, dict):
        for k, v in payload.items():
            if k in FORBIDDEN_BLIND_KEYS:
                raise ValueError(f"BLINDNESS LEAK DETECTED at {path}.{k}!")
            assert_blind_payload_clean(v, f"{path}.{k}")
    elif isinstance(payload, list):
        for i, item in enumerate(payload):
            assert_blind_payload_clean(item, f"{path}[{i}]")


def build_blind_e2_annotation_package(
    videos: list[dict[str, Any]],
    annotator_ids: list[str],
    out_dir: Path | str,
    seed: int = 20261125,
) -> dict[str, Path]:
    """Create per-annotator blind E2 task bundles + isolated unblinding key.

    Each entry in `videos`:
      {
        "query_id": str,
        "video_path": str,
        "sampled_frame_paths": list[str] (optional),
        "captions": [
          {"arm": str, "model_id": str, "caption": str, "claims": list[str]}
        ]
      }
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key_map: dict[str, dict[str, Any]] = {}
    outputs: dict[str, Path] = {}

    for ann_id in annotator_ids:
        rng = random.Random(int(sha256(f"{seed}:{ann_id}".encode()).hexdigest()[:12], 16))
        video_order = list(videos)
        rng.shuffle(video_order)

        blind_tasks: list[dict[str, Any]] = []
        for v_idx, v in enumerate(video_order):
            qid = str(v["query_id"])
            blind_vid = f"V{v_idx + 1:03d}_{sha256(f'{ann_id}:{qid}'.encode()).hexdigest()[:6]}"
            cap_entries = list(v["captions"])
            rng.shuffle(cap_entries)

            blind_caps: list[dict[str, Any]] = []
            for c_idx, c in enumerate(cap_entries):
                bcid = f"{blind_vid}_C{c_idx + 1:02d}"
                raw_claims = c.get("claims", [])
                claim_objs = []
                claim_id_map: dict[str, dict[str, Any]] = {}
                for fi, cl_entry in enumerate(raw_claims):
                    if isinstance(cl_entry, dict):
                        cl_txt = str(cl_entry.get("claim", cl_entry.get("text", ""))).strip()
                        auto_cid = str(
                            cl_entry.get("claim_id")
                            or f"{qid}::{c['arm']}::{c['model_id']}::C{fi + 1:02d}"
                        )
                    else:
                        cl_txt = str(cl_entry).strip()
                        auto_cid = f"{qid}::{c['arm']}::{c['model_id']}::C{fi + 1:02d}"
                    if not cl_txt:
                        continue
                    blind_cid = f"{bcid}_F{len(claim_objs) + 1:02d}"
                    claim_objs.append(
                        {
                            "claim_id": blind_cid,
                            "text": cl_txt,
                            "norm_key": normalize_claim_key(cl_txt),
                            "exact_claim_key": sha256(cl_txt.encode("utf-8")).hexdigest(),
                        }
                    )
                    claim_id_map[blind_cid] = {
                        "auto_claim_id": auto_cid,
                        "claim_index": fi,
                        "claim_text": cl_txt,
                    }
                blind_caps.append(
                    {
                        "blind_caption_id": bcid,
                        "caption": str(c["caption"]).strip(),
                        "claims": claim_objs,
                    }
                )
                key_map[bcid] = {
                    "annotator_id": ann_id,
                    "query_id": qid,
                    "blind_video_id": blind_vid,
                    "arm": c["arm"],
                    "model_id": c["model_id"],
                    "claim_id_map": claim_id_map,
                    **{field:c[field] for field in ("caption_sha256","claim_split_sha256","reference_description_sha256","core_lock_sha256","evaluator_id") if field in c},
                }

            blind_tasks.append(
                {
                    "task_type": "E2_FACT_AUDIT",
                    "blind_video_id": blind_vid,
                    "video_path": str(v["video_path"]),
                    "sampled_frame_paths": list(v.get("sampled_frame_paths", [])),
                    "phase1_core_events_required": True,
                    "human_reference_spans":v.get("human_reference_spans",[]),
                    "frozen_reference_core":v.get("frozen_reference_core"),
                    "captions": blind_caps,
                }
            )

        bundle = {
            "schema_version": "uarcap-q2-blind-e2-v2",
            "bundle_version": "e2-v2",
            "annotator_id": ann_id,
            "instructions": (
                "Phase 1: Watch the muted video ONLY and record 1-2 main core events before revealing captions. "
                "Phase 2: Evaluate each anonymous caption's claims strictly against what is visible in the video."
            ),
            "tasks": blind_tasks,
        }
        bundle["bundle_sha256"] = primitive_hash({"annotator_id": ann_id, "tasks": blind_tasks})
        assert_blind_payload_clean(bundle)
        ann_path = atomic_write_json(out / f"blind_tasks_e2_{ann_id}.json", bundle)
        outputs[f"tasks_{ann_id}"] = ann_path

    package_sha = primitive_hash(key_map)
    key_path = atomic_write_json(
        out / "blind_key_e2_DO_NOT_OPEN.json",
        {
            "schema_version": "uarcap-q2-blind-key-e2-v2",
            "bundle_version": "e2-v2",
            "package_sha256": package_sha,
            "mapping": key_map,
        },
    )
    outputs["blind_key"] = key_path
    return outputs


def build_blind_e3_annotation_package(
    items: list[dict[str, Any]],
    annotator_ids: list[str],
    out_dir: Path | str,
    seed: int = 20261125,
) -> dict[str, Path]:
    """Create per-annotator blind E3 dual-target verification bundles.

    Each item in `items`:
      {
        "query_id": str, "arm": str, "model_id": str, "video_path": str,
        "caption": str, "target_true_fact": str, "target_conflict_fact": str
      }
    Randomizes which target is presented as Target A vs Target B so annotators don't know which is true/conflict!
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key_map: dict[str, dict[str, Any]] = {}
    outputs: dict[str, Path] = {}

    for ann_id in annotator_ids:
        rng = random.Random(int(sha256(f"e3:{seed}:{ann_id}".encode()).hexdigest()[:12], 16))
        shuffled = list(items)
        rng.shuffle(shuffled)

        blind_tasks: list[dict[str, Any]] = []
        for idx, it in enumerate(shuffled):
            raw_sig = f"{ann_id}:{it['query_id']}:{it['arm']}:{it['model_id']}"
            btid = f"E3_{idx + 1:04d}_{sha256(raw_sig.encode()).hexdigest()[:6]}"
            swap = rng.choice([True, False])
            tgt_a = it["target_conflict_fact"] if swap else it["target_true_fact"]
            tgt_b = it["target_true_fact"] if swap else it["target_conflict_fact"]
            blind_tasks.append(
                {
                    "task_type": "E3_DUAL_TARGET",
                    "blind_task_id": btid,
                    "video_path": str(it["video_path"]),
                    "caption": str(it["caption"]).strip(),
                    "target_a": str(tgt_a).strip(),
                    "target_b": str(tgt_b).strip(),
                }
            )
            key_map[btid] = {
                "annotator_id": ann_id,
                "query_id": it["query_id"],
                "arm": it["arm"],
                "model_id": it["model_id"],
                "target_a_role": "conflict" if swap else "true",
                "target_b_role": "true" if swap else "conflict",
            }

        bundle = {
            "schema_version": "uarcap-q2-blind-e3-v2",
            "bundle_version": "e3-v2",
            "annotator_id": ann_id,
            "instructions": (
                "Judge independently whether the anonymous caption semantically expresses Target A and/or Target B. "
                "Both may be true simultaneously. Mark negation, hedging, partial match, or generic hypernym flags."
            ),
            "tasks": blind_tasks,
        }
        bundle["bundle_sha256"] = primitive_hash({"annotator_id": ann_id, "tasks": blind_tasks})
        assert_blind_payload_clean(bundle)
        ann_path = atomic_write_json(out / f"blind_tasks_e3_{ann_id}.json", bundle)
        outputs[f"tasks_{ann_id}"] = ann_path

    package_sha = primitive_hash(key_map)
    key_path = atomic_write_json(
        out / "blind_key_e3_DO_NOT_OPEN.json",
        {
            "schema_version": "uarcap-q2-blind-key-e3-v2",
            "bundle_version": "e3-v2",
            "package_sha256": package_sha,
            "mapping": key_map,
        },
    )
    outputs["blind_key"] = key_path
    return outputs


def unblind_annotations(
    saved_records: list[dict[str, Any]],
    blind_key_path: Path | str,
) -> list[dict[str, Any]]:
    """Re-attach hidden query_id/arm/model_id/auto_claim_id strictly AFTER human annotation finishes.

    Supports BOTH flat caption/task records AND raw web-UI AnnotationStore task exports (which nest
    captions under `rec["captions"][blind_caption_id]` or E3 answers under `rec["e3"]`).
    """
    key_obj = read_json(blind_key_path)
    key_map = key_obj["mapping"]
    pkg_sha = key_obj.get("package_sha256", "")
    out: list[dict[str, Any]] = []

    for rec in saved_records:
        # Handle raw web UI E2 video-level export (containing nested `captions` dict)
        if rec.get("task_type") == "E2_FACT_AUDIT" and isinstance(rec.get("captions"), dict):
            for bcid, crec in rec["captions"].items():
                if bcid not in key_map:
                    raise KeyError(f"Unknown blind caption ID during unblinding: {bcid}")
                meta = key_map[bcid]
                cid_map = meta.get("claim_id_map", {})
                claims_by_auto_id: dict[str, str] = {}
                for blind_cid, lbl in dict(crec.get("claims", {})).items():
                    if blind_cid in cid_map:
                        auto_cid = cid_map[blind_cid]["auto_claim_id"]
                    else:
                        auto_cid = blind_cid
                    claims_by_auto_id[auto_cid] = str(lbl)
                # Map source_claim_ids in split/merge revisions from blind_cid -> auto_cid
                mapped_revs = []
                for rev in crec.get("claim_revisions", []):
                    src_mapped = [
                        cid_map[sc]["auto_claim_id"] if sc in cid_map else sc
                        for sc in rev.get("source_claim_ids", [])
                    ]
                    mapped_revs.append({**rev, "source_claim_ids": src_mapped})
                out.append(
                    {
                        "blind_caption_id": bcid,
                        "annotator_id": meta["annotator_id"],
                        "query_id": meta["query_id"],
                        "arm": meta["arm"],
                        "model_id": meta["model_id"],
                        "package_sha256": pkg_sha,
                        "claim_labels_by_id": claims_by_auto_id,
                        "added_claims": list(crec.get("added_claims", [])),
                        "claim_revisions": mapped_revs,
                        "core_events": list(rec.get("core_events", [])),
                        "elapsed_sec": float(rec.get("elapsed_sec", 0.0)),
                    }
                )
            continue

        # Handle raw web UI E3 export (nesting answers under `rec["e3"]` and `task_key`)
        if rec.get("task_type") == "E3_DUAL_TARGET" and isinstance(rec.get("e3"), dict):
            bid = rec.get("blind_task_id") or rec.get("task_key")
            if bid not in key_map:
                raise KeyError(f"Unknown blind E3 task ID during unblinding: {bid}")
            meta = key_map[bid]
            e3 = rec["e3"]
            a_val = int(bool(e3.get("target_a_expressed", False)))
            b_val = int(bool(e3.get("target_b_expressed", False)))
            neg_a = bool(e3.get("negated_a", False))
            neg_b = bool(e3.get("negated_b", False))
            if meta["target_a_role"] == "true":
                yt, yc, nt, nc = a_val, b_val, neg_a, neg_b
            else:
                yt, yc, nt, nc = b_val, a_val, neg_b, neg_a
            out.append(
                {
                    "blind_task_id": bid,
                    "annotator_id": meta["annotator_id"],
                    "query_id": meta["query_id"],
                    "arm": meta["arm"],
                    "model_id": meta["model_id"],
                    "package_sha256": pkg_sha,
                    "y_true": yt,
                    "y_conflict": yc,
                    "negated_true": nt,
                    "negated_conflict": nc,
                    "hedged": bool(e3.get("hedged", False)),
                    "partial": bool(e3.get("partial", False)),
                    "generic_only": bool(e3.get("generic_only", False)),
                    "elapsed_sec": float(rec.get("elapsed_sec", 0.0)),
                }
            )
            continue

        bid = rec.get("blind_caption_id") or rec.get("blind_task_id")
        if bid not in key_map:
            raise KeyError(f"Unknown blind ID during unblinding: {bid}")
        meta = key_map[bid]
        merged = {**rec, **meta, "package_sha256": pkg_sha}
        if "claim_labels_by_id" in rec and isinstance(rec["claim_labels_by_id"], dict):
            cid_map = meta.get("claim_id_map", {})
            merged["claim_labels_by_id"] = {
                (cid_map[k]["auto_claim_id"] if k in cid_map else k): str(v)
                for k, v in rec["claim_labels_by_id"].items()
            }
        # Map randomized target_a/target_b back to y_true / y_conflict for E3
        if "target_a_role" in meta:
            a_val = int(bool(rec.get("target_a_expressed", False)))
            b_val = int(bool(rec.get("target_b_expressed", False)))
            if meta["target_a_role"] == "true":
                merged["y_true"], merged["y_conflict"] = a_val, b_val
            else:
                merged["y_true"], merged["y_conflict"] = b_val, a_val
        out.append(merged)
    return out
