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

from ..common.io_utils import atomic_write_json, read_json

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
                claims_list = [str(cl).strip() for cl in c.get("claims", []) if str(cl).strip()]
                claim_objs = [
                    {
                        "claim_id": f"{bcid}_F{fi + 1:02d}",
                        "text": cl,
                        "norm_key": normalize_claim_key(cl),
                    }
                    for fi, cl in enumerate(claims_list)
                ]
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
                }

            blind_tasks.append(
                {
                    "task_type": "E2_FACT_AUDIT",
                    "blind_video_id": blind_vid,
                    "video_path": str(v["video_path"]),
                    "sampled_frame_paths": list(v.get("sampled_frame_paths", [])),
                    "phase1_core_events_required": True,
                    "captions": blind_caps,
                }
            )

        bundle = {
            "schema_version": "uarcap-q2-blind-e2-v1",
            "annotator_id": ann_id,
            "instructions": (
                "Phase 1: Watch the muted video ONLY and record 1-2 main core events before revealing captions. "
                "Phase 2: Evaluate each anonymous caption's claims strictly against what is visible in the video."
            ),
            "tasks": blind_tasks,
        }
        assert_blind_payload_clean(bundle)
        ann_path = atomic_write_json(out / f"blind_tasks_e2_{ann_id}.json", bundle)
        outputs[f"tasks_{ann_id}"] = ann_path

    key_path = atomic_write_json(out / "blind_key_e2_DO_NOT_OPEN.json", {"mapping": key_map})
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
            "schema_version": "uarcap-q2-blind-e3-v1",
            "annotator_id": ann_id,
            "instructions": (
                "Judge independently whether the anonymous caption semantically expresses Target A and/or Target B. "
                "Both may be true simultaneously. Mark negation, hedging, partial match, or generic hypernym flags."
            ),
            "tasks": blind_tasks,
        }
        assert_blind_payload_clean(bundle)
        ann_path = atomic_write_json(out / f"blind_tasks_e3_{ann_id}.json", bundle)
        outputs[f"tasks_{ann_id}"] = ann_path

    key_path = atomic_write_json(out / "blind_key_e3_DO_NOT_OPEN.json", {"mapping": key_map})
    outputs["blind_key"] = key_path
    return outputs


def unblind_annotations(
    saved_records: list[dict[str, Any]],
    blind_key_path: Path | str,
) -> list[dict[str, Any]]:
    """Re-attach hidden query_id/arm/model_id strictly AFTER human annotation finishes."""
    key_map = read_json(blind_key_path)["mapping"]
    out: list[dict[str, Any]] = []
    for rec in saved_records:
        bid = rec.get("blind_caption_id") or rec.get("blind_task_id")
        if bid not in key_map:
            raise KeyError(f"Unknown blind ID during unblinding: {bid}")
        meta = key_map[bid]
        merged = {**rec, **meta}
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
