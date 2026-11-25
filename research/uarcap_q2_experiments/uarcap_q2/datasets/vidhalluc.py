"""VidHalluc dataset adapter for E3 controlled target-fact intervention.

Primary-source facts pinned in protocol v1.2-final:
  - HF repo: chaoyuli/VidHalluc (gated: False, license: CC-BY-NC-ND 4.0)
  - Actual verified archive size: data/ACH_videos.zip = 16,754,119,398 bytes (15.60 GiB / 16.75 GB)
  - Clip filenames carry 11-char YouTube IDs (e.g., vcCwvRYqU2I_clip_4) enabling exact source-ID overlap
    auditing against VideoUFO and MSR-VTT/MSVD, plus strict 1-clip-per-YouTube-ID deduplication.
  - Scientific safeguard: MCQ distractors are NEVER assumed strictly contradictory before human video verification;
    every parsed candidate starts as EligibilityStatus.CANDIDATE_PENDING_VERIFICATION.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import re
from typing import Any

from ..common.io_utils import atomic_write_json
from ..common.schemas import EligibilityStatus, QueryRecord, SourceIdStatus

VIDHALLUC_HF_REPO = "chaoyuli/VidHalluc"
VIDHALLUC_HF_URL = "https://huggingface.co/datasets/chaoyuli/VidHalluc"
VIDHALLUC_LICENSE = "cc-by-nc-nd-4.0"
VIDHALLUC_ACH_ZIP_BYTES = 16754119398  # 15.60 GiB / 16.75 GB (verified via HF Tree API)
VIDHALLUC_TSH_ZIP_BYTES = 5217405904
VIDHALLUC_STH_ZIP_BYTES = 2073560312

YOUTUBE_CLIP_RE = re.compile(r"^([A-Za-z0-9_-]{11})_clip_(\d+)(?:\.mp4)?$")


def extract_youtube_id(clip_name: str) -> str | None:
    stem = Path(clip_name).stem
    m = YOUTUBE_CLIP_RE.match(stem)
    if m:
        return m.group(1)
    # Fallback: explicit 11-char prefix before _clip
    if "_clip_" in stem:
        prefix = stem.split("_clip_", 1)[0]
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", prefix):
            return prefix
    return None


def load_vidhalluc_ach_mcq(
    ach_mcq_path: Path | str,
    video_root: Path | str,
) -> list[QueryRecord]:
    """Parse VidHalluc ach_mcq.json into unified QueryRecords with CANDIDATE_PENDING_VERIFICATION status."""
    raw = json.loads(Path(ach_mcq_path).read_text(encoding="utf-8"))
    vroot = Path(video_root)
    seen_clips: set[str] = set()
    records: list[QueryRecord] = []

    for pair_group_id, clip_map in raw.items():
        if not isinstance(clip_map, dict):
            continue
        for clip_key, item in clip_map.items():
            if clip_key in seen_clips:
                continue
            seen_clips.add(clip_key)
            yt_id = extract_youtube_id(clip_key)
            choices = item.get("Choices", {})
            correct_letter = str(item.get("Correct Answer", "")).strip()
            true_action = str(choices.get(correct_letter, "")).strip()
            distractors = [
                str(val).strip()
                for letter, val in sorted(choices.items())
                if str(letter).strip() != correct_letter and str(val).strip()
            ]
            if not true_action or not distractors:
                continue
            # Prefer paired-group sibling's true action as the hardest scene-confusable distractor if available
            sibling_actions = []
            for other_clip, other_item in clip_map.items():
                if other_clip == clip_key:
                    continue
                oc = other_item.get("Choices", {})
                ol = str(other_item.get("Correct Answer", "")).strip()
                oa = str(oc.get(ol, "")).strip()
                if oa and oa != true_action:
                    sibling_actions.append(oa)
            primary_conflict = next((a for a in sibling_actions if a in distractors), distractors[0])

            vpath = vroot / f"{clip_key}.mp4"
            rec = QueryRecord(
                dataset_id="VidHalluc-ACH",
                query_id=f"vidhalluc_{clip_key}",
                source_id=yt_id,
                source_id_status=(
                    SourceIdStatus.VERIFIED_EXACT.value if yt_id else SourceIdStatus.UNKNOWN_OPAQUE.value
                ),
                video_path=vpath.as_posix(),
                video_sha256=None,
                duration=None,
                domain="ACH",
                reference_description=true_action,
                reference_events=[true_action],
                core_events=[true_action],
                target_true_fact=true_action,
                candidate_conflict_fact=primary_conflict,
                split="unassigned",
                # CRITICAL: Distractor is NEVER auto-certified as strictly absent from the clip!
                eligibility_status=EligibilityStatus.CANDIDATE_PENDING_VERIFICATION.value,
                metadata={
                    "pair_group_id": str(pair_group_id),
                    "clip_key": clip_key,
                    "all_distractors": distractors,
                    "question": item.get("Question", ""),
                    "license": VIDHALLUC_LICENSE,
                    "redistribution_allowed": False,
                },
            )
            rec.validate()
            records.append(rec)

    records.sort(key=lambda r: r.query_id)
    return records


def deduplicate_and_filter_vidhalluc(
    records: list[QueryRecord],
    videoufo_source_ids: set[str],
    historical_source_ids: set[str] | None = None,
) -> tuple[list[QueryRecord], dict[str, Any]]:
    """Apply exact YouTube-ID overlap exclusion and max 1-clip-per-YouTube-ID deduplication."""
    hist_ids = historical_source_ids or set()
    total_before = len(records)
    overlap_videoufo = 0
    overlap_historical = 0
    same_source_dups = 0
    unknown_source_count = 0

    kept: list[QueryRecord] = []
    seen_sources: set[str] = set()
    excluded_log: list[dict[str, str]] = []

    for r in sorted(records, key=lambda x: x.query_id):
        sid = r.source_id
        if not sid:
            unknown_source_count += 1
            excluded_log.append({"query_id": r.query_id, "reason": "UNKNOWN_SOURCE_ID"})
            continue
        if sid in videoufo_source_ids:
            overlap_videoufo += 1
            excluded_log.append({"query_id": r.query_id, "source_id": sid, "reason": "OVERLAP_VIDEOUFO"})
            continue
        if sid in hist_ids:
            overlap_historical += 1
            excluded_log.append({"query_id": r.query_id, "source_id": sid, "reason": "OVERLAP_HISTORICAL_BENCHMARK"})
            continue
        if sid in seen_sources:
            same_source_dups += 1
            excluded_log.append({"query_id": r.query_id, "source_id": sid, "reason": "SAME_SOURCE_DUPLICATE"})
            continue
        seen_sources.add(sid)
        kept.append(r)

    audit = {
        "dataset_id": "VidHalluc-ACH",
        "count_before": total_before,
        "overlap_videoufo_count": overlap_videoufo,
        "overlap_historical_count": overlap_historical,
        "same_source_duplicate_count": same_source_dups,
        "unknown_source_excluded_count": unknown_source_count,
        "eligible_unique_source_count": len(kept),
        "excluded_examples": excluded_log[:25],
    }
    return kept, audit


def sample_vidhalluc_pilot_candidates(
    unique_source_records: list[QueryRecord],
    candidate_count: int = 30,
    seed: int = 20261125,
) -> tuple[list[QueryRecord], list[QueryRecord]]:
    """Sample candidate_count unique-source VidHalluc clips for Pilot A3 5-item human applicability screening.

    All sampled pilot source IDs are permanently excluded from the Formal E3 pool.
    """
    pool = sorted(unique_source_records, key=lambda r: r.query_id)
    if len(pool) < candidate_count:
        raise ValueError(f"Insufficient unique-source VidHalluc records: need {candidate_count}, have {len(pool)}")
    rng = random.Random(seed)
    chosen = sorted(rng.sample(pool, candidate_count), key=lambda r: r.query_id)
    chosen_sources = {r.source_id for r in chosen if r.source_id}
    chosen_ids = {r.query_id for r in chosen}

    pilot = [
        QueryRecord.from_dict(
            {
                **r.to_dict(),
                "split": "pilot_a3_candidate",
                "eligibility_status": EligibilityStatus.CANDIDATE_PENDING_VERIFICATION.value,
            }
        )
        for r in chosen
    ]
    formal_pool = [
        QueryRecord.from_dict({**r.to_dict(), "split": "formal_e3_pool"})
        for r in pool
        if r.query_id not in chosen_ids and (r.source_id not in chosen_sources)
    ]
    return pilot, formal_pool


def write_vidhalluc_pilot_exclusion_manifest(pilot_candidates: list[QueryRecord], out_path: Path | str, seed: int) -> Path:
    payload = {
        "dataset_id": "VidHalluc-ACH",
        "purpose": "Permanent exclusion list of Pilot A3 candidate queries and YouTube source IDs from Formal E3",
        "sampling_seed": seed,
        "count": len(pilot_candidates),
        "excluded_query_ids": sorted(r.query_id for r in pilot_candidates),
        "excluded_source_ids": sorted({r.source_id for r in pilot_candidates if r.source_id}),
    }
    return atomic_write_json(out_path, payload)
