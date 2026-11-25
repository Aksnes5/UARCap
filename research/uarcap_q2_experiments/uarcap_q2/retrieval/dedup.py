"""Source-overlap audit and two-stage conservative near-duplicate screening.

Rules enforced from protocol v1.2-final:
  - VidHalluc: exact 11-char YouTube ID exclusion vs. VideoUFO and historical benchmarks.
  - DREAM-1K: opaque integer vid 1..1000 lacks verified source IDs -> NEVER claim exact source-ID zero-overlap!
    Use two-stage conservative screening (X-CLIP cosine + frame dHash + caption Jaccard) -> manual review queue
    + conservative exclusion of high-risk candidates, explicitly declaring residual uncertainty.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from ..common.schemas import QueryRecord, SourceIdStatus
from ..datasets.video_decoder import hamming_distance_hex


@dataclass
class NearDupFlag:
    query_id: str
    domain: str
    top_corpus_clip_id: str
    top_corpus_caption: str
    xclip_cosine: float
    min_dhash_hamming: int | None
    caption_jaccard: float
    risk_tier: str  # HIGH_CONSERVATIVE_EXCLUDE | MEDIUM_MANUAL_REVIEW | LOW_PASS
    requires_human_review: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def caption_word_jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def screen_e2_near_duplicates(
    records: list[QueryRecord],
    top1_visual_matches: dict[str, dict[str, Any]],
    query_dhashes: dict[str, list[str]] | None = None,
    corpus_dhashes: dict[str, list[str]] | None = None,
    xclip_review_threshold: float = 0.88,
    xclip_conservative_exclude_threshold: float = 0.94,
    dhash_close_threshold: int = 8,
    caption_jaccard_threshold: float = 0.55,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    """Two-stage conservative near-duplicate screening for E2 (CaReBench / DREAM-1K) vs VideoUFO.

    NOTE: Similarity thresholds produce candidate review flags and conservative exclusions,
    NEVER claimed as proven exact duplicates or exact source-ID zero-overlap.
    """
    resolved_ds = dataset_id or (records[0].dataset_id if records else "CaReBench")
    q_dh = query_dhashes or {}
    c_dh = corpus_dhashes or {}

    flags: list[NearDupFlag] = []
    conservative_exclude_ids: list[str] = []
    manual_review_ids: list[str] = []
    exact_source_checked = 0

    for rec in records:
        if rec.source_id_status == SourceIdStatus.VERIFIED_EXACT.value and rec.source_id:
            exact_source_checked += 1

        match = top1_visual_matches.get(rec.query_id)
        if not match:
            continue
        clip_id = str(match.get("clip_id", ""))
        corpus_cap = str(match.get("caption", ""))
        cos = float(match.get("cosine", 0.0))
        jac = caption_word_jaccard(rec.reference_description, corpus_cap)

        min_ham: int | None = None
        if rec.query_id in q_dh and clip_id in c_dh and q_dh[rec.query_id] and c_dh[clip_id]:
            min_ham = min(
                hamming_distance_hex(h1, h2)
                for h1 in q_dh[rec.query_id]
                for h2 in c_dh[clip_id]
            )

        reasons: list[str] = []
        if cos >= xclip_review_threshold:
            reasons.append(f"xclip_cos={cos:.3f}>={xclip_review_threshold}")
        if min_ham is not None and min_ham <= dhash_close_threshold:
            reasons.append(f"min_dhash_hamming={min_ham}<={dhash_close_threshold}")
        if jac >= caption_jaccard_threshold:
            reasons.append(f"caption_jaccard={jac:.3f}>={caption_jaccard_threshold}")

        # Conservative risk tiering (never equate threshold directly to proven exact identity!)
        if (cos >= xclip_conservative_exclude_threshold) or (
            cos >= xclip_review_threshold and min_ham is not None and min_ham <= dhash_close_threshold
        ) or (cos >= xclip_review_threshold and jac >= caption_jaccard_threshold):
            tier = "HIGH_CONSERVATIVE_EXCLUDE"
            conservative_exclude_ids.append(rec.query_id)
            manual_review_ids.append(rec.query_id)
            req_rev = True
        elif reasons:
            tier = "MEDIUM_MANUAL_REVIEW"
            manual_review_ids.append(rec.query_id)
            req_rev = True
        else:
            tier = "LOW_PASS"
            req_rev = False

        flags.append(
            NearDupFlag(
                query_id=rec.query_id,
                domain=rec.domain,
                top_corpus_clip_id=clip_id,
                top_corpus_caption=corpus_cap,
                xclip_cosine=round(cos, 5),
                min_dhash_hamming=min_ham,
                caption_jaccard=round(jac, 5),
                risk_tier=tier,
                requires_human_review=req_rev,
                reason="; ".join(reasons) if reasons else "below_screening_thresholds",
            )
        )

    if resolved_ds.lower() == "carebench":
        residual_stmt = (
            "CaReBench public metadata uses anonymous FineAction parent IDs (v_XXXXXXXX) without verified original YouTube URLs/IDs. "
            "Parent-video dedup removes known within-FineAction multi-clip clustering, and near-duplicate screening produces conservative exclusions only; "
            "residual unknown source overlap with VideoUFO cannot be ruled out."
        )
    else:
        residual_stmt = (
            "DREAM-1K public metadata provides opaque integer IDs (1..1000) without verified original source URLs/YouTube IDs. "
            "Screening flags are conservative near-duplicate exclusions only and do not guarantee 100% source-level zero overlap."
        )

    return {
        "dataset_id": resolved_ds,
        "exact_source_zero_overlap_claimed": False,
        "residual_overlap_uncertainty_statement": residual_stmt,
        "total_screened": len(records),
        "exact_source_checked_count": exact_source_checked,
        "high_conservative_exclude_count": len(conservative_exclude_ids),
        "manual_review_queue_count": len(manual_review_ids),
        "conservative_exclude_query_ids": sorted(conservative_exclude_ids),
        "manual_review_query_ids": sorted(manual_review_ids),
        "flags": [f.to_dict() for f in flags],
    }


def screen_dream1k_near_duplicates(
    records: list[QueryRecord],
    top1_visual_matches: dict[str, dict[str, Any]],
    query_dhashes: dict[str, list[str]] | None = None,
    corpus_dhashes: dict[str, list[str]] | None = None,
    xclip_review_threshold: float = 0.88,
    xclip_conservative_exclude_threshold: float = 0.94,
    dhash_close_threshold: int = 8,
    caption_jaccard_threshold: float = 0.55,
) -> dict[str, Any]:
    """Backward-compatible wrapper for DREAM-1K backup near-duplicate screening."""
    return screen_e2_near_duplicates(
        records=records,
        top1_visual_matches=top1_visual_matches,
        query_dhashes=query_dhashes,
        corpus_dhashes=corpus_dhashes,
        xclip_review_threshold=xclip_review_threshold,
        xclip_conservative_exclude_threshold=xclip_conservative_exclude_threshold,
        dhash_close_threshold=dhash_close_threshold,
        caption_jaccard_threshold=caption_jaccard_threshold,
        dataset_id="DREAM-1K",
    )
