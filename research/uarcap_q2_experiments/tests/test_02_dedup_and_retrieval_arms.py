"""Tests 02: Near-duplicate screening, faithful D reranker, A0 empty rule, A1/A3/A7 prefix consistency, and E3 visual lock."""

from __future__ import annotations

from pathlib import Path

import pytest

from uarcap_q2.common.schemas import ExpArm
from uarcap_q2.datasets.dream1k import load_dream1k_records
from uarcap_q2.retrieval.arm_builder import (
    build_e3_intervention_arms,
    build_natural_retrieval_arm,
    verify_e3_visual_lock,
    verify_natural_prefix_consistency,
)
from uarcap_q2.retrieval.dedup import screen_dream1k_near_duplicates
from uarcap_q2.retrieval.reranker import (
    MockXCLIPRetriever,
    group_external_candidates_strip_only,
    rerank_d_consensus,
)


def test_dream1k_near_duplicate_screening_never_claims_exact_zero_overlap(synthetic_dream1k_dir: Path) -> None:
    recs = load_dream1k_records(synthetic_dream1k_dir / "json" / "metadata.json", synthetic_dream1k_dir / "video")
    matches = {
        recs[0].query_id: {"clip_id": "YT_HIGH.0", "caption": recs[0].reference_description, "cosine": 0.96},
        recs[1].query_id: {"clip_id": "YT_MED.0", "caption": "different scene entirely", "cosine": 0.89},
        recs[2].query_id: {"clip_id": "YT_LOW.0", "caption": "completely unrelated", "cosine": 0.35},
    }
    report = screen_dream1k_near_duplicates(recs[:3], matches)
    assert report["exact_source_zero_overlap_claimed"] is False
    assert recs[0].query_id in report["conservative_exclude_query_ids"]
    assert recs[1].query_id in report["manual_review_query_ids"]
    assert recs[2].query_id not in report["manual_review_query_ids"]


def test_strip_only_grouping_and_d_consensus_reranking() -> None:
    # Strip-only preserves punctuation/case differences (exact v0.8 predecessor fidelity!)
    raw = [
        {"caption": "  a dog runs in the park  ", "score": 0.80, "clip_id": "A.0"},
        {"caption": "a dog runs in the park", "score": 0.75, "clip_id": "A.1"},
        {"caption": "A dog runs in the park.", "score": 0.70, "clip_id": "B.0"},
        {"caption": "a dog runs in the green park", "score": 0.65, "clip_id": "C.0"},
        {"caption": "unrelated spaceship flies", "score": 0.95, "clip_id": "D.0"},
    ]
    grouped = group_external_candidates_strip_only(raw, top_m=20)
    assert len(grouped) == 4  # First two merged; case+period variant remains distinct!
    reranked = rerank_d_consensus(grouped)
    # Outlier with high similarity (0.95) gets demoted by consensus ROUGE-L!
    assert reranked[-1].caption == "unrelated spaceship flies"
    assert reranked[0].rerank_rank == 0 and reranked[0].rerank_score is not None


def test_natural_arm_a0_empty_and_prefix_consistency(synthetic_dream1k_dir: Path) -> None:
    recs = load_dream1k_records(synthetic_dream1k_dir / "json" / "metadata.json", synthetic_dream1k_dir / "video")
    reranked = rerank_d_consensus(MockXCLIPRetriever().retrieve_top20(recs[0]))
    packets = {
        arm.value: build_natural_retrieval_arm(recs[0].query_id, arm, reranked)
        for arm in (ExpArm.ARM_A0, ExpArm.ARM_A1, ExpArm.ARM_A3, ExpArm.ARM_A7)
    }
    verify_natural_prefix_consistency(packets)
    assert packets["ARM_A0"].evcap_cap_entry == [[], [], []]
    assert packets["ARM_A0"].evcap_sims_entry == [[], []]
    assert packets["ARM_A7"].text_evidence[:1] == packets["ARM_A1"].text_evidence
    assert packets["ARM_A7"].text_evidence[:3] == packets["ARM_A3"].text_evidence


def test_e3_visual_lock_base_a0_and_base_b3(synthetic_dream1k_dir: Path) -> None:
    recs = load_dream1k_records(synthetic_dream1k_dir / "json" / "metadata.json", synthetic_dream1k_dir / "video")
    reranked = rerank_d_consensus(MockXCLIPRetriever().retrieve_top20(recs[0]))

    p_a0 = build_e3_intervention_arms(
        "vidhalluc_1", "a person throws grass", "a person rakes leaves", "a person stands outside", base_visual_mode="BASE_A0"
    )
    verify_e3_visual_lock(p_a0)
    assert all(p.visual_topn == 0 and p.visual_clip_ids == [] for p in p_a0.values())

    p_b3 = build_e3_intervention_arms(
        "vidhalluc_1",
        "a person throws grass",
        "a person rakes leaves",
        "a person stands outside",
        base_visual_mode="BASE_B3",
        b3_visual_prefix=reranked[:3],
    )
    verify_e3_visual_lock(p_b3)
    assert all(p.visual_topn == 3 and len(p.visual_clip_ids) == 3 for p in p_b3.values())
