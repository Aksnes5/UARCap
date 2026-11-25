"""Tests 03: Frozen Video-LLaMA & Modern VLM adapters, E3 candidate balance/workflow, and E3-Bridge threshold."""

from __future__ import annotations

from pathlib import Path

import pytest

from uarcap_q2.common.schemas import EligibilityStatus, ExpArm, PilotStage
from uarcap_q2.datasets.dream1k import load_dream1k_records
from uarcap_q2.datasets.vidhalluc import load_vidhalluc_ach_mcq
from uarcap_q2.generators.modern_vlm_adapter import ModernVLMConfig, Qwen25VLAdapter, render_modern_vlm_prompt
from uarcap_q2.generators.videollama_adapter import (
    FROZEN_CHECKPOINT_SHA256,
    FrozenVideoLLaMAAdapter,
    FrozenVideoLLaMAConfig,
)
from uarcap_q2.interventions.e3_bridge import NaturalConflictObservation, build_e3_bridge_cohort
from uarcap_q2.interventions.vidhalluc_e3 import (
    E3ApplicabilityChecklist,
    apply_human_applicability_review,
    build_vidhalluc_e3_candidate,
    filter_runnable_e3_candidates,
)
from uarcap_q2.retrieval.arm_builder import build_natural_retrieval_arm
from uarcap_q2.retrieval.reranker import MockXCLIPRetriever, rerank_d_consensus


def test_videollama_dry_run_argv_and_a0_wire_format(synthetic_dream1k_dir: Path, tmp_path: Path) -> None:
    recs = load_dream1k_records(synthetic_dream1k_dir / "json" / "metadata.json", synthetic_dream1k_dir / "video")
    reranked = rerank_d_consensus(MockXCLIPRetriever().retrieve_top20(recs[0]))
    pkt_a0 = build_natural_retrieval_arm(recs[0].query_id, ExpArm.ARM_A0, reranked)

    cfg = FrozenVideoLLaMAConfig(
        python_executable="/media/bn/.../venv/bin/python",
        evcap_work_root="/home/bn/uar_v06_runtime/evcap_work",
        checkpoint_path="/media/bn/.../ckpt.pt",
        llm_model_dir="/media/bn/.../vicuna-7b-v1.3",
        video_feature_dir="/home/bn/videoufo_research/features/videollama_faithful_v2",
        torch_home="/media/bn/.../torch_cache",
        hf_hub_cache="/media/bn/.../hf_cache",
    )
    adapter = FrozenVideoLLaMAAdapter(cfg)
    dry = adapter.run_batch_subprocess(tmp_path / "test.yaml", [pkt_a0], "ARM_A0", tmp_path / "vl_out", dry_run=True)
    assert dry["dry_run"] is True
    assert dry["expected_checkpoint_sha256"] == FROZEN_CHECKPOINT_SHA256
    assert "--text_topn" in dry["argv"] and "0" in dry["argv"]


def test_modern_vlm_config_prompt_and_trace(synthetic_dream1k_dir: Path) -> None:
    recs = load_dream1k_records(synthetic_dream1k_dir / "json" / "metadata.json", synthetic_dream1k_dir / "video")
    reranked = rerank_d_consensus(MockXCLIPRetriever().retrieve_top20(recs[0]))
    pkt_a0 = build_natural_retrieval_arm(recs[0].query_id, ExpArm.ARM_A0, reranked)
    pkt_a7 = build_natural_retrieval_arm(recs[0].query_id, ExpArm.ARM_A7, reranked)

    with pytest.raises(ValueError, match="max_new_tokens"):
        ModernVLMConfig(local_model_path="/tmp/qwen", prompt_regime="short_aligned", max_new_tokens=128).validate()

    cfg = ModernVLMConfig(local_model_path="/tmp/qwen25vl", pinned_revision="rev_test_01")
    adapter = Qwen25VLAdapter(cfg)
    from PIL import Image

    dummy_frames = [Image.new("RGB", (32, 32), color=(i * 10, 20, 30)) for i in range(16)]
    msgs0, rend0 = adapter.build_messages(pkt_a0, dummy_frames)
    msgs7, rend7 = adapter.build_messages(pkt_a7, dummy_frames)
    assert len(msgs0) == 2 and len(msgs7) == 2
    assert rend0["user"] != rend7["user"]
    p0 = render_modern_vlm_prompt(pkt_a0, "short_aligned")
    assert "8-18 words" in p0["user"]


def test_vidhalluc_e3_candidate_pending_until_5_item_human_approval(synthetic_vidhalluc_json: Path, tmp_path: Path) -> None:
    recs = load_vidhalluc_ach_mcq(synthetic_vidhalluc_json, tmp_path)
    cand = build_vidhalluc_e3_candidate(recs[0], neu_text="In this video clip, a person is outside in daylight.")
    assert cand.eligibility_status == EligibilityStatus.CANDIDATE_PENDING_VERIFICATION.value
    assert filter_runnable_e3_candidates([cand]) == []

    # Reject if distractor co-occurs
    bad_chk = E3ApplicabilityChecklist(
        distractor_strictly_absent=False,
        target_in_sampled_window=True,
        action_clear_in_sampled_frames=True,
        neu_strictly_neutral_and_compatible=True,
        surface_balance_acceptable=True,
        cooccurrence_risk_flagged=True,
        reviewer_id="A1",
    )
    rej = apply_human_applicability_review(cand, bad_chk)
    assert rej.eligibility_status == EligibilityStatus.REJECTED.value
    assert filter_runnable_e3_candidates([rej]) == []

    # Approve only when all 5 checks pass
    good_chk = E3ApplicabilityChecklist(
        distractor_strictly_absent=True,
        target_in_sampled_window=True,
        action_clear_in_sampled_frames=True,
        neu_strictly_neutral_and_compatible=True,
        surface_balance_acceptable=True,
        cooccurrence_risk_flagged=False,
        reviewer_id="A1",
    )
    ok_cand = apply_human_applicability_review(cand, good_chk)
    assert ok_cand.eligibility_status == EligibilityStatus.APPROVED.value
    assert len(filter_runnable_e3_candidates([ok_cand])) == 1


def test_e3_bridge_enforces_min_25_pairs_or_disables_gracefully() -> None:
    obs_small = [
        NaturalConflictObservation(
            query_id=f"dream1k_{i:04d}",
            domain="youtube",
            verified_true_fact="throwing grass into a bin",
            natural_videoufo_caption="person raking leaves",
            natural_conflict_fact="raking dry leaves on grass",
            single_target_clean=True,
            conflict_strictly_absent_in_video=True,
            proposed_neu_text="In the video, a person works outdoors today.",
            reviewer_id="A1",
        )
        for i in range(10)
    ]
    res_small = build_e3_bridge_cohort(obs_small, min_clean_pairs=25)
    assert res_small["enabled"] is False
    assert res_small["status"] == "DISABLED_INSUFFICIENT_CLEAN_PAIRS"
    assert "Narrow mechanistic claims" in res_small["claim_scope_directive"]

    obs_enough = [
        NaturalConflictObservation(
            query_id=f"dream1k_{i:04d}",
            domain="youtube",
            verified_true_fact="throwing grass into a bin",
            natural_videoufo_caption="person raking leaves",
            natural_conflict_fact="raking dry leaves on grass",
            single_target_clean=True,
            conflict_strictly_absent_in_video=True,
            proposed_neu_text="In the video, a person works outdoors today.",
            reviewer_id="A1",
        )
        for i in range(28)
    ]
    res_ok = build_e3_bridge_cohort(obs_enough, min_clean_pairs=25)
    assert res_ok["enabled"] is True and res_ok["selected_count"] == 28
