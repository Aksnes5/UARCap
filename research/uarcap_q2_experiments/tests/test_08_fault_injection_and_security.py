"""Tests 08: Active fault-injection verification (10 mandatory failure rejections) + secret & license compliance."""

from __future__ import annotations

from pathlib import Path

import pytest

from uarcap_q2.common.gates import (
    AuthorizationError,
    ProtocolIsolationError,
    check_public_release_compliance,
    enforce_pilot_formal_exclusion,
    scan_directory_for_secrets,
    scan_text_for_secrets,
    verify_stage_authorization,
)
from uarcap_q2.common.io_utils import atomic_write_json
from uarcap_q2.common.schemas import ExpArm, PilotStage, VerdictStatus
from uarcap_q2.datasets.dream1k import load_dream1k_records
from uarcap_q2.generators.base import TraceVerificationError, verify_actual_input_trace
from uarcap_q2.generators.mock_generator import MockGeneratorAdapter
from uarcap_q2.retrieval.arm_builder import (
    build_e3_intervention_arms,
    build_natural_retrieval_arm,
    verify_e3_visual_lock,
    verify_natural_prefix_consistency,
)
from uarcap_q2.retrieval.reranker import MockXCLIPRetriever, rerank_d_consensus
from uarcap_q2.runner.orchestrator import RunSpec, run_natural_retrieval_stage


def test_fault_injection_wrong_checkpoint_non_empty_a0_visual_drift_and_oom(
    synthetic_dream1k_dir: Path, tmp_path: Path
) -> None:
    recs = load_dream1k_records(synthetic_dream1k_dir / "json" / "metadata.json", synthetic_dream1k_dir / "video")
    q = recs[0]
    ret = MockXCLIPRetriever()

    # 1. Wrong checkpoint SHA-256 rejected!
    s_ckpt = run_natural_retrieval_stage(
        RunSpec("fi_ckpt", PilotStage.PILOT_A1, "DREAM-1K", [ExpArm.ARM_A0], tmp_path / "fi_ckpt", allow_cpu_mock=True),
        [q], ret, MockGeneratorAdapter(fault_mode="wrong_checkpoint"), skip_video_decode=True,
    )
    assert s_ckpt["verdict"] == VerdictStatus.FAIL.value and s_ckpt["all_traces_verified"] is False

    # 2. Non-empty ARM_A0 external evidence rejected!
    s_a0 = run_natural_retrieval_stage(
        RunSpec("fi_a0", PilotStage.PILOT_A1, "DREAM-1K", [ExpArm.ARM_A0], tmp_path / "fi_a0", allow_cpu_mock=True),
        [q], ret, MockGeneratorAdapter(fault_mode="non_empty_a0"), skip_video_decode=True,
    )
    assert s_a0["verdict"] == VerdictStatus.FAIL.value and s_a0["a0_zero_evidence_verified"] is False

    # 3. Visual input hash drift rejected!
    s_vis = run_natural_retrieval_stage(
        RunSpec("fi_vis", PilotStage.PILOT_A1, "DREAM-1K", [ExpArm.ARM_A1], tmp_path / "fi_vis", allow_cpu_mock=True),
        [q], ret, MockGeneratorAdapter(fault_mode="visual_drift"), skip_video_decode=True,
    )
    assert s_vis["verdict"] == VerdictStatus.FAIL.value and s_vis["all_traces_verified"] is False

    # 4. CUDA OOM halts safely without crashing corruptly!
    s_oom = run_natural_retrieval_stage(
        RunSpec("fi_oom", PilotStage.PILOT_A1, "DREAM-1K", [ExpArm.ARM_A0, ExpArm.ARM_A7], tmp_path / "fi_oom", allow_cpu_mock=True),
        recs[:3], ret, MockGeneratorAdapter(fault_mode="oom", fault_on_query_id=recs[1].query_id), skip_video_decode=True,
    )
    assert s_oom["verdict"] == VerdictStatus.FAIL.value and s_oom["oom_halted"] is True

    # 5. Empty output exceeds 3% gate -> FAIL!
    s_emp = run_natural_retrieval_stage(
        RunSpec("fi_emp", PilotStage.PILOT_A1, "DREAM-1K", [ExpArm.ARM_A0], tmp_path / "fi_emp", allow_cpu_mock=True),
        [q], ret, MockGeneratorAdapter(fault_mode="empty_output"), skip_video_decode=True,
    )
    assert s_emp["verdict"] == VerdictStatus.FAIL.value and s_emp["empty_output_rate"] == 1.0


def test_fault_injection_query_mismatch_prefix_mismatch_and_e3_visual_drift(synthetic_dream1k_dir: Path) -> None:
    recs = load_dream1k_records(synthetic_dream1k_dir / "json" / "metadata.json", synthetic_dream1k_dir / "video")
    reranked = rerank_d_consensus(MockXCLIPRetriever().retrieve_top20(recs[0]))
    reranked2 = rerank_d_consensus(MockXCLIPRetriever().retrieve_top20(recs[1]))

    # 6. A1 vs A7 prefix mismatch rejected!
    p_a1 = build_natural_retrieval_arm(recs[0].query_id, ExpArm.ARM_A1, reranked)
    p_a7_wrong = build_natural_retrieval_arm(recs[0].query_id, ExpArm.ARM_A7, reranked2)
    with pytest.raises(ValueError, match="Prefix consistency violation"):
        verify_natural_prefix_consistency({"ARM_A1": p_a1, "ARM_A7": p_a7_wrong})

    # 7. E3 visual contract drift across arms rejected!
    e3 = build_e3_intervention_arms("q1", "s", "c", "n", base_visual_mode="BASE_B3", b3_visual_prefix=reranked[:3])
    e3["E3_CON"].visual_contract_sha256 = "TAMPERED_VISUAL_CONTRACT"
    with pytest.raises(ValueError, match="E3 visual lock violation"):
        verify_e3_visual_lock(e3)

    # 8. Wrong query mapping in actual trace rejected!
    out = MockGeneratorAdapter().generate_one(recs[0], p_a1, PilotStage.PILOT_A1, "vsha", list(range(16)))
    wrong_pkt = build_natural_retrieval_arm(recs[1].query_id, ExpArm.ARM_A1, reranked)
    with pytest.raises(TraceVerificationError, match="Query mismatch"):
        verify_actual_input_trace(out.input_trace, wrong_pkt, "mock_ckpt_sha256_0001", "vsha")


def test_fault_injection_pilot_leak_and_unauthorized_formal_b_blocked(tmp_path: Path) -> None:
    # 9. Pilot query leaked into Formal B rejected!
    with pytest.raises(ProtocolIsolationError, match="Pilot-to-Formal contamination"):
        enforce_pilot_formal_exclusion(["dream1k_0001", "dream1k_0002"], {"dream1k_0001"})

    # 10. Unauthorized Formal B blocked even when allow_cpu_mock=True!
    with pytest.raises(AuthorizationError, match="HARD-BLOCKED"):
        verify_stage_authorization(PilotStage.FORMAL_B, auth_receipt_path=None, allow_cpu_mock=True)

    # Partial authorization (missing A3 REAL_RUNTIME_PASS receipt) still blocked!
    auth = atomic_write_json(
        tmp_path / "formal_auth.json",
        {
            "stage": "FORMAL_B",
            "approved": True,
            "approval_id": "TEST_ID",
            "frozen_pilot_params_confirmed": True,
            "frozen_formal_config_sha256": "abc123",
            "evaluator_version": "mistral-24b-v1",
            "asset_manifest_sha256": "def456",
            "pilot_exclusion_manifest_sha256": "ghi789",
        },
    )
    a1 = atomic_write_json(tmp_path / "a1.json", {"status": "REAL_RUNTIME_PASS"})
    a2 = atomic_write_json(tmp_path / "a2.json", {"status": "REAL_RUNTIME_PASS", "human_calibration_completed": True})
    with pytest.raises(AuthorizationError, match="missing prerequisite receipt for PILOT_A3"):
        verify_stage_authorization(PilotStage.FORMAL_B, auth, prior_receipts={"PILOT_A1": a1, "PILOT_A2": a2})


def test_secret_leak_scanner_and_public_release_license_compliance() -> None:
    exp_root = Path(__file__).resolve().parents[1]
    # Synthetic token string must be detected by scanner
    fake_hf = "hf_" + "A1b2C3d4E5f6G7h8I9j0K1L2"
    assert len(scan_text_for_secrets(f"token={fake_hf}")) == 1

    # Actual codebase must have ZERO leaked secrets!
    repo_hits = scan_directory_for_secrets(exp_root)
    assert repo_hits == [], f"Secret tokens found in repo: {repo_hits}"

    # Restricted VidHalluc CC-BY-NC-ND JSON or raw videos must be flagged as forbidden for public release
    viols = check_public_release_compliance(["data/vidhalluc/ach_mcq.json", "videos/clip.mp4", "uarcap_q2/common/gates.py"])
    assert len(viols) == 2
