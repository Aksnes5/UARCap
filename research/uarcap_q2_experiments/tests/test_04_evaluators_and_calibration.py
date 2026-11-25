"""Tests 04: 3-Way Reference-Fact Evaluator validation, extraction audit, Blind E3 evaluator, and 3x4 calibration redlines."""

from __future__ import annotations

import pytest

from uarcap_q2.common.schemas import ExpArm, RefRelation, VerdictStatus, VideoTruthLabel
from uarcap_q2.evaluators.backends import MockJudgeBackend
from uarcap_q2.evaluators.calibration import AlignedAtomRecord, audit_evaluator_calibration
from uarcap_q2.evaluators.e3_semantic_eval import BlindE3SemanticEvaluator
from uarcap_q2.evaluators.reference_fact_eval import (
    PROTOCOL_LAYER,
    ThreeWayReferenceFactEvaluator,
    audit_atomic_extraction,
)


def test_3way_evaluator_clean_and_cross_family_enforcement() -> None:
    with pytest.raises(ValueError, match="Cross-family"):
        ThreeWayReferenceFactEvaluator(MockJudgeBackend(model_family="qwen"), generator_model_family="qwen")

    ev = ThreeWayReferenceFactEvaluator(MockJudgeBackend(model_family="mistral"), generator_model_family="qwen")
    res = ev.evaluate_caption(
        query_id="dream1k_0001",
        arm=ExpArm.ARM_A7.value,
        model_id="qwen25vl",
        caption="A person is raking leaves and standing near a bicycle.",
        reference_description="A person is throwing grass in the garbage.",
        core_events=["A person is throwing grass in the garbage."],
    )
    assert res.protocol_layer == PROTOCOL_LAYER
    assert res.parse_status == "OK"
    assert res.n_contra_ref == 1
    assert res.n_unment_ref == 1


@pytest.mark.parametrize(
    "fault",
    ["invalid_json", "empty", "refusal", "truncated", "illegal_label", "missing_field", "no_evidence", "mismatched_claim", "duplicate_claims"],
)
def test_3way_evaluator_never_defaults_parse_failure_to_true_or_false(fault: str) -> None:
    ev = ThreeWayReferenceFactEvaluator(MockJudgeBackend(fault_mode=fault))
    res = ev.evaluate_caption(
        query_id="q1",
        arm=ExpArm.ARM_A0.value,
        model_id="m1",
        caption="A dog runs fast.",
        reference_description="A dog runs fast.",
        core_events=["A dog runs fast."],
    )
    assert res.parse_status == "PARSE_FAILED"
    assert res.n_contra_ref is None and res.n_entail_ref is None and res.p_unment_ref is None


def test_atomic_extraction_audit_detects_false_claim_omission() -> None:
    extracted = {"item1": ["A man walks down the street"]}
    human = {
        "item1": [
            {"claim": "A man walks down the street", "is_video_error": False},
            {"claim": "A pink elephant flies overhead", "is_video_error": True},
        ]
    }
    aud = audit_atomic_extraction(extracted, human)
    assert aud["extraction_recall"] == 0.5
    assert aud["false_claim_omission_rate"] == 1.0  # Caught dropped false claim!


def test_blind_e3_evaluator_cooccurrence_negation_hedging_and_anticircularity() -> None:
    with pytest.raises(ValueError, match="Circular"):
        BlindE3SemanticEvaluator(MockJudgeBackend(model_id="builder_m"), intervention_builder_model_id="builder_m")

    ev = BlindE3SemanticEvaluator(MockJudgeBackend(model_id="judge_m"), intervention_builder_model_id="builder_m")

    # Co-occurrence (1, 1) explicitly allowed!
    both = ev.evaluate_caption(
        "q1", ExpArm.E3_CON.value, "m1",
        "A person is throwing grass in the garbage and raking leaves.",
        "throwing grass in the garbage", "raking leaves",
    )
    assert both.y_true == 1 and both.y_conflict == 1

    # Explicit negation of conflict -> y_conflict = 0, negated_conflict = True
    neg = ev.evaluate_caption(
        "q2", ExpArm.E3_CON.value, "m1",
        "A person is throwing grass in the garbage, not raking leaves.",
        "throwing grass in the garbage", "raking leaves",
    )
    assert neg.y_true == 1 and neg.y_conflict == 0 and neg.negated_conflict is True

    # Generic hypernym -> (0, 0)
    gen = ev.evaluate_caption(
        "q3", ExpArm.E3_NEU.value, "m1",
        "A person is working in the yard.",
        "throwing grass in the garbage", "raking leaves",
    )
    assert gen.y_true == 0 and gen.y_conflict == 0 and gen.generic_only is True


def test_calibration_returns_inconclusive_on_zero_errors_and_fails_on_a7_omission_bias() -> None:
    # 1. Zero true errors -> INCONCLUSIVE (NEVER fake PASS!)
    zero_err_recs = [
        AlignedAtomRecord(f"v{i}", arm, "m1", RefRelation.ENTAILED.value, VideoTruthLabel.VIDEO_SUPPORTED.value)
        for i in range(10)
        for arm in (ExpArm.ARM_A0.value, ExpArm.ARM_A7.value)
    ]
    res_inc = audit_evaluator_calibration(zero_err_recs, n_boot=100)
    assert res_inc["verdict"] == VerdictStatus.INCONCLUSIVE.value

    # 2. A7 systematic omission bias: A0 catches errors, A7 hides errors inside Unmentioned -> FAIL!
    biased_recs: list[AlignedAtomRecord] = []
    for i in range(12):
        # A0: 1 true error caught cleanly as Contradicted + 2 true supported Entailed
        biased_recs.append(AlignedAtomRecord(f"v{i}", ExpArm.ARM_A0.value, "m1", RefRelation.CONTRADICTED.value, VideoTruthLabel.VIDEO_CONTRADICTED.value))
        biased_recs.append(AlignedAtomRecord(f"v{i}", ExpArm.ARM_A0.value, "m1", RefRelation.ENTAILED.value, VideoTruthLabel.VIDEO_SUPPORTED.value))
        # A7: 2 true errors, BOTH missed into Unmentioned!
        biased_recs.append(AlignedAtomRecord(f"v{i}", ExpArm.ARM_A7.value, "m1", RefRelation.UNMENTIONED.value, VideoTruthLabel.VIDEO_CONTRADICTED.value))
        biased_recs.append(AlignedAtomRecord(f"v{i}", ExpArm.ARM_A7.value, "m1", RefRelation.UNMENTIONED.value, VideoTruthLabel.VIDEO_CONTRADICTED.value))
        biased_recs.append(AlignedAtomRecord(f"v{i}", ExpArm.ARM_A7.value, "m1", RefRelation.ENTAILED.value, VideoTruthLabel.VIDEO_SUPPORTED.value))
    res_fail = audit_evaluator_calibration(biased_recs, n_boot=100)
    assert res_fail["verdict"] == VerdictStatus.FAIL.value
    assert any("A7 systematic omission bias" in r for r in res_fail["fail_reasons"])

    # 3. Well-calibrated unbiased evaluator -> PASS!
    good_recs: list[AlignedAtomRecord] = []
    for i in range(12):
        # Vary paired delta across videos so Spearman rho is well-defined and high
        good_recs.append(AlignedAtomRecord(f"v{i}", ExpArm.ARM_A0.value, "m1", RefRelation.ENTAILED.value, VideoTruthLabel.VIDEO_SUPPORTED.value))
        if i % 3 == 0:
            good_recs.append(AlignedAtomRecord(f"v{i}", ExpArm.ARM_A0.value, "m1", RefRelation.CONTRADICTED.value, VideoTruthLabel.VIDEO_CONTRADICTED.value))
        good_recs.append(AlignedAtomRecord(f"v{i}", ExpArm.ARM_A7.value, "m1", RefRelation.ENTAILED.value, VideoTruthLabel.VIDEO_SUPPORTED.value))
        for _ in range(i % 3):
            good_recs.append(AlignedAtomRecord(f"v{i}", ExpArm.ARM_A7.value, "m1", RefRelation.CONTRADICTED.value, VideoTruthLabel.VIDEO_CONTRADICTED.value))
    res_pass = audit_evaluator_calibration(good_recs, n_boot=100)
    assert res_pass["verdict"] == VerdictStatus.PASS.value
