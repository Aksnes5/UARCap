"""Tests 06: E1 explorer, paired/cluster/Holm/McNemar/DiD stats, two-phase IPW correction, and power planner."""

from __future__ import annotations

import pytest

from uarcap_q2.statistics.e1_historical import analyze_e1_selection_heterogeneity
from uarcap_q2.statistics.paired_stats import (
    classify_e3_joint_pattern,
    cluster_paired_effect,
    cross_model_did_effect,
    holm_bonferroni,
    paired_binary_mcnemar_effect,
)
from uarcap_q2.statistics.power_planner import extrapolate_formal_b_resources, plan_formal_dream1k_sample_size
from uarcap_q2.statistics.two_phase_correction import TwoPhaseUnit, estimate_two_phase_corrected_delta


def test_e1_retrospective_selection_and_missing_arms() -> None:
    ref_a = {
        "q1": {"ARM_A1": 0.50, "ARM_A7": 0.70},
        "q2": {"ARM_A1": 0.60, "ARM_A7": 0.40},
        "q3": {"ARM_A7": 0.80},  # missing baseline ARM_A1
    }
    ref_b = {
        "q1": {"ARM_A1": 0.52, "ARM_A7": 0.68},
        "q2": {"ARM_A1": 0.58, "ARM_A7": 0.42},
        "q3": {"ARM_A7": 0.75},
    }
    res = analyze_e1_selection_heterogeneity(ref_a, ref_b, {"q1": "S1", "q2": "S2", "q3": "S3"}, n_boot=200)
    assert res["n_queries"] == 2 and res["missing_arms_count"] == 1
    assert res["same_ref_oracle_mean_gain"] == pytest.approx(0.10, abs=1e-4)
    assert res["cross_ref_selection_mean_gain"] == pytest.approx(0.08, abs=1e-4)


def test_paired_stats_edge_cases_all_zero_single_direction_clusters_and_holm() -> None:
    # 1. All-zero differences -> p = 1.0, CI = (0, 0)
    zero = cluster_paired_effect({f"q{i}": 0.0 for i in range(10)}, n_boot=200, n_perm=500)
    assert zero["mean_diff"] == 0.0 and zero["p_value"] == 1.0 and zero["ci95"] == (0.0, 0.0)

    # 2. Single-direction positive diffs with duplicate sources clustered properly
    diffs = {f"q{i}": 1.0 for i in range(12)}
    clusters = {f"q{i}": f"SRC_{i // 2}" for i in range(12)}  # 6 unique sources
    pos = cluster_paired_effect(diffs, cluster_id_by_query=clusters, n_boot=200, n_perm=500)
    assert pos["n_queries"] == 12 and pos["n_clusters"] == 6
    assert pos["mean_diff"] == 1.0 and pos["p_value"] < 0.05

    # 3. Holm-Bonferroni preserves None and steps down monotonically
    hb = holm_bonferroni({"h1": 0.005, "h2": 0.02, "h3": 0.04, "h4": None}, alpha=0.05)
    assert hb["h1"]["significant"] is True and hb["h1"]["holm_adj_p"] == pytest.approx(0.015, abs=1e-5)
    assert hb["h2"]["significant"] is True and hb["h2"]["holm_adj_p"] == pytest.approx(0.04, abs=1e-5)
    assert hb["h4"]["holm_adj_p"] is None and hb["h4"]["significant"] is False


def test_mcnemar_e3_joint_quadrants_and_cross_model_did() -> None:
    con = {f"q{i}": 1 if i < 14 else 0 for i in range(20)}
    neu = {f"q{i}": 1 if i < 2 else 0 for i in range(20)}
    res = paired_binary_mcnemar_effect(con, neu, n_boot=300)
    assert res["n_paired"] == 20 and res["risk_diff"] == pytest.approx(0.60, abs=1e-5)
    assert res["mcnemar_p"] < 0.01

    assert classify_e3_joint_pattern(0.3, True, -0.25, True) == "SUBSTITUTION_CONTAMINATION"
    assert classify_e3_joint_pattern(0.3, True, -0.02, False) == "ADDITIVE_OR_HEDGED_CONTAMINATION"
    assert classify_e3_joint_pattern(0.01, False, -0.25, True) == "SUPPRESSION_OR_DISTRACTION"
    assert classify_e3_joint_pattern(0.01, False, 0.01, False) == "VISUAL_ROBUST_OR_NULL_EFFECT"

    did = cross_model_did_effect({"q1": 0.5, "q2": 0.6, "q3": 0.4}, {"q1": 0.1, "q2": 0.1, "q3": 0.1, "q4": 0.2}, n_boot=200, n_perm=500)
    assert did["n_queries"] == 3 and did["missing_non_common_queries"] == 1
    assert did["mean_diff"] > 0.35


def test_two_phase_ipw_bias_correction_and_calibration_isolation() -> None:
    # Simulate an auto proxy that systematically underestimates true human delta by -0.25!
    units: list[TwoPhaseUnit] = []
    for i in range(100):
        stratum = "enriched" if i < 30 else "base"
        pi = 0.50 if stratum == "enriched" else 0.20
        sampled = (i % 2 == 0) if stratum == "enriched" else (i % 5 == 0)
        auto_d = 0.10
        human_d = 0.35 if sampled else None  # True delta is 0.35 (auto underestimates by 0.25)
        units.append(TwoPhaseUnit(f"q{i}", stratum, auto_d, sampled, pi, human_d))

    res = estimate_two_phase_corrected_delta(units, calibration_excluded_ids={"pilot_001"})
    assert res["status"] == "OK"
    assert res["auto_uncorrected_mean"] == pytest.approx(0.10, abs=1e-4)
    assert res["corrected_mean"] == pytest.approx(0.35, abs=1e-4)

    # Leakage of calibration ID into validation sample MUST raise!
    with pytest.raises(ValueError, match="Calibration-to-Validation leakage"):
        estimate_two_phase_corrected_delta(units, calibration_excluded_ids={"q0"})


def test_power_planner_and_a6000_resource_extrapolation() -> None:
    pw = plan_formal_dream1k_sample_size(pilot_delta_sd=0.60, min_detectable_effect=0.08)
    assert 600 <= pw["recommended_formal_n"] <= 800
    ext = extrapolate_formal_b_resources(
        pilot_a6000_sec_per_video_vl=2.1,
        pilot_a6000_sec_per_video_vlm=3.5,
        pilot_a6000_sec_per_caption_judge=4.0,
        pilot_two_annotator_min_per_video=18.0,
        formal_dream1k_n=700,
        formal_vidhalluc_n=170,
    )
    assert ext["projected_a6000_gpu_hours"] < 45.0
    assert ext["resource_gate_pass"] is True
