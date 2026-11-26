"""CPU tests for the unified downstream analysis/statistics/figures and formal E3 gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from uarcap_q2.final_analysis import reporting
from uarcap_q2.final_analysis.core import (
    all_transitions, cluster_bootstrap, holm, judge_reliability, missing_value_sensitivity,
    paired_arm_stats, per_model_arm_descriptives, random_agreed_sample,
)
from uarcap_q2.final_analysis.pipelines import formal_e3_plan, load_historical_metrics


def _rows():
    rows = []
    for qid, caps in {"q1": ["a man cuts wood", "a man cuts wood with an axe", "a man chops wood"],
                      "q2": ["a woman cooks rice", "a woman cooks rice", "a woman boils rice"]}.items():
        for arm, cap, claims in zip(
            ("ARM_A0", "ARM_A1", "ARM_A7"),
            caps,
            (
                [("a man cuts wood", "entailed"), ("a man uses a knife", "contradicted"), ("the sky is blue", "unmentioned")],
                [("a man cuts wood", "entailed"), ("a man uses a knife", "contradicted")],
                [("a man cuts wood with an axe", "entailed"), ("a man uses an axe", "entailed"), ("a man chops wood", "entailed")],
            ),
        ):
            # second qid gets analogous labels
            cs = claims if qid == "q1" else [("a woman cooks rice", "entailed"), ("a woman uses a pan", "contradicted"), ("it is winter", "unmentioned")]
            label_counts = [sum(1 for c, l in cs if l == k) for k in ("entailed", "contradicted", "unmentioned")]
            rows.append({"query_id": qid, "arm": arm, "model_id": "m", "caption": cap, "parse_status": "OK",
                         "claims": [{"claim": c, "label": l, "claim_id": f"{qid}::{arm}::{i}"} for i, (c, l) in enumerate(cs)],
                         "n_claims": len(cs), "n_entail_ref": label_counts[0], "n_contra_ref": label_counts[1],
                         "n_unment_ref": label_counts[2], "p_entail_ref": label_counts[0] / max(1, len(cs)),
                         "r_core_ref": 1.0 if arm == "ARM_A7" else 0.5,
                         "auxiliary_dimension_breakdown": {"object_entity": {"total": 1}, "mixed": {"total": 1}}})
    return rows


def test_descriptives_and_transitions():
    rows = _rows()
    d = per_model_arm_descriptives(rows)
    assert d["ARM_A0"]["claims"] == 6 and d["ARM_A7"]["support_rate"] == 0.6667
    tr = all_transitions(rows)
    # "a man cuts wood" -> "a man cuts wood with an axe" must be a semantic match (no add+delete)
    assert tr["ARM_A0->ARM_A1"]["removed_unmentioned"] == 1
    assert "core_coverage_delta" in tr["ARM_A0->ARM_A7"]


def test_judge_reliability_kappa_and_direction():
    rows = _rows()
    labels = []
    for r in rows:
        for c in r["claims"]:
            lab = c["label"] if r["arm"] != "ARM_A1" else ("contradicted" if c["label"] == "unmentioned" else c["label"])
            import re
            key = " ".join(t for t in re.sub(r"[^a-z0-9'\s]", " ", c["claim"].lower()).split() if t not in {"a", "an", "the"})
            labels.append({"query_id": r["query_id"], "norm_key": key, "llama_label": lab})
    rel = judge_reliability(rows, labels)
    assert rel["n_claims"] > 0 and 0 <= rel["observed_agreement"] <= 1
    assert rel["budget_differential"].get("disagreement_rate_A7_minus_A0") is not None
    assert random_agreed_sample(rows, labels, n=2)


def test_stats_holm_and_sensitivity():
    rows = _rows()
    st = paired_arm_stats(rows, "support_rate")
    assert "ARM_A0->ARM_A1" in st and st["ARM_A0->ARM_A1"]["n_clusters"] == 2
    assert cluster_bootstrap({"q1": 1.0, "q2": 0.0})["ci95"][0] <= 0.5
    h = holm({"a": 0.01, "b": 0.04})
    assert "a" in h and h["a"]["p_holm"] <= h["b"]["p_holm"]
    assert "ARM_A0" in missing_value_sensitivity(rows)


def test_figures_and_tables(tmp_path: Path):
    d = {"m": {a: {"core_coverage_mean": 0.5, "dimensions": {"object_entity": 3}} for a in ("ARM_A0", "ARM_A1", "ARM_A7")}}
    assert reporting.plot_core_coverage(d, ["m"], tmp_path)
    tr = {"ARM_A0->ARM_A1": {"added_correct": 1, "lost_correct": 2, "added_error": 0, "removed_error": 1, "replacements": 1, "label_flips": 1}}
    assert reporting.plot_e2_fact_changes(tr, tmp_path)
    out = reporting.export_table([{"a": 1, "b": 2}], tmp_path, "t")
    assert out["status"] == "OK"
    # missing figure inputs must be PENDING, never fabricated
    assert reporting.plot_budget_curve([], tmp_path) == []
    assert (tmp_path / "fig_budget_curve.PENDING.txt").is_file()


def test_formal_e3_plan_refuses_under_floor(tmp_path: Path):
    p = tmp_path / "approved.json"
    p.write_text(json.dumps({"n_approved": 3, "approved": [{}] * 3}))
    plan = formal_e3_plan(min_approved=150, candidates_path=p)
    assert plan["eligible"] is False and plan["status"] == "AWAITING_SUFFICIENT_CANDIDATES"


def test_historical_loader_handles_missing():
    rows = load_historical_metrics([Path("/nonexistent/path")])
    assert rows == []
