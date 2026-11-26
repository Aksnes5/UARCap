"""V2 final engineering audit tests: reserve continuation, failure isolation, and queue integrity."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "laptop_final_handoff_20261126"
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "deployment")]

from e2_metric_bridge import compute_metrics
from e3_reserve_pool import MINIMUM, POLICY, continue_screen
from evaluate_e3_v2 import JUDGES, describe_panels
from q2_controlled_sensitivity import analyze
from q2_laptop_queue import build, validate
from q2_module_report import report_provenance
from q2_release_report import collect
from uarcap_q2.common.io_utils import atomic_write_json, primitive_hash, read_json, sha256_file
from uarcap_q2.interventions.neutral_repair import V2


def test_e3_reserve_continue_insufficient_blocks_cleanly_and_isolates_unrelated(tmp_path):
    """Scenario 1: E3 < 150 without reserve approval exits 3 and writes E3_INSUFFICIENT_VALID_CANDIDATES."""
    base_approved = [
        {"query_id": f"orig_{i}", "source_id": f"src_{i:07d}", "human_verification": "PENDING"}
        for i in range(MINIMUM - 3)
    ]
    base_path = atomic_write_json(
        tmp_path / "ORIGINAL_SCREEN_FREEZE.json",
        {"schema": V2, "config_sha256": "cfg1", "approved": base_approved, "rejected": [], "n_approved": len(base_approved)},
    )
    out_path = tmp_path / "E3_FORMAL_APPROVED_V2.json"
    args = SimpleNamespace(
        base=base_path,
        pool=tmp_path / "missing_pool.json",
        approval=tmp_path / "missing_approval.json",
        out=out_path,
    )
    rc = continue_screen(args)
    assert rc == 3
    doc = read_json(out_path)
    assert doc["status"] == "E3_INSUFFICIENT_VALID_CANDIDATES"
    assert doc["eligible"] is False
    assert doc["reserve_state"] == "BLOCKED_MISSING_RESERVE_OR_APPROVAL"
    assert doc["n_approved"] == MINIMUM - 3

    # Verify queue isolation: A3/A5 budget, stress, visual signatures, and final snapshot never depend on E3!
    tasks = {t["id"]: t for t in build("v2")["tasks"]}
    e3_blockers = {"q2l_v2_e3_freeze", "q2l_v2_e3_reserve_continue", "q2l_v2_e3_videollama", "q2l_v2_e3_qwen"}
    for unrelated in (
        "q2l_v2_budget_videollama",
        "q2l_v2_budget_qwen",
        "q2l_v2_budget_report_videollama",
        "q2l_v2_budget_report_qwen",
        "q2l_v2_budget_report",
        "q2l_v2_stress_gemma",
        "q2l_v2_visual_signature_generation",
        "q2l_v2_final_snapshot",
    ):
        assert not (e3_blockers & set(tasks[unrelated]["depends_on"])), f"{unrelated} unexpectedly depends on E3"


def test_e3_reserve_continue_validates_query_id_regex_without_nameerror(tmp_path):
    """Regression test for missing `import re` in e3_reserve_pool.continue_screen()."""
    base_approved = [
        {"query_id": f"orig_{i}", "source_id": f"src_{i:07d}", "human_verification": "PENDING"}
        for i in range(MINIMUM - 1)
    ]
    base_path = atomic_write_json(
        tmp_path / "base.json",
        {"schema": V2, "config_sha256": "cfg1", "approved": base_approved, "rejected": [], "n_approved": len(base_approved)},
    )
    pool_data = {
        "policy": POLICY,
        "candidates": [{"query_id": "../unsafe_escape", "source_id": "12345678901", "video_path": str(tmp_path / "x.mp4")}],
    }
    pool_path = atomic_write_json(tmp_path / "pool.json", pool_data)
    auth_path = atomic_write_json(
        tmp_path / "auth.json",
        {
            "approved": True,
            "approved_by": "cpu-audit-test",
            "protocol_revision": "v2-test",
            "policy": POLICY,
            "pool_sha256": sha256_file(pool_path),
            "max_additional_candidates": 1,
        },
    )
    args = SimpleNamespace(
        base=base_path,
        pool=pool_path,
        approval=auth_path,
        out=tmp_path / "combined.json",
    )
    # Must raise ValueError("Unsafe VidHalluc query binding"), NOT NameError("name 're' is not defined")!
    with pytest.raises(ValueError, match="Unsafe VidHalluc query binding"):
        continue_screen(args)


def test_single_model_and_single_judge_isolation_in_queue_and_reports():
    """Scenarios 2, 3, and 5: Single-model, single-judge, and 64-frame failure isolation."""
    tasks = {t["id"]: t for t in build("v2")["tasks"]}

    # Scenario 2: Single-model budget reports do not depend on the other model or Llama!
    assert tasks["q2l_v2_budget_metrics_videollama"]["depends_on"] == ["q2l_v2_budget_videollama_gemma"]
    assert tasks["q2l_v2_budget_metrics_qwen"]["depends_on"] == ["q2l_v2_budget_qwen_gemma"]
    assert tasks["q2l_v2_budget_report_videollama"]["depends_on"] == ["q2l_v2_budget_metrics_videollama"]
    assert tasks["q2l_v2_budget_report_qwen"]["depends_on"] == ["q2l_v2_budget_metrics_qwen"]

    # Scenario 3: Single-judge E3 / controlled / provenance / historical / relevance reports never depend on Llama!
    for gemma_only_task in (
        "q2l_v2_e3_videollama_analyze_single",
        "q2l_v2_e3_qwen_analyze_single",
        "q2l_v2_bridge_videollama_analyze_single",
        "q2l_v2_bridge_qwen_analyze_single",
        "q2l_v2_report_visual32_gemma",
        "q2l_v2_report_length_gemma",
        "q2l_v2_report_factorial_gemma",
        "q2l_v2_provenance_report_gemma",
        "q2l_v2_historical_report_gemma",
        "q2l_v2_relevance_report_gemma",
    ):
        deps = tasks[gemma_only_task]["depends_on"]
        assert all("_llama" not in d and "second" not in d for d in deps), f"{gemma_only_task} depends on second judge: {deps}"

    # Verify E3 panels & provenance keep single-judge honest (never disguised as dual-judge)
    gemma_id, llama_id = JUDGES
    partial_panels = {
        gemma_id: {
            ("Qwen/Qwen3.8-27B-FP8", "q1", arm, target): {"expressed": 1 if arm == "E3_SUP" else 0}
            for arm in ("E3_BASE", "E3_SUP", "E3_CON", "E3_NEU")
            for target in ("f_plus", "f_minus")
        },
        llama_id: {},
    }
    desc = describe_panels(partial_panels, models=("Qwen/Qwen3.8-27B-FP8",), expected_qids=["q1"])
    assert desc[f"{gemma_id}|Qwen/Qwen3.8-27B-FP8|f_plus"]["complete_four_arm_queries"] == 1
    assert desc[f"{llama_id}|Qwen/Qwen3.8-27B-FP8|f_plus"]["complete_four_arm_queries"] == 0

    prov_gemma = {
        "expected_jobs": 1,
        "rows": [{
            "job_id": "j1",
            "model_id": "Qwen/Qwen3.8-27B-FP8",
            "query_id": "q1",
            "arm": "ARM_A1",
            "before": [],
            "after": [{"claim": "a dog runs", "label": "entailed"}],
            "evidence": [{"rank": 1, "text": "a dog runs"}],
            "caption_sha256": ["a", "b"],
            "relations": {"added_indices": [0], "evidence_labels": [["entailed"]]},
        }],
    }
    prov_summary = report_provenance(prov_gemma, None)
    assert prov_summary["primary_jobs_parsed"] == 1 and prov_summary["second_jobs_parsed"] == 0

    # Scenario 5: 64-frame failure never blocks 32-frame reports, length reports, factorial reports, or budget_report!
    def transitive_deps(tid: str) -> set[str]:
        seen, stack = set(), list(tasks[tid]["depends_on"])
        while stack:
            cur = stack.pop()
            if cur not in seen:
                seen.add(cur)
                stack.extend(tasks[cur]["depends_on"])
        return seen

    for safe_task in (
        "q2l_v2_report_visual32_gemma",
        "q2l_v2_report_visual32_dual",
        "q2l_v2_report_length_gemma",
        "q2l_v2_report_length_dual",
        "q2l_v2_report_factorial_gemma",
        "q2l_v2_report_factorial_dual",
        "q2l_v2_budget_report",
        "q2l_v2_final_snapshot",
    ):
        deps = transitive_deps(safe_task)
        assert not any("visual64" in d for d in deps), f"{safe_task} unexpectedly depends on visual64: {deps}"


def test_spice_failure_does_not_block_other_traditional_metrics(monkeypatch):
    """Scenario 4: Even when SPICE raises an error, BLEU/METEOR/ROUGE-L/CIDEr succeed cleanly."""
    import e2_metric_bridge

    class FakeBleu:
        def __init__(self, n):
            self.n = n
        def compute_score(self, refs, preds):
            k = len(refs)
            return [0.5, 0.4, 0.3, 0.2], [[0.5] * k, [0.4] * k, [0.3] * k, [0.2] * k]

    class FakeScalar:
        def __init__(self, val):
            self.val = val
        def compute_score(self, refs, preds):
            return self.val, [self.val] * len(refs)

    # Inject fake pycocoevalcap modules so test works on Windows without pycocoevalcap/Java SPICE
    import types
    fake_pkg = types.ModuleType("pycocoevalcap")
    fake_bleu = types.ModuleType("pycocoevalcap.bleu.bleu")
    fake_bleu.Bleu = FakeBleu
    fake_meteor = types.ModuleType("pycocoevalcap.meteor.meteor")
    fake_meteor.Meteor = lambda: FakeScalar(0.31)
    fake_rouge = types.ModuleType("pycocoevalcap.rouge.rouge")
    fake_rouge.Rouge = lambda: FakeScalar(0.42)
    fake_cider = types.ModuleType("pycocoevalcap.cider.cider")
    fake_cider.Cider = lambda: FakeScalar(0.67)

    for name, mod in [
        ("pycocoevalcap", fake_pkg),
        ("pycocoevalcap.bleu", types.ModuleType("pycocoevalcap.bleu")),
        ("pycocoevalcap.bleu.bleu", fake_bleu),
        ("pycocoevalcap.meteor", types.ModuleType("pycocoevalcap.meteor")),
        ("pycocoevalcap.meteor.meteor", fake_meteor),
        ("pycocoevalcap.rouge", types.ModuleType("pycocoevalcap.rouge")),
        ("pycocoevalcap.rouge.rouge", fake_rouge),
        ("pycocoevalcap.cider", types.ModuleType("pycocoevalcap.cider")),
        ("pycocoevalcap.cider.cider", fake_cider),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    def broken_spice():
        raise RuntimeError("Java heap / Stanford CoreNLP unavailable")

    monkeypatch.setattr(e2_metric_bridge, "Spice_fn", broken_spice)
    corpus, per_q = compute_metrics({"q1": ["a dog runs"], "q2": ["a cat sits"]}, {"q1": ["a dog runs"], "q2": ["a cat sits"]}, skip_spice=False)
    assert corpus["BLEU_1"] == 0.5 and corpus["CIDEr"] == 0.67
    assert corpus["SPICE"] is None
    assert corpus["SPICE_status"].startswith("BLOCKED: RuntimeError:")
    assert per_q["q1"]["CIDEr"] == 0.67 and "SPICE" not in per_q["q1"]

    # Also verify SPICE dict-output parsing when SPICE succeeds!
    monkeypatch.setattr(e2_metric_bridge, "Spice_fn", lambda: (lambda refs, preds: (0.25, [{"All": {"f": 0.2}}, {"All": {"f": 0.3}}])))
    corpus_ok, per_q_ok = compute_metrics({"q1": ["a dog runs"], "q2": ["a cat sits"]}, {"q1": ["a dog runs"], "q2": ["a cat sits"]}, skip_spice=False)
    assert corpus_ok["SPICE_status"] == "OK" and corpus_ok["SPICE"] == 0.25
    assert per_q_ok["q1"]["SPICE"] == 0.2 and per_q_ok["q2"]["SPICE"] == 0.3


def test_human_pending_keeps_confirmatory_blocked_while_exploratory_runs(tmp_path):
    """Scenario 6 & 7: Exploratory snapshot works without human labels; confirmatory stays hard-blocked."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_final_analysis.py"), "--confirmatory"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 3

    queue_path = HANDOFF / "REMAINING_GPU_TASK_QUEUE.json"
    out_dir = tmp_path / "release_snapshot"
    matrix = collect(queue_path, None, out_dir)
    assert matrix["schema"] == "uarcap_q2.release_matrix.v2"
    assert matrix["explicit_scientific_gates"]["a2_human_calibration_completed"] is False
    assert matrix["explicit_scientific_gates"]["a3_human_validation"] is False
    assert matrix["explicit_scientific_gates"]["formal_b_authorized"] is False
    assert (out_dir / "FINAL_EXPERIMENT_MATRIX.json").is_file()
    assert (out_dir / "MANUSCRIPT_EVIDENCE.md").is_file()


def test_committed_v2_queue_and_expected_artifacts_have_zero_drift():
    """Verify emitted REMAINING_GPU_TASK_QUEUE.json and EXPECTED_OUTPUT_ARTIFACTS.json match code 1-to-1."""
    generated = build("v2")
    validate(generated["tasks"])
    committed = read_json(HANDOFF / "REMAINING_GPU_TASK_QUEUE.json")
    assert generated == committed, "REMAINING_GPU_TASK_QUEUE.json drifted from deployment/q2_laptop_queue.py"

    expected_doc = read_json(HANDOFF / "EXPECTED_OUTPUT_ARTIFACTS.json")
    assert expected_doc["task_count"] == len(committed["tasks"]) == 111
    assert [a["id"] for a in expected_doc["artifacts"]] == [t["id"] for t in committed["tasks"]]
    assert [a["output"] for a in expected_doc["artifacts"]] == [t["output"] for t in committed["tasks"]]
