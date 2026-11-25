"""Tests 07: Full end-to-end CPU simulation, resume/skip durability, and cooperative GPU lock contention."""

from __future__ import annotations

from pathlib import Path

import pytest

from uarcap_q2.annotation.package_builder import build_blind_e2_annotation_package
from uarcap_q2.common.gpu_lock import GPULockBusyError, GPULockConfig, acquire_cooperative_gpu_lock
from uarcap_q2.common.io_utils import read_jsonl
from uarcap_q2.common.schemas import ExpArm, PilotStage, RefRelation, VerdictStatus, VideoTruthLabel
from uarcap_q2.datasets.dream1k import load_dream1k_records, sample_dream1k_stratified_pilot
from uarcap_q2.evaluators.backends import MockJudgeBackend
from uarcap_q2.evaluators.calibration import AlignedAtomRecord, audit_evaluator_calibration
from uarcap_q2.evaluators.reference_fact_eval import ThreeWayReferenceFactEvaluator
from uarcap_q2.generators.mock_generator import MockGeneratorAdapter
from uarcap_q2.retrieval.reranker import MockXCLIPRetriever
from uarcap_q2.runner.orchestrator import RunSpec, run_natural_retrieval_stage
from uarcap_q2.statistics.paired_stats import cluster_paired_effect


def test_full_end_to_end_cpu_pipeline_and_resume(synthetic_dream1k_dir: Path, tmp_path: Path) -> None:
    # 1. Load & sample synthetic DREAM-1K
    recs = load_dream1k_records(
        synthetic_dream1k_dir / "json" / "metadata.json",
        synthetic_dream1k_dir / "video",
        require_video_exists=True,
    )
    pilot_recs, _ = sample_dream1k_stratified_pilot(recs, per_domain=1, seed=20261125)
    assert len(pilot_recs) == 5

    # 2. Run Pilot A1 orchestrator (5 videos x 3 arms = 15 generations, real PyAV 16-frame decode!)
    out_dir = tmp_path / "pilot_a1_run"
    lock_file = tmp_path / "gpu.lock"
    spec = RunSpec(
        run_id="e2e_pilot_a1",
        stage=PilotStage.PILOT_A1,
        dataset_id="DREAM-1K",
        arms=[ExpArm.ARM_A0, ExpArm.ARM_A1, ExpArm.ARM_A7],
        out_dir=out_dir,
        gpu_lock_path=lock_file,
        require_nvidia_smi_idle=False,
        allow_cpu_mock=True,
    )
    gen = MockGeneratorAdapter()
    ret = MockXCLIPRetriever()
    summary = run_natural_retrieval_stage(spec, pilot_recs, ret, gen)
    assert summary["verdict"] == VerdictStatus.PASS.value
    assert summary["completed_generations"] == 15
    assert (out_dir / "PILOT_A1_EXECUTION_REPORT.md").is_file()
    assert (out_dir / "PILOT_A1_INTEGRITY_REPORT.md").is_file()
    assert (out_dir / "PILOT_A1_GO_NO_GO.md").is_file()

    # 3. Verify safe resume skips all 15 already-completed items without re-running!
    resumed = run_natural_retrieval_stage(spec, pilot_recs, ret, gen)
    assert resumed["resumed_skipped"] == 15 and resumed["newly_completed"] == 0

    # 4. Run Layer 3 3-way reference-fact evaluator on all 15 mock predictions
    preds = read_jsonl(out_dir / "predictions.jsonl")
    ref_by_qid = {r.query_id: r for r in pilot_recs}
    evaluator = ThreeWayReferenceFactEvaluator(MockJudgeBackend(model_family="mock_mistral"), generator_model_family="mock_vl")
    eval_rows = [
        evaluator.evaluate_caption(
            query_id=p["query_id"],
            arm=p["arm"],
            model_id=p["model_id"],
            caption=p["caption"],
            reference_description=ref_by_qid[p["query_id"]].reference_description,
            core_events=ref_by_qid[p["query_id"]].core_events,
        )
        for p in preds
    ]
    assert all(e.parse_status == "OK" for e in eval_rows)

    # 5. Build blind annotation package from outputs
    vid_entries = []
    for q in pilot_recs:
        caps = [
            {"arm": e.arm, "model_id": e.model_id, "caption": e.caption, "claims": [c.claim for c in e.claims]}
            for e in eval_rows
            if e.query_id == q.query_id
        ]
        vid_entries.append({"query_id": q.query_id, "video_path": q.video_path, "captions": caps})
    pkg = build_blind_e2_annotation_package(vid_entries, ["A1", "A2"], tmp_path / "blind_pkg")
    assert pkg["tasks_A1"].is_file() and pkg["tasks_A2"].is_file() and pkg["blind_key"].is_file()

    # 6. Run paired statistics & calibration audit on aligned mock records
    a7_map = {e.query_id: float(e.p_unment_ref or 0.0) for e in eval_rows if e.arm == ExpArm.ARM_A7.value}
    a0_map = {e.query_id: float(e.p_unment_ref or 0.0) for e in eval_rows if e.arm == ExpArm.ARM_A0.value}
    stat = cluster_paired_effect({q: a7_map[q] - a0_map[q] for q in a0_map}, n_boot=200, n_perm=300)
    assert stat["n_queries"] == 5

    aligned = [
        AlignedAtomRecord(e.query_id, e.arm, e.model_id, c.label, VideoTruthLabel.VIDEO_SUPPORTED.value)
        for e in eval_rows
        for c in e.claims
    ]
    cal = audit_evaluator_calibration(aligned, n_boot=100)
    assert cal["verdict"] in {v.value for v in VerdictStatus}


def test_cooperative_gpu_lock_rejects_concurrent_holder_and_preserves_lock_file(tmp_path: Path) -> None:
    lock_file = tmp_path / "shared_gpu.lock"
    cfg1 = GPULockConfig(lock_path=lock_file, run_id="run_1", require_nvidia_smi_idle=False, sidecar_dir=tmp_path)
    cfg2 = GPULockConfig(lock_path=lock_file, run_id="run_2", require_nvidia_smi_idle=False, sidecar_dir=tmp_path)

    with acquire_cooperative_gpu_lock(cfg1):
        assert (tmp_path / "gpu_lock_holder_run_1.json").is_file()
        with pytest.raises(GPULockBusyError):
            with acquire_cooperative_gpu_lock(cfg2):
                pass

    # Sidecar cleaned up, shared lock file NEVER deleted, and lock now re-acquirable!
    assert not (tmp_path / "gpu_lock_holder_run_1.json").exists()
    assert lock_file.exists()
    with acquire_cooperative_gpu_lock(cfg2):
        assert (tmp_path / "gpu_lock_holder_run_2.json").is_file()
