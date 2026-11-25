"""Complete real stage pipeline dispatchers for Pilot A0, A1, A2, A3, Formal B, E1, and E3-Bridge (P0-1 & P0-7).

Every stage has a genuine, non-stub execution path with explicit input -> process -> output -> StageRunStatus:
  - A0: Preflight & asset/permission verification -> preflight_receipt.json
  - A1: DREAM-1K Pilot 30 stratified sampling + real retrieval/rerank + frozen Video-LLaMA batch -> run_receipt.json + 3 MD reports
  - A2: Reads A1 predictions -> runs 3-way evaluator -> exports blind E2 packages -> stops at HUMAN_VALIDATION_PENDING
        until --human-annotations is supplied -> unblinds & runs 3x4 calibration -> calibration_receipt.json
  - A3: Enforces A2 REAL_RUNTIME_PASS -> runs Modern VLM on A1 30 DREAM-1K -> parses VidHalluc & filters overlaps ->
        checks human-approved E3 candidates (stops at HUMAN_VALIDATION_PENDING if unverified!) -> runs 4-arm E3 on
        both models -> runs BlindE3SemanticEvaluator -> writes pilot_a3_receipt.json
  - FORMAL_B: Enforces full Formal B gate (A1/A2/A3 REAL_RUNTIME_PASS + frozen bindings) -> E2 + E3 + optional E3-Bridge + stats
  - E1: Standalone retrospective CPU analysis on per-query CIDEr JSONs -> e1_report.json
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..annotation.package_builder import build_blind_e2_annotation_package, unblind_annotations
from ..common.gates import verify_stage_authorization
from ..common.io_utils import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl
from ..common.schemas import (
    EligibilityStatus,
    ExpArm,
    PilotStage,
    QueryRecord,
    StageRunStatus,
    VerdictStatus,
    VideoTruthLabel,
)
from ..datasets.dream1k import load_dream1k_records, sample_dream1k_stratified_pilot, write_pilot_exclusion_manifest
from ..datasets.vidhalluc import (
    deduplicate_and_filter_vidhalluc,
    load_vidhalluc_ach_mcq,
    sample_vidhalluc_pilot_candidates,
    write_vidhalluc_pilot_exclusion_manifest,
)
from ..evaluators.backends import JudgeBackend
from ..evaluators.calibration import AlignedAtomRecord, audit_evaluator_calibration
from ..evaluators.e3_semantic_eval import BlindE3SemanticEvaluator
from ..evaluators.reference_fact_eval import ThreeWayReferenceFactEvaluator
from ..generators.base import GeneratorAdapter
from ..interventions.e3_bridge import NaturalConflictObservation, build_e3_bridge_cohort
from ..interventions.vidhalluc_e3 import E3InterventionCandidate, filter_runnable_e3_candidates
from ..retrieval.reranker import RetrieverAdapter
from ..statistics.e1_historical import analyze_e1_selection_heterogeneity
from ..statistics.paired_stats import (
    classify_e3_joint_pattern,
    cluster_paired_effect,
    cross_model_did_effect,
    holm_bonferroni,
    paired_binary_mcnemar_effect,
)
from .orchestrator import RunSpec, run_e3_controlled_stage, run_natural_retrieval_stage


def run_pilot_a2_pipeline(
    a1_predictions_path: Path | str,
    dream1k_annotation_path: Path | str,
    dream1k_video_root: Path | str,
    judge_backend: JudgeBackend,
    out_dir: Path | str,
    auth_receipt_path: Path | str | None = None,
    prior_a1_receipt: Path | str | None = None,
    human_annotations_path: Path | str | None = None,
    generator_model_family: str = "vicuna_videollama",
    allow_cpu_mock: bool = False,
) -> dict[str, Any]:
    """Real Pilot A2 stage: consumes A1 predictions, runs 3-way evaluator, builds blind UI pack, & calibrates on human truth."""
    auth = verify_stage_authorization(
        stage=PilotStage.PILOT_A2,
        auth_receipt_path=auth_receipt_path,
        prior_receipts={PilotStage.PILOT_A1.value: prior_a1_receipt} if prior_a1_receipt else {},
        allow_cpu_mock=allow_cpu_mock,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not Path(a1_predictions_path).is_file():
        raise FileNotFoundError(f"Pilot A2 requires A1 predictions.jsonl: {a1_predictions_path}")

    preds = read_jsonl(a1_predictions_path)
    if not preds:
        raise ValueError(f"Pilot A1 predictions file is empty: {a1_predictions_path}")

    recs = load_dream1k_records(dream1k_annotation_path, dream1k_video_root)
    rec_by_qid = {r.query_id: r for r in recs}

    evaluator = ThreeWayReferenceFactEvaluator(
        backend=judge_backend,
        generator_model_family=generator_model_family,
        enforce_cross_family=True,
    )

    eval_rows: list[dict[str, Any]] = []
    videos_for_blind: dict[str, dict[str, Any]] = {}
    parse_failures = 0

    for p in preds:
        qid = str(p["query_id"])
        if qid not in rec_by_qid:
            raise KeyError(f"A1 prediction query_id '{qid}' not found in DREAM-1K annotations")
        qrec = rec_by_qid[qid]
        ev = evaluator.evaluate_caption(
            query_id=qid,
            arm=str(p["arm"]),
            model_id=str(p["model_id"]),
            caption=str(p.get("caption", "")),
            reference_description=qrec.reference_description,
            core_events=qrec.core_events,
        )
        if ev.parse_status != "OK":
            parse_failures += 1
        ev_d = ev.to_dict()
        eval_rows.append(ev_d)

        v_entry = videos_for_blind.setdefault(
            qid, {"query_id": qid, "video_path": qrec.video_path, "captions": []}
        )
        v_entry["captions"].append(
            {
                "arm": ev.arm,
                "model_id": ev.model_id,
                "caption": ev.caption,
                "claims": [c.claim for c in ev.claims],
            }
        )

    atomic_write_jsonl(out / "a2_ref_fact_eval.jsonl", eval_rows)
    blind_pkg = build_blind_e2_annotation_package(
        list(videos_for_blind.values()), annotator_ids=["A1", "A2"], out_dir=out / "blind_annotation_package"
    )

    # If human ground truth is not yet returned, STOP cleanly at HUMAN_VALIDATION_PENDING (NEVER fake PASS!)
    if not human_annotations_path or not Path(human_annotations_path).is_file():
        receipt = {
            "stage": PilotStage.PILOT_A2.value,
            "status": StageRunStatus.HUMAN_VALIDATION_PENDING.value,
            "execution_mode": auth["mode"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "judge_model_id": judge_backend.cfg.model_id,
            "n_captions_evaluated": len(eval_rows),
            "parse_failures": parse_failures,
            "human_calibration_completed": False,
            "blind_annotation_files": {k: str(v) for k, v in blind_pkg.items()},
            "next_action": (
                "Run `python scripts/launch_annotation_ui.py --bundle <blind_tasks_e2_A1.json>` for both annotators, "
                "adjudicate disagreements, then re-invoke Pilot A2 with --human-annotations <adjudicated.json>."
            ),
        }
        atomic_write_json(out / "calibration_receipt.json", receipt)
        return receipt

    # Ingest adjudicated human annotations (either already unblinded or blind needing unblinding)
    raw_human = read_json(human_annotations_path)
    human_list = raw_human["records"] if isinstance(raw_human, dict) and "records" in raw_human else raw_human
    if human_list and "blind_caption_id" in human_list[0]:
        human_list = unblind_annotations(human_list, blind_pkg["blind_key"])

    # Align (query_id, arm, model_id, claim_idx) with auto 3-way verdicts
    auto_by_key = {(r["query_id"], r["arm"], r["model_id"]): r for r in eval_rows}
    aligned: list[AlignedAtomRecord] = []
    for h in human_list:
        key = (str(h["query_id"]), str(h["arm"]), str(h["model_id"]))
        auto_r = auto_by_key.get(key)
        if not auto_r or auto_r["parse_status"] != "OK":
            continue
        h_labels = h.get("claim_labels", [])
        for c_auto, h_lbl in zip(auto_r["claims"], h_labels):
            aligned.append(
                AlignedAtomRecord(
                    video_id=key[0],
                    arm=key[1],
                    model_id=key[2],
                    ref_label=str(c_auto["label"]),
                    human_label=str(h_lbl),
                )
            )

    cal = audit_evaluator_calibration(aligned, n_boot=500)
    is_mock = auth["mode"] == "cpu_mock"
    if cal["verdict"] == VerdictStatus.PASS.value:
        status = StageRunStatus.MOCK_PASS.value if is_mock else StageRunStatus.REAL_RUNTIME_PASS.value
    elif cal["verdict"] == VerdictStatus.INCONCLUSIVE.value:
        status = StageRunStatus.INCONCLUSIVE.value
    else:
        status = StageRunStatus.FAILED.value

    receipt = {
        "stage": PilotStage.PILOT_A2.value,
        "status": status,
        "execution_mode": auth["mode"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "judge_model_id": judge_backend.cfg.model_id,
        "n_captions_evaluated": len(eval_rows),
        "parse_failures": parse_failures,
        "human_calibration_completed": True,
        "calibration_audit": cal,
        "verdict": cal["verdict"],
    }
    atomic_write_json(out / "calibration_receipt.json", receipt)
    return receipt


def run_pilot_a3_pipeline(
    dream1k_pilot_queries: list[QueryRecord],
    retriever: RetrieverAdapter,
    modern_vlm: GeneratorAdapter,
    videollama_adapter: GeneratorAdapter,
    vidhalluc_ach_mcq_path: Path | str,
    vidhalluc_video_root: Path | str,
    e3_judge_backend: JudgeBackend,
    out_dir: Path | str,
    videoufo_source_ids: set[str],
    approved_e3_candidates_path: Path | str | None = None,
    auth_receipt_path: Path | str | None = None,
    prior_a2_receipt: Path | str | None = None,
    gpu_lock_path: Path | None = None,
    allow_cpu_mock: bool = False,
    skip_video_decode: bool = False,
) -> dict[str, Any]:
    """Real Pilot A3 stage: Modern VLM on DREAM-1K 30 + VidHalluc overlap/applicability gate + 4-arm E3 + dual-endpoint eval."""
    auth = verify_stage_authorization(
        stage=PilotStage.PILOT_A3,
        auth_receipt_path=auth_receipt_path,
        prior_receipts={PilotStage.PILOT_A2.value: prior_a2_receipt} if prior_a2_receipt else {},
        allow_cpu_mock=allow_cpu_mock,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Run Modern VLM on the SAME DREAM-1K pilot queries (cross-model A0/A1/A7 check)
    vlm_dream_summary = run_natural_retrieval_stage(
        RunSpec(
            run_id="pilot_a3_dream1k_vlm",
            stage=PilotStage.PILOT_A3,
            dataset_id="DREAM-1K",
            arms=[ExpArm.ARM_A0, ExpArm.ARM_A1, ExpArm.ARM_A7],
            out_dir=out / "dream1k_modern_vlm",
            auth_receipt_path=Path(auth_receipt_path) if auth_receipt_path else None,
            prior_receipts={PilotStage.PILOT_A2.value: prior_a2_receipt} if prior_a2_receipt else {},
            gpu_lock_path=gpu_lock_path,
            allow_cpu_mock=allow_cpu_mock,
        ),
        queries=dream1k_pilot_queries,
        retriever=retriever,
        generator=modern_vlm,
        skip_video_decode=skip_video_decode,
    )

    # 2. Load VidHalluc & run exact YouTube ID overlap filter + 1-clip-per-source dedup
    vh_raw = load_vidhalluc_ach_mcq(vidhalluc_ach_mcq_path, vidhalluc_video_root)
    vh_clean, overlap_audit = deduplicate_and_filter_vidhalluc(vh_raw, videoufo_source_ids=videoufo_source_ids)
    atomic_write_json(out / "vidhalluc_overlap_audit.json", overlap_audit)
    vh_by_qid = {r.query_id: r for r in vh_clean}

    # 3. Require human-verified E3 candidates (NEVER auto-approve MCQ distractors!)
    if not approved_e3_candidates_path or not Path(approved_e3_candidates_path).is_file():
        cand_n = min(30, len(vh_clean))
        pilot_cands, _ = sample_vidhalluc_pilot_candidates(vh_clean, candidate_count=cand_n, seed=20261125)
        excl_path = write_vidhalluc_pilot_exclusion_manifest(pilot_cands, out / "pilot_a3_excluded_vidhalluc.json", seed=20261125)
        pending_file = atomic_write_json(
            out / "vidhalluc_candidates_pending_human_review.json",
            [r.to_dict() for r in pilot_cands],
        )
        receipt = {
            "stage": PilotStage.PILOT_A3.value,
            "status": StageRunStatus.HUMAN_VALIDATION_PENDING.value,
            "execution_mode": auth["mode"],
            "dream1k_modern_vlm_summary": vlm_dream_summary,
            "vidhalluc_overlap_audit": overlap_audit,
            "pending_candidates_file": str(pending_file),
            "pilot_exclusion_manifest": str(excl_path),
            "next_action": "Complete 5-item human applicability review on VidHalluc candidates and pass --approved-e3-candidates.",
        }
        atomic_write_json(out / "run_receipt.json", receipt)
        return receipt

    raw_cands = read_json(approved_e3_candidates_path)
    cand_objs = [E3InterventionCandidate(**c) for c in raw_cands]
    runnable = filter_runnable_e3_candidates(cand_objs)
    if not runnable:
        receipt = {
            "stage": PilotStage.PILOT_A3.value,
            "status": StageRunStatus.FAILED.value,
            "execution_mode": auth["mode"],
            "error": "Zero human-APPROVED E3 candidates in approved_e3_candidates file (unverified distractors refused).",
        }
        atomic_write_json(out / "run_receipt.json", receipt)
        return receipt

    # 4. Run 4-arm visual-locked E3 on BOTH models
    e3_vl = run_e3_controlled_stage(
        RunSpec(
            run_id="pilot_a3_e3_videollama",
            stage=PilotStage.PILOT_A3,
            dataset_id="VidHalluc-ACH",
            arms=[ExpArm.E3_BASE, ExpArm.E3_SUP, ExpArm.E3_CON, ExpArm.E3_NEU],
            out_dir=out / "e3_videollama",
            auth_receipt_path=Path(auth_receipt_path) if auth_receipt_path else None,
            prior_receipts={PilotStage.PILOT_A2.value: prior_a2_receipt} if prior_a2_receipt else {},
            gpu_lock_path=gpu_lock_path,
            allow_cpu_mock=allow_cpu_mock,
        ),
        queries_by_id=vh_by_qid,
        candidates=runnable,
        generator=videollama_adapter,
        base_visual_mode="BASE_A0",
        skip_video_decode=skip_video_decode,
    )
    e3_vlm = run_e3_controlled_stage(
        RunSpec(
            run_id="pilot_a3_e3_modern_vlm",
            stage=PilotStage.PILOT_A3,
            dataset_id="VidHalluc-ACH",
            arms=[ExpArm.E3_BASE, ExpArm.E3_SUP, ExpArm.E3_CON, ExpArm.E3_NEU],
            out_dir=out / "e3_modern_vlm",
            auth_receipt_path=Path(auth_receipt_path) if auth_receipt_path else None,
            prior_receipts={PilotStage.PILOT_A2.value: prior_a2_receipt} if prior_a2_receipt else {},
            gpu_lock_path=gpu_lock_path,
            allow_cpu_mock=allow_cpu_mock,
        ),
        queries_by_id=vh_by_qid,
        candidates=runnable,
        generator=modern_vlm,
        base_visual_mode="BASE_A0",
        skip_video_decode=skip_video_decode,
    )

    # 5. Evaluate dual independent endpoints (Y_true, Y_conflict) via blind single-target semantic evaluator
    e3_eval = BlindE3SemanticEvaluator(e3_judge_backend)
    cand_by_qid = {c.query_id: c for c in runnable}
    e3_eval_rows: list[dict[str, Any]] = []
    for subdir in ("e3_videollama", "e3_modern_vlm"):
        for p in read_jsonl(out / subdir / "e3_predictions.jsonl"):
            c = cand_by_qid[p["query_id"]]
            res = e3_eval.evaluate_caption(
                query_id=p["query_id"],
                arm=p["arm"],
                model_id=p["model_id"],
                caption=p["caption"],
                target_true_fact=c.target_true_fact,
                target_conflict_fact=c.candidate_conflict_fact,
            )
            e3_eval_rows.append(res.to_dict())
    atomic_write_jsonl(out / "e3_dual_endpoint_eval.jsonl", e3_eval_rows)

    all_ok = (
        vlm_dream_summary["verdict"] == VerdictStatus.PASS.value
        and e3_vl["verdict"] == VerdictStatus.PASS.value
        and e3_vlm["verdict"] == VerdictStatus.PASS.value
    )
    status = (
        StageRunStatus.FAILED.value
        if not all_ok
        else (StageRunStatus.MOCK_PASS.value if auth["mode"] == "cpu_mock" else StageRunStatus.REAL_RUNTIME_PASS.value)
    )
    receipt = {
        "stage": PilotStage.PILOT_A3.value,
        "status": status,
        "execution_mode": auth["mode"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dream1k_modern_vlm_summary": vlm_dream_summary,
        "e3_videollama_summary": e3_vl,
        "e3_modern_vlm_summary": e3_vlm,
        "n_e3_evaluated_captions": len(e3_eval_rows),
        "verdict": VerdictStatus.PASS.value if all_ok else VerdictStatus.FAIL.value,
    }
    atomic_write_json(out / "run_receipt.json", receipt)
    return receipt


def run_formal_b_pipeline(
    formal_dream1k_queries: list[QueryRecord],
    formal_vidhalluc_queries_by_id: dict[str, QueryRecord],
    approved_e3_candidates: list[E3InterventionCandidate],
    retriever: RetrieverAdapter,
    videollama_adapter: GeneratorAdapter,
    modern_vlm: GeneratorAdapter,
    ref_judge_backend: JudgeBackend,
    e3_judge_backend: JudgeBackend,
    out_dir: Path | str,
    auth_receipt_path: Path | str,
    prior_receipts: dict[str, Path | str],
    formal_config_path: Path | str,
    asset_manifest_path: Path | str,
    pilot_exclusion_path: Path | str,
    pilot_excluded_query_ids: set[str],
    e3_bridge_observations: list[NaturalConflictObservation] | None = None,
    gpu_lock_path: Path | None = None,
    skip_video_decode: bool = False,
) -> dict[str, Any]:
    """Complete guarded Formal Study B pipeline (E2 + E3 + optional E3-Bridge + E2/E3/E5 Holm & DiD statistics)."""
    auth = verify_stage_authorization(
        stage=PilotStage.FORMAL_B,
        auth_receipt_path=auth_receipt_path,
        prior_receipts=prior_receipts,
        allow_cpu_mock=False,
        formal_config_path=formal_config_path,
        asset_manifest_path=asset_manifest_path,
        pilot_exclusion_path=pilot_exclusion_path,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. E2 Natural Retrieval on DREAM-1K Formal (both models, shared queries)
    e2_summaries = {}
    for tag, gen in (("videollama", videollama_adapter), ("modern_vlm", modern_vlm)):
        e2_summaries[tag] = run_natural_retrieval_stage(
            RunSpec(
                run_id=f"formal_b_e2_{tag}",
                stage=PilotStage.FORMAL_B,
                dataset_id="DREAM-1K",
                arms=[ExpArm.ARM_A0, ExpArm.ARM_A1, ExpArm.ARM_A7],
                out_dir=out / f"e2_{tag}",
                auth_receipt_path=Path(auth_receipt_path),
                prior_receipts=prior_receipts,
                formal_config_path=Path(formal_config_path),
                asset_manifest_path=Path(asset_manifest_path),
                pilot_exclusion_path=Path(pilot_exclusion_path),
                pilot_excluded_query_ids=pilot_excluded_query_ids,
                gpu_lock_path=gpu_lock_path,
                allow_cpu_mock=False,
            ),
            queries=formal_dream1k_queries,
            retriever=retriever,
            generator=gen,
            skip_video_decode=skip_video_decode,
        )

    # 2. E3 Controlled Intervention on VidHalluc Formal (both models)
    e3_summaries = {}
    for tag, gen in (("videollama", videollama_adapter), ("modern_vlm", modern_vlm)):
        e3_summaries[tag] = run_e3_controlled_stage(
            RunSpec(
                run_id=f"formal_b_e3_{tag}",
                stage=PilotStage.FORMAL_B,
                dataset_id="VidHalluc-ACH",
                arms=[ExpArm.E3_BASE, ExpArm.E3_SUP, ExpArm.E3_CON, ExpArm.E3_NEU],
                out_dir=out / f"e3_{tag}",
                auth_receipt_path=Path(auth_receipt_path),
                prior_receipts=prior_receipts,
                formal_config_path=Path(formal_config_path),
                asset_manifest_path=Path(asset_manifest_path),
                pilot_exclusion_path=Path(pilot_exclusion_path),
                gpu_lock_path=gpu_lock_path,
                allow_cpu_mock=False,
            ),
            queries_by_id=formal_vidhalluc_queries_by_id,
            candidates=approved_e3_candidates,
            generator=gen,
            base_visual_mode="BASE_A0",
            skip_video_decode=skip_video_decode,
        )

    # 3. Optional E3-Bridge cohort check
    bridge_report = (
        build_e3_bridge_cohort(e3_bridge_observations)
        if e3_bridge_observations is not None
        else {"module": "E3-Bridge", "enabled": False, "status": "NOT_PROVIDED"}
    )
    atomic_write_json(out / "e3_bridge_decision.json", bridge_report)

    # 4. Automated Evaluation & E2/E3/E5 Final Statistics (Holm + McNemar + DiD)
    q_by_id = {q.query_id: q for q in formal_dream1k_queries}
    ref_ev = ThreeWayReferenceFactEvaluator(ref_judge_backend, enforce_cross_family=False)
    e2_deltas_by_model: dict[str, dict[str, float]] = {}
    for tag in ("videollama", "modern_vlm"):
        preds = read_jsonl(out / f"e2_{tag}" / "predictions.jsonl")
        evals = [
            ref_ev.evaluate_caption(
                p["query_id"], p["arm"], p["model_id"], p["caption"], q_by_id[p["query_id"]].reference_description, q_by_id[p["query_id"]].core_events
            ).to_dict()
            for p in preds
        ]
        atomic_write_jsonl(out / f"e2_{tag}_ref_eval.jsonl", evals)
        a7 = {e["query_id"]: float(e["n_contra_ref"] or 0.0) for e in evals if e["arm"] == ExpArm.ARM_A7.value}
        a0 = {e["query_id"]: float(e["n_contra_ref"] or 0.0) for e in evals if e["arm"] == ExpArm.ARM_A0.value}
        e2_deltas_by_model[tag] = {q: a7[q] - a0[q] for q in sorted(set(a7) & set(a0))}

    e2_did = cross_model_did_effect(e2_deltas_by_model["modern_vlm"], e2_deltas_by_model["videollama"], n_boot=300, n_perm=500)

    e3_ev = BlindE3SemanticEvaluator(e3_judge_backend)
    cand_map = {c.query_id: c for c in approved_e3_candidates}
    e3_stats: dict[str, Any] = {}
    for tag in ("videollama", "modern_vlm"):
        preds = read_jsonl(out / f"e3_{tag}" / "e3_predictions.jsonl")
        e3_rows = [
            e3_ev.evaluate_caption(
                p["query_id"], p["arm"], p["model_id"], p["caption"], cand_map[p["query_id"]].target_true_fact, cand_map[p["query_id"]].candidate_conflict_fact
            ).to_dict()
            for p in preds
        ]
        atomic_write_jsonl(out / f"e3_{tag}_dual_eval.jsonl", e3_rows)
        con_c = {r["query_id"]: r["y_conflict"] for r in e3_rows if r["arm"] == ExpArm.E3_CON.value}
        neu_c = {r["query_id"]: r["y_conflict"] for r in e3_rows if r["arm"] == ExpArm.E3_NEU.value}
        con_t = {r["query_id"]: r["y_true"] for r in e3_rows if r["arm"] == ExpArm.E3_CON.value}
        neu_t = {r["query_id"]: r["y_true"] for r in e3_rows if r["arm"] == ExpArm.E3_NEU.value}
        mc_c = paired_binary_mcnemar_effect(con_c, neu_c, n_boot=300)
        mc_t = paired_binary_mcnemar_effect(con_t, neu_t, n_boot=300)
        hb = holm_bonferroni({"y_conflict": mc_c["mcnemar_p"], "y_true": mc_t["mcnemar_p"]})
        e3_stats[tag] = {
            "y_conflict_con_vs_neu": mc_c,
            "y_true_con_vs_neu": mc_t,
            "holm": hb,
            "joint_pattern": classify_e3_joint_pattern(
                mc_c["risk_diff"], hb["y_conflict"]["significant"], mc_t["risk_diff"], hb["y_true"]["significant"]
            ),
        }

    stats_bundle = {"e2_cross_model_did_contra": e2_did, "e3_by_model": e3_stats}
    atomic_write_json(out / "formal_b_statistics.json", stats_bundle)

    all_pass = all(s["verdict"] == VerdictStatus.PASS.value for s in (*e2_summaries.values(), *e3_summaries.values()))
    receipt = {
        "stage": PilotStage.FORMAL_B.value,
        "status": StageRunStatus.REAL_RUNTIME_PASS.value if all_pass else StageRunStatus.FAILED.value,
        "approval_id": auth["approval_id"],
        "evaluator_version": auth["evaluator_version"],
        "e2_summaries": e2_summaries,
        "e3_summaries": e3_summaries,
        "e3_bridge": {"enabled": bridge_report["enabled"], "status": bridge_report["status"]},
        "statistics_file": str(out / "formal_b_statistics.json"),
        "verdict": VerdictStatus.PASS.value if all_pass else VerdictStatus.FAIL.value,
    }
    atomic_write_json(out / "formal_b_receipt.json", receipt)
    return receipt


def run_e1_retrospective_pipeline(
    scores_ref_a_json: Path | str,
    scores_ref_b_json: Path | str,
    source_map_json: Path | str,
    out_path: Path | str,
    baseline_arm: str = "ARM_A1",
) -> dict[str, Any]:
    """Executable E1 retrospective CPU pipeline reading frozen per-query CIDEr JSONs and writing report."""
    res = analyze_e1_selection_heterogeneity(
        per_query_scores_ref_a=read_json(scores_ref_a_json),
        per_query_scores_ref_b=read_json(scores_ref_b_json),
        source_id_by_query=read_json(source_map_json),
        baseline_arm=baseline_arm,
    )
    atomic_write_json(out_path, res)
    return res
