"""Complete real stage pipeline dispatchers for Pilot A0, A1, A2, A3, Formal B, E1, and E3-Bridge.

Every stage has a genuine, non-stub execution path with explicit input -> process -> output -> StageRunStatus:
  - A0: Preflight & asset/permission verification -> preflight_receipt.json
  - A1: DREAM-1K Pilot 30 stratified sampling + real retrieval/rerank + frozen Video-LLaMA batch -> run_receipt.json + 3 MD reports
  - A2: Reads A1 predictions -> runs 3-way evaluator with stable claim_ids -> exports version-bound blind E2 packages ->
        reuses existing auto-eval & blind keys on re-entry -> strict claim_id reconciliation (omitted claims + split/merge +
        missing/duplicate rejection) -> 3x4 calibration -> calibration_receipt.json
  - A3: Enforces A2 REAL_RUNTIME_PASS -> runs Modern VLM on A1 30 DREAM-1K -> parses VidHalluc & filters overlaps ->
        pauses at HUMAN_VALIDATION_PENDING if E3 candidates unapproved -> runs 4-arm E3 on both models ->
        runs BlindE3SemanticEvaluator + exports blind E3 & cross-model DREAM-1K annotation bundles ->
        pauses at HUMAN_VALIDATION_PENDING until human E3 verification is returned -> computes independent Y_true &
        Y_conflict agreement, Con vs Neu differential bias, and cross-model DREAM-1K bias -> pilot_a3 receipt
  - FORMAL_B: Enforces full Formal B gate + stratified probability sampling + near-dup/Pilot/same-source isolation ->
        E2 + E3 + real E3-Bridge execution (when enabled) + full missingness-preserving stats (A1-A0, A7-A1, A7-A0,
        Core/Unmentioned, Two-Phase IPW correction, E3 secondary comparisons, MSR-VTT historical bridge, Holm & DiD)
  - E1: Standalone retrospective CPU analysis on per-query CIDEr JSONs -> e1_report.json
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..annotation.package_builder import (
    build_blind_e2_annotation_package,
    build_blind_e3_annotation_package,
    unblind_annotations,
)
from ..common.gates import verify_stage_authorization
from ..common.gpu_runtime import release_gpu_memory
from ..common.io_utils import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl, sha256_file
from ..common.schemas import (
    ExpArm,
    PilotStage,
    QueryRecord,
    StageRunStatus,
    VerdictStatus,
    VideoTruthLabel,
)
from ..datasets.carebench import load_carebench_records
from ..datasets.dream1k import load_dream1k_records
from ..datasets.formal_sampling import (
    FormalSamplingConfig,
    build_formal_e2_manifest,
    build_formal_vidhalluc_manifest,
)
from ..datasets.vidhalluc import (
    deduplicate_and_filter_vidhalluc,
    load_vidhalluc_ach_mcq,
    sample_vidhalluc_pilot_candidates,
    write_vidhalluc_pilot_exclusion_manifest,
)
from ..evaluators.backends import JudgeBackend
from ..evaluators.calibration import (
    AlignedAtomRecord,
    audit_cross_model_e2_bias,
    audit_e3_evaluator_agreement,
    audit_evaluator_calibration,
    make_stable_claim_id,
    reconcile_a2_human_and_auto,
)
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
from ..statistics.two_phase_correction import TwoPhaseUnit, estimate_two_phase_corrected_delta
from .orchestrator import RunSpec, run_e3_controlled_stage, run_natural_retrieval_stage


def _attach_stable_claim_ids(ev_dict: dict[str, Any]) -> dict[str, Any]:
    """Attach deterministic claim_id to every auto-extracted claim in a CaptionRefFactEval dict."""
    qid, arm, mid = str(ev_dict["query_id"]), str(ev_dict["arm"]), str(ev_dict["model_id"])
    claims = []
    for idx, c in enumerate(ev_dict.get("claims", [])):
        c_copy = dict(c)
        c_copy.setdefault("claim_id", make_stable_claim_id(qid, arm, mid, idx))
        claims.append(c_copy)
    ev_dict["claims"] = claims
    return ev_dict


def _extract_valid_metric_map(
    eval_rows: list[dict[str, Any]],
    arm: str,
    metric_key: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Extract valid numeric metric map for an arm WITHOUT coercing parse failures / None to 0.0!"""
    arm_rows = [e for e in eval_rows if e.get("arm") == arm]
    valid: dict[str, float] = {}
    parse_failed_ids: list[str] = []
    missing_metric_ids: list[str] = []
    for e in arm_rows:
        qid = str(e["query_id"])
        if e.get("parse_status") != "OK":
            parse_failed_ids.append(qid)
            continue
        val = e.get(metric_key)
        if val is None:
            missing_metric_ids.append(qid)
            continue
        valid[qid] = float(val)
    total = len(arm_rows)
    n_missing = len(parse_failed_ids) + len(missing_metric_ids)
    report = {
        "arm": arm,
        "metric": metric_key,
        "total_n": total,
        "valid_n": len(valid),
        "missing_n": n_missing,
        "missing_rate": round(n_missing / total, 6) if total > 0 else 0.0,
        "parse_failed_ids": sorted(parse_failed_ids),
        "missing_metric_ids": sorted(missing_metric_ids),
    }
    return valid, report


def _extract_valid_e3_binary_map(
    e3_rows: list[dict[str, Any]],
    arm: str,
    endpoint: str,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Extract valid binary E3 endpoint (y_true or y_conflict) WITHOUT coercing parse failures to 0!"""
    arm_rows = [r for r in e3_rows if r.get("arm") == arm]
    valid: dict[str, int] = {}
    parse_failed_ids: list[str] = []
    for r in arm_rows:
        qid = str(r["query_id"])
        if r.get("parse_status", "OK") != "OK" or r.get(endpoint) is None:
            parse_failed_ids.append(qid)
            continue
        valid[qid] = int(r[endpoint])
    total = len(arm_rows)
    report = {
        "arm": arm,
        "endpoint": endpoint,
        "total_n": total,
        "valid_n": len(valid),
        "parse_failed_n": len(parse_failed_ids),
        "missing_rate": round(len(parse_failed_ids) / total, 6) if total > 0 else 0.0,
        "parse_failed_ids": sorted(parse_failed_ids),
    }
    return valid, report


def _paired_arm_diff_analysis(
    eval_rows: list[dict[str, Any]],
    arm_a: str,
    arm_b: str,
    metric_key: str,
    n_boot: int = 300,
    n_perm: int = 500,
    max_missing_rate_fail: float | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Compute paired (arm_a - arm_b) effect preserving explicit missingness (NEVER filling None with 0.0)."""
    va, rep_a = _extract_valid_metric_map(eval_rows, arm_a, metric_key)
    vb, rep_b = _extract_valid_metric_map(eval_rows, arm_b, metric_key)
    common = sorted(set(va) & set(vb))
    diffs = {q: va[q] - vb[q] for q in common}
    total_union = len(set(va) | set(vb) | set(rep_a["parse_failed_ids"]) | set(rep_b["parse_failed_ids"]) | set(rep_a["missing_metric_ids"]) | set(rep_b["missing_metric_ids"]))
    dropped_due_to_missing = total_union - len(common)
    missing_rate = (dropped_due_to_missing / total_union) if total_union > 0 else 0.0

    if max_missing_rate_fail is not None and missing_rate > max_missing_rate_fail:
        stat: dict[str, Any] = {
            "status": "FAILED_MISSINGNESS_EXCEEDED",
            "n_queries": len(common),
            "mean_diff": None,
            "ci95": (None, None),
            "p_value": None,
        }
    else:
        stat = cluster_paired_effect(diffs, n_boot=n_boot, n_perm=n_perm)
        stat["status"] = "OK" if common else "MISSING_ALL_PAIRS"

    stat["missingness"] = {
        "total_union_queries": total_union,
        "paired_valid_n": len(common),
        "dropped_missing_or_parse_failed_n": dropped_due_to_missing,
        "pair_missing_rate": round(missing_rate, 6),
        "arm_a": rep_a,
        "arm_b": rep_b,
    }
    return diffs, stat


def _load_e2_records_by_dataset(
    e2_dataset: str,
    annotation_path: Path | str,
    video_root: Path | str,
    require_video_exists: bool = False,
) -> tuple[str, list[QueryRecord]]:
    ds = (e2_dataset or "carebench").strip().lower().replace("-", "")
    if ds == "carebench":
        return "CaReBench", load_carebench_records(annotation_path, video_root, require_video_exists=require_video_exists)
    if ds == "dream1k":
        return "DREAM-1K", load_dream1k_records(annotation_path, video_root, require_video_exists=require_video_exists)
    raise ValueError(f"Unsupported E2 dataset '{e2_dataset}' (allowed: carebench, dream1k)")


def run_pilot_a2_pipeline(
    a1_predictions_path: Path | str,
    dream1k_annotation_path: Path | str | None = None,
    dream1k_video_root: Path | str | None = None,
    judge_backend: JudgeBackend | None = None,
    out_dir: Path | str = ".",
    auth_receipt_path: Path | str | None = None,
    prior_a1_receipt: Path | str | None = None,
    human_annotations_path: Path | str | None = None,
    generator_model_family: str = "vicuna_videollama",
    allow_cpu_mock: bool = False,
    e2_dataset: str | None = None,
    e2_annotation_path: Path | str | None = None,
    e2_video_root: Path | str | None = None,
) -> dict[str, Any]:
    """Real Pilot A2 stage (supports CaReBench primary & DREAM-1K backup): consumes A1 predictions, reuses generated
    auto-eval/blind bundles on re-entry, and performs strict claim_id reconciliation."""
    if judge_backend is None:
        raise ValueError("judge_backend is required for Pilot A2")
    ann_path = e2_annotation_path or dream1k_annotation_path
    vid_root = e2_video_root or dream1k_video_root
    if ann_path is None or vid_root is None:
        raise ValueError("Pilot A2 requires e2_annotation_path/e2_video_root (or dream1k_* aliases)")

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

    preds_peek = read_jsonl(a1_predictions_path)
    if not preds_peek:
        raise ValueError(f"Pilot A1 predictions file is empty: {a1_predictions_path}")
    # Infer dataset cleanly when caller omitted e2_dataset (NEVER mislabel CaReBench as DREAM-1K!)
    if e2_dataset is not None:
        resolved_e2_key = e2_dataset
    else:
        first_ds = str(preds_peek[0].get("dataset_id", "")).lower().replace("-", "")
        first_qid = str(preds_peek[0].get("query_id", "")).lower()
        if first_ds == "carebench" or first_qid.startswith("carebench_"):
            resolved_e2_key = "carebench"
        elif first_ds == "dream1k" or first_qid.startswith("dream1k_"):
            resolved_e2_key = "dream1k"
        else:
            resolved_e2_key = "carebench"
    dataset_label = "CaReBench" if resolved_e2_key.lower().replace("-", "") == "carebench" else "DREAM-1K"

    a1_sha = sha256_file(a1_predictions_path)
    eval_path = out / "a2_ref_fact_eval.jsonl"
    manifest_path = out / "a2_bundle_manifest.json"
    reused_existing = False

    # Reuse previously generated auto-eval + blind mapping when re-entering A2 with human annotations!
    if manifest_path.is_file() and eval_path.is_file():
        prev_man = read_json(manifest_path)
        prev_key = Path(prev_man["blind_annotation_files"]["blind_key"])
        if prev_man.get("a1_predictions_sha256") == a1_sha and prev_key.is_file():
            eval_rows = read_jsonl(eval_path)
            blind_pkg = {k: Path(v) for k, v in prev_man["blind_annotation_files"].items()}
            parse_failures = sum(1 for r in eval_rows if r.get("parse_status") != "OK")
            pkg_sha = str(prev_man.get("package_sha256", ""))
            dataset_label = str(prev_man.get("dataset_id", dataset_label))
            reused_existing = True

    if not reused_existing:
        preds = preds_peek
        dataset_label, recs = _load_e2_records_by_dataset(resolved_e2_key, ann_path, vid_root)
        rec_by_qid = {r.query_id: r for r in recs}

        evaluator = ThreeWayReferenceFactEvaluator(
            backend=judge_backend,
            generator_model_family=generator_model_family,
            enforce_cross_family=True,
        )

        eval_rows = []
        videos_for_blind: dict[str, dict[str, Any]] = {}
        parse_failures = 0

        for p in preds:
            qid = str(p["query_id"])
            if qid not in rec_by_qid:
                raise KeyError(f"A1 prediction query_id '{qid}' not found in {dataset_label} annotations")
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
            ev_d = _attach_stable_claim_ids(ev.to_dict())
            ev_d["dataset_id"] = dataset_label
            eval_rows.append(ev_d)

            v_entry = videos_for_blind.setdefault(
                qid, {"query_id": qid, "dataset_id": dataset_label, "video_path": qrec.video_path, "captions": []}
            )
            v_entry["captions"].append(
                {
                    "arm": ev.arm,
                    "model_id": ev.model_id,
                    "caption": ev.caption,
                    "claims": [{"claim_id": c["claim_id"], "claim": c["claim"]} for c in ev_d["claims"]],
                }
            )

        atomic_write_jsonl(eval_path, eval_rows)
        blind_pkg = build_blind_e2_annotation_package(
            list(videos_for_blind.values()), annotator_ids=["A1", "A2"], out_dir=out / "blind_annotation_package"
        )
        pkg_sha = str(read_json(blind_pkg["blind_key"]).get("package_sha256", ""))
        atomic_write_json(
            manifest_path,
            {
                "dataset_id": dataset_label,
                "a1_predictions_path": str(a1_predictions_path),
                "a1_predictions_sha256": a1_sha,
                "auto_eval_path": str(eval_path),
                "auto_eval_sha256": sha256_file(eval_path),
                "package_sha256": pkg_sha,
                "judge_model_id": judge_backend.cfg.model_id,
                "blind_annotation_files": {k: str(v) for k, v in blind_pkg.items()},
            },
        )

    # If human ground truth is not yet returned, STOP cleanly at HUMAN_VALIDATION_PENDING (NEVER fake PASS!)
    if not human_annotations_path or not Path(human_annotations_path).is_file():
        receipt = {
            "stage": PilotStage.PILOT_A2.value,
            "dataset_id": dataset_label,
            "status": StageRunStatus.HUMAN_VALIDATION_PENDING.value,
            "execution_mode": auth["mode"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "judge_model_id": judge_backend.cfg.model_id,
            "n_captions_evaluated": len(eval_rows),
            "parse_failures": parse_failures,
            "reused_existing_auto_eval_and_bundle": reused_existing,
            "package_sha256": pkg_sha,
            "human_calibration_completed": False,
            "blind_annotation_files": {k: str(v) for k, v in blind_pkg.items()},
            "next_action": (
                "Run `python scripts/launch_annotation_ui.py --bundle <blind_tasks_e2_A1.json>` for A1 & A2, "
                "adjudicate via `python scripts/adjudicate_annotations.py --task e2 ...`, then re-invoke Pilot A2 "
                "with --human-annotations <adjudicated_e2.json>."
            ),
        }
        atomic_write_json(out / "calibration_receipt.json", receipt)
        return receipt

    # Ingest adjudicated human annotations (either raw blind export needing unblinding or already unblinded)
    raw_human = read_json(human_annotations_path)
    human_list = raw_human["records"] if isinstance(raw_human, dict) and "records" in raw_human else raw_human
    if human_list and (
        "blind_caption_id" in human_list[0]
        or (human_list[0].get("task_type") == "E2_FACT_AUDIT" and "captions" in human_list[0])
    ):
        unblinded = unblind_annotations(human_list, blind_pkg["blind_key"])
        human_payload: dict[str, Any] = {
            "package_sha256": raw_human.get("package_sha256", pkg_sha) if isinstance(raw_human, dict) else pkg_sha,
            "records": unblinded,
        }
    else:
        human_payload = raw_human if isinstance(raw_human, dict) else {"records": human_list}

    # Strict claim_id reconciliation (rejects missing/duplicate claims, positional zip truncation, or bundle mismatch!)
    reconciled = reconcile_a2_human_and_auto(eval_rows, human_payload, expected_package_sha256=pkg_sha or None)
    aligned: list[AlignedAtomRecord] = reconciled["aligned_records"]
    atomic_write_json(
        out / "a2_aligned_records.json",
        [
            {
                "video_id": a.video_id,
                "arm": a.arm,
                "model_id": a.model_id,
                "ref_label": a.ref_label,
                "human_label": a.human_label,
            }
            for a in aligned
        ],
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
        "dataset_id": dataset_label,
        "status": status,
        "execution_mode": auth["mode"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "judge_model_id": judge_backend.cfg.model_id,
        "n_captions_evaluated": len(eval_rows),
        "parse_failures": parse_failures,
        "reused_existing_auto_eval_and_bundle": reused_existing,
        "package_sha256": pkg_sha,
        "human_calibration_completed": True,
        "reconciliation_audit": reconciled["reconciliation_audit"],
        "calibration_audit": cal,
        "aligned_records_path": str(out / "a2_aligned_records.json"),
        "verdict": cal["verdict"],
    }
    atomic_write_json(out / "calibration_receipt.json", receipt)
    return receipt


def run_pilot_a3_pipeline(
    dream1k_pilot_queries: list[QueryRecord] | None = None,
    retriever: RetrieverAdapter | None = None,
    modern_vlm: GeneratorAdapter | None = None,
    videollama_adapter: GeneratorAdapter | None = None,
    vidhalluc_ach_mcq_path: Path | str = "",
    vidhalluc_video_root: Path | str = "",
    e3_judge_backend: JudgeBackend | None = None,
    out_dir: Path | str = ".",
    videoufo_source_ids: set[str] | None = None,
    approved_e3_candidates_path: Path | str | None = None,
    e3_human_annotations_path: Path | str | None = None,
    dream1k_vlm_human_annotations_path: Path | str | None = None,
    a2_aligned_records_path: Path | str | None = None,
    ref_judge_backend: JudgeBackend | None = None,
    require_cross_model_dream1k_human: bool = False,
    auth_receipt_path: Path | str | None = None,
    prior_a2_receipt: Path | str | None = None,
    gpu_lock_path: Path | None = None,
    allow_cpu_mock: bool = False,
    skip_video_decode: bool = False,
    e2_pilot_queries: list[QueryRecord] | None = None,
    e2_vlm_human_annotations_path: Path | str | None = None,
    require_cross_model_e2_human: bool = False,
) -> dict[str, Any]:
    """Real Pilot A3 stage with FULL human evaluation closed-loop (supports CaReBench primary & DREAM-1K backup)."""
    pilot_queries = e2_pilot_queries if e2_pilot_queries is not None else (dream1k_pilot_queries or [])
    if not pilot_queries:
        raise ValueError("Pilot A3 requires non-empty e2_pilot_queries (or dream1k_pilot_queries)")
    if retriever is None or modern_vlm is None or videollama_adapter is None or e3_judge_backend is None:
        raise ValueError("Pilot A3 requires retriever, modern_vlm, videollama_adapter, and e3_judge_backend")
    e2_vlm_h_path = e2_vlm_human_annotations_path or dream1k_vlm_human_annotations_path
    req_cross_e2 = bool(require_cross_model_e2_human or require_cross_model_dream1k_human)
    vufo_sids = videoufo_source_ids or set()

    e2_dataset_id = pilot_queries[0].dataset_id or "CaReBench"
    is_carebench = e2_dataset_id.lower() == "carebench"
    e2_slug = "carebench" if is_carebench else "dream1k"

    auth = verify_stage_authorization(
        stage=PilotStage.PILOT_A3,
        auth_receipt_path=auth_receipt_path,
        prior_receipts={PilotStage.PILOT_A2.value: prior_a2_receipt} if prior_a2_receipt else {},
        allow_cpu_mock=allow_cpu_mock,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Run Modern VLM on E2 pilot queries (reuse cleanly if already completed!)
    vlm_e2_dir = out / f"{e2_slug}_modern_vlm"
    vlm_e2_rec_p = vlm_e2_dir / "run_receipt.json"
    if vlm_e2_rec_p.is_file() and read_json(vlm_e2_rec_p).get("verdict") == VerdictStatus.PASS.value:
        vlm_e2_summary = read_json(vlm_e2_rec_p)
    else:
        vlm_e2_summary = run_natural_retrieval_stage(
            RunSpec(
                run_id=f"pilot_a3_{e2_slug}_vlm",
                stage=PilotStage.PILOT_A3,
                dataset_id=e2_dataset_id,
                arms=[ExpArm.ARM_A0, ExpArm.ARM_A1, ExpArm.ARM_A7],
                out_dir=vlm_e2_dir,
                auth_receipt_path=Path(auth_receipt_path) if auth_receipt_path else None,
                prior_receipts={PilotStage.PILOT_A2.value: prior_a2_receipt} if prior_a2_receipt else {},
                gpu_lock_path=gpu_lock_path,
                allow_cpu_mock=allow_cpu_mock,
            ),
            queries=pilot_queries,
            retriever=retriever,
            generator=modern_vlm,
            skip_video_decode=skip_video_decode,
        )
        release_gpu_memory()

    # 1b. Evaluate Modern VLM E2 captions + build blind package for cross-model E2 bias check (reusable!)
    vlm_eval_path = out / f"{e2_slug}_vlm_ref_eval.jsonl"
    vlm_blind_manifest_path = out / f"{e2_slug}_vlm_blind_manifest.json"
    q_by_id = {q.query_id: q for q in pilot_queries}
    r_judge = ref_judge_backend or e3_judge_backend
    if vlm_blind_manifest_path.is_file() and vlm_eval_path.is_file():
        vlm_blind_man = read_json(vlm_blind_manifest_path)
        vlm_eval_rows = read_jsonl(vlm_eval_path)
    else:
        vlm_preds = read_jsonl(vlm_e2_dir / "predictions.jsonl")
        ref_ev = ThreeWayReferenceFactEvaluator(r_judge, enforce_cross_family=False)
        vlm_eval_rows = []
        vids_blind: dict[str, dict[str, Any]] = {}
        for p in vlm_preds:
            qrec = q_by_id[p["query_id"]]
            ev_d = _attach_stable_claim_ids(
                ref_ev.evaluate_caption(
                    p["query_id"], p["arm"], p["model_id"], p["caption"], qrec.reference_description, qrec.core_events
                ).to_dict()
            )
            ev_d["dataset_id"] = e2_dataset_id
            vlm_eval_rows.append(ev_d)
            ve = vids_blind.setdefault(
                qrec.query_id,
                {"query_id": qrec.query_id, "dataset_id": e2_dataset_id, "video_path": qrec.video_path, "captions": []},
            )
            ve["captions"].append(
                {
                    "arm": ev_d["arm"],
                    "model_id": ev_d["model_id"],
                    "caption": ev_d["caption"],
                    "claims": [{"claim_id": c["claim_id"], "claim": c["claim"]} for c in ev_d["claims"]],
                }
            )
        atomic_write_jsonl(vlm_eval_path, vlm_eval_rows)
        vlm_bpkg = build_blind_e2_annotation_package(
            list(vids_blind.values()), annotator_ids=["A1", "A2"], out_dir=out / f"{e2_slug}_vlm_blind_annotation_package"
        )
        vlm_blind_man = {
            "dataset_id": e2_dataset_id,
            "package_sha256": read_json(vlm_bpkg["blind_key"]).get("package_sha256", ""),
            "blind_annotation_files": {k: str(v) for k, v in vlm_bpkg.items()},
        }
        atomic_write_json(vlm_blind_manifest_path, vlm_blind_man)

    # 2. Load VidHalluc & run known YouTube ID overlap filter + 1-clip-per-source dedup
    vh_raw = load_vidhalluc_ach_mcq(vidhalluc_ach_mcq_path, vidhalluc_video_root)
    vh_clean, overlap_audit = deduplicate_and_filter_vidhalluc(vh_raw, videoufo_source_ids=vufo_sids)
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
            "e2_dataset_id": e2_dataset_id,
            "status": StageRunStatus.HUMAN_VALIDATION_PENDING.value,
            "pending_reason": "VIDHALLUC_CANDIDATE_APPLICABILITY_REVIEW",
            "execution_mode": auth["mode"],
            "e2_modern_vlm_summary": vlm_e2_summary,
            "vidhalluc_overlap_audit": overlap_audit,
            "pending_candidates_file": str(pending_file),
            "pilot_exclusion_manifest": str(excl_path),
            "e3_human_validation_completed": False,
            "next_action": "Complete 5-item human applicability review on VidHalluc candidates and pass --approved-e3-candidates.",
        }
        if not is_carebench:
            receipt["dream1k_modern_vlm_summary"] = vlm_e2_summary
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
            "e3_human_validation_completed": False,
            "error": "Zero human-APPROVED E3 candidates in approved_e3_candidates file (unverified distractors refused).",
        }
        atomic_write_json(out / "run_receipt.json", receipt)
        return receipt

    # 4 & 5. Run 4-arm visual-locked E3 on BOTH models + blind auto-eval + blind E3 bundle (reused on re-entry!)
    e3_eval_path = out / "e3_dual_endpoint_eval.jsonl"
    e3_blind_manifest_path = out / "a3_e3_blind_manifest.json"
    cands_sha = sha256_file(approved_e3_candidates_path)

    if e3_blind_manifest_path.is_file() and e3_eval_path.is_file() and read_json(e3_blind_manifest_path).get("approved_candidates_sha256") == cands_sha:
        e3_blind_man = read_json(e3_blind_manifest_path)
        e3_eval_rows = read_jsonl(e3_eval_path)
        e3_vl = read_json(out / "e3_videollama" / "e3_run_receipt.json")
        e3_vlm = read_json(out / "e3_modern_vlm" / "e3_run_receipt.json")
        reused_e3_bundle = True
    else:
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
        release_gpu_memory()

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
        release_gpu_memory()

        e3_eval = BlindE3SemanticEvaluator(e3_judge_backend)
        cand_by_qid = {c.query_id: c for c in runnable}
        e3_eval_rows = []
        e3_blind_items: list[dict[str, Any]] = []
        for subdir in ("e3_videollama", "e3_modern_vlm"):
            for p in read_jsonl(out / subdir / "e3_predictions.jsonl"):
                c = cand_by_qid[p["query_id"]]
                res = e3_eval.evaluate_caption_safe(
                    query_id=p["query_id"],
                    arm=p["arm"],
                    model_id=p["model_id"],
                    caption=p["caption"],
                    target_true_fact=c.target_true_fact,
                    target_conflict_fact=c.candidate_conflict_fact,
                )
                e3_eval_rows.append(res)
                e3_blind_items.append(
                    {
                        "query_id": p["query_id"],
                        "arm": p["arm"],
                        "model_id": p["model_id"],
                        "video_path": vh_by_qid[p["query_id"]].video_path,
                        "caption": p["caption"],
                        "target_true_fact": c.target_true_fact,
                        "target_conflict_fact": c.candidate_conflict_fact,
                    }
                )
        atomic_write_jsonl(e3_eval_path, e3_eval_rows)
        e3_bpkg = build_blind_e3_annotation_package(
            e3_blind_items, annotator_ids=["A1", "A2"], out_dir=out / "e3_blind_annotation_package"
        )
        e3_blind_man = {
            "approved_candidates_sha256": cands_sha,
            "package_sha256": read_json(e3_bpkg["blind_key"]).get("package_sha256", ""),
            "blind_annotation_files": {k: str(v) for k, v in e3_bpkg.items()},
        }
        atomic_write_json(e3_blind_manifest_path, e3_blind_man)
        reused_e3_bundle = False

    gen_all_ok = (
        vlm_e2_summary["verdict"] == VerdictStatus.PASS.value
        and e3_vl["verdict"] == VerdictStatus.PASS.value
        and e3_vlm["verdict"] == VerdictStatus.PASS.value
    )
    if not gen_all_ok:
        receipt = {
            "stage": PilotStage.PILOT_A3.value,
            "e2_dataset_id": e2_dataset_id,
            "status": StageRunStatus.FAILED.value,
            "execution_mode": auth["mode"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "e2_modern_vlm_summary": vlm_e2_summary,
            "e3_videollama_summary": e3_vl,
            "e3_modern_vlm_summary": e3_vlm,
            "e3_human_validation_completed": False,
            "verdict": VerdictStatus.FAIL.value,
        }
        if not is_carebench:
            receipt["dream1k_modern_vlm_summary"] = vlm_e2_summary
        atomic_write_json(out / "run_receipt.json", receipt)
        return receipt

    # 6. NEVER pass A3 on generation alone! Pause at HUMAN_VALIDATION_PENDING until E3 human verification returns!
    need_e3_human = not e3_human_annotations_path or not Path(e3_human_annotations_path).is_file()
    need_vlm_e2_human = req_cross_e2 and (not e2_vlm_h_path or not Path(e2_vlm_h_path).is_file())
    if need_e3_human or need_vlm_e2_human:
        receipt = {
            "stage": PilotStage.PILOT_A3.value,
            "e2_dataset_id": e2_dataset_id,
            "status": StageRunStatus.HUMAN_VALIDATION_PENDING.value,
            "pending_reason": "E3_AND_CROSS_MODEL_HUMAN_EVALUATION_REQUIRED",
            "execution_mode": auth["mode"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "e2_modern_vlm_summary": vlm_e2_summary,
            "e3_videollama_summary": e3_vl,
            "e3_modern_vlm_summary": e3_vlm,
            "n_e3_evaluated_captions": len(e3_eval_rows),
            "e3_human_validation_completed": False,
            "e3_blind_annotation_files": e3_blind_man["blind_annotation_files"],
            "e3_package_sha256": e3_blind_man["package_sha256"],
            "e2_vlm_blind_annotation_files": vlm_blind_man["blind_annotation_files"],
            "e2_vlm_package_sha256": vlm_blind_man["package_sha256"],
            "next_action": (
                "Launch local web UI on e3_blind_annotation_package/blind_tasks_e3_A1.json & A2.json, "
                "adjudicate via `python scripts/adjudicate_annotations.py --task e3 ...`, then re-run Pilot A3 "
                "with --e3-human-annotations <adjudicated_e3.json> (and optional --e2-vlm-human-annotations)."
            ),
        }
        if not is_carebench:
            receipt["dream1k_modern_vlm_summary"] = vlm_e2_summary
            receipt["dream1k_vlm_blind_annotation_files"] = vlm_blind_man["blind_annotation_files"]
            receipt["dream1k_vlm_package_sha256"] = vlm_blind_man["package_sha256"]
        atomic_write_json(out / "run_receipt.json", receipt)
        return receipt

    # 7. Ingest E3 human verification (unblind if needed) + compute independent Y_true/Y_conflict & Con vs Neu bias!
    raw_e3_h = read_json(e3_human_annotations_path)
    e3_h_list = raw_e3_h["records"] if isinstance(raw_e3_h, dict) and "records" in raw_e3_h else raw_e3_h
    if e3_h_list and (
        "blind_task_id" in e3_h_list[0] or (e3_h_list[0].get("task_type") == "E3_DUAL_TARGET" and "e3" in e3_h_list[0])
    ):
        e3_h_unblinded = unblind_annotations(e3_h_list, e3_blind_man["blind_annotation_files"]["blind_key"])
        e3_h_payload: dict[str, Any] = {
            "package_sha256": raw_e3_h.get("package_sha256", e3_blind_man["package_sha256"]) if isinstance(raw_e3_h, dict) else e3_blind_man["package_sha256"],
            "records": e3_h_unblinded,
        }
    else:
        e3_h_payload = raw_e3_h if isinstance(raw_e3_h, dict) else {"records": e3_h_list}

    e3_agreement = audit_e3_evaluator_agreement(
        auto_e3_rows=e3_eval_rows,
        human_e3_payload=e3_h_payload,
        expected_package_sha256=e3_blind_man["package_sha256"] or None,
    )

    # 8. Optional/Required cross-model E2 evaluation bias check
    cross_model_bias: dict[str, Any] | None = None
    if e2_vlm_h_path and Path(e2_vlm_h_path).is_file():
        raw_vlm_h = read_json(e2_vlm_h_path)
        vlm_h_list = raw_vlm_h["records"] if isinstance(raw_vlm_h, dict) and "records" in raw_vlm_h else raw_vlm_h
        if vlm_h_list and (
            "blind_caption_id" in vlm_h_list[0]
            or (vlm_h_list[0].get("task_type") == "E2_FACT_AUDIT" and "captions" in vlm_h_list[0])
        ):
            vlm_h_payload: dict[str, Any] = {
                "package_sha256": vlm_blind_man["package_sha256"],
                "records": unblind_annotations(vlm_h_list, vlm_blind_man["blind_annotation_files"]["blind_key"]),
            }
        else:
            vlm_h_payload = raw_vlm_h if isinstance(raw_vlm_h, dict) else {"records": vlm_h_list}
        vlm_rec = reconcile_a2_human_and_auto(
            vlm_eval_rows, vlm_h_payload, expected_package_sha256=vlm_blind_man["package_sha256"] or None
        )
        vlm_aligned: list[AlignedAtomRecord] = vlm_rec["aligned_records"]
        if a2_aligned_records_path and Path(a2_aligned_records_path).is_file():
            vl_aligned = [AlignedAtomRecord(**r) for r in read_json(a2_aligned_records_path)]
            cross_model_bias = audit_cross_model_e2_bias(vl_aligned, vlm_aligned, dataset_id=e2_dataset_id)
        else:
            cross_model_bias = {
                "dataset_id": e2_dataset_id,
                "verdict": audit_evaluator_calibration(vlm_aligned, n_boot=300)["verdict"],
                "note": f"Modern VLM {e2_dataset_id} calibration audited standalone (pass --a2-aligned-records for cross-model gap).",
                "modern_vlm_audit": audit_evaluator_calibration(vlm_aligned, n_boot=300),
            }

    validation_report = {
        "stage": PilotStage.PILOT_A3.value,
        "e2_dataset_id": e2_dataset_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reused_e3_bundle": reused_e3_bundle,
        "e3_evaluator_agreement": e3_agreement,
        "cross_model_e2_bias": cross_model_bias,
    }
    if not is_carebench:
        validation_report["cross_model_dream1k_bias"] = cross_model_bias
    atomic_write_json(out / "a3_evaluator_validation_report.json", validation_report)

    verdicts = [e3_agreement["verdict"]]
    if cross_model_bias is not None:
        verdicts.append(cross_model_bias["verdict"])

    if VerdictStatus.FAIL.value in verdicts:
        final_verdict = VerdictStatus.FAIL.value
        status = StageRunStatus.FAILED.value
    elif VerdictStatus.INCONCLUSIVE.value in verdicts:
        final_verdict = VerdictStatus.INCONCLUSIVE.value
        status = StageRunStatus.INCONCLUSIVE.value
    else:
        final_verdict = VerdictStatus.PASS.value
        status = StageRunStatus.MOCK_PASS.value if auth["mode"] == "cpu_mock" else StageRunStatus.REAL_RUNTIME_PASS.value

    receipt = {
        "stage": PilotStage.PILOT_A3.value,
        "e2_dataset_id": e2_dataset_id,
        "status": status,
        "execution_mode": auth["mode"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "e2_modern_vlm_summary": vlm_e2_summary,
        "e3_videollama_summary": e3_vl,
        "e3_modern_vlm_summary": e3_vlm,
        "n_e3_evaluated_captions": len(e3_eval_rows),
        "e3_human_validation_completed": True,
        "cross_model_e2_bias_checked": cross_model_bias is not None,
        "e3_evaluator_agreement_verdict": e3_agreement["verdict"],
        "cross_model_e2_bias_verdict": cross_model_bias["verdict"] if cross_model_bias else None,
        "validation_report_path": str(out / "a3_evaluator_validation_report.json"),
        "verdict": final_verdict,
    }
    if not is_carebench:
        receipt["dream1k_modern_vlm_summary"] = vlm_e2_summary
        receipt["cross_model_dream1k_bias_checked"] = cross_model_bias is not None
        receipt["cross_model_dream1k_bias_verdict"] = cross_model_bias["verdict"] if cross_model_bias else None
    atomic_write_json(out / "run_receipt.json", receipt)
    return receipt


def run_formal_b_pipeline(
    formal_dream1k_queries: list[QueryRecord] | None = None,
    formal_vidhalluc_queries_by_id: dict[str, QueryRecord] | None = None,
    approved_e3_candidates: list[E3InterventionCandidate] | None = None,
    retriever: RetrieverAdapter | None = None,
    videollama_adapter: GeneratorAdapter | None = None,
    modern_vlm: GeneratorAdapter | None = None,
    ref_judge_backend: JudgeBackend | None = None,
    e3_judge_backend: JudgeBackend | None = None,
    out_dir: Path | str = ".",
    auth_receipt_path: Path | str = "",
    prior_receipts: dict[str, Path | str] | None = None,
    formal_config_path: Path | str = "",
    asset_manifest_path: Path | str = "",
    pilot_exclusion_path: Path | str = "",
    pilot_excluded_query_ids: set[str] | None = None,
    e3_bridge_observations: list[NaturalConflictObservation] | None = None,
    sampling_cfg: FormalSamplingConfig | None = None,
    explicit_near_dup_excluded_ids: set[str] | None = None,
    videoufo_source_ids: set[str] | None = None,
    pilot_excluded_vidhalluc_query_ids: set[str] | None = None,
    pilot_excluded_vidhalluc_source_ids: set[str] | None = None,
    e2_validation_human_annotations_path: Path | str | None = None,
    msrvtt_historical_bridge: dict[str, Path | str] | None = None,
    missing_policy_max_rate: float | None = None,
    e3_bridge_min_clean_pairs: int = 25,
    gpu_lock_path: Path | None = None,
    skip_video_decode: bool = False,
    formal_e2_queries: list[QueryRecord] | None = None,
    pilot_excluded_parent_ids: set[str] | None = None,
    extra_debug_query_ids: set[str] | None = None,
    extra_debug_parent_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Complete guarded Formal Study B pipeline (supports CaReBench primary & DREAM-1K backup):
      - Parent-video dedup (CaReBench) + stratified probability sampling + Pilot/debug/near-dup/same-source isolation
      - E2 (both models) + E3 (both models) + real E3-Bridge execution (when enabled)
      - Full missingness-preserving stats (NEVER `or 0.0`!):
          * E2 A1-A0, A7-A1, A7-A0 across n_contra_ref, n_entail_ref, r_core_ref, p_unment_ref, n_unment_ref,
            plus auxiliary reference support/conflict rates and static/dynamic/mixed/unclassified diagnostic counts
          * E2 independent human validation set + Two-Phase IPW error correction
          * E2 cross-model DiD + Holm-Bonferroni (primary family preserved unchanged!)
          * E3 primary (Con vs Neu) + secondary comparisons + cross-model DiD
          * MSR-VTT historical output bridge
          * E3-Bridge execution stats (when enabled)
    """
    e2_queries_in = formal_e2_queries if formal_e2_queries is not None else (formal_dream1k_queries or [])
    if not e2_queries_in:
        raise ValueError("FORMAL_B requires non-empty formal_e2_queries (or formal_dream1k_queries)")
    if (
        formal_vidhalluc_queries_by_id is None
        or approved_e3_candidates is None
        or retriever is None
        or videollama_adapter is None
        or modern_vlm is None
        or ref_judge_backend is None
        or e3_judge_backend is None
        or prior_receipts is None
    ):
        raise ValueError("FORMAL_B missing required pipeline dependencies")

    pilot_excl_q = set(pilot_excluded_query_ids or set())
    e2_dataset_id = e2_queries_in[0].dataset_id or "CaReBench"
    is_carebench = e2_dataset_id.lower() == "carebench"
    e2_slug = "carebench" if is_carebench else "dream1k"

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

    # 0. Formal Sample Isolation, Parent-Video Dedup (CaReBench), Stratified Sampling & Analysis Manifest Freeze
    scfg = sampling_cfg or FormalSamplingConfig()
    e2_manifest_path = out / f"formal_{e2_slug}_analysis_manifest.json"
    selected_e2, e2_manifest = build_formal_e2_manifest(
        all_records=e2_queries_in,
        pilot_excluded_ids=pilot_excl_q,
        cfg=scfg,
        explicit_near_dup_excluded_ids=explicit_near_dup_excluded_ids,
        out_manifest_path=e2_manifest_path,
        pilot_excluded_parent_ids=pilot_excluded_parent_ids,
        extra_debug_query_ids=extra_debug_query_ids,
        extra_debug_parent_ids=extra_debug_parent_ids,
    )
    vh_selected, vh_manifest = build_formal_vidhalluc_manifest(
        raw_records=list(formal_vidhalluc_queries_by_id.values()),
        videoufo_source_ids=videoufo_source_ids or set(),
        pilot_excluded_query_ids=pilot_excluded_vidhalluc_query_ids or set(),
        pilot_excluded_source_ids=pilot_excluded_vidhalluc_source_ids or set(),
        cfg=scfg,
        out_manifest_path=out / "formal_vidhalluc_analysis_manifest.json",
    )
    vh_allowed_ids = {r.query_id for r in vh_selected}
    vh_filtered_map = {r.query_id: r for r in vh_selected}
    filtered_e3_cands = [c for c in approved_e3_candidates if c.query_id in vh_allowed_ids]

    # 1. E2 Natural Retrieval on E2 Formal (both models, shared stratified cohort)
    e2_summaries = {}
    for tag, gen in (("videollama", videollama_adapter), ("modern_vlm", modern_vlm)):
        e2_summaries[tag] = run_natural_retrieval_stage(
            RunSpec(
                run_id=f"formal_b_e2_{e2_slug}_{tag}",
                stage=PilotStage.FORMAL_B,
                dataset_id=e2_dataset_id,
                arms=[ExpArm.ARM_A0, ExpArm.ARM_A1, ExpArm.ARM_A7],
                out_dir=out / f"e2_{tag}",
                auth_receipt_path=Path(auth_receipt_path),
                prior_receipts=prior_receipts,
                formal_config_path=Path(formal_config_path),
                asset_manifest_path=Path(asset_manifest_path),
                pilot_exclusion_path=Path(pilot_exclusion_path),
                pilot_excluded_query_ids=pilot_excl_q,
                gpu_lock_path=gpu_lock_path,
                allow_cpu_mock=False,
            ),
            queries=selected_e2,
            retriever=retriever,
            generator=gen,
            skip_video_decode=skip_video_decode,
        )
        release_gpu_memory()

    # 2. E3 Controlled Intervention on VidHalluc Formal (both models, isolated non-pilot sources)
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
            queries_by_id=vh_filtered_map,
            candidates=filtered_e3_cands,
            generator=gen,
            base_visual_mode="BASE_A0",
            skip_video_decode=skip_video_decode,
        )
        release_gpu_memory()

    # 3. Real E3-Bridge cohort screening AND actual execution when enabled!
    bridge_report = (
        build_e3_bridge_cohort(
            e3_bridge_observations,
            min_clean_pairs=e3_bridge_min_clean_pairs,
            source_dataset_id=e2_dataset_id,
        )
        if e3_bridge_observations is not None
        else {"module": "E3-Bridge", "source_dataset_id": e2_dataset_id, "enabled": False, "status": "NOT_PROVIDED"}
    )
    atomic_write_json(out / "e3_bridge_decision.json", bridge_report)

    q_by_id = {q.query_id: q for q in selected_e2}
    e3_bridge_execution: dict[str, Any] = {
        "source_dataset_id": e2_dataset_id,
        "enabled": bool(bridge_report.get("enabled")),
        "executed": False,
    }
    if bridge_report.get("enabled") and bridge_report.get("candidates"):
        bridge_cands = [E3InterventionCandidate(**cd) for cd in bridge_report["candidates"] if cd["query_id"] in q_by_id]
        if bridge_cands:
            bridge_runs: dict[str, Any] = {}
            bridge_stats: dict[str, Any] = {}
            e3_ev_bridge = BlindE3SemanticEvaluator(e3_judge_backend)
            bc_map = {c.query_id: c for c in bridge_cands}
            for tag, gen in (("videollama", videollama_adapter), ("modern_vlm", modern_vlm)):
                bridge_runs[tag] = run_e3_controlled_stage(
                    RunSpec(
                        run_id=f"formal_b_e3_bridge_{e2_slug}_{tag}",
                        stage=PilotStage.FORMAL_B,
                        dataset_id=f"{e2_dataset_id}-E3-Bridge",
                        arms=[ExpArm.E3_BASE, ExpArm.E3_SUP, ExpArm.E3_CON, ExpArm.E3_NEU],
                        out_dir=out / f"e3_bridge_{tag}",
                        auth_receipt_path=Path(auth_receipt_path),
                        prior_receipts=prior_receipts,
                        formal_config_path=Path(formal_config_path),
                        asset_manifest_path=Path(asset_manifest_path),
                        pilot_exclusion_path=Path(pilot_exclusion_path),
                        gpu_lock_path=gpu_lock_path,
                        allow_cpu_mock=False,
                    ),
                    queries_by_id=q_by_id,
                    candidates=bridge_cands,
                    generator=gen,
                    base_visual_mode="BASE_A0",
                    skip_video_decode=skip_video_decode,
                )
                release_gpu_memory()
                b_preds = read_jsonl(out / f"e3_bridge_{tag}" / "e3_predictions.jsonl")
                b_rows = [
                    e3_ev_bridge.evaluate_caption_safe(
                        p["query_id"], p["arm"], p["model_id"], p["caption"], bc_map[p["query_id"]].target_true_fact, bc_map[p["query_id"]].candidate_conflict_fact
                    )
                    for p in b_preds
                ]
                atomic_write_jsonl(out / f"e3_bridge_{tag}_dual_eval.jsonl", b_rows)
                con_c, _ = _extract_valid_e3_binary_map(b_rows, ExpArm.E3_CON.value, "y_conflict")
                neu_c, _ = _extract_valid_e3_binary_map(b_rows, ExpArm.E3_NEU.value, "y_conflict")
                con_t, _ = _extract_valid_e3_binary_map(b_rows, ExpArm.E3_CON.value, "y_true")
                neu_t, _ = _extract_valid_e3_binary_map(b_rows, ExpArm.E3_NEU.value, "y_true")
                bridge_stats[tag] = {
                    "y_conflict_con_vs_neu": paired_binary_mcnemar_effect(con_c, neu_c, n_boot=300),
                    "y_true_con_vs_neu": paired_binary_mcnemar_effect(con_t, neu_t, n_boot=300),
                }
            e3_bridge_execution = {
                "source_dataset_id": e2_dataset_id,
                "enabled": True,
                "executed": True,
                "n_candidates_executed": len(bridge_cands),
                "runs": bridge_runs,
                "stats": bridge_stats,
            }

    # 4. E2 Automated Evaluation & Complete Missingness-Preserving Paired Statistics
    ref_ev = ThreeWayReferenceFactEvaluator(ref_judge_backend, enforce_cross_family=False)
    e2_evals_by_model: dict[str, list[dict[str, Any]]] = {}
    e2_stats_by_model: dict[str, Any] = {}
    e2_deltas_store: dict[str, dict[str, dict[str, float]]] = {}

    e2_endpoints = ("n_contra_ref", "n_entail_ref", "r_core_ref", "p_unment_ref", "n_unment_ref")
    e2_contrasts = (
        ("a1_minus_a0", ExpArm.ARM_A1.value, ExpArm.ARM_A0.value),
        ("a7_minus_a1", ExpArm.ARM_A7.value, ExpArm.ARM_A1.value),
        ("a7_minus_a0", ExpArm.ARM_A7.value, ExpArm.ARM_A0.value),
    )

    for tag in ("videollama", "modern_vlm"):
        preds = read_jsonl(out / f"e2_{tag}" / "predictions.jsonl")
        evals = [
            _attach_stable_claim_ids(
                ref_ev.evaluate_caption(
                    p["query_id"],
                    p["arm"],
                    p["model_id"],
                    p["caption"],
                    q_by_id[p["query_id"]].reference_description,
                    q_by_id[p["query_id"]].core_events,
                ).to_dict()
            )
            for p in preds
        ]
        atomic_write_jsonl(out / f"e2_{tag}_ref_eval.jsonl", evals)
        e2_evals_by_model[tag] = evals

        model_contrasts: dict[str, Any] = {}
        e2_deltas_store[tag] = {}
        primary_pvals: dict[str, float | None] = {}

        for ep in e2_endpoints:
            model_contrasts[ep] = {}
            for cname, arm_a, arm_b in e2_contrasts:
                diffs, stat = _paired_arm_diff_analysis(
                    evals, arm_a, arm_b, ep, n_boot=300, n_perm=500, max_missing_rate_fail=missing_policy_max_rate
                )
                model_contrasts[ep][cname] = stat
                e2_deltas_store[tag][f"{ep}__{cname}"] = diffs
                if ep in ("n_contra_ref", "n_entail_ref") and cname in ("a1_minus_a0", "a7_minus_a1"):
                    primary_pvals[f"{ep}__{cname}"] = stat.get("p_value")

        # Per-arm summary for core coverage & unmentioned rates (preserving missingness!)
        arm_descriptives: dict[str, Any] = {}
        for arm in (ExpArm.ARM_A0.value, ExpArm.ARM_A1.value, ExpArm.ARM_A7.value):
            arm_descriptives[arm] = {}
            for ep in e2_endpoints:
                vmap, mrep = _extract_valid_metric_map(evals, arm, ep)
                arm_descriptives[arm][ep] = {
                    "mean": round(sum(vmap.values()) / len(vmap), 6) if vmap else None,
                    "valid_n": mrep["valid_n"],
                    "missing_n": mrep["missing_n"],
                    "missing_rate": mrep["missing_rate"],
                }

        e2_stats_by_model[tag] = {
            "parse_failures_total": sum(1 for e in evals if e.get("parse_status") != "OK"),
            "arm_descriptives": arm_descriptives,
            "paired_contrasts": model_contrasts,
            "holm_primary_family_a1_a0_and_a7_a1": holm_bonferroni(primary_pvals),
        }

    # 4b. E2 Cross-Model DiD across A7-A0, A1-A0, A7-A1 for primary endpoints
    e2_did_all: dict[str, Any] = {}
    did_pvals: dict[str, float | None] = {}
    for ep in ("n_contra_ref", "n_entail_ref"):
        for cname in ("a7_minus_a0", "a1_minus_a0", "a7_minus_a1"):
            k = f"{ep}__{cname}"
            dres = cross_model_did_effect(e2_deltas_store["modern_vlm"][k], e2_deltas_store["videollama"][k], n_boot=300, n_perm=500)
            e2_did_all[k] = dres
            if cname == "a7_minus_a0":
                did_pvals[k] = dres.get("p_value")
    e2_did_all["holm_a7_minus_a0"] = holm_bonferroni(did_pvals)

    # 4c. E2 Independent Human Validation Set + Two-Phase Sampling Error-Correction Candidate Analysis
    two_phase_results: dict[str, Any] = {}
    val_ids = set(e2_manifest["e2_validation_set"]["validation_query_ids"])
    if val_ids:
        # Always export blind validation bundle for the independent E2 validation subset
        val_blind_vids: dict[str, dict[str, Any]] = {}
        for tag in ("videollama", "modern_vlm"):
            for ev_d in e2_evals_by_model[tag]:
                if ev_d["query_id"] not in val_ids or ev_d.get("parse_status") != "OK":
                    continue
                qrec = q_by_id[ev_d["query_id"]]
                ve = val_blind_vids.setdefault(
                    qrec.query_id,
                    {"query_id": qrec.query_id, "dataset_id": e2_dataset_id, "video_path": qrec.video_path, "captions": []},
                )
                ve["captions"].append(
                    {
                        "arm": ev_d["arm"],
                        "model_id": ev_d["model_id"],
                        "caption": ev_d["caption"],
                        "claims": [{"claim_id": c["claim_id"], "claim": c["claim"]} for c in ev_d["claims"]],
                    }
                )
        val_bpkg = build_blind_e2_annotation_package(
            list(val_blind_vids.values()), annotator_ids=["A1", "A2"], out_dir=out / "e2_validation_blind_package"
        )
        two_phase_results["validation_blind_package"] = {k: str(v) for k, v in val_bpkg.items()}

    if e2_validation_human_annotations_path and Path(e2_validation_human_annotations_path).is_file():
        raw_val_h = read_json(e2_validation_human_annotations_path)
        val_h_recs = raw_val_h["records"] if isinstance(raw_val_h, dict) and "records" in raw_val_h else raw_val_h
        # Compute human video-error counts per (model_id, query_id, arm)
        hum_err_by_mqa: dict[tuple[str, str, str], int] = {}
        for r in val_h_recs:
            lbls = list(r.get("claim_labels_by_id", {}).values())
            n_err = sum(1 for v in lbls if v == VideoTruthLabel.VIDEO_CONTRADICTED.value)
            n_err += sum(1 for ac in r.get("added_claims", []) if ac.get("label") == VideoTruthLabel.VIDEO_CONTRADICTED.value)
            hum_err_by_mqa[(str(r["model_id"]), str(r["query_id"]), str(r["arm"]))] = n_err

        for tag, model_id in (("videollama", videollama_adapter.model_id), ("modern_vlm", modern_vlm.model_id)):
            auto_deltas = e2_deltas_store[tag]["n_contra_ref__a7_minus_a0"]
            tp_units: list[TwoPhaseUnit] = []
            for qid, auto_d in sorted(auto_deltas.items()):
                um = e2_manifest["units"][qid]
                in_val = bool(um["in_e2_human_validation"])
                h_delta: float | None = None
                if in_val:
                    k7 = (model_id, qid, ExpArm.ARM_A7.value)
                    k0 = (model_id, qid, ExpArm.ARM_A0.value)
                    if k7 in hum_err_by_mqa and k0 in hum_err_by_mqa:
                        h_delta = float(hum_err_by_mqa[k7] - hum_err_by_mqa[k0])
                tp_units.append(
                    TwoPhaseUnit(
                        query_id=qid,
                        stratum=str(um["stratum"]),
                        auto_delta=float(auto_d),
                        in_human_sample=(in_val and h_delta is not None),
                        inclusion_prob=float(um["val_inclusion_prob"]) if in_val else 0.0,
                        human_delta=h_delta,
                    )
                )
            two_phase_results[tag] = estimate_two_phase_corrected_delta(
                tp_units, calibration_excluded_ids=pilot_excl_q
            )
    else:
        two_phase_results["status"] = "PENDING_E2_VALIDATION_HUMAN_ANNOTATIONS"

    # 5. E3 Dual Endpoints (Primary Con vs Neu + Secondary Intervention Comparisons + Cross-Model DiD)
    e3_ev = BlindE3SemanticEvaluator(e3_judge_backend)
    cand_map = {c.query_id: c for c in filtered_e3_cands}
    e3_stats: dict[str, Any] = {}
    e3_con_neu_diffs: dict[str, dict[str, dict[str, float]]] = {"y_conflict": {}, "y_true": {}}
    secondary_pairs = (
        ("sup_vs_neu", ExpArm.E3_SUP.value, ExpArm.E3_NEU.value),
        ("con_vs_sup", ExpArm.E3_CON.value, ExpArm.E3_SUP.value),
        ("con_vs_base", ExpArm.E3_CON.value, ExpArm.E3_BASE.value),
        ("sup_vs_base", ExpArm.E3_SUP.value, ExpArm.E3_BASE.value),
        ("neu_vs_base", ExpArm.E3_NEU.value, ExpArm.E3_BASE.value),
    )

    for tag in ("videollama", "modern_vlm"):
        preds = read_jsonl(out / f"e3_{tag}" / "e3_predictions.jsonl")
        e3_rows = [
            e3_ev.evaluate_caption_safe(
                p["query_id"],
                p["arm"],
                p["model_id"],
                p["caption"],
                cand_map[p["query_id"]].target_true_fact,
                cand_map[p["query_id"]].candidate_conflict_fact,
            )
            for p in preds
        ]
        atomic_write_jsonl(out / f"e3_{tag}_dual_eval.jsonl", e3_rows)

        con_c, rep_cc = _extract_valid_e3_binary_map(e3_rows, ExpArm.E3_CON.value, "y_conflict")
        neu_c, rep_nc = _extract_valid_e3_binary_map(e3_rows, ExpArm.E3_NEU.value, "y_conflict")
        con_t, rep_ct = _extract_valid_e3_binary_map(e3_rows, ExpArm.E3_CON.value, "y_true")
        neu_t, rep_nt = _extract_valid_e3_binary_map(e3_rows, ExpArm.E3_NEU.value, "y_true")

        mc_c = paired_binary_mcnemar_effect(con_c, neu_c, n_boot=300)
        mc_t = paired_binary_mcnemar_effect(con_t, neu_t, n_boot=300)
        hb = holm_bonferroni({"y_conflict": mc_c["mcnemar_p"], "y_true": mc_t["mcnemar_p"]})

        e3_con_neu_diffs["y_conflict"][tag] = {q: float(con_c[q] - neu_c[q]) for q in sorted(set(con_c) & set(neu_c))}
        e3_con_neu_diffs["y_true"][tag] = {q: float(con_t[q] - neu_t[q]) for q in sorted(set(con_t) & set(neu_t))}

        sec_comparisons: dict[str, Any] = {}
        for sname, arm_a, arm_b in secondary_pairs:
            ac, _ = _extract_valid_e3_binary_map(e3_rows, arm_a, "y_conflict")
            bc, _ = _extract_valid_e3_binary_map(e3_rows, arm_b, "y_conflict")
            at, _ = _extract_valid_e3_binary_map(e3_rows, arm_a, "y_true")
            bt, _ = _extract_valid_e3_binary_map(e3_rows, arm_b, "y_true")
            sec_comparisons[sname] = {
                "y_conflict": paired_binary_mcnemar_effect(ac, bc, n_boot=300),
                "y_true": paired_binary_mcnemar_effect(at, bt, n_boot=300),
            }

        e3_stats[tag] = {
            "parse_failures_total": sum(1 for r in e3_rows if r.get("parse_status", "OK") != "OK"),
            "missingness_primary": {"con_y_conflict": rep_cc, "neu_y_conflict": rep_nc, "con_y_true": rep_ct, "neu_y_true": rep_nt},
            "y_conflict_con_vs_neu": mc_c,
            "y_true_con_vs_neu": mc_t,
            "holm": hb,
            "joint_pattern": classify_e3_joint_pattern(
                mc_c["risk_diff"], hb["y_conflict"]["significant"], mc_t["risk_diff"], hb["y_true"]["significant"]
            ),
            "secondary_comparisons": sec_comparisons,
        }

    e3_did = {
        "y_conflict_con_vs_neu": cross_model_did_effect(
            e3_con_neu_diffs["y_conflict"]["modern_vlm"], e3_con_neu_diffs["y_conflict"]["videollama"], n_boot=300, n_perm=500
        ),
        "y_true_con_vs_neu": cross_model_did_effect(
            e3_con_neu_diffs["y_true"]["modern_vlm"], e3_con_neu_diffs["y_true"]["videollama"], n_boot=300, n_perm=500
        ),
    }
    e3_did["holm"] = holm_bonferroni(
        {"y_conflict": e3_did["y_conflict_con_vs_neu"].get("p_value"), "y_true": e3_did["y_true_con_vs_neu"].get("p_value")}
    )

    # 6. Optional MSR-VTT Historical Output Bridge
    msrvtt_bridge_report: dict[str, Any] = {"status": "NOT_PROVIDED"}
    if msrvtt_historical_bridge:
        if msrvtt_historical_bridge.get("summary_json") and Path(msrvtt_historical_bridge["summary_json"]).is_file():
            msrvtt_bridge_report = {"status": "LOADED_FROZEN_SUMMARY", "summary": read_json(msrvtt_historical_bridge["summary_json"])}
        elif all(
            msrvtt_historical_bridge.get(k) and Path(msrvtt_historical_bridge[k]).is_file()
            for k in ("e1_ref_a", "e1_ref_b", "e1_source_map")
        ):
            e1_res = run_e1_retrospective_pipeline(
                msrvtt_historical_bridge["e1_ref_a"],
                msrvtt_historical_bridge["e1_ref_b"],
                msrvtt_historical_bridge["e1_source_map"],
                out / "msrvtt_historical_bridge_e1.json",
            )
            msrvtt_bridge_report = {"status": "COMPUTED_FROM_FROZEN_PER_QUERY_JSONS", "e1_report": e1_res}

    # Auxiliary static/dynamic/mixed/unclassified diagnostic summary (strictly descriptive, NO added formal tests!)
    e2_aux_dim_summary: dict[str, Any] = {}
    for tag in ("videollama", "modern_vlm"):
        by_arm_dim: dict[str, Any] = {}
        for arm in (ExpArm.ARM_A0.value, ExpArm.ARM_A1.value, ExpArm.ARM_A7.value):
            agg: dict[str, dict[str, int]] = {
                d: {"total": 0, "entailed": 0, "contradicted": 0, "unmentioned": 0}
                for d in ("static_entity_attribute", "dynamic_action_temporal", "mixed", "unclassified")
            }
            for ev in e2_evals_by_model[tag]:
                if ev.get("arm") != arm or ev.get("parse_status") != "OK":
                    continue
                bd = ev.get("auxiliary_dimension_breakdown") or {}
                for d, counts in bd.items():
                    if d in agg:
                        for k in ("total", "entailed", "contradicted", "unmentioned"):
                            agg[d][k] += int(counts.get(k, 0))
            by_arm_dim[arm] = agg
        e2_aux_dim_summary[tag] = by_arm_dim

    manifests_info: dict[str, str] = {
        "e2_dataset_id": e2_dataset_id,
        "e2_manifest": str(e2_manifest_path),
        "vidhalluc_manifest": str(out / "formal_vidhalluc_analysis_manifest.json"),
    }
    if not is_carebench:
        manifests_info["dream1k_manifest"] = str(e2_manifest_path)

    stats_bundle = {
        "e2_dataset_id": e2_dataset_id,
        "e3_dataset_id": "VidHalluc-ACH",
        "e3_causal_scope_notice": (
            "VidHalluc-ACH E3 controlled causal findings apply primarily to action facts and are NOT "
            f"automatically extrapolated to all static object/attribute errors observed in {e2_dataset_id} E2."
        ),
        "formal_sampling_manifests": manifests_info,
        "e2_by_model": e2_stats_by_model,
        "e2_auxiliary_static_dynamic_diagnostics": e2_aux_dim_summary,
        "e2_cross_model_did": e2_did_all,
        "e2_cross_model_did_contra": e2_did_all["n_contra_ref__a7_minus_a0"],  # Backward-compatible alias
        "e2_two_phase_error_correction": two_phase_results,
        "e3_by_model": e3_stats,
        "e3_cross_model_did": e3_did,
        "e3_bridge": e3_bridge_execution,
        "msrvtt_historical_bridge": msrvtt_bridge_report,
    }
    atomic_write_json(out / "formal_b_statistics.json", stats_bundle)

    all_pass = all(s["verdict"] == VerdictStatus.PASS.value for s in (*e2_summaries.values(), *e3_summaries.values()))
    if e3_bridge_execution.get("executed"):
        all_pass = all_pass and all(
            r["verdict"] == VerdictStatus.PASS.value for r in e3_bridge_execution["runs"].values()
        )
    receipt = {
        "stage": PilotStage.FORMAL_B.value,
        "e2_dataset_id": e2_dataset_id,
        "e3_dataset_id": "VidHalluc-ACH",
        "status": StageRunStatus.REAL_RUNTIME_PASS.value if all_pass else StageRunStatus.FAILED.value,
        "approval_id": auth["approval_id"],
        "evaluator_version": auth["evaluator_version"],
        "e2_manifest_sha256": e2_manifest.get("manifest_sha256", ""),
        "vidhalluc_manifest_sha256": vh_manifest.get("manifest_sha256", ""),
        "e2_summaries": e2_summaries,
        "e3_summaries": e3_summaries,
        "e3_bridge": {
            "source_dataset_id": e2_dataset_id,
            "enabled": bridge_report["enabled"],
            "executed": e3_bridge_execution.get("executed", False),
            "status": bridge_report["status"],
        },
        "msrvtt_historical_bridge_status": msrvtt_bridge_report["status"],
        "statistics_file": str(out / "formal_b_statistics.json"),
        "verdict": VerdictStatus.PASS.value if all_pass else VerdictStatus.FAIL.value,
    }
    if not is_carebench:
        receipt["dream1k_manifest_sha256"] = e2_manifest.get("manifest_sha256", "")
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
