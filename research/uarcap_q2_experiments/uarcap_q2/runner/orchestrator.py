"""Unified batch-capable experiment orchestrator (P0-3, P0-4, P0-5, P0-6 remediated).

Key guarantees:
  1. Uses generator.generate_batch(...) per arm so 7B Video-LLaMA loads ONCE per arm AND passes real decoded
     PIL `frame_images` to Modern VLM!
  2. Trace-before-commit: ONLY outputs passing verify_actual_input_trace(...) are committed; trace failures
     NEVER enter predictions.jsonl, input_traces.jsonl, or completed_keys!
  3. Paired commit + config-bound resume: every prediction & trace carries `run_config_fingerprint` and
     `trace_sha256`. Resume re-validates every existing line (catching truncation, missing trace pairs,
     duplicate/extra keys, or config/model drift) and computes final metrics over the FULL run!
  4. Strict StageRunStatus: stamps MOCK_PASS vs REAL_RUNTIME_PASS vs FAILED (MOCK_PASS can NEVER unlock real stages!).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from ..common.gates import enforce_pilot_formal_exclusion, verify_stage_authorization
from ..common.gpu_lock import GPULockConfig, acquire_cooperative_gpu_lock
from ..common.io_utils import (
    append_jsonl_durable,
    atomic_write_json,
    primitive_hash,
    read_jsonl,
)
from ..common.schemas import (
    EligibilityStatus,
    ExpArm,
    FailureCategory,
    PilotStage,
    QueryRecord,
    StageRunStatus,
    VerdictStatus,
)
from ..datasets.video_decoder import decode_and_sample_video
from ..generators.base import (
    BatchItem,
    GenerationOutput,
    GeneratorAdapter,
    GeneratorOOMError,
    TraceVerificationError,
    verify_actual_input_trace,
)
from ..interventions.vidhalluc_e3 import E3InterventionCandidate
from ..retrieval.arm_builder import (
    ArmInputPacket,
    build_e3_intervention_arms,
    build_natural_retrieval_arm,
    verify_e3_visual_lock,
    verify_natural_prefix_consistency,
)
from ..retrieval.reranker import RetrieverAdapter, rerank_d_consensus
from .report_generator import write_pilot_a1_reports


class ResumeIntegrityError(ValueError):
    """Raised when existing run outputs are truncated, unpaired, or from a different config/model version."""


@dataclass
class RunSpec:
    run_id: str
    stage: PilotStage
    dataset_id: str
    arms: list[ExpArm]
    out_dir: Path
    auth_receipt_path: Path | None = None
    prior_receipts: dict[str, Path | str] | None = None
    formal_config_path: Path | None = None
    asset_manifest_path: Path | None = None
    pilot_exclusion_path: Path | None = None
    pilot_excluded_query_ids: set[str] | None = None
    pilot_excluded_source_ids: set[str] | None = None
    gpu_lock_path: Path | None = None
    gpu_uuid: str | None = None
    require_nvidia_smi_idle: bool = False
    allow_cpu_mock: bool = False
    limit: int | None = None
    max_retries: int = 1
    num_frames: int = 16


def _compute_run_config_fingerprint(
    spec: RunSpec,
    generator: GeneratorAdapter,
    retriever_id: str,
) -> str:
    return primitive_hash(
        {
            "stage": spec.stage.value,
            "dataset_id": spec.dataset_id,
            "model_id": generator.model_id,
            "checkpoint_sha256": generator.expected_checkpoint_sha256,
            "injection_interface": generator.injection_interface,
            "retriever_id": retriever_id,
            "num_frames": spec.num_frames,
        }
    )


def _load_and_verify_existing_records(
    preds_path: Path,
    traces_path: Path,
    expected_fingerprint: str,
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], list[tuple[str, str, str]]]:
    """Re-validate existing predictions & paired traces on resume. Rejects truncation, unpaired traces, or config drift."""
    if not preds_path.is_file() and not traces_path.is_file():
        return {}, []
    if preds_path.is_file() != traces_path.is_file():
        raise ResumeIntegrityError("Unpaired resume state: predictions.jsonl and input_traces.jsonl must both exist.")

    try:
        preds = read_jsonl(preds_path)
        traces = read_jsonl(traces_path)
    except ValueError as exc:
        raise ResumeIntegrityError(f"Truncated or malformed JSONL during resume: {exc}") from exc

    if len(preds) != len(traces):
        raise ResumeIntegrityError(
            f"Paired record count mismatch on resume: predictions={len(preds)} != traces={len(traces)}"
        )

    valid_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    duplicates: list[tuple[str, str, str]] = []
    for idx, (p, t) in enumerate(zip(preds, traces)):
        if p.get("run_config_fingerprint") != expected_fingerprint:
            raise ResumeIntegrityError(
                f"Config/model version drift at line {idx + 1}: existing fingerprint "
                f"{p.get('run_config_fingerprint')!r} != current {expected_fingerprint!r}. Refusing to reuse stale outputs!"
            )
        if t.get("run_config_fingerprint") != expected_fingerprint:
            raise ResumeIntegrityError(f"Trace config fingerprint drift at line {idx + 1}.")
        t_clean = {k: v for k, v in t.items() if k not in ("run_config_fingerprint", "trace_sha256")}
        expected_tsha = primitive_hash(t_clean)
        if p.get("trace_sha256") != expected_tsha or t.get("trace_sha256") != expected_tsha:
            raise ResumeIntegrityError(f"Prediction/Trace pairing SHA-256 mismatch at line {idx + 1}.")
        key = (str(p["query_id"]), str(p["arm"]), str(p["model_id"]))
        if key in valid_by_key:
            duplicates.append(key)
        valid_by_key[key] = p
    return valid_by_key, duplicates


def _commit_verified_pair(
    preds_path: Path,
    traces_path: Path,
    out_obj: GenerationOutput,
    fingerprint: str,
) -> dict[str, Any]:
    """Commit prediction + trace as a hash-linked pair ONLY AFTER trace verification has succeeded."""
    trace_dict = out_obj.input_trace.to_dict()
    trace_sha = primitive_hash(trace_dict)
    pred_row = {
        **out_obj.to_dict(),
        "run_config_fingerprint": fingerprint,
        "trace_sha256": trace_sha,
    }
    trace_row = {
        **trace_dict,
        "run_config_fingerprint": fingerprint,
        "trace_sha256": trace_sha,
    }
    # Write trace first, then prediction; resume loader strictly enforces 1:1 count and hash match
    append_jsonl_durable(traces_path, trace_row)
    append_jsonl_durable(preds_path, pred_row)
    return pred_row


def run_natural_retrieval_stage(
    spec: RunSpec,
    queries: list[QueryRecord],
    retriever: RetrieverAdapter,
    generator: GeneratorAdapter,
    skip_video_decode: bool = False,
) -> dict[str, Any]:
    """Execute a natural-retrieval stage (Pilot A1 or Formal B E2) with batch execution, trace-before-commit, & strict resume."""
    t_start = perf_counter()

    # 1. Hard authorization check BEFORE touching data, GPU lock, or models!
    auth = verify_stage_authorization(
        stage=spec.stage,
        auth_receipt_path=spec.auth_receipt_path,
        prior_receipts=spec.prior_receipts,
        allow_cpu_mock=spec.allow_cpu_mock,
        formal_config_path=spec.formal_config_path,
        asset_manifest_path=spec.asset_manifest_path,
        pilot_exclusion_path=spec.pilot_exclusion_path,
    )

    selected_queries = list(queries[: spec.limit]) if spec.limit is not None else list(queries)
    if spec.stage == PilotStage.FORMAL_B:
        enforce_pilot_formal_exclusion(
            formal_query_ids=[q.query_id for q in selected_queries],
            pilot_excluded_ids=spec.pilot_excluded_query_ids or set(),
            formal_source_ids=[q.source_id for q in selected_queries],
            pilot_excluded_sources=spec.pilot_excluded_source_ids,
        )

    out_dir = Path(spec.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "predictions.jsonl"
    traces_path = out_dir / "input_traces.jsonl"
    fails_path = out_dir / "failures.jsonl"
    retrieval_path = out_dir / "retrieval_candidates.jsonl"

    fingerprint = _compute_run_config_fingerprint(spec, generator, retriever.retriever_id)
    existing_by_key, duplicate_keys = _load_and_verify_existing_records(preds_path, traces_path, fingerprint)

    atomic_write_json(
        out_dir / "query_manifest.json",
        {
            "run_id": spec.run_id,
            "stage": spec.stage.value,
            "dataset_id": spec.dataset_id,
            "model_id": generator.model_id,
            "run_config_fingerprint": fingerprint,
            "arms": [a.value for a in spec.arms],
            "count": len(selected_queries),
            "query_ids": [q.query_id for q in selected_queries],
        },
    )

    expected_keys = {(q.query_id, arm.value, generator.model_id) for q in selected_queries for arm in spec.arms}
    resumed_skipped = 0
    newly_completed = 0
    failed_count = 0
    all_videos_ok = True
    all_traces_ok = True
    a0_verified = True
    prefix_verified = True
    oom_halted = False

    def _run_loop() -> None:
        nonlocal resumed_skipped, newly_completed, failed_count
        nonlocal all_videos_ok, all_traces_ok, a0_verified, prefix_verified, oom_halted

        # Phase 1: Decode frames ONCE per query + run retrieval + D rerank + verify arm packets
        pending_by_arm: dict[str, list[BatchItem]] = {a.value: [] for a in spec.arms}
        for q in selected_queries:
            q_arms_needed = [a for a in spec.arms if (q.query_id, a.value, generator.model_id) not in existing_by_key]
            if not q_arms_needed:
                resumed_skipped += len(spec.arms)
                continue
            resumed_skipped += len(spec.arms) - len(q_arms_needed)

            if skip_video_decode:
                frame_imgs: list[Any] = []
                vis_sha = f"mock_vis_sha_{q.query_id}"
                frame_idx = list(range(spec.num_frames))
            else:
                frame_imgs, rep = decode_and_sample_video(q.video_path, num_frames=spec.num_frames)
                if not rep.ok:
                    all_videos_ok = False
                    for arm in q_arms_needed:
                        failed_count += 1
                        append_jsonl_durable(
                            fails_path,
                            {
                                "query_id": q.query_id,
                                "arm": arm.value,
                                "model_id": generator.model_id,
                                "error_type": f"VIDEO_DECODE_{rep.status}",
                                "error": rep.error_message,
                            },
                        )
                    continue
                vis_sha = rep.combined_visual_sha256
                frame_idx = rep.sampled_indices

            raw_top20 = retriever.retrieve_top20(q)
            reranked = rerank_d_consensus(raw_top20)
            append_jsonl_durable(
                retrieval_path,
                {
                    "query_id": q.query_id,
                    "retriever_id": retriever.retriever_id,
                    "original_top20": [c.to_dict() for c in raw_top20],
                    "reranked_d": [c.to_dict() for c in reranked],
                },
            )

            packets: dict[str, ArmInputPacket] = {
                arm.value: build_natural_retrieval_arm(q.query_id, arm, reranked) for arm in spec.arms
            }
            try:
                verify_natural_prefix_consistency(packets)
            except ValueError as exc:
                prefix_verified = False
                a0_verified = False
                raise RuntimeError(f"Arm input invariant failed for {q.query_id}: {exc}") from exc

            for arm in q_arms_needed:
                pending_by_arm[arm.value].append(
                    BatchItem(
                        query=q,
                        packet=packets[arm.value],
                        visual_input_sha256=vis_sha,
                        frame_indices=frame_idx,
                        frame_images=frame_imgs,
                    )
                )

        # Phase 2: Execute each arm via batch adapter (loads 7B Video-LLaMA ONCE per arm!)
        for arm in spec.arms:
            batch_items = pending_by_arm[arm.value]
            if not batch_items:
                continue

            # Distinguish batch-native subclass vs per-item generator cleanly
            is_custom_batch = type(generator).generate_batch is not GeneratorAdapter.generate_batch
            if is_custom_batch:
                try:
                    batch_outs = generator.generate_batch(batch_items, spec.stage, out_dir, run_id=spec.run_id)
                except GeneratorOOMError as exc:
                    oom_halted = True
                    failed_count += len(batch_items)
                    append_jsonl_durable(
                        fails_path,
                        {"arm": arm.value, "model_id": generator.model_id, "error_type": "CUDA_OOM_SAFE_STOP", "error": str(exc)},
                    )
                    return
                except Exception as exc:
                    if isinstance(exc, TraceVerificationError):
                        all_traces_ok = False
                        if arm == ExpArm.ARM_A0 or "a0" in str(exc).lower():
                            a0_verified = False
                    failed_count += len(batch_items)
                    append_jsonl_durable(
                        fails_path,
                        {"arm": arm.value, "model_id": generator.model_id, "error_type": type(exc).__name__, "error": str(exc)},
                    )
                    continue

                item_by_qid = {bi.query.query_id: bi for bi in batch_items}
                for cand_out in batch_outs:
                    bi = item_by_qid.get(cand_out.query_id)
                    verified_out: GenerationOutput | None = None
                    try:
                        if bi is None:
                            raise TraceVerificationError(f"Unexpected query_id in batch output: {cand_out.query_id}")
                        verify_actual_input_trace(
                            cand_out.input_trace, bi.packet, generator.expected_checkpoint_sha256, bi.visual_input_sha256
                        )
                        verified_out = cand_out
                    except Exception as exc:
                        all_traces_ok = False
                        if arm == ExpArm.ARM_A0 or "a0" in str(exc).lower():
                            a0_verified = False
                        failed_count += 1
                        append_jsonl_durable(
                            fails_path,
                            {
                                "query_id": cand_out.query_id,
                                "arm": arm.value,
                                "model_id": generator.model_id,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            },
                        )
                    if verified_out is not None:
                        row = _commit_verified_pair(preds_path, traces_path, verified_out, fingerprint)
                        existing_by_key[(verified_out.query_id, arm.value, generator.model_id)] = row
                        newly_completed += 1
            else:
                for bi in batch_items:
                    verified_out = None
                    last_err: Exception | None = None
                    attempt = 0
                    while attempt <= spec.max_retries:
                        attempt += 1
                        try:
                            cand_out = generator.generate_one(
                                bi.query,
                                bi.packet,
                                spec.stage,
                                bi.visual_input_sha256,
                                bi.frame_indices,
                                frame_images=bi.frame_images,
                            )
                            # CRITICAL P0-6 FIX: verify trace BEFORE assigning verified_out!
                            verify_actual_input_trace(
                                cand_out.input_trace, bi.packet, generator.expected_checkpoint_sha256, bi.visual_input_sha256
                            )
                            verified_out = cand_out
                            break
                        except GeneratorOOMError as exc:
                            oom_halted = True
                            failed_count += 1
                            append_jsonl_durable(
                                fails_path,
                                {
                                    "query_id": bi.query.query_id,
                                    "arm": arm.value,
                                    "model_id": generator.model_id,
                                    "error_type": "CUDA_OOM_SAFE_STOP",
                                    "error": str(exc),
                                },
                            )
                            return
                        except Exception as exc:
                            verified_out = None
                            last_err = exc
                            if isinstance(exc, TraceVerificationError):
                                all_traces_ok = False
                                if arm == ExpArm.ARM_A0 or "a0" in str(exc).lower():
                                    a0_verified = False
                                break

                    if verified_out is None:
                        failed_count += 1
                        append_jsonl_durable(
                            fails_path,
                            {
                                "query_id": bi.query.query_id,
                                "arm": arm.value,
                                "model_id": generator.model_id,
                                "error_type": type(last_err).__name__ if last_err else "UNKNOWN",
                                "error": str(last_err) if last_err else "unknown error",
                            },
                        )
                        continue

                    row = _commit_verified_pair(preds_path, traces_path, verified_out, fingerprint)
                    existing_by_key[(verified_out.query_id, arm.value, generator.model_id)] = row
                    newly_completed += 1

    if spec.gpu_lock_path is not None:
        lcfg = GPULockConfig(
            lock_path=spec.gpu_lock_path,
            run_id=spec.run_id,
            gpu_uuid=spec.gpu_uuid,
            require_nvidia_smi_idle=spec.require_nvidia_smi_idle,
            sidecar_dir=out_dir,
        )
        with acquire_cooperative_gpu_lock(lcfg):
            _run_loop()
    else:
        _run_loop()

    wall_sec = round(perf_counter() - t_start, 4)

    # Full-run completeness & metrics audit over ALL verified records (NOT just newly_completed!)
    completed_keys = set(existing_by_key.keys())
    missing_keys = sorted(expected_keys - completed_keys)
    extra_keys = sorted(completed_keys - expected_keys)
    full_records = [existing_by_key[k] for k in sorted(expected_keys & completed_keys)]
    total_empty = sum(1 for r in full_records if (not r.get("ok")) or (not str(r.get("caption", "")).strip()))
    empty_rate = (total_empty / len(full_records)) if full_records else 0.0
    peak_vram = max((int(r.get("peak_vram_bytes", 0)) for r in full_records), default=0)

    gate_reasons: list[str] = []
    failure_cat: str | None = None
    if oom_halted:
        failure_cat = FailureCategory.MODEL_INCOMPATIBLE.value
        gate_reasons.append("Halted safely due to CUDA OOM.")
    if not all_videos_ok:
        failure_cat = failure_cat or FailureCategory.DATA_UNAVAILABLE.value
        gate_reasons.append("One or more videos failed strict 16-frame decode / repeated-first-frame validation.")
    if not all_traces_ok or not a0_verified or not prefix_verified:
        failure_cat = failure_cat or FailureCategory.MODEL_INCOMPATIBLE.value
        gate_reasons.append("Input trace / checkpoint / A0 / prefix verification failed.")
    if missing_keys or extra_keys or duplicate_keys or failed_count > 0:
        failure_cat = failure_cat or FailureCategory.MODEL_INCOMPATIBLE.value
        gate_reasons.append(
            f"Completeness mismatch: missing={len(missing_keys)}, extra={len(extra_keys)}, "
            f"duplicates={len(duplicate_keys)}, failed={failed_count}."
        )
    if empty_rate >= 0.03:
        failure_cat = failure_cat or FailureCategory.MODEL_INCOMPATIBLE.value
        gate_reasons.append(f"Full-run empty output rate {empty_rate:.3f} >= 0.03 threshold.")

    is_mock = auth["mode"] == "cpu_mock"
    if gate_reasons:
        stage_status = StageRunStatus.FAILED.value
        verdict = VerdictStatus.FAIL.value
    elif is_mock:
        stage_status = StageRunStatus.MOCK_PASS.value
        verdict = VerdictStatus.PASS.value
    else:
        stage_status = StageRunStatus.REAL_RUNTIME_PASS.value
        verdict = VerdictStatus.PASS.value

    summary: dict[str, Any] = {
        "run_id": spec.run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": spec.stage.value,
        "status": stage_status,  # MOCK_PASS | REAL_RUNTIME_PASS | FAILED (strictly isolated!)
        "dataset_id": spec.dataset_id,
        "model_id": generator.model_id,
        "checkpoint_sha256": generator.expected_checkpoint_sha256,
        "retriever_id": retriever.retriever_id,
        "run_config_fingerprint": fingerprint,
        "arms": [a.value for a in spec.arms],
        "execution_mode": auth["mode"],
        "n_queries": len(selected_queries),
        "expected_generations": len(expected_keys),
        "completed_generations": len(expected_keys & completed_keys),
        "newly_completed": newly_completed,
        "resumed_skipped": resumed_skipped,
        "failed_generations": failed_count,
        "missing_keys_count": len(missing_keys),
        "extra_keys_count": len(extra_keys),
        "duplicate_keys_count": len(duplicate_keys),
        "empty_output_rate": round(empty_rate, 6),
        "all_videos_decoded_ok": all_videos_ok,
        "a0_zero_evidence_verified": a0_verified,
        "prefix_consistency_verified": prefix_verified,
        "all_traces_verified": all_traces_ok,
        "oom_halted": oom_halted,
        "wall_seconds": wall_sec,
        "peak_vram_bytes": peak_vram,
        "verdict": verdict,
        "go_no_go_verdict": verdict,
        "failure_category": failure_cat,
        "gate_reasons": gate_reasons,
    }
    atomic_write_json(out_dir / "run_receipt.json", summary)
    if spec.stage == PilotStage.PILOT_A1:
        write_pilot_a1_reports(summary, out_dir)
    return summary


def run_e3_controlled_stage(
    spec: RunSpec,
    queries_by_id: dict[str, QueryRecord],
    candidates: list[E3InterventionCandidate],
    generator: GeneratorAdapter,
    base_visual_mode: str = "BASE_A0",
    skip_video_decode: bool = False,
) -> dict[str, Any]:
    """Execute 4 visual-locked E3 arms (Base/Sup/Con/Neu) strictly on human-APPROVED candidates only."""
    t_start = perf_counter()
    auth = verify_stage_authorization(
        stage=spec.stage,
        auth_receipt_path=spec.auth_receipt_path,
        prior_receipts=spec.prior_receipts,
        allow_cpu_mock=spec.allow_cpu_mock,
        formal_config_path=spec.formal_config_path,
        asset_manifest_path=spec.asset_manifest_path,
        pilot_exclusion_path=spec.pilot_exclusion_path,
    )

    unapproved = [c.query_id for c in candidates if c.eligibility_status != EligibilityStatus.APPROVED.value]
    if unapproved:
        raise ValueError(
            f"Refusing E3 run: {len(unapproved)} candidate(s) are not human-APPROVED (e.g., {unapproved[:3]})."
        )

    selected = list(candidates[: spec.limit]) if spec.limit is not None else list(candidates)
    e3_arms = [ExpArm.E3_BASE, ExpArm.E3_SUP, ExpArm.E3_CON, ExpArm.E3_NEU]
    out_dir = Path(spec.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "e3_predictions.jsonl"
    traces_path = out_dir / "e3_input_traces.jsonl"
    fails_path = out_dir / "e3_failures.jsonl"

    fingerprint = _compute_run_config_fingerprint(spec, generator, f"e3_{base_visual_mode}")
    existing_by_key, dup_keys = _load_and_verify_existing_records(preds_path, traces_path, fingerprint)
    expected_keys = {(c.query_id, a.value, generator.model_id) for c in selected for a in e3_arms}

    newly_completed = 0
    failed_count = 0
    all_visual_locked = True

    def _run_e3() -> None:
        nonlocal newly_completed, failed_count, all_visual_locked
        pending_by_arm: dict[str, list[BatchItem]] = {a.value: [] for a in e3_arms}
        for cand in selected:
            q = queries_by_id[cand.query_id]
            if skip_video_decode:
                imgs: list[Any] = []
                vis_sha = f"mock_e3_vis_{q.query_id}"
                fidx = list(range(spec.num_frames))
            else:
                imgs, rep = decode_and_sample_video(q.video_path, num_frames=spec.num_frames)
                if not rep.ok:
                    failed_count += 4
                    continue
                vis_sha = rep.combined_visual_sha256
                fidx = rep.sampled_indices
            pkts = build_e3_intervention_arms(
                q.query_id, cand.sup_text, cand.con_text, cand.neu_text, base_visual_mode=base_visual_mode
            )
            verify_e3_visual_lock(pkts)
            for arm in e3_arms:
                if (q.query_id, arm.value, generator.model_id) in existing_by_key:
                    continue
                pending_by_arm[arm.value].append(BatchItem(q, pkts[arm.value], vis_sha, fidx, imgs))

        vis_sha_by_qid: dict[str, set[str]] = {}
        for arm in e3_arms:
            items = pending_by_arm[arm.value]
            if not items:
                continue
            outs = generator.generate_batch(items, spec.stage, out_dir, run_id=f"{spec.run_id}_e3")
            item_map = {bi.query.query_id: bi for bi in items}
            for o in outs:
                bi = item_map[o.query_id]
                try:
                    verify_actual_input_trace(o.input_trace, bi.packet, generator.expected_checkpoint_sha256, bi.visual_input_sha256)
                    vis_sha_by_qid.setdefault(o.query_id, set()).add(o.input_trace.visual_input_sha256)
                    row = _commit_verified_pair(preds_path, traces_path, o, fingerprint)
                    existing_by_key[(o.query_id, arm.value, generator.model_id)] = row
                    newly_completed += 1
                except Exception as exc:
                    failed_count += 1
                    append_jsonl_durable(
                        fails_path,
                        {"query_id": o.query_id, "arm": arm.value, "error": str(exc)},
                    )
        if any(len(s) > 1 for s in vis_sha_by_qid.values()):
            all_visual_locked = False

    if spec.gpu_lock_path is not None:
        with acquire_cooperative_gpu_lock(
            GPULockConfig(spec.gpu_lock_path, spec.run_id, spec.gpu_uuid, spec.require_nvidia_smi_idle, out_dir)
        ):
            _run_e3()
    else:
        _run_e3()

    completed = set(existing_by_key.keys())
    missing = sorted(expected_keys - completed)
    ok = (not missing) and (failed_count == 0) and (not dup_keys) and all_visual_locked
    status = (
        StageRunStatus.FAILED.value
        if not ok
        else (StageRunStatus.MOCK_PASS.value if auth["mode"] == "cpu_mock" else StageRunStatus.REAL_RUNTIME_PASS.value)
    )
    summary = {
        "run_id": spec.run_id,
        "stage": spec.stage.value,
        "status": status,
        "execution_mode": auth["mode"],
        "model_id": generator.model_id,
        "n_candidates": len(selected),
        "expected_generations": len(expected_keys),
        "completed_generations": len(expected_keys & completed),
        "newly_completed": newly_completed,
        "failed_generations": failed_count,
        "all_visual_locked": all_visual_locked,
        "wall_seconds": round(perf_counter() - t_start, 4),
        "verdict": VerdictStatus.PASS.value if ok else VerdictStatus.FAIL.value,
    }
    atomic_write_json(out_dir / "e3_run_receipt.json", summary)
    return summary
