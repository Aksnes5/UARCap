"""Unified multi-stage, multi-arm experiment orchestrator with resume, OOM stop, lock, and trace audit."""

from __future__ import annotations

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
    read_jsonl,
)
from ..common.schemas import ExpArm, FailureCategory, PilotStage, QueryRecord, VerdictStatus
from ..datasets.video_decoder import decode_and_sample_video
from ..generators.base import (
    GenerationOutput,
    GeneratorAdapter,
    GeneratorOOMError,
    TraceVerificationError,
    verify_actual_input_trace,
)
from ..retrieval.arm_builder import (
    ArmInputPacket,
    build_natural_retrieval_arm,
    verify_natural_prefix_consistency,
)
from ..retrieval.reranker import RetrieverAdapter, rerank_d_consensus
from .report_generator import write_pilot_a1_reports


@dataclass
class RunSpec:
    run_id: str
    stage: PilotStage
    dataset_id: str
    arms: list[ExpArm]
    out_dir: Path
    auth_receipt_path: Path | None = None
    prior_receipts: dict[str, Path | str] | None = None
    pilot_excluded_query_ids: set[str] | None = None
    pilot_excluded_source_ids: set[str] | None = None
    gpu_lock_path: Path | None = None
    gpu_uuid: str | None = None
    require_nvidia_smi_idle: bool = False
    allow_cpu_mock: bool = False
    limit: int | None = None
    max_retries: int = 1
    num_frames: int = 16


def _load_completed_keys(predictions_jsonl: Path) -> set[tuple[str, str, str]]:
    if not predictions_jsonl.is_file():
        return set()
    done: set[tuple[str, str, str]] = set()
    for row in read_jsonl(predictions_jsonl):
        if row.get("ok") is True:
            done.add((str(row["query_id"]), str(row["arm"]), str(row["model_id"])))
    return done


def run_natural_retrieval_stage(
    spec: RunSpec,
    queries: list[QueryRecord],
    retriever: RetrieverAdapter,
    generator: GeneratorAdapter,
    skip_video_decode: bool = False,
) -> dict[str, Any]:
    """Execute a natural-retrieval stage (e.g., Pilot A1 or Formal B E2) with full audit & resume."""
    t_start = perf_counter()

    # 1. Hard authorization check BEFORE touching data, GPU lock, or models!
    auth = verify_stage_authorization(
        stage=spec.stage,
        auth_receipt_path=spec.auth_receipt_path,
        prior_receipts=spec.prior_receipts,
        allow_cpu_mock=spec.allow_cpu_mock,
    )

    # 2. Hard Pilot-to-Formal exclusion check
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

    atomic_write_json(
        out_dir / "query_manifest.json",
        {
            "run_id": spec.run_id,
            "stage": spec.stage.value,
            "dataset_id": spec.dataset_id,
            "model_id": generator.model_id,
            "arms": [a.value for a in spec.arms],
            "count": len(selected_queries),
            "query_ids": [q.query_id for q in selected_queries],
        },
    )

    completed_keys = _load_completed_keys(preds_path)
    resumed_skipped = 0
    newly_completed = 0
    failed_count = 0
    empty_count = 0
    peak_vram = 0
    all_videos_ok = True
    all_traces_ok = True
    a0_verified = True
    prefix_verified = True
    oom_halted = False

    def _run_loop() -> None:
        nonlocal resumed_skipped, newly_completed, failed_count, empty_count, peak_vram
        nonlocal all_videos_ok, all_traces_ok, a0_verified, prefix_verified, oom_halted

        for q in selected_queries:
            # Frame decode & validation
            if skip_video_decode:
                vis_sha = f"mock_vis_sha_{q.query_id}"
                frame_idx = list(range(spec.num_frames))
            else:
                _imgs, rep = decode_and_sample_video(q.video_path, num_frames=spec.num_frames)
                if not rep.ok:
                    all_videos_ok = False
                    for arm in spec.arms:
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

            # Retrieval + D consensus reranking
            raw_top20 = retriever.retrieve_top20(q)
            reranked = rerank_d_consensus(raw_top20)
            append_jsonl_durable(
                retrieval_path,
                {
                    "query_id": q.query_id,
                    "original_top20": [c.to_dict() for c in raw_top20],
                    "reranked_d": [c.to_dict() for c in reranked],
                },
            )

            # Build and verify all arm packets together
            packets: dict[str, ArmInputPacket] = {}
            for arm in spec.arms:
                packets[arm.value] = build_natural_retrieval_arm(q.query_id, arm, reranked)
            try:
                verify_natural_prefix_consistency(packets)
            except ValueError as exc:
                prefix_verified = False
                a0_verified = False
                raise RuntimeError(f"Arm input invariant failed for {q.query_id}: {exc}") from exc

            for arm in spec.arms:
                key = (q.query_id, arm.value, generator.model_id)
                if key in completed_keys:
                    resumed_skipped += 1
                    continue
                pkt = packets[arm.value]
                attempt = 0
                out_obj: GenerationOutput | None = None
                last_err: Exception | None = None
                while attempt <= spec.max_retries:
                    attempt += 1
                    try:
                        out_obj = generator.generate_one(q, pkt, spec.stage, vis_sha, frame_idx)
                        verify_actual_input_trace(
                            out_obj.input_trace, pkt, generator.expected_checkpoint_sha256, vis_sha
                        )
                        break
                    except GeneratorOOMError as exc:
                        oom_halted = True
                        failed_count += 1
                        append_jsonl_durable(
                            fails_path,
                            {
                                "query_id": q.query_id,
                                "arm": arm.value,
                                "model_id": generator.model_id,
                                "error_type": "CUDA_OOM_SAFE_STOP",
                                "error": str(exc),
                            },
                        )
                        return  # Immediate safe halt on OOM!
                    except Exception as exc:
                        last_err = exc
                        if isinstance(exc, TraceVerificationError):
                            all_traces_ok = False
                            if arm == ExpArm.ARM_A0 or "a0" in str(exc).lower():
                                a0_verified = False
                            break

                if out_obj is None:
                    failed_count += 1
                    append_jsonl_durable(
                        fails_path,
                        {
                            "query_id": q.query_id,
                            "arm": arm.value,
                            "model_id": generator.model_id,
                            "error_type": type(last_err).__name__ if last_err else "UNKNOWN",
                            "error": str(last_err) if last_err else "unknown error",
                        },
                    )
                    continue

                if not out_obj.ok or not out_obj.caption.strip():
                    empty_count += 1
                peak_vram = max(peak_vram, out_obj.peak_vram_bytes)
                append_jsonl_durable(preds_path, out_obj.to_dict())
                append_jsonl_durable(traces_path, out_obj.input_trace.to_dict())
                completed_keys.add(key)
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
    expected_total = len(selected_queries) * len(spec.arms)
    total_done = len(completed_keys)
    empty_rate = (empty_count / max(newly_completed, 1)) if newly_completed else 0.0

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
    if total_done < expected_total or failed_count > 0:
        failure_cat = failure_cat or FailureCategory.MODEL_INCOMPATIBLE.value
        gate_reasons.append(f"Incomplete generation outputs: completed={total_done}, expected={expected_total}, failed={failed_count}.")
    if empty_rate >= 0.03:
        failure_cat = failure_cat or FailureCategory.MODEL_INCOMPATIBLE.value
        gate_reasons.append(f"Empty output rate {empty_rate:.3f} >= 0.03 threshold.")

    verdict = VerdictStatus.PASS.value if not gate_reasons else VerdictStatus.FAIL.value
    summary: dict[str, Any] = {
        "run_id": spec.run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": spec.stage.value,
        "dataset_id": spec.dataset_id,
        "model_id": generator.model_id,
        "checkpoint_sha256": generator.expected_checkpoint_sha256,
        "arms": [a.value for a in spec.arms],
        "execution_mode": auth["mode"],
        "n_queries": len(selected_queries),
        "expected_generations": expected_total,
        "completed_generations": total_done,
        "newly_completed": newly_completed,
        "resumed_skipped": resumed_skipped,
        "failed_generations": failed_count,
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
