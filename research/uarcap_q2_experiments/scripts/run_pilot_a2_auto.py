#!/usr/bin/env python3
"""Pilot A2 automatic stage: real cross-family fact judge -> blind human annotation package.

Runs inside the existing BN vLLM environment (one model load per run, not per prompt).
Writes artifacts compatible with the repository's Pilot A2 re-entry path, so that after
human annotation the standard pipeline can ingest `--human-annotations` and reuse the
auto-eval + blind mapping.

Status ends at PILOT_A2_HUMAN_VALIDATION_PENDING. It never claims scientific calibration.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uarcap_q2.annotation.package_builder import build_blind_e2_annotation_package  # noqa: E402
from uarcap_q2.common.gates import verify_stage_authorization  # noqa: E402
from uarcap_q2.common.gpu_lock import GPULockConfig, acquire_cooperative_gpu_lock  # noqa: E402
from uarcap_q2.common.io_utils import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl, sha256_file  # noqa: E402
from uarcap_q2.common.schemas import PilotStage  # noqa: E402
from uarcap_q2.datasets.carebench import load_carebench_records  # noqa: E402
from uarcap_q2.evaluators.backends import JudgeBackendConfig, LocalVLLMJudgeBackend  # noqa: E402
from uarcap_q2.evaluators.batched_reference_fact_eval import evaluate_captions_batched  # noqa: E402


def _extract_frames_from_a1_tsv(a1_dir: Path, qids: set[str], frames_root: Path) -> dict[str, list[str]]:
    """Decode the exact 16 JPEG frames used as A1 model input from the ARM_A0 TSV bundle."""
    tsvs = sorted(a1_dir.glob("videollama_batch_ARM_A0/tsv/*.img.tsv"))
    if not tsvs:
        tsvs = sorted(a1_dir.glob("videollama_batch_*/tsv/*.img.tsv"))
    if not tsvs:
        raise FileNotFoundError(f"No A1 frame TSV under {a1_dir}")
    tsv = tsvs[0]
    frames_root.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[str]] = {}
    with tsv.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid = parts[0]
            if qid not in qids:
                continue
            n = int(parts[1])
            b64s = parts[2 : 2 + n]
            d = frames_root / qid
            d.mkdir(parents=True, exist_ok=True)
            paths: list[str] = []
            for i, b in enumerate(b64s):
                p = d / f"frame_{i:02d}.jpg"
                p.write_bytes(base64.b64decode(b))
                paths.append(str(p))
            if len(paths) != 16:
                raise ValueError(f"Expected 16 frames for {qid}, got {len(paths)}")
            out[qid] = paths
    missing = qids - set(out)
    if missing:
        raise FileNotFoundError(f"Missing A1 TSV frames for {len(missing)} queries: {sorted(missing)[:5]}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--paths-config", type=Path, required=True)
    ap.add_argument("--auth-receipt", type=Path, required=True)
    ap.add_argument("--prior-a1", type=Path, required=True)
    ap.add_argument("--judge-model-id", required=True)
    ap.add_argument("--judge-model-path", type=Path, required=True)
    ap.add_argument("--judge-family", required=True)
    ap.add_argument("--judge-quantization", default="")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--chunk-size", type=int, default=24)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--run-id", default="pilot_a2_carebench_eval_v1")
    ap.add_argument("--generator-family", default="vicuna_videollama")
    args = ap.parse_args(argv)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    paths_cfg = yaml.safe_load(args.paths_config.read_text(encoding="utf-8"))
    host = paths_cfg.get("host", {})
    hist = paths_cfg["historical_frozen_assets"]

    auth = verify_stage_authorization(
        stage=PilotStage.PILOT_A2,
        auth_receipt_path=args.auth_receipt,
        prior_receipts={PilotStage.PILOT_A1.value: args.prior_a1},
        allow_cpu_mock=False,
    )
    if args.judge_family.lower() == args.generator_family.lower():
        raise SystemExit("Cross-family judge policy violated")

    verification = read_json(out / "a2_input_verification.json")
    if not verification.get("all_verified"):
        raise SystemExit("Pilot A1 inputs did not pass SHA-256 verification; refusing A2")
    lock = read_json(out / "a2_core_event_lock.json")
    if not lock.get("created_before_any_judge_call"):
        raise SystemExit("Core-event lock missing or not pre-registered")
    core_by_qid = {e["query_id"]: e for e in lock["entries"]}
    if lock.get("metadata_sha256") != verification["metadata"]["actual_sha256"]:
        raise SystemExit("Core-event lock/metadata SHA mismatch")

    preds = read_jsonl(args.a1_dir / "predictions.jsonl")
    if len(preds) != 90:
        raise SystemExit(f"Expected 90 A1 predictions, found {len(preds)}")
    frozen = read_json(args.a1_dir / "FROZEN_SAMPLING_MANIFEST.json")
    frozen_ids = [e["query_id"] for e in frozen["selected"]]
    pred_qids = {str(p["query_id"]) for p in preds}
    if pred_qids != set(frozen_ids):
        raise SystemExit("A1 prediction query set differs from the frozen sampling manifest")

    data_root = Path(paths_cfg["q2_writable_workspace"]["data_root"])
    metadata_path = data_root / "carebench" / "json" / "metadata.json"
    video_root = data_root / "carebench" / "videos"
    records = load_carebench_records(metadata_path, video_root, require_video_exists=False)
    rec_by_qid = {r.query_id: r for r in records}

    items = []
    for p in preds:
        qid = str(p["query_id"])
        rec = rec_by_qid[qid]
        core = core_by_qid[qid]["core_event_candidate"]
        meta = rec.metadata or {}
        ref_parts = []
        if rec.reference_description:
            ref_parts.append(f"Caption: {rec.reference_description}")
        if meta.get("spatial_caption"):
            ref_parts.append(f"Spatial description: {meta['spatial_caption']}")
        if meta.get("temporal_caption"):
            ref_parts.append(f"Temporal description: {meta['temporal_caption']}")
        reference_text = "\n".join(ref_parts) if ref_parts else rec.reference_description
        items.append(
            {
                "query_id": qid,
                "arm": str(p["arm"]),
                "model_id": str(p["model_id"]),
                "caption": str(p.get("caption", "")),
                "reference_description": reference_text,
                "reference_fields": list(meta.get("human_reference_fields", [])),
                "core_events": [core] if core else [],
            }
        )

    frames_by_qid = _extract_frames_from_a1_tsv(args.a1_dir, set(frozen_ids), out / "blind_annotation_package" / "frames")

    backend = LocalVLLMJudgeBackend(
        JudgeBackendConfig(
            backend_type="vllm_offline",
            model_id=args.judge_model_id,
            model_family=args.judge_family,
            local_model_path=str(args.judge_model_path),
            temperature=0.0,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            quantization=args.judge_quantization,
            language_model_only=True,
        )
    )

    lock_cfg = GPULockConfig(
        lock_path=Path(host["shared_gpu_lock"]),
        run_id=args.run_id,
        gpu_uuid=host.get("gpu_uuid"),
        require_nvidia_smi_idle=True,
        sidecar_dir=out,
    )
    t0 = time.time()
    peak_sample = {"mib": 0}
    sample_stop = threading.Event()

    def _sample_gpu() -> None:
        while not sample_stop.wait(1.0):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=5,
                ).strip().splitlines()
                peak_sample["mib"] = max(peak_sample["mib"], int(out[0].strip()))
            except Exception:  # noqa: BLE001 - best-effort telemetry only
                pass

    sampler = threading.Thread(target=_sample_gpu, daemon=True)
    with acquire_cooperative_gpu_lock(lock_cfg):
        sampler.start()
        try:
            res = evaluate_captions_batched(backend, items, max_retries=args.max_retries, chunk_size=args.chunk_size)
        finally:
            sample_stop.set()
            sampler.join(timeout=2)
        peak_vram = int(peak_sample["mib"]) * 1024 * 1024
        try:
            gpu_name = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, timeout=5
            ).strip().splitlines()[0]
        except Exception:  # noqa: BLE001
            gpu_name = "unknown"
    elapsed = round(time.time() - t0, 1)

    rows = res["rows"]
    atomic_write_jsonl(out / "a2_ref_fact_eval.jsonl", rows)
    atomic_write_jsonl(out / "a2_judge_raw_calls.jsonl", res["raw_calls"])

    import torch
    import transformers
    import vllm

    judge_receipt = {
        "schema_version": "uarcap_q2.pilot_a2_judge_run.v1",
        "stage": "PILOT_A2",
        "run_id": args.run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_mode": auth["mode"],
        "approval_id": auth.get("approval_id"),
        "judge_model_id": args.judge_model_id,
        "judge_model_path": str(args.judge_model_path),
        "judge_family": args.judge_family,
        "judge_backend": "vllm_offline_in_process",
        "vllm_version": getattr(vllm, "__version__", "unknown"),
        "transformers_version": getattr(transformers, "__version__", "unknown"),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "quantization": args.judge_quantization or "auto-detected",
        "temperature": 0.0,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "chunk_size": args.chunk_size,
        "max_retries": args.max_retries,
        "n_items": len(items),
        "n_unique_captions_evaluated": res["unique_captions"],
        "n_duplicate_caption_reuses": res["n_duplicate_reuses"],
        "n_retries": res["n_retries"],
        "n_parse_failures": res["parse_failures"],
        "n_raw_judge_calls": len(res["raw_calls"]),
        "elapsed_sec": elapsed,
        "peak_vram_bytes": peak_vram,
        "gpu_name": gpu_name,
        "core_event_lock_sha256": lock.get("lock_sha256"),
        "a1_predictions_sha256": verification["a1_files"]["predictions.jsonl"]["actual_sha256"],
        "notes": [
            "Identical (query_id, caption) pairs share one evaluation input/result; per-arm claim_ids and arm mapping are preserved.",
            "UNMENTIONED is not counted as a factual error.",
            "CaReBench GPT-extracted events/objects are not used as Gold.",
        ],
    }
    atomic_write_json(out / "a2_judge_run_receipt.json", judge_receipt)

    by_qid: dict[str, dict] = {}
    for r in rows:
        qid = r["query_id"]
        v = by_qid.setdefault(
            qid,
            {
                "query_id": qid,
                "dataset_id": "CaReBench",
                "video_path": rec_by_qid[qid].video_path,
                "sampled_frame_paths": frames_by_qid[qid],
                "captions": [],
            },
        )
        v["captions"].append(
            {
                "arm": r["arm"],
                "model_id": r["model_id"],
                "caption": r["caption"],
                "claims": [{"claim_id": c["claim_id"], "claim": c["claim"]} for c in r.get("claims", [])],
            }
        )

    blind_pkg = build_blind_e2_annotation_package(
        [by_qid[q] for q in frozen_ids], annotator_ids=["A1", "A2"], out_dir=out / "blind_annotation_package"
    )
    pkg_sha = str(read_json(blind_pkg["blind_key"]).get("package_sha256", ""))
    atomic_write_json(
        out / "a2_bundle_manifest.json",
        {
            "dataset_id": "CaReBench",
            "a1_predictions_path": str(args.a1_dir / "predictions.jsonl"),
            "a1_predictions_sha256": judge_receipt["a1_predictions_sha256"],
            "auto_eval_path": str(out / "a2_ref_fact_eval.jsonl"),
            "auto_eval_sha256": sha256_file(out / "a2_ref_fact_eval.jsonl"),
            "package_sha256": pkg_sha,
            "judge_model_id": args.judge_model_id,
            "blind_annotation_files": {k: str(v) for k, v in blind_pkg.items()},
            "input_verification_path": str(out / "a2_input_verification.json"),
            "core_event_lock_path": str(out / "a2_core_event_lock.json"),
            "core_event_lock_sha256": lock.get("lock_sha256"),
            "judge_run_receipt_path": str(out / "a2_judge_run_receipt.json"),
        },
    )

    receipt = {
        "stage": "PILOT_A2",
        "dataset_id": "CaReBench",
        "status": "PILOT_A2_HUMAN_VALIDATION_PENDING",
        "execution_mode": auth["mode"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "judge_model_id": args.judge_model_id,
        "judge_family": args.judge_family,
        "n_captions_evaluated": len(rows),
        "n_unique_captions_evaluated": res["unique_captions"],
        "n_duplicate_caption_reuses": res["n_duplicate_reuses"],
        "parse_failures": res["parse_failures"],
        "reused_existing_auto_eval_and_bundle": False,
        "package_sha256": pkg_sha,
        "human_calibration_completed": False,
        "blind_annotation_files": {k: str(v) for k, v in blind_pkg.items()},
        "next_action": (
            "Launch the blind UI for annotator A1 and A2 with separate --save-dir directories, "
            "export both annotations, adjudicate with scripts/adjudicate_annotations.py --task e2, "
            "then re-run run_pilot_a2.sh with --human-annotations <adjudicated_e2.json>."
        ),
    }
    atomic_write_json(out / "calibration_receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "n_captions_evaluated", "n_unique_captions_evaluated", "parse_failures", "package_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
