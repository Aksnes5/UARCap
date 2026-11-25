"""Real Unified Stage Runner CLI (Final Offline Handoff — Reuses Existing BN Models, Zero Stubs!).

Exit codes:
  0  = REAL_RUNTIME_PASS or MOCK_PASS (only when outputs & receipt actually written and all gates pass)
  2  = FAILED / INCONCLUSIVE / missing required real assets or runtime error
  3  = BLOCKED (authorization gate or prerequisite REAL_RUNTIME_PASS check failed)
  10 = HUMAN_VALIDATION_PENDING (stage paused cleanly awaiting required human annotation/verification)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment.bn_preflight_check import run_preflight
from uarcap_q2.common.gates import AuthorizationError, verify_stage_authorization
from uarcap_q2.common.io_utils import atomic_write_json, read_json
from uarcap_q2.common.schemas import ExpArm, PilotStage, StageRunStatus
from uarcap_q2.datasets.dream1k import load_dream1k_records, sample_dream1k_stratified_pilot, write_pilot_exclusion_manifest
from uarcap_q2.datasets.formal_sampling import FormalSamplingConfig
from uarcap_q2.datasets.vidhalluc import load_vidhalluc_ach_mcq
from uarcap_q2.evaluators.backends import (
    JudgeBackend,
    JudgeBackendConfig,
    LocalTransformersJudgeBackend,
    LocalVLLMJudgeBackend,
    MockJudgeBackend,
    OpenAICompatJudgeBackend,
    SubprocessTransformersJudgeBackend,
)
from uarcap_q2.generators.mock_generator import MockGeneratorAdapter
from uarcap_q2.generators.modern_vlm_adapter import ModernVLMConfig, Qwen25VLAdapter, SubprocessModernVLMAdapter
from uarcap_q2.generators.qwen3_vllm_adapter import (
    DEFAULT_BN_QWEN38_PATH,
    Qwen3VLLMConfig,
    Qwen3VLLMMultimodalAdapter,
)
from uarcap_q2.generators.videollama_adapter import FrozenVideoLLaMAAdapter, FrozenVideoLLaMAConfig
from uarcap_q2.interventions.e3_bridge import NaturalConflictObservation
from uarcap_q2.interventions.vidhalluc_e3 import E3InterventionCandidate, filter_runnable_e3_candidates
from uarcap_q2.retrieval.reranker import MockXCLIPRetriever, RetrieverAdapter
from uarcap_q2.retrieval.videoufo_retriever import (
    SubprocessVideoUFOXCLIPRetriever,
    VideoUFOIndexConfig,
    VideoUFOXCLIPRetriever,
)
from uarcap_q2.runner.orchestrator import RunSpec, run_natural_retrieval_stage
from uarcap_q2.runner.stage_pipelines import (
    run_e1_retrospective_pipeline,
    run_formal_b_pipeline,
    run_pilot_a2_pipeline,
    run_pilot_a3_pipeline,
)


def _exit_code_for_status(status: str) -> int:
    if status in (StageRunStatus.REAL_RUNTIME_PASS.value, StageRunStatus.MOCK_PASS.value):
        return 0
    if status == StageRunStatus.HUMAN_VALIDATION_PENDING.value:
        return 10
    if status == StageRunStatus.BLOCKED.value:
        return 3
    return 2


def _build_real_retriever(
    paths_cfg: dict[str, Any],
    use_subprocess: bool = False,
    gpu_lock: Path | None = None,
) -> RetrieverAdapter:
    hist = paths_cfg["historical_frozen_assets"]
    host = paths_cfg.get("host", {})
    vcfg = VideoUFOIndexConfig(
        videoufo_csv_path=hist["videoufo_csv_path"],
        clips_full_tsv_path=hist["clips_full_tsv_path"],
        brief_text_emb_npy_path=hist["brief_text_emb_npy_path"],
        xclip_repo_dir=hist["xclip_repo_dir"],
        xclip_ckpt_path=hist["xclip_ckpt_path"],
        video_feature_dir=hist["video_feature_dir"],
    )
    if use_subprocess:
        return SubprocessVideoUFOXCLIPRetriever(
            cfg=vcfg,
            python_executable=hist["frozen_videollama_python"],
            gpu_lock_path=gpu_lock,
            gpu_uuid=host.get("gpu_uuid"),
            require_nvidia_smi_idle=bool(gpu_lock),
            verify_feature_files_exist=True,
        )
    return VideoUFOXCLIPRetriever(vcfg, verify_feature_files_exist=True)


def _build_real_videollama(paths_cfg: dict[str, Any]) -> FrozenVideoLLaMAAdapter:
    hist = paths_cfg["historical_frozen_assets"]
    host = paths_cfg.get("host", {})
    return FrozenVideoLLaMAAdapter(
        FrozenVideoLLaMAConfig(
            python_executable=hist["frozen_videollama_python"],
            evcap_work_root=hist["v06_runtime_evcap_work"],
            checkpoint_path=hist["canonical_checkpoint"],
            llm_model_dir=hist["vicuna_7b_dir"],
            video_feature_dir=hist["video_feature_dir"],
            torch_home=hist["torch_home"],
            hf_hub_cache=hist["hf_hub_cache"],
            device=host.get("device", "cuda:0"),
            expected_checkpoint_sha256=hist["canonical_checkpoint_sha256"],
        )
    )


def _build_real_modern_vlm(args: argparse.Namespace, paths_cfg: dict[str, Any], gpu_lock: Path | None) -> Any:
    """Build primary modern video VLM adapter reusing existing BN Qwen3.8-27B-FP8 (or Qwen3.5-27B) via vLLM."""
    host = paths_cfg.get("host", {})
    existing = paths_cfg.get("existing_bn_modern_models", {})
    model_path = args.modern_vlm_path or existing.get("primary_modern_vlm_dir", DEFAULT_BN_QWEN38_PATH)
    model_id = args.modern_vlm_id or existing.get("primary_modern_vlm_id", "Qwen/Qwen3.8-27B-FP8")
    backend_mode = args.modern_vlm_backend

    # Optional legacy Qwen2.5-VL path ONLY if explicitly requested
    if backend_mode in ("legacy_qwen25vl", "legacy_qwen25vl_subprocess"):
        mcfg = ModernVLMConfig(
            model_id=model_id,
            local_model_path=model_path,
            pinned_revision=args.modern_vlm_revision,
            device=host.get("device", "cuda:0"),
        )
        if backend_mode == "legacy_qwen25vl_subprocess":
            return SubprocessModernVLMAdapter(
                mcfg,
                python_executable=args.vllm_python or None,
                conda_env=args.vllm_conda_env or paths_cfg.get("isolated_conda_envs", {}).get("modern_vlm_env", "uarcap_q2_modern_vlm"),
                gpu_lock_path=gpu_lock,
                gpu_uuid=host.get("gpu_uuid"),
            )
        return Qwen25VLAdapter(mcfg)

    vllm_py = args.vllm_python or existing.get("existing_vllm_python", "")
    vllm_conda = args.vllm_conda_env or existing.get("existing_vllm_conda_env", "")
    api_url = args.modern_vlm_api_url or (existing.get("local_vllm_api_base_url", "") if backend_mode == "vllm_openai_http" else "")
    qcfg = Qwen3VLLMConfig(
        model_id=model_id,
        local_model_path=model_path,
        backend_mode=backend_mode,
        api_base_url=api_url,
        vllm_python_executable=vllm_py,
        vllm_conda_env=vllm_conda,
        pinned_revision=args.modern_vlm_revision,
        enable_thinking=bool(args.enable_thinking),
        quantization=args.modern_vlm_quantization,
        device=host.get("device", "cuda:0"),
    )
    return Qwen3VLLMMultimodalAdapter(
        qcfg,
        gpu_lock_path=gpu_lock,
        gpu_uuid=host.get("gpu_uuid"),
        require_nvidia_smi_idle=bool(gpu_lock and backend_mode == "vllm_subprocess"),
    )


def _build_real_judge(args: argparse.Namespace, paths_cfg: dict[str, Any], gpu_lock: Path | None) -> JudgeBackend:
    """Build cross-family text factuality judge reusing existing BN models (default: Mistral Small 3.2 24B)."""
    existing = paths_cfg.get("existing_bn_modern_models", {})
    host = paths_cfg.get("host", {})
    model_id = args.judge_model_id or existing.get("primary_judge_model_id", "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    family = args.judge_family or existing.get("primary_judge_family", "mistral")
    model_path = args.judge_model_path or existing.get("primary_judge_dir", "")
    mode = args.judge_backend_mode

    if args.judge_api_url or mode == "openai_compat":
        return OpenAICompatJudgeBackend(
            JudgeBackendConfig(
                backend_type="openai_compat",
                model_id=model_id,
                model_family=family,
                api_base_url=args.judge_api_url or existing.get("local_vllm_api_base_url", ""),
            )
        )
    if not model_path or not Path(model_path).is_dir():
        raise FileNotFoundError(
            f"Real text judge requires an existing local BN model directory (--judge-model-path) or --judge-api-url; missing: {model_path!r}"
        )
    if mode == "vllm_offline":
        return LocalVLLMJudgeBackend(
            JudgeBackendConfig(
                backend_type="vllm_offline",
                model_id=model_id,
                model_family=family,
                local_model_path=model_path,
            )
        )
    if mode in ("vllm_subprocess", "transformers_subprocess"):
        btype = "vllm_offline" if mode == "vllm_subprocess" else "local_transformers"
        return SubprocessTransformersJudgeBackend(
            JudgeBackendConfig(
                backend_type=btype,
                model_id=model_id,
                model_family=family,
                local_model_path=model_path,
            ),
            python_executable=args.judge_python or existing.get("existing_vllm_python") or None,
            conda_env=args.judge_conda_env or existing.get("existing_vllm_conda_env") or paths_cfg.get("isolated_conda_envs", {}).get("text_evaluator_env", "uarcap_q2_evaluator"),
            gpu_lock_path=gpu_lock,
            gpu_uuid=host.get("gpu_uuid"),
            require_nvidia_smi_idle=bool(gpu_lock),
        )
    return LocalTransformersJudgeBackend(
        JudgeBackendConfig(
            backend_type="local_transformers",
            model_id=model_id,
            model_family=family,
            local_model_path=model_path,
        )
    )


def _load_videoufo_source_ids(paths_cfg: dict[str, Any]) -> set[str]:
    tsv_path = Path(paths_cfg["historical_frozen_assets"]["clips_full_tsv_path"])
    if not tsv_path.is_file():
        raise FileNotFoundError(f"VideoUFO clips_full.tsv missing for overlap audit: {tsv_path}")
    return {
        ln.split("\t")[0].strip().rsplit(".", 1)[0]
        for ln in tsv_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("clip_id")
    }


def dispatch_stage(
    args: argparse.Namespace,
    injected_retriever: RetrieverAdapter | None = None,
    injected_videollama: Any | None = None,
    injected_modern_vlm: Any | None = None,
    injected_ref_judge: JudgeBackend | None = None,
    injected_e3_judge: JudgeBackend | None = None,
) -> tuple[int, dict[str, Any]]:
    """Real stage dispatcher (supports dependency injection ONLY for CPU contract testing; zero fake-pass paths!)."""
    if args.stage == "E1":
        res = run_e1_retrospective_pipeline(
            args.e1_ref_a, args.e1_ref_b, args.e1_source_map, args.out_dir / "e1_report.json"
        )
        return 0, res

    stage = PilotStage(args.stage)
    paths_cfg = yaml.safe_load(args.paths_config.read_text(encoding="utf-8"))
    stage_cfg = yaml.safe_load(args.stage_config.read_text(encoding="utf-8"))

    priors: dict[str, Path | str] = {}
    if args.prior_a1:
        priors[PilotStage.PILOT_A1.value] = args.prior_a1
    if args.prior_a2:
        priors[PilotStage.PILOT_A2.value] = args.prior_a2
    if args.prior_a3:
        priors[PilotStage.PILOT_A3.value] = args.prior_a3

    if args.dry_run:
        plan = {
            "mode": "dry_run",
            "stage": stage.value,
            "run_id": args.run_id,
            "paths_config": str(args.paths_config),
            "stage_config": str(args.stage_config),
            "out_dir": str(args.out_dir),
            "prior_receipts": {k: str(v) for k, v in priors.items()},
            "modern_vlm_default": args.modern_vlm_id,
            "judge_default": args.judge_model_id,
            "gpu_touched": False,
        }
        return 0, plan

    # Enforce unified state-machine authorization BEFORE any real asset loading!
    try:
        verify_stage_authorization(
            stage=stage,
            auth_receipt_path=args.auth_receipt,
            prior_receipts=priors,
            allow_cpu_mock=args.cpu_mock,
            formal_config_path=args.stage_config if stage == PilotStage.FORMAL_B else None,
            asset_manifest_path=args.asset_manifest,
            pilot_exclusion_path=args.pilot_exclusion,
        )
    except AuthorizationError as exc:
        err = {"stage": stage.value, "status": StageRunStatus.BLOCKED.value, "error": str(exc)}
        return 3, err

    q2w = paths_cfg["q2_writable_workspace"]
    host = paths_cfg.get("host", {})
    dream_dir = Path(q2w["dream1k_dir"])
    ann_file = dream_dir / "json" / "metadata.json"
    vid_dir = dream_dir / "video"
    gpu_lock = Path(host["shared_gpu_lock"]) if (not args.cpu_mock and host.get("shared_gpu_lock")) else None

    # ---- PILOT A0 ----
    if stage == PilotStage.PILOT_A0:
        rep = run_preflight(args.paths_config, out_receipt=args.out_dir / "preflight_receipt.json")
        st = StageRunStatus.MOCK_PASS.value if args.cpu_mock else (
            StageRunStatus.REAL_RUNTIME_PASS.value if rep["bn_live_ready"] else StageRunStatus.FAILED.value
        )
        rep["status"] = st
        atomic_write_json(args.out_dir / "preflight_receipt.json", rep)
        return _exit_code_for_status(st), rep

    # ---- PILOT A1 ----
    if stage == PilotStage.PILOT_A1:
        if not ann_file.is_file():
            return 2, {"stage": stage.value, "status": StageRunStatus.FAILED.value, "error": f"DREAM-1K metadata missing: {ann_file}"}
        recs = load_dream1k_records(ann_file, vid_dir, require_video_exists=True)
        a1_cfg = stage_cfg.get("stage_a1", {})
        per_dom = int(a1_cfg.get("sampling", {}).get("per_domain", 6))
        seed = int(a1_cfg.get("sampling", {}).get("seed", 20261125))
        pilot_recs, _ = sample_dream1k_stratified_pilot(recs, per_domain=min(per_dom, len(recs) // 5), seed=seed)
        write_pilot_exclusion_manifest(pilot_recs, args.out_dir / "pilot_a_excluded_dream1k_vids.json", seed=seed)

        retriever = injected_retriever or (
            MockXCLIPRetriever() if args.cpu_mock else _build_real_retriever(paths_cfg, use_subprocess=args.xclip_subprocess, gpu_lock=gpu_lock)
        )
        generator = injected_videollama or (MockGeneratorAdapter() if args.cpu_mock else _build_real_videollama(paths_cfg))

        spec = RunSpec(
            run_id=args.run_id,
            stage=PilotStage.PILOT_A1,
            dataset_id="DREAM-1K",
            arms=[ExpArm.ARM_A0, ExpArm.ARM_A1, ExpArm.ARM_A7],
            out_dir=args.out_dir,
            auth_receipt_path=args.auth_receipt,
            gpu_lock_path=gpu_lock,
            gpu_uuid=host.get("gpu_uuid"),
            require_nvidia_smi_idle=(not args.cpu_mock and injected_videollama is None),
            allow_cpu_mock=args.cpu_mock,
            limit=args.limit,
        )
        summary = run_natural_retrieval_stage(spec, pilot_recs, retriever, generator)
        return _exit_code_for_status(summary["status"]), summary

    # ---- PILOT A2 ----
    if stage == PilotStage.PILOT_A2:
        if not args.a1_predictions or not Path(args.a1_predictions).is_file():
            return 2, {"stage": stage.value, "status": StageRunStatus.FAILED.value, "error": "Pilot A2 requires --a1-predictions"}
        judge = injected_ref_judge or (
            MockJudgeBackend() if args.cpu_mock else _build_real_judge(args, paths_cfg, gpu_lock)
        )
        rec = run_pilot_a2_pipeline(
            a1_predictions_path=args.a1_predictions,
            dream1k_annotation_path=ann_file,
            dream1k_video_root=vid_dir,
            judge_backend=judge,
            out_dir=args.out_dir,
            auth_receipt_path=args.auth_receipt,
            prior_a1_receipt=args.prior_a1,
            human_annotations_path=args.human_annotations,
            allow_cpu_mock=args.cpu_mock,
        )
        return _exit_code_for_status(rec["status"]), rec

    # ---- PILOT A3 ----
    if stage == PilotStage.PILOT_A3:
        recs = load_dream1k_records(ann_file, vid_dir, require_video_exists=True)
        pilot_recs, _ = sample_dream1k_stratified_pilot(recs, per_domain=min(6, len(recs) // 5), seed=20261125)
        vh_dir = Path(q2w["vidhalluc_dir"])
        vh_mcq = vh_dir / "ach_mcq.json"
        if not vh_mcq.is_file():
            return 2, {"stage": stage.value, "status": StageRunStatus.FAILED.value, "error": f"VidHalluc ach_mcq.json missing: {vh_mcq}"}

        retriever = injected_retriever or (
            MockXCLIPRetriever() if args.cpu_mock else _build_real_retriever(paths_cfg, use_subprocess=args.xclip_subprocess, gpu_lock=gpu_lock)
        )
        vl_gen = injected_videollama or (MockGeneratorAdapter() if args.cpu_mock else _build_real_videollama(paths_cfg))
        if injected_modern_vlm is not None:
            vlm_gen = injected_modern_vlm
        elif args.cpu_mock:
            vlm_gen = MockGeneratorAdapter(model_id="mock_modern_vlm_cpu")
        else:
            vlm_gen = _build_real_modern_vlm(args, paths_cfg, gpu_lock)
        e3_judge = injected_e3_judge or (
            MockJudgeBackend() if args.cpu_mock else _build_real_judge(args, paths_cfg, gpu_lock)
        )
        ref_judge = injected_ref_judge or e3_judge
        vufo_sids = set() if args.cpu_mock else _load_videoufo_source_ids(paths_cfg)
        rec = run_pilot_a3_pipeline(
            dream1k_pilot_queries=pilot_recs[: args.limit] if args.limit else pilot_recs,
            retriever=retriever,
            modern_vlm=vlm_gen,
            videollama_adapter=vl_gen,
            vidhalluc_ach_mcq_path=vh_mcq,
            vidhalluc_video_root=vh_dir / "data",
            e3_judge_backend=e3_judge,
            ref_judge_backend=ref_judge,
            out_dir=args.out_dir,
            videoufo_source_ids=vufo_sids,
            approved_e3_candidates_path=args.approved_e3_candidates,
            e3_human_annotations_path=args.e3_human_annotations,
            dream1k_vlm_human_annotations_path=args.dream1k_vlm_human_annotations,
            a2_aligned_records_path=args.a2_aligned_records,
            require_cross_model_dream1k_human=bool(args.require_cross_model_dream1k_human),
            auth_receipt_path=args.auth_receipt,
            prior_a2_receipt=args.prior_a2,
            gpu_lock_path=gpu_lock,
            allow_cpu_mock=args.cpu_mock,
        )
        return _exit_code_for_status(rec["status"]), rec

    # ---- FORMAL B ----
    if stage == PilotStage.FORMAL_B:
        if not args.pilot_exclusion or not Path(args.pilot_exclusion).is_file():
            return 3, {"stage": stage.value, "status": StageRunStatus.BLOCKED.value, "error": "FORMAL_B requires --pilot-exclusion"}
        if not args.approved_e3_candidates or not Path(args.approved_e3_candidates).is_file():
            return 2, {"stage": stage.value, "status": StageRunStatus.FAILED.value, "error": "FORMAL_B requires --approved-e3-candidates"}

        excl_ids = set(read_json(args.pilot_exclusion).get("excluded_query_ids", []))
        all_dream = load_dream1k_records(ann_file, vid_dir, require_video_exists=True)

        vh_pilot_q: set[str] = set()
        vh_pilot_s: set[str] = set()
        if args.pilot_vidhalluc_exclusion and Path(args.pilot_vidhalluc_exclusion).is_file():
            vh_ex = read_json(args.pilot_vidhalluc_exclusion)
            vh_pilot_q = set(vh_ex.get("excluded_query_ids", []))
            vh_pilot_s = set(vh_ex.get("excluded_source_ids", []))

        near_dup_ids: set[str] = set()
        if args.dream1k_near_dup_exclusion and Path(args.dream1k_near_dup_exclusion).is_file():
            nd = read_json(args.dream1k_near_dup_exclusion)
            near_dup_ids = set(nd.get("excluded_query_ids", nd) if isinstance(nd, dict) else nd)

        sampling_cfg = FormalSamplingConfig(
            dream1k_target_n=args.formal_dream1k_n or args.limit,
            e2_validation_target_n=args.formal_e2_val_n,
            vidhalluc_target_n=args.formal_vidhalluc_n,
        )

        vh_dir = Path(q2w["vidhalluc_dir"])
        vh_raw = load_vidhalluc_ach_mcq(vh_dir / "ach_mcq.json", vh_dir / "data")
        vh_by_qid = {r.query_id: r for r in vh_raw}
        e3_cands = filter_runnable_e3_candidates([E3InterventionCandidate(**c) for c in read_json(args.approved_e3_candidates)])

        e3_bridge_obs = None
        if args.e3_bridge_observations and Path(args.e3_bridge_observations).is_file():
            e3_bridge_obs = [NaturalConflictObservation(**o) for o in read_json(args.e3_bridge_observations)]

        msrvtt_bridge: dict[str, Path | str] | None = None
        if args.msrvtt_historical_summary:
            msrvtt_bridge = {"summary_json": args.msrvtt_historical_summary}
        elif args.e1_ref_a and args.e1_ref_b and args.e1_source_map:
            msrvtt_bridge = {"e1_ref_a": args.e1_ref_a, "e1_ref_b": args.e1_ref_b, "e1_source_map": args.e1_source_map}

        retriever = injected_retriever or _build_real_retriever(paths_cfg, use_subprocess=args.xclip_subprocess, gpu_lock=gpu_lock)
        vl_gen = injected_videollama or _build_real_videollama(paths_cfg)
        vlm_gen = injected_modern_vlm or _build_real_modern_vlm(args, paths_cfg, gpu_lock)
        ref_j = injected_ref_judge or _build_real_judge(args, paths_cfg, gpu_lock)
        e3_j = injected_e3_judge or ref_j
        vufo_sids = _load_videoufo_source_ids(paths_cfg) if paths_cfg.get("historical_frozen_assets", {}).get("clips_full_tsv_path") and Path(paths_cfg["historical_frozen_assets"]["clips_full_tsv_path"]).is_file() else set()

        rec = run_formal_b_pipeline(
            formal_dream1k_queries=all_dream,
            formal_vidhalluc_queries_by_id=vh_by_qid,
            approved_e3_candidates=e3_cands,
            retriever=retriever,
            videollama_adapter=vl_gen,
            modern_vlm=vlm_gen,
            ref_judge_backend=ref_j,
            e3_judge_backend=e3_j,
            out_dir=args.out_dir,
            auth_receipt_path=args.auth_receipt,
            prior_receipts=priors,
            formal_config_path=args.stage_config,
            asset_manifest_path=args.asset_manifest,
            pilot_exclusion_path=args.pilot_exclusion,
            pilot_excluded_query_ids=excl_ids,
            e3_bridge_observations=e3_bridge_obs,
            sampling_cfg=sampling_cfg,
            explicit_near_dup_excluded_ids=near_dup_ids,
            videoufo_source_ids=vufo_sids,
            pilot_excluded_vidhalluc_query_ids=vh_pilot_q,
            pilot_excluded_vidhalluc_source_ids=vh_pilot_s,
            e2_validation_human_annotations_path=args.e2_validation_human_annotations,
            msrvtt_historical_bridge=msrvtt_bridge,
            gpu_lock_path=gpu_lock,
        )
        return _exit_code_for_status(rec["status"]), rec

    return 2, {"stage": str(args.stage), "status": StageRunStatus.FAILED.value, "error": "Unsupported stage"}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="UARCap Q2 Real Unified Stage Runner CLI (BN Existing Model Reuse)")
    ap.add_argument("--stage", choices=[s.value for s in PilotStage] + ["E1"], required=True)
    ap.add_argument("--paths-config", type=Path, required=True)
    ap.add_argument("--stage-config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--run-id", default="uarcap_q2_run")
    ap.add_argument("--auth-receipt", type=Path, default=None)
    ap.add_argument("--prior-a1", type=Path, default=None)
    ap.add_argument("--prior-a2", type=Path, default=None)
    ap.add_argument("--prior-a3", type=Path, default=None)
    ap.add_argument("--asset-manifest", type=Path, default=None)
    ap.add_argument("--pilot-exclusion", type=Path, default=None)
    ap.add_argument("--pilot-vidhalluc-exclusion", type=Path, default=None)
    ap.add_argument("--dream1k-near-dup-exclusion", type=Path, default=None)
    ap.add_argument("--a1-predictions", type=Path, default=None)
    ap.add_argument("--a2-aligned-records", type=Path, default=None)
    ap.add_argument("--human-annotations", type=Path, default=None)
    ap.add_argument("--approved-e3-candidates", type=Path, default=None)
    ap.add_argument("--e3-human-annotations", type=Path, default=None)
    ap.add_argument("--dream1k-vlm-human-annotations", type=Path, default=None)
    ap.add_argument("--require-cross-model-dream1k-human", action="store_true")
    ap.add_argument("--e2-validation-human-annotations", type=Path, default=None)
    ap.add_argument("--e3-bridge-observations", type=Path, default=None)
    ap.add_argument("--msrvtt-historical-summary", type=Path, default=None)
    ap.add_argument("--formal-dream1k-n", type=int, default=None)
    ap.add_argument("--formal-e2-val-n", type=int, default=None)
    ap.add_argument("--formal-vidhalluc-n", type=int, default=None)
    # Existing BN cross-family judge options (default: Mistral Small 3.2 24B)
    ap.add_argument("--judge-model-path", default="")
    ap.add_argument("--judge-api-url", default="")
    ap.add_argument("--judge-model-id", default="mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    ap.add_argument("--judge-family", default="mistral")
    ap.add_argument(
        "--judge-backend-mode",
        choices=["vllm_subprocess", "vllm_offline", "openai_compat", "local_transformers", "transformers_subprocess"],
        default="vllm_subprocess",
    )
    ap.add_argument("--judge-python", default="")
    ap.add_argument("--judge-conda-env", default="")
    # Existing BN modern VLM options (default: Qwen/Qwen3.8-27B-FP8 via vLLM 0.29.0; NO Qwen2.5-VL download!)
    ap.add_argument("--modern-vlm-path", default="")
    ap.add_argument("--modern-vlm-id", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--modern-vlm-revision", default="LOCAL_BN_QWEN38_FP8_66SHARDS")
    ap.add_argument(
        "--modern-vlm-backend",
        choices=["vllm_subprocess", "vllm_offline", "vllm_openai_http", "legacy_qwen25vl", "legacy_qwen25vl_subprocess"],
        default="vllm_subprocess",
    )
    ap.add_argument("--modern-vlm-api-url", default="")
    ap.add_argument("--modern-vlm-quantization", default="fp8")
    ap.add_argument("--vllm-python", default="")
    ap.add_argument("--vllm-conda-env", default="")
    ap.add_argument("--enable-thinking", action="store_true", help="Enable Qwen3 thinking mode (default False!)")
    ap.add_argument("--xclip-subprocess", action="store_true", help="Run X-CLIP inside frozen Video-LLaMA Python subprocess")
    ap.add_argument("--e1-ref-a", type=Path, default=None)
    ap.add_argument("--e1-ref-b", type=Path, default=None)
    ap.add_argument("--e1-source-map", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cpu-mock", action="store_true", help="CPU mock mode (stamps MOCK_PASS; NEVER unlocks real stages)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rc, payload = dispatch_stage(args)
    except Exception as exc:
        print(json.dumps({"stage": args.stage, "status": StageRunStatus.FAILED.value, "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
