"""Real Unified Stage Runner CLI (P0-1 & P0-5 remediated: ZERO fake READY_FOR_BN_REAL_ADAPTER return-0 stub!).

Exit codes:
  0  = REAL_RUNTIME_PASS or MOCK_PASS (only when outputs & receipt actually written and all gates pass)
  2  = FAILED / missing required real assets or runtime error
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
from uarcap_q2.datasets.vidhalluc import load_vidhalluc_ach_mcq
from uarcap_q2.evaluators.backends import (
    JudgeBackend,
    JudgeBackendConfig,
    LocalTransformersJudgeBackend,
    MockJudgeBackend,
    OpenAICompatJudgeBackend,
)
from uarcap_q2.generators.mock_generator import MockGeneratorAdapter
from uarcap_q2.generators.modern_vlm_adapter import ModernVLMConfig, Qwen25VLAdapter
from uarcap_q2.generators.videollama_adapter import FrozenVideoLLaMAAdapter, FrozenVideoLLaMAConfig
from uarcap_q2.interventions.vidhalluc_e3 import E3InterventionCandidate, filter_runnable_e3_candidates
from uarcap_q2.retrieval.reranker import MockXCLIPRetriever, RetrieverAdapter
from uarcap_q2.retrieval.videoufo_retriever import VideoUFOIndexConfig, VideoUFOXCLIPRetriever
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


def _build_real_retriever(paths_cfg: dict[str, Any]) -> VideoUFOXCLIPRetriever:
    hist = paths_cfg["historical_frozen_assets"]
    vcfg = VideoUFOIndexConfig(
        videoufo_csv_path=hist["videoufo_csv_path"],
        clips_full_tsv_path=hist["clips_full_tsv_path"],
        brief_text_emb_npy_path=hist["brief_text_emb_npy_path"],
        xclip_repo_dir=hist["xclip_repo_dir"],
        xclip_ckpt_path=hist["xclip_ckpt_path"],
        video_feature_dir=hist["video_feature_dir"],
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


def _build_real_judge(judge_model_path: str | None, judge_api_url: str | None, judge_model_id: str, judge_family: str) -> JudgeBackend:
    if judge_api_url:
        return OpenAICompatJudgeBackend(
            JudgeBackendConfig(
                backend_type="openai_compat",
                model_id=judge_model_id,
                model_family=judge_family,
                api_base_url=judge_api_url,
            )
        )
    if judge_model_path and Path(judge_model_path).is_dir():
        return LocalTransformersJudgeBackend(
            JudgeBackendConfig(
                backend_type="local_transformers",
                model_id=judge_model_id,
                model_family=judge_family,
                local_model_path=judge_model_path,
            )
        )
    raise FileNotFoundError(
        "Real text judge requires either --judge-model-path (existing local directory) or --judge-api-url."
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

        retriever = injected_retriever or (MockXCLIPRetriever() if args.cpu_mock else _build_real_retriever(paths_cfg))
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
            MockJudgeBackend()
            if args.cpu_mock
            else _build_real_judge(args.judge_model_path, args.judge_api_url, args.judge_model_id, args.judge_family)
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

        retriever = injected_retriever or (MockXCLIPRetriever() if args.cpu_mock else _build_real_retriever(paths_cfg))
        vl_gen = injected_videollama or (MockGeneratorAdapter() if args.cpu_mock else _build_real_videollama(paths_cfg))
        if injected_modern_vlm is not None:
            vlm_gen = injected_modern_vlm
        elif args.cpu_mock:
            vlm_gen = MockGeneratorAdapter(model_id="mock_modern_vlm_cpu")
        else:
            if not args.modern_vlm_path:
                return 2, {"stage": stage.value, "status": StageRunStatus.FAILED.value, "error": "Pilot A3 real mode requires --modern-vlm-path"}
            vlm_gen = Qwen25VLAdapter(
                ModernVLMConfig(
                    model_id=args.modern_vlm_id,
                    local_model_path=args.modern_vlm_path,
                    pinned_revision=args.modern_vlm_revision,
                    device=host.get("device", "cuda:0"),
                )
            )
        e3_judge = injected_e3_judge or (
            MockJudgeBackend()
            if args.cpu_mock
            else _build_real_judge(args.judge_model_path, args.judge_api_url, args.judge_model_id, args.judge_family)
        )
        vufo_sids = set() if args.cpu_mock else _load_videoufo_source_ids(paths_cfg)
        rec = run_pilot_a3_pipeline(
            dream1k_pilot_queries=pilot_recs[: args.limit] if args.limit else pilot_recs,
            retriever=retriever,
            modern_vlm=vlm_gen,
            videollama_adapter=vl_gen,
            vidhalluc_ach_mcq_path=vh_mcq,
            vidhalluc_video_root=vh_dir / "data",
            e3_judge_backend=e3_judge,
            out_dir=args.out_dir,
            videoufo_source_ids=vufo_sids,
            approved_e3_candidates_path=args.approved_e3_candidates,
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
        formal_dream = [r for r in all_dream if r.query_id not in excl_ids]
        if args.limit:
            formal_dream = formal_dream[: args.limit]

        vh_dir = Path(q2w["vidhalluc_dir"])
        vh_raw = load_vidhalluc_ach_mcq(vh_dir / "ach_mcq.json", vh_dir / "data")
        vh_by_qid = {r.query_id: r for r in vh_raw}
        e3_cands = filter_runnable_e3_candidates([E3InterventionCandidate(**c) for c in read_json(args.approved_e3_candidates)])

        retriever = injected_retriever or _build_real_retriever(paths_cfg)
        vl_gen = injected_videollama or _build_real_videollama(paths_cfg)
        vlm_gen = injected_modern_vlm or Qwen25VLAdapter(
            ModernVLMConfig(
                model_id=args.modern_vlm_id,
                local_model_path=args.modern_vlm_path,
                pinned_revision=args.modern_vlm_revision,
                device=host.get("device", "cuda:0"),
            )
        )
        ref_j = injected_ref_judge or _build_real_judge(args.judge_model_path, args.judge_api_url, args.judge_model_id, args.judge_family)
        e3_j = injected_e3_judge or ref_j

        rec = run_formal_b_pipeline(
            formal_dream1k_queries=formal_dream,
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
            gpu_lock_path=gpu_lock,
        )
        return _exit_code_for_status(rec["status"]), rec

    return 2, {"stage": str(args.stage), "status": StageRunStatus.FAILED.value, "error": "Unsupported stage"}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="UARCap Q2 Real Unified Stage Runner CLI")
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
    ap.add_argument("--a1-predictions", type=Path, default=None)
    ap.add_argument("--human-annotations", type=Path, default=None)
    ap.add_argument("--approved-e3-candidates", type=Path, default=None)
    ap.add_argument("--judge-model-path", default="")
    ap.add_argument("--judge-api-url", default="")
    ap.add_argument("--judge-model-id", default="mistralai/Mistral-Small-24B-Instruct-2501")
    ap.add_argument("--judge-family", default="mistral")
    ap.add_argument("--modern-vlm-path", default="")
    ap.add_argument("--modern-vlm-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--modern-vlm-revision", default="PENDING_PIN_AT_DOWNLOAD")
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
