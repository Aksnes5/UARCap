"""Unified stage runner CLI for BN scripts (supports --dry-run, --resume, --limit, --auth-receipt)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uarcap_q2.common.gates import AuthorizationError, verify_stage_authorization
from uarcap_q2.common.schemas import ExpArm, PilotStage
from uarcap_q2.datasets.dream1k import load_dream1k_records, sample_dream1k_stratified_pilot, write_pilot_exclusion_manifest
from uarcap_q2.generators.mock_generator import MockGeneratorAdapter
from uarcap_q2.retrieval.reranker import MockXCLIPRetriever
from uarcap_q2.runner.orchestrator import RunSpec, run_natural_retrieval_stage


def main() -> int:
    ap = argparse.ArgumentParser(description="UARCap Q2 Unified Stage Runner CLI")
    ap.add_argument("--stage", choices=[s.value for s in PilotStage], required=True)
    ap.add_argument("--paths-config", type=Path, required=True)
    ap.add_argument("--stage-config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--run-id", default="uarcap_q2_run")
    ap.add_argument("--auth-receipt", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cpu-mock", action="store_true", help="Run offline CPU mock pipeline (never claims GPU validity)")
    args = ap.parse_args()

    stage = PilotStage(args.stage)
    paths_cfg = yaml.safe_load(args.paths_config.read_text(encoding="utf-8"))
    stage_cfg = yaml.safe_load(args.stage_config.read_text(encoding="utf-8"))

    if args.dry_run:
        plan = {
            "mode": "dry_run",
            "stage": stage.value,
            "run_id": args.run_id,
            "paths_config": str(args.paths_config),
            "stage_config": str(args.stage_config),
            "out_dir": str(args.out_dir),
            "gpu_touched": False,
            "authorization_checked_on_real_run": True,
        }
        print(json.dumps(plan, indent=2))
        return 0

    try:
        verify_stage_authorization(
            stage=stage,
            auth_receipt_path=args.auth_receipt,
            allow_cpu_mock=args.cpu_mock,
        )
    except AuthorizationError as exc:
        print(f"[BLOCKED] {exc}", file=sys.stderr)
        return 3

    if not args.cpu_mock:
        print(
            json.dumps(
                {
                    "status": "READY_FOR_BN_REAL_ADAPTER",
                    "stage": stage.value,
                    "note": (
                        "Real GPU execution binds FrozenVideoLLaMAAdapter / Qwen25VLAdapter / "
                        "LocalTransformersJudgeBackend on BN Linux with cooperative gpu.lock."
                    ),
                },
                indent=2,
            )
        )
        return 0

    # CPU mock mode: loads staged annotation file if present
    dream_dir = Path(paths_cfg["q2_writable_workspace"]["dream1k_dir"])
    ann_file = dream_dir / "json" / "metadata.json"
    recs = load_dream1k_records(ann_file, dream_dir / "video")
    pilot_recs, _ = sample_dream1k_stratified_pilot(recs, per_domain=min(6, len(recs) // 5))
    write_pilot_exclusion_manifest(pilot_recs, args.out_dir / "pilot_a_excluded_dream1k_vids.json", seed=20261125)

    spec = RunSpec(
        run_id=args.run_id,
        stage=stage,
        dataset_id="DREAM-1K",
        arms=[ExpArm.ARM_A0, ExpArm.ARM_A1, ExpArm.ARM_A7],
        out_dir=args.out_dir,
        allow_cpu_mock=True,
        limit=args.limit,
    )
    summary = run_natural_retrieval_stage(spec, pilot_recs, MockXCLIPRetriever(), MockGeneratorAdapter())
    print(json.dumps(summary, indent=2))
    return 0 if summary["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
