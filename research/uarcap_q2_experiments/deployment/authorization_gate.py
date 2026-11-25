"""CLI Authorization Gate Enforcer invoked by BN bash entrypoints (P0-5 remediated)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uarcap_q2.common.gates import AuthorizationError, verify_stage_authorization
from uarcap_q2.common.schemas import PilotStage


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify unified stage authorization & real prerequisite receipts")
    ap.add_argument("--stage", choices=[s.value for s in PilotStage], required=True)
    ap.add_argument("--auth-receipt", type=Path, default=None)
    ap.add_argument("--prior-a1", type=Path, default=None)
    ap.add_argument("--prior-a2", type=Path, default=None)
    ap.add_argument("--prior-a3", type=Path, default=None)
    ap.add_argument("--formal-config", type=Path, default=None)
    ap.add_argument("--asset-manifest", type=Path, default=None)
    ap.add_argument("--pilot-exclusion", type=Path, default=None)
    ap.add_argument("--allow-cpu-mock", action="store_true")
    args = ap.parse_args(argv)

    priors = {}
    if args.prior_a1:
        priors[PilotStage.PILOT_A1.value] = args.prior_a1
    if args.prior_a2:
        priors[PilotStage.PILOT_A2.value] = args.prior_a2
    if args.prior_a3:
        priors[PilotStage.PILOT_A3.value] = args.prior_a3

    try:
        res = verify_stage_authorization(
            stage=args.stage,
            auth_receipt_path=args.auth_receipt,
            prior_receipts=priors,
            allow_cpu_mock=args.allow_cpu_mock,
            formal_config_path=args.formal_config,
            asset_manifest_path=args.asset_manifest,
            pilot_exclusion_path=args.pilot_exclusion,
        )
        print(json.dumps(res, indent=2))
        return 0
    except AuthorizationError as exc:
        print(f"[AUTHORIZATION_BLOCKED] {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
