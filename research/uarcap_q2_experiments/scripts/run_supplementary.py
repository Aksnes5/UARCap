#!/usr/bin/env python3
"""Supplementary Q2 pipeline: CPU analyses, E3-Bridge candidate/job/approval steps.

All outputs are POST_HOC_EXPLORATORY or MACHINE_ASSISTED; no human gate is claimed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uarcap_q2.final_analysis import supplementary as S  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true", help="run all CPU descriptive analyses")
    ap.add_argument("--e3-bridge-candidates", action="store_true")
    ap.add_argument("--e3-bridge-jobs", action="store_true")
    ap.add_argument("--e3-bridge-merge", type=Path, help="verdict JSONL from the VLM verifier")
    ap.add_argument("--min-valid",type=int,default=30)
    ap.add_argument("--screened-plan",type=Path)
    ap.add_argument("--candidate-file",type=Path)
    ap.add_argument("--approved-out",type=Path)
    args = ap.parse_args(argv)
    if args.e3_bridge_candidates:
        p = S.build_e3_bridge_candidates_v2()
        print(json.dumps({"n_candidates": p["n_candidates"], "status": p["status"]}, indent=2))
        return 0
    if args.e3_bridge_jobs:
        cp = json.loads((S.SUPP / "E3_BRIDGE_CANDIDATES_V2.json").read_text())
        print(json.dumps({"n_jobs": S.verification_jobs(cp)["n_jobs"]}, indent=2))
        return 0
    if args.e3_bridge_merge:
        if not (args.screened_plan and args.candidate_file and args.approved_out and args.e3_bridge_merge.is_file()):
            ap.exit(3,"HARD_BLOCK V2 freeze requires candidates, screened plan, visual results and a new output\n")
        from uarcap_q2.common.io_utils import read_json,atomic_write_json
        cp = read_json(args.candidate_file)
        verdicts = S.read_jsonl(args.e3_bridge_merge)
        p = S.merge_e3_bridge_verification(cp,verdicts,screened_plan=read_json(args.screened_plan),min_valid=args.min_valid)
        if args.approved_out.exists() and read_json(args.approved_out)!=p:
            raise ValueError("Use a new versioned freeze")
        atomic_write_json(args.approved_out,p)
        print(json.dumps({k: p[k] for k in ("n_candidates", "n_approved", "n_rejected", "eligible", "status")}, indent=2))
        return 0 if p["eligible"] else 3
    out = S.run_all()
    print(json.dumps({"status": "POST_HOC_EXPLORATORY", "keys": list(out),
                      "e3_bridge": out["e3_bridge_candidates"], "out_dir": str(S.SUPP)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
