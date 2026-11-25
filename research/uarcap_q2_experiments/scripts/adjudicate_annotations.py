"""CLI for one-click dual-annotator unblinding, disagreement extraction, and adjudication (E2 & E3)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uarcap_q2.annotation.adjudicator import adjudicate_e2_annotations, adjudicate_e3_annotations


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Adjudicate dual-annotator blind exports (E2 or E3)")
    ap.add_argument("--task", choices=["e2", "e3"], required=True)
    ap.add_argument("--ann1", type=Path, required=True, help="Annotator 1 export JSON/JSONL or progress.json")
    ap.add_argument("--ann2", type=Path, required=True, help="Annotator 2 export JSON/JSONL or progress.json")
    ap.add_argument("--blind-key", type=Path, required=True, help="blind_key_e2_DO_NOT_OPEN.json or blind_key_e3_DO_NOT_OPEN.json")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--resolutions", type=Path, default=None, help="Optional resolved disagreements JSON")
    ap.add_argument("--joint-sec", type=float, default=0.0, help="Joint adjudication wall seconds")
    args = ap.parse_args(argv)

    fn = adjudicate_e2_annotations if args.task == "e2" else adjudicate_e3_annotations
    res = fn(
        ann1_export_path=args.ann1,
        ann2_export_path=args.ann2,
        blind_key_path=args.blind_key,
        out_dir=args.out_dir,
        resolutions_path=args.resolutions,
        adjudication_joint_sec=args.joint_sec,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res["status"] == "ADJUDICATED_COMPLETE" else 10


if __name__ == "__main__":
    raise SystemExit(main())
