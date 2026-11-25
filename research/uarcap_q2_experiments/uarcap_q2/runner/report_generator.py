"""Automatic Markdown report generators for Pilot A1 (Execution, Integrity, and Go/No-Go)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common.io_utils import atomic_write_text


def write_pilot_a1_reports(summary: dict[str, Any], out_dir: Path | str) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_q = summary["n_queries"]
    arms = summary["arms"]
    expected_total = summary["expected_generations"]
    completed_total = summary["completed_generations"]
    failed_total = summary["failed_generations"]
    empty_rate = summary["empty_output_rate"]
    verdict = summary["go_no_go_verdict"]
    failure_cat = summary.get("failure_category")

    exec_md = f"""# PILOT A1 EXECUTION REPORT

- **Run ID**: `{summary['run_id']}`
- **Stage**: `{summary['stage']}`
- **Dataset**: `{summary['dataset_id']}` ({n_q} queries)
- **Model**: `{summary['model_id']}` (`{summary['checkpoint_sha256']}`)
- **Arms**: `{', '.join(arms)}`
- **Mode**: `{summary['execution_mode']}`
- **Expected Generations**: `{expected_total}`
- **Completed Generations**: `{completed_total}`
- **Failed Generations**: `{failed_total}`
- **Skipped (Resumed)**: `{summary['resumed_skipped']}`
- **Total Wall Seconds**: `{summary['wall_seconds']}`
- **Peak VRAM (Bytes)**: `{summary['peak_vram_bytes']}`
"""

    integ_md = f"""# PILOT A1 INTEGRITY REPORT

- **All Queries Decoded Cleanly (16 uniform frames, no repeated-first-frame)**: `{summary['all_videos_decoded_ok']}`
- **ARM_A0 Strictly Zero External Evidence Verified**: `{summary['a0_zero_evidence_verified']}`
- **Prefix Consistency (`A1 ⊂ A3 ⊂ A7`) Verified**: `{summary['prefix_consistency_verified']}`
- **All Actual Input Traces Match Declared Packets**: `{summary['all_traces_verified']}`
- **Empty Output Rate**: `{empty_rate:.4f}` (threshold `< 0.03`)
- **Near-Duplicate High-Risk Rate**: `{summary.get('near_dup_high_risk_rate', 0.0):.4f}` (threshold `<= 0.25`)
- **Output Completeness (`completed == expected`)**: `{completed_total == expected_total}`
"""

    reasons = summary.get("gate_reasons", [])
    reasons_block = "\n".join(f"- {r}" for r in reasons) if reasons else "- All Pilot A1 technical gates passed."
    gng_md = f"""# PILOT A1 GO / NO-GO DECISION

- **Verdict**: **`{verdict}`**
- **Failure Category**: `{failure_cat or 'NONE'}`
- **Unlock Next Stage (`PILOT_A2`)**: `{verdict == 'PASS'}`

## Gate Evaluation Details

{reasons_block}
"""

    p_exec = atomic_write_text(out / "PILOT_A1_EXECUTION_REPORT.md", exec_md)
    p_int = atomic_write_text(out / "PILOT_A1_INTEGRITY_REPORT.md", integ_md)
    p_gng = atomic_write_text(out / "PILOT_A1_GO_NO_GO.md", gng_md)
    return {"execution_report": p_exec, "integrity_report": p_int, "go_no_go": p_gng}
