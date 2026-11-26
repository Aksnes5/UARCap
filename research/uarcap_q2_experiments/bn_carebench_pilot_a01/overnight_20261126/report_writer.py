#!/usr/bin/env python3
"""Independent report writer: every 30 min, summarize local state/results into the
morning report + minimal human-review TODO. No Claude/API/network dependency."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

BASE = Path(os.environ.get("OVERNIGHT_BASE", "/home/bn/uarcap_q2_runtime/overnight/20261126"))
Q2 = Path("/home/bn/uarcap_q2_runtime")
OUT = Q2 / "outputs"
REPORT = BASE / "OVERNIGHT_UARCAP_Q2_EXECUTION_REPORT.md"
TODO = BASE / "USER_REVIEW_TODO.md"
LOG = BASE / "logs" / "report_writer.log"


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def rj(p, d=None):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception: return d if d is not None else {}


def count_lines(p):
    try: return sum(1 for _ in Path(p).open("rb"))
    except Exception: return 0


def log(m):
    with LOG.open("a", encoding="utf-8") as f: f.write(f"{utcnow()} {m}\n")


def write_report():
    st = rj(BASE / "task_state.json", {})
    hb = rj(BASE / "heartbeat.json", {})
    shb = rj(BASE / "supervisor_heartbeat.json", {})
    dl = rj(BASE / "download_state.json", {})
    tasks = st.get("tasks", {})
    counts = {}
    for v in tasks.values(): counts[v.get("status")] = counts.get(v.get("status"), 0) + 1
    started = st.get("started_utc", "")
    elapsed_h = 0.0
    try:
        from datetime import datetime as dt
        elapsed_h = (dt.now(timezone.utc) - dt.fromisoformat(started)).total_seconds() / 3600 if started else 0
    except Exception: pass
    a2ag = rj(OUT / "pilot_a2_machine/A2_CROSS_JUDGE_AGREEMENT.json", {})
    a2vis = rj(OUT / "pilot_a2_machine_visual/A2_VISUAL_VERIFICATION_SUMMARY.json", {})
    a2rec = rj(OUT / "pilot_a2_machine/A2_HUMAN_REVIEW_REQUIRED.json", {})
    a3r = rj(OUT / "pilot_a3_exploratory/carebench_qwen38/run_receipt.json", {})
    a3e = rj(OUT / "pilot_a3_exploratory/carebench_qwen38_eval/a2_judge_run_receipt.json", {})
    a3c = rj(OUT / "pilot_a3_machine_cross_judge/A2_CROSS_JUDGE_AGREEMENT.json", {})
    fm = rj(OUT / "formal_candidate_unvalidated/FORMAL_MANIFEST_RECEIPT.json", {})
    fv = rj(OUT / "formal_candidate_unvalidated/videollama/run_receipt.json", {})
    fq = rj(OUT / "formal_candidate_unvalidated/qwen/run_receipt.json", {})
    fe1 = rj(OUT / "formal_candidate_unvalidated/videollama_eval/a2_judge_run_receipt.json", {})
    fe2 = rj(OUT / "formal_candidate_unvalidated/qwen_eval/a2_judge_run_receipt.json", {})
    e3r = rj(OUT / "pilot_e3_exploratory/E3_CROSS_JUDGE.json", {})
    e3a = rj(OUT / "pilot_e3_exploratory/E3_APPROVED_CANDIDATES.json", {})
    e3hr = rj(OUT / "pilot_e3_exploratory/E3_HUMAN_REVIEW_CANDIDATES.json", {})
    pending = rj(BASE / "pending_repairs.json", {"items": []})
    recov = count_lines(BASE / "recovery_actions.jsonl")
    checks = count_lines(BASE / "health_checks.jsonl")
    lines = [
        "# OVERNIGHT UARCap Q2 Execution Report",
        "",
        f"- Generated: {utcnow()}",
        f"- Queue started: {started}; elapsed: **{elapsed_h:.2f} h**",
        f"- Orchestrator: {shb.get('orchestrator_alive')} (pid {hb.get('current_task_pid')}); current task: `{hb.get('current_task')}`",
        f"- Task counts: {counts}",
        f"- GPU: {hb.get('gpu') or shb.get('gpu')}",
        f"- Supervisor checks: {checks}; recovery actions: {recov}; pending repairs: {len(pending.get('items', []))}",
        "",
        "## Network / download",
        f"- VidHalluc download: status={dl.get('status')} bytes={dl.get('bytes_on_disk')} percent={dl.get('percent')} speed_kBps={dl.get('speed_kbps')}",
        f"- Download errors: {dl.get('last_error')}",
        "",
        "## A2 machine-assisted calibration",
        f"- Cross-judge: {a2ag.get('n_evaluated')}/{a2ag.get('n_unique_claims')} unique claims; agreement={a2ag.get('observed_agreement_unique')}; kappa={a2ag.get('cohen_kappa_unique')}; disagreements={a2ag.get('n_disagreements')}",
        f"- Visual verification: verified={a2vis.get('n_verified')} of {a2vis.get('n_flagged')}; video_truth={a2vis.get('video_truth_counts')}; still uncertain={a2vis.get('n_still_uncertain')}",
        f"- Human review items: {a2rec.get('n_items')}",
        "",
        "## A3 CaReBench cross-model (exploratory)",
        f"- Qwen3.8 generation: {a3r.get('n_predictions')} predictions, status={a3r.get('status')}, peak={a3r.get('peak_vram_mib')} MiB",
        f"- Gemma eval: rows={a3e.get('n_rows')}, parse_failures={a3e.get('n_parse_failures')}",
        f"- Llama cross-judge: evaluated={a3c.get('n_evaluated')}, agreement={a3c.get('observed_agreement_unique')}, kappa={a3c.get('cohen_kappa_unique')}",
        "",
        "## Formal candidate generation (UNVALIDATED)",
        f"- Frozen manifest: {fm.get('n_selected')} candidates ({fm.get('n_validation')} validation); hash={fm.get('manifest_sha256')}",
        f"- Video-LLaMA: {fv.get('n_predictions')} predictions; per_arm={fv.get('per_arm')}; failures={fv.get('n_failures')}",
        f"- Qwen3.8: {fq.get('n_predictions')} predictions; per_arm={fq.get('per_arm')}; failures={fq.get('n_failures')}",
        f"- Gemma eval old/new: {fe1.get('n_rows')} / {fe2.get('n_rows')}",
        "",
        "## E3 exploratory (VidHalluc)",
        f"- Approved candidates: {e3a.get('n_approved')} (machine-assisted); review candidates: {e3hr.get('n_items')}",
        f"- Cross-judge: pairs={e3r.get('n_pairs')}; true_agree={e3r.get('true_agreement')}; conflict_agree={e3r.get('conflict_agreement')}",
        "",
        "## Integrity / gates",
        "- A2 remains `PILOT_A2_HUMAN_VALIDATION_PENDING`; no human signatures or agreement were fabricated.",
        "- A3/E3/formal outputs are exploratory or `FORMAL_CANDIDATE_GENERATION_UNVALIDATED`; no confirmatory statistics.",
        "",
        "## Recovery",
        "```bash",
        f"cd {BASE} && bash run_overnight.sh   # resumes from task_state.json",
        "touch STOP_REQUESTED                 # graceful stop after current task",
        "```",
        "",
        "## Artifacts",
        f"- Overnight base: `{BASE}`",
        f"- Task state/logs: `{BASE}/task_state.json`, `{BASE}/logs/`",
        f"- A2 machine: `{OUT}/pilot_a2_machine*`",
        f"- A3 exploratory: `{OUT}/pilot_a3_exploratory`, `{OUT}/pilot_a3_machine*`",
        f"- VidHalluc/E3: `{OUT}/pilot_e3_exploratory`",
        f"- Formal candidate: `{OUT}/formal_candidate_unvalidated`",
        f"- Branch/commit: `antigravity/uarcap-q2-bn-carebench-pilot-a01` (latest commit at session end)",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    todo = [f"# USER REVIEW TODO — {utcnow()}", "",
            f"- A2 machine-assisted human review items: **{a2rec.get('n_items', 0)}** — `{OUT}/pilot_a2_machine/A2_HUMAN_REVIEW_REQUIRED.json`",
            f"- E3 candidate human review: **{e3hr.get('n_items', 0)}** — `{OUT}/pilot_e3_exploratory/E3_HUMAN_REVIEW_CANDIDATES.json`",
            f"- Pending repairs: **{len(pending.get('items', []))}** — `{BASE}/pending_repairs.json`"]
    TODO.write_text("\n".join(todo) + "\n", encoding="utf-8")
    log("reports written")


def main():
    hours = 16
    end = time.time() + hours * 3600
    while time.time() < end:
        if (BASE / "STOP_REQUESTED").exists():
            write_report(); return 0
        try: write_report()
        except Exception as e: log(f"error {e}")
        time.sleep(1800)


if __name__ == "__main__":
    raise SystemExit(main())
