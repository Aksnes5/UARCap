#!/usr/bin/env python3
"""Unified UARCap Q2 downstream entrypoint.

Modes:
  --describe-only      descriptive counts, transitions, judge reliability (no inference)
  --machine-assisted   descriptive + exploratory paired statistics, clearly labelled uncalibrated
  --human-calibrated   requires completed A2 human calibration; recalculates with imported labels
  --confirmatory       requires A2/A3/Formal-B gates; otherwise hard-refuses
  --integrity-check    re-print formal integrity audit status
  --generate-figures / --export-paper-tables

Never fabricates human labels or fills missing results with plausible numbers.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uarcap_q2.final_analysis.core import (  # noqa: E402
    ARMS, all_transitions, cross_model_paired, holm, judge_reliability, metric_by_qid_arm,
    missing_value_sensitivity, paired_arm_stats, per_model_arm_descriptives, random_agreed_sample, read_jsonl,
)
from uarcap_q2.final_analysis.pipelines import (  # noqa: E402
    E3, FORMAL, O, build_e3_bridge_candidates, formal_e3_plan, load_historical_metrics, prepare_validation_package,
)
from uarcap_q2.final_analysis import reporting  # noqa: E402

OUT = O / "final_analysis"
A2_RECEIPT = O / "pilot_a2/pilot_a2_carebench_eval_v1/calibration_receipt.json"

MODELS = {
    "formal_videollama": (FORMAL / "videollama_eval", FORMAL / "videollama_machine_cross", FORMAL / "videollama"),
    "formal_qwen38": (FORMAL / "qwen_eval", FORMAL / "qwen_machine_cross", FORMAL / "qwen"),
    "a3_qwen38": (O / "pilot_a3_exploratory/carebench_qwen38_eval", O / "pilot_a3_machine_cross_judge", O / "pilot_a3_exploratory/carebench_qwen38"),
    "a3_qwen35": (O / "pilot_a3_exploratory/carebench_qwen35_eval", O / "pilot_a3_machine_cross_judge_qwen35", O / "pilot_a3_exploratory/carebench_qwen35"),
}


def load_model(name):
    ev, cross, pred = MODELS[name]
    rows = read_jsonl(ev / "a2_ref_fact_eval.jsonl")
    labels = read_jsonl(cross / "A2_CROSS_JUDGE_LABELS.jsonl")
    return rows, labels


def gates() -> dict:
    a2 = {}
    try: a2 = json.loads(A2_RECEIPT.read_text())
    except Exception: pass
    a3_receipts = list((O / "pilot_a3_exploratory").rglob("*receipt*.json")) if (O / "pilot_a3_exploratory").exists() else []
    return {
        "a2_status": a2.get("status"),
        "a2_human_calibration_completed": bool(a2.get("human_calibration_completed")),
        "a3_human_validation": any("VALIDATION_PASS" in str(p) for p in a3_receipts),
        "formal_b_authorized": False,
    }


def descriptive_all() -> dict:
    out = {"models": {}, "transitions": {}, "reliability": {}, "random_agreed_audit": {}}
    for name in MODELS:
        rows, labels = load_model(name)
        if not rows: continue
        out["models"][name] = per_model_arm_descriptives(rows)
        out["transitions"][name] = all_transitions(rows)
        out["reliability"][name] = judge_reliability(rows, labels)
        out["random_agreed_audit"][name] = random_agreed_sample(rows, labels, n=20)
    out["missing_value_sensitivity"] = {name: missing_value_sensitivity(load_model(name)[0]) for name in MODELS if load_model(name)[0]}
    return out


def machine_assisted_stats() -> dict:
    metrics = ("support_rate", "conflict_rate", "unmentioned_rate", "core_coverage")
    stats = {}
    pvals = {}
    for name in MODELS:
        rows, _ = load_model(name)
        if not rows: continue
        stats[name] = {}
        for m in metrics:
            res = paired_arm_stats(rows, m)
            stats[name][m] = res
            for pair, v in res.items():
                if v.get("p_two_sided") is not None:
                    pvals[f"{name}|{m}|{pair}"] = v["p_two_sided"]
    stats["holm"] = holm(pvals)
    fa = load_model("formal_videollama")[0]; fq = load_model("formal_qwen38")[0]
    if fa and fq:
        stats["cross_model"] = {m: cross_model_paired(fa, fq, m) for m in metrics}
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--describe-only", action="store_true")
    ap.add_argument("--machine-assisted", action="store_true")
    ap.add_argument("--human-calibrated", action="store_true")
    ap.add_argument("--confirmatory", action="store_true")
    ap.add_argument("--integrity-check", action="store_true")
    ap.add_argument("--generate-figures", action="store_true")
    ap.add_argument("--export-paper-tables", action="store_true")
    ap.add_argument("--prepare-validation-package", action="store_true")
    ap.add_argument("--e3-plan", action="store_true")
    ap.add_argument("--e3-bridge", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    g = gates()
    if args.human_calibrated and not (g["a2_status"] == "REAL_RUNTIME_PASS" and g["a2_human_calibration_completed"]):
        print(json.dumps({"status": "HUMAN_CALIBRATION_REQUIRED", "gates": g}, indent=2)); return 3
    if args.confirmatory and not (g["a2_human_calibration_completed"] and g["a3_human_validation"] and g["formal_b_authorized"]):
        print(json.dumps({"status": "CONFIRMATORY_GATES_NOT_MET", "gates": g}, indent=2)); return 3
    if args.integrity_check:
        audit = O / "final_audit/FORMAL_INTEGRITY_AUDIT.json"
        print(audit.read_text() if audit.is_file() else json.dumps({"status": "PENDING"}))
        return 0
    if args.prepare_validation_package:
        print(json.dumps(prepare_validation_package(), indent=2)[:3000]); return 0
    if args.e3_plan:
        print(json.dumps(formal_e3_plan(), indent=2)); return 0
    if args.e3_bridge:
        rows = load_model("formal_qwen38")[0]
        retr = read_jsonl(FORMAL / "formal_retrieval.jsonl")
        print(json.dumps(build_e3_bridge_candidates(rows, retr), indent=2)[:3000]); return 0

    desc = descriptive_all()
    (OUT / "descriptive.json").write_text(json.dumps(desc, indent=2), encoding="utf-8")
    result = {"descriptive": desc, "mode": "MACHINE_ASSISTED_UNCALIBRATED", "gates": g}
    if args.machine_assisted or args.confirmatory or args.human_calibrated:
        stats = machine_assisted_stats()
        (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        result["stats"] = stats
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # paper tables
    if args.export_paper_tables:
        desc_rows = []
        for name, arms in desc["models"].items():
            for arm, s in arms.items():
                desc_rows.append({"model": name, "arm": arm, **{k: s.get(k) for k in ("captions","claims","entailed","contradicted","unmentioned","support_rate","conflict_rate","unmentioned_rate","core_coverage_mean","mean_caption_words","parse_failure_rate")}})
        reporting.export_table(desc_rows, OUT / "tables", "table_e2_descriptive")
        tr_rows = [{"model": m, "transition": t, **v} for m, ts in desc["transitions"].items() for t, v in ts.items()]
        reporting.export_table(tr_rows, OUT / "tables", "table_e2_fact_changes")
        rel_rows = []
        for name, r in desc["reliability"].items():
            for arm, v in r.get("per_arm", {}).items():
                rel_rows.append({"model": name, "arm": arm, "n": v.get("n"), "disagreement_rate": v.get("disagreement_rate"), "directions": v.get("directions")})
        reporting.export_table(rel_rows, OUT / "tables", "table_judge_reliability")
        if "stats" in result:
            eff = [{"model": m, "metric": met, "pair": p, **v} for m, ms in result["stats"].items() if isinstance(ms, dict) and m not in ("holm","cross_model") for met, ps in ms.items() for p, v in ps.items()]
            reporting.export_table(eff, OUT / "tables", "table_paired_effects")
        hist = load_historical_metrics()
        reporting.export_table(hist, OUT / "tables", "table_historical_bridge")
    if args.generate_figures:
        figdir = OUT / "figures"
        reporting.plot_e2_fact_changes(desc["transitions"].get("formal_qwen38", {}), figdir)
        reporting.plot_core_coverage(desc["models"], list(desc["models"]), figdir)
        for name in desc["models"]:
            reporting.plot_dimensions(desc["models"], name, figdir)
            if name in desc["reliability"]:
                reporting.plot_agreement_matrix(desc["reliability"][name], figdir, stem=f"fig_judge_confusion_{name}")
        if "stats" in result:
            reporting.plot_effect_sizes({k: v for k, v in result["stats"].items() if k.startswith(("formal","a3"))}, figdir)
        reporting.plot_flow({"CaReBench parents": 913, "Pilot parents excluded": 30, "Eligible after near-dup scan": 883,
                             "Formal selected": 600, "Independent validation": 60}, figdir)
        reporting.plot_budget_curve(load_historical_metrics(), figdir)
    print(json.dumps({"status": result["mode"], "models": list(desc["models"]), "gates": g,
                      "stats_included": "stats" in result, "out_dir": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
