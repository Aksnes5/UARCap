"""Build frozen sensitivity designs and summarize aligned within-query contrasts."""

import argparse
from collections import Counter
from math import ceil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uarcap_q2.common.io_utils import atomic_write_json, read_json, read_jsonl, sha256_file
from uarcap_q2.final_analysis.core import cluster_bootstrap
from uarcap_q2.final_analysis.validation_identity import index, verify_caption_pair

OUTCOMES = ("words", "n_entail_ref", "n_contra_ref", "n_unment_ref", "r_core_ref", "llama_n_contra")


def plan(length, visual, short):
    length_ids, visual_ids = (read_json(p)["query_ids"] for p in (length, visual))
    if len(length_ids) != 120 or len(set(length_ids)) != 120 or len(visual_ids) != 150 or len(set(visual_ids)) != 150:
        raise ValueError("Unexpected frozen parent plans")
    baseline = {(r["query_id"], r["arm"]): r for r in read_jsonl(short)}
    rows = [baseline[q, a] for q in length_ids for a in ("ARM_A0", "ARM_A1", "ARM_A7")]
    if not all(r["ok"] and r["input_trace"]["decoding_config"]["prompt_regime"] == "short_aligned" for r in rows):
        raise ValueError("Require valid archived Qwen short controls")
    words = sorted(len(r["caption"].split()) for r in rows)
    lo, hi = [words[ceil(len(words) * p) - 1] for p in (.25, .75)]
    if not 1 <= lo <= hi <= 30:
        raise ValueError("Unusable archived short length range")
    return {"schema": "uarcap_q2.controlled_sensitivity.v2", "query_ids": length_ids,
            "qwen_factorial_query_ids": length_ids[:40], "visual_nested_query_ids": visual_ids[:60],
            "word_range": [lo, hi], "length_quantiles": [.25, .75], "length_baseline_n": len(words),
            "input_sha256": {str(p): sha256_file(p) for p in (length, visual, short)}}


def check_length_revision(files, old_task_state, legacy_output):
    previous = read_json(old_task_state)
    known = previous["tasks"].get("q2l_v1_length_qwen", {})
    if known and (known.get("status") != "PENDING" or any(known.get(k) for k in ("started_utc", "last_start_utc", "attempts", "retries", "attempt"))):
        raise ValueError("Prior detailed-length experiment has started or lacks affirmative pending state")
    if any((legacy_output / name).exists() for name in ("predictions.jsonl", "failures.jsonl", "input_traces.jsonl", "run_receipt.json", "smoke_receipt.json")):
        raise ValueError("Historical detailed regime has runtime artifacts and cannot be replaced")
    return {**files, "revision": "DETAILED_TO_UNIFORM_LENGTH", "legacy_state_sha256": sha256_file(old_task_state),
            "legacy_generation_artifacts_present": False}


def prepare(out, length, visual, short, state=None, legacy_output=None):
    design = plan(length, visual, short)
    out.mkdir(parents=True, exist_ok=True)
    for name, ids in (("factorial", design["qwen_factorial_query_ids"]), ("visual", design["visual_nested_query_ids"])):
        atomic_write_json(out / (name + ".json"), {"query_ids": ids})
    if state is not None:
        try:
            design = check_length_revision(design, state, legacy_output)
        except ValueError as exc:
            design = {**design, "revision": "BLOCKED", "length_blocker": str(exc)}
    atomic_write_json(out / "design.json", design)


def evaluated_folder(folder, evaluation=None, cross=None, include_cross=True):
    pred = index(read_jsonl(folder / "predictions.jsonl"))
    evs = index(read_jsonl(evaluation or folder / "eval/a2_ref_fact_eval.jsonl"))
    if set(pred) != set(evs):
        raise ValueError("Prediction and judge grids differ")
    from q2_budget_report import norm
    path = cross or folder / "cross/A2_CROSS_JUDGE_LABELS.jsonl"
    labels = {(r["query_id"], r["norm_key"]): r["llama_label"] for r in read_jsonl(path)} if include_cross and path.is_file() else {}
    values = {}
    for key, row in pred.items():
        model, q, arm = key
        if not model.startswith("Qwen/") or (q, arm) in values:
            raise ValueError("Sensitivity must use one pinned Qwen system")
        r = evs[key]
        claims = verify_caption_pair(row, r)
        second = [labels.get((q, norm(c["claim"]))) for c in claims]
        values[q, arm] = {"words": len(row["caption"].split()), "trace": row["input_trace"],
                          "llama_n_contra": second.count("contradicted") if None not in second else None,
                          **{k: r[k] for k in OUTCOMES[1:-1]}}
    return values


def paired(ids, cells, coefficients):
    result = {}
    for endpoint in OUTCOMES:
        data = {}
        for q in ids:
            scores = [cell(q)[endpoint] for cell in cells]
            if all(x is not None for x in scores):
                data[q] = sum(x * c for x, c in zip(scores, coefficients))
        result[endpoint] = cluster_bootstrap(data) if data else {"n": 0, "status": "UNAVAILABLE"}
    return result


def cell(table, arm):
    return lambda q: table[q, arm]


def analyze(kind, design, baseline, additions):
    if kind == "length":
        ids = design["query_ids"]
        uniform = additions[0]
        arms = ("ARM_A0", "ARM_A1", "ARM_A7")
        lo, hi = design["word_range"]
        results = {}
        for arm in arms:
            range_counts = Counter("below" if uniform[q, arm]["words"] < lo else "above" if uniform[q, arm]["words"] > hi else "within" for q in ids)
            results[arm] = {"n": len(ids), "word_counts": {q: uniform[q, arm]["words"] for q in ids},
                            "compliance": dict(range_counts), "uniform_minus_short": paired(ids, [cell(uniform, arm), cell(baseline, arm)], [1, -1])}
        return results
    if kind == "factorial":
        result = {}
        ids = design["qwen_factorial_query_ids"]
        factorial = additions[0]
        for q in ids:
            grid = [factorial[q,a]["trace"] for a in ("T0F0","T1F0","T0F1","T1F1")]
            if any(any(t[k]!=grid[0][k] for k in ("visual_input_sha256","frame_indices","decoding_config","checkpoint_sha256")) for t in grid):
                raise ValueError("Qwen factorial query frames or decoding differ")
            if any(grid[2][k]!=grid[3][k] for k in ("visual_evidence_ids","visual_evidence_scores")) or grid[1]["text_evidence"]!=grid[3]["text_evidence"]:
                raise ValueError("Qwen factorial text or clip differs across corresponding conditions")
            media=[t["actual_processor_telemetry"] for t in grid]
            if media[2]["external_frames_sha256"]!=media[3]["external_frames_sha256"] or media[2]["external_video_sha256"]!=media[3]["external_video_sha256"]:
                raise ValueError("Qwen factorial retrieved frames changed")
        a, b, c, d = [cell(factorial, arm) for arm in ("T0F0", "T1F0", "T0F1", "T1F1")]
        result["word_counts"] = {q: {arm: factorial[q, arm]["words"] for arm in ("T0F0", "T1F0", "T0F1", "T1F1")} for q in ids}
        result.update({key: paired(ids, [a, b, c, d], co) for key, co in
                       {"interaction": [1,-1,-1,1], "text_without_frames": [-1,1,0,0],
                        "frames_without_text": [-1,0,1,0], "text_with_frames": [0,0,-1,1],
                        "frames_with_text": [0,-1,0,1]}.items()})
        return result
    ids = design["visual_nested_query_ids"]
    a0, a7 = additions
    for q in ids:
        left, right = (table[q, arm]["trace"] for table, arm in ((a0, "ARM_A0"), (a7, "ARM_A7")))
        if any(left[k] != right[k] for k in ("visual_input_sha256", "frame_indices", "checkpoint_sha256", "decoding_config")):
            raise ValueError("Dense A0/A7 input changed")
        baseline_a0, baseline_a7 = (baseline[q,arm]["trace"] for arm in ("ARM_A0","ARM_A7"))
        if any(baseline_a0[k]!=baseline_a7[k] for k in ("visual_input_sha256","frame_indices","checkpoint_sha256","decoding_config")):
            raise ValueError("Archived A0/A7 control inputs differ")
        if left["checkpoint_sha256"]!=baseline_a0["checkpoint_sha256"]:
            raise ValueError("Dense and baseline checkpoints differ")
    return {"interaction": paired(ids, [cell(a7,"ARM_A7"), cell(a0,"ARM_A0"), cell(baseline,"ARM_A7"), cell(baseline,"ARM_A0")], [1,-1,-1,1]),
            "budget_at_dense": paired(ids, [cell(a7,"ARM_A7"), cell(a0,"ARM_A0")], [1,-1]),
            "budget_at_16": paired(ids, [cell(baseline,"ARM_A7"), cell(baseline,"ARM_A0")], [1,-1])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("plan")
    for flag in ("length", "visual", "short", "out", "state", "legacy-output"):
        prep.add_argument("--" + flag, type=Path, required=flag not in ("state", "legacy-output"))
    report = sub.add_parser("report")
    report.add_argument("--kind", choices=("length", "factorial", "visual32", "visual64"), required=True)
    for flag in ("design", "baseline", "baseline-eval", "baseline-cross", "out"):
        report.add_argument("--" + flag, type=Path, required=flag != "baseline-cross")
    report.add_argument("--addition", action="append", type=Path, required=True)
    args = p.parse_args()
    if args.command == "plan":
        prepare(args.out, args.length, args.visual, args.short, args.state, args.legacy_output)
    else:
        design = read_json(args.design)
        include_cross=bool(args.baseline_cross)
        base = evaluated_folder(args.baseline, args.baseline_eval, args.baseline_cross, include_cross)
        result = analyze(args.kind, design, base, [evaluated_folder(p,include_cross=include_cross) for p in args.addition])
        inputs = [args.design, args.baseline / "predictions.jsonl", args.baseline_eval]
        for folder in args.addition:
            inputs.extend((folder / "predictions.jsonl", folder / "eval/a2_ref_fact_eval.jsonl", folder / "cross/A2_CROSS_JUDGE_LABELS.jsonl"))
        if args.baseline_cross:
            inputs.append(args.baseline_cross)
        atomic_write_json(args.out, {"schema": "uarcap_q2.controlled_sensitivity_report.v2", "kind": args.kind,
                                     "status": "MACHINE_ASSISTED_UNCALIBRATED", "contrasts": result,
                                     "judge_scope":"DUAL_JUDGE" if include_cross else "GEMMA_ONLY",
                                     "input_sha256": {str(p): sha256_file(p) if p.is_file() else None for p in inputs}})
        from q2_module_report import plot
        effects={}
        for key, group in result.items():
            if args.kind=="length":
                group=group["uniform_minus_short"]
            if isinstance(group,dict):
                for field in ("n_entail_ref","n_contra_ref","llama_n_contra"):
                    if field in group:
                        effects[f"{key} · {field}"]=group[field]
        plot(effects,args.out,"Paired change in written-reference facts (95% bootstrap interval)")


if __name__ == "__main__":
    main()
