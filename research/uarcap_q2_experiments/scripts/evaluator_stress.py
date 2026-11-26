"""Closed-form text diagnostics using the reference evaluator's production prompts and parsers."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uarcap_q2.common.io_utils import append_jsonl_durable, atomic_write_json, primitive_hash, read_jsonl, sha256_file
from uarcap_q2.evaluators.batched_reference_fact_eval import EXTRACT_SYS, NLI_SYS
from uarcap_q2.evaluators.reference_fact_eval import parse_and_validate_verdicts, parse_extracted_claims

NAMES = (("Ari","Bea"), ("Cara","Dev"), ("Eli","Fay"), ("Gus","Hana"), ("Ivo","Jia"),
         ("Kai","Luz"), ("Moe","Nia"), ("Oli","Paz"), ("Quin","Rae"), ("Sol","Taj"),
         ("Uma","Val"), ("Wes","Xia"), ("Yan","Zoe"), ("Ada","Ben"))


def fixture_cases():
    rows = []
    for i, (a,b) in enumerate(NAMES):
        n = i % 4 + 2
        ref = (f"{a} gave {b} exactly {n} red cups. Only {a} gave cups. The cups were red and no other color. "
               f"{a} gave the cups before {b} rang a bell. Neither person danced. "
               f"{b} rang the bell and did not throw it.")
        main = f"{a} gave {b} cups."
        cases = [
            ("correct", main, "entailed"),
            ("paraphrase", f"{b} received cups from {a}.", "entailed"),
            ("wrong_action", f"{b} threw the bell.", "contradicted"),
            ("role_swap", f"{b} gave cups to {a}.", "contradicted"),
            ("count", f"{a} gave {b} exactly {n+1} cups.", "contradicted"),
            ("color", f"The cups were blue.", "contradicted"),
            ("order", f"{b} rang the bell before {a} gave the cups.", "contradicted"),
            ("negation", f"{a} did not give {b} any cups.", "contradicted"),
            ("double_negation", f"It is false that {a} did not give {b} any cups.", "entailed"),
            ("vague", f"{a} did something.", "entailed"),
            ("generic", "Something happened involving cups.", "entailed"),
            ("related_object", "There was a bell.", "entailed"),
            ("unmentioned", "The bell was made of brass.", "unmentioned"),
        ]
        for cat, claim, label in cases:
            rows.append({"id": f"{i:02d}:{cat}", "group": i, "scope": "CONTROLLED_TEXT_NLI", "reference": ref,
                         "category": cat, "claim": claim, "expected": label})
        atoms = [f"{a} danced.", f"{b} threw the bell."]
        rows.append({"id": f"{i:02d}:atomic", "group": i, "scope": "CONTROLLED_TEXT_EXTRACTION",
                     "category": "atomic_split_merge", "reference": ref,
                     "caption": f"{a} danced and {b} threw the bell.", "canonical_atoms": atoms})
    random.Random(20261126).shuffle(rows)
    return rows


def score(backend, outfile, binding, cases=None, chunk=16):
    rows = fixture_cases() if cases is None else cases
    by_id = {r["id"]: r for r in rows}
    done = set()
    for r in read_jsonl(outfile):
        if r["binding"] != binding or r["case_id"] not in by_id or r["case_id"] in done:
            raise ValueError("Stress ledger is stale or has duplicate records")
        done.add(r["case_id"])
    for start in range(0,len(rows),chunk):
        batch = [r for r in rows[start:start+chunk] if r["id"] not in done]
        prompts = []
        for case in batch:
            if case["scope"] == "CONTROLLED_TEXT_NLI":
                prompts.append((NLI_SYS, "[TASK:3WAY_REF_NLI]\nREFERENCE: " + case["reference"] +
                                "\nCLAIMS_JSON: " + json.dumps([case["claim"]])))
            else:
                prompts.append((EXTRACT_SYS, "[TASK:EXTRACT_CLAIMS]\nCAPTION: " + case["caption"]))
        if not prompts:
            continue
        outputs = backend.complete_batch(prompts)
        if len(outputs) != len(batch):
            raise ValueError("Backend returned incomplete stress batch")
        for case, raw in zip(batch, outputs):
            row = {"case_id": case["id"], "binding": binding, "raw": raw, "parse_ok": False}
            try:
                if case["scope"] == "CONTROLLED_TEXT_NLI":
                    row["label"] = parse_and_validate_verdicts(raw, [case["claim"]])[0].label
                else:
                    row["atoms"] = parse_extracted_claims(raw)
                row["parse_ok"] = True
            except ValueError as exc:
                row["parse_error"] = str(exc)
            append_jsonl_durable(outfile, row)


def summarize(paths, cases=None):
    cases = fixture_cases() if cases is None else cases
    result = {}
    errors = {}
    review = []
    for name, path in paths.items():
        if not path.is_file():
            continue
        records = read_jsonl(path)
        rows = {r["case_id"]:r for r in records}
        if len(rows) != len(records) or len({r["binding"] for r in records}) != 1 or set(rows) - {c["id"] for c in cases}:
            raise ValueError("Mixed or invalid stress ledger")
        stats = defaultdict(Counter)
        matrix = defaultdict(Counter)
        errors[name] = set()
        atomic = Counter()
        stable = Counter()
        for c in cases:
            category = c["category"]
            r = rows.get(c["id"])
            summary = stats[category]
            summary["planned"] += 1
            if not r:
                continue
            summary["observed"] += 1
            summary["parsed"] += r["parse_ok"]
            if "expected" not in c:
                atoms = r.get("atoms") or []
                atomic["observed"] += 1
                atomic["wrong_atom_count"] += len(atoms) != len(c["canonical_atoms"])
                exact = {s.lower().rstrip(".") for s in atoms} == {s.lower().rstrip(".") for s in c["canonical_atoms"]}
                atomic["canonical_exact"] += exact
                if not exact:
                    review.append({"judge": name, "case": c, "row": r, "reason": "CHECK_OMISSION_MERGE_OR_PARAPHRASE"})
                continue
            label = r.get("label", "PARSE_FAILED")
            matrix[c["expected"]][label] += 1
            summary["correct"] += label == c["expected"]
            if label != c["expected"]:
                errors[name].add(c["id"])
                review.append({"judge": name, "case": c, "row": r, "reason": "INCORRECT_OR_UNPARSED_NLI"})
            if category == "paraphrase":
                other = rows.get(f"{c['group']:02d}:correct", {})
                if r.get("parse_ok") and other.get("parse_ok"):
                    stable["paired"] += 1
                    stable["same_label"] += other["label"] == label
        result[name] = {"sha256": sha256_file(path), "observed": len(rows),
                        "parse_failure_count": sum(not r["parse_ok"] for r in rows.values()),
                        "parse_failure_rate": sum(not r["parse_ok"] for r in rows.values()) / len(rows),
                        "category": {k: {**v, "accuracy": v["correct"]/v["observed"] if v["observed"] and k != "atomic_split_merge" else None}
                                     for k,v in stats.items()},
                        "confusion": dict(matrix), "paraphrase_stability": dict(stable), "atomic_split_checks": dict(atomic)}
    common = set.intersection(*errors.values()) if len(errors) >= 2 else None
    return {"schema": "uarcap_q2.evaluator_stress.v1", "status": "CONTROLLED_TEXT_ONLY", "n_planned": len(cases),
            "suite_sha256": primitive_hash(cases), "models": result, "shared_errors": sorted(common) if common is not None else None,
            "needs_review": review, "video_truth": "AWAITING_INDEPENDENT_HUMAN_OBSERVATIONS"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for k in ("weights", "out", "lock", "raw"):
        run.add_argument("--"+k, type=Path, required=True)
    for k in ("gpu-uuid", "model-id"):
        run.add_argument("--"+k, required=True)
    run.add_argument("--family", choices=("gemma","llama"), required=True)
    report = sub.add_parser("report")
    for k in ("gemma", "llama", "out"):
        report.add_argument("--"+k,type=Path,required=True)
    args = p.parse_args()
    if args.command == "run":
        from uarcap_q2.common.gpu_lock import GPULockConfig, acquire_cooperative_gpu_lock
        from uarcap_q2.evaluators.backends import JudgeBackendConfig, LocalVLLMJudgeBackend
        cfg = JudgeBackendConfig("vllm_offline", args.model_id, args.family, str(args.weights),
                                 gpu_memory_utilization=.95 if args.family=="llama" else .85,
                                 language_model_only=args.family=="gemma")
        binding = primitive_hash([fixture_cases(), vars(cfg), sha256_file(Path(__file__))])
        with acquire_cooperative_gpu_lock(GPULockConfig(args.lock, "q2_evaluator_stress", args.gpu_uuid, True, args.out.parent)):
            score(LocalVLLMJudgeBackend(cfg), args.raw, binding, chunk=8 if args.family=="llama" else 16)
        report_paths = {args.family: args.raw}
    else:
        report_paths = {"gemma": args.gemma, "llama": args.llama}
    atomic_write_json(args.out, summarize(report_paths))


if __name__ == "__main__":
    main()
