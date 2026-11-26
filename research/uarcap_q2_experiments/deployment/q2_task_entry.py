"""Scheduler command wrapper. Successful exit requires validated output."""

import argparse
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from q2_laptop_queue import ROOT,GPU,UUID
from uarcap_q2.common.io_utils import read_json,read_jsonl,primitive_hash,sha256_file


def unique(rows, keys):
    values=set()
    for row in rows:
        key=tuple(row.get(k) for k in keys)
        if any(v is None for v in key) or key in values:
            raise ValueError("Output has missing or repeated key")
        values.add(key)
    return values


def accept(rule, path):
    kind=rule["kind"]
    if not path.is_file():
        raise ValueError(f"Output missing: {path}")
    if kind in ("rows","comparison_ledger","e3_evaluation"):
        rows=read_jsonl(path)
        if kind=="rows":
            key=unique(rows,rule["keys"])
            if len(key)!=rule["count"] or ("parse_status" in rule and any(r.get("parse_status")!=rule["parse_status"] for r in rows)):
                raise ValueError("Row count or parser status differs from expectation")
        elif kind=="comparison_ledger":
            actual=unique(rows,["job_id"])
            expected={(j["job_id"],) for j in read_json(rule["jobs"])["jobs"]}
            if actual!=expected:
                raise ValueError("Visual ledger incomplete")
        else:
            accepted=[r for r in rows if r["parse_status"]=="OK"]
            keys=unique(accepted,["model_id","query_id","arm","judge","target_id"])
            cohort=read_json(rule["dynamic_cohort"])
            if len(keys)!=rule.get("rows_per_candidate",16)*cohort["n_approved"]:
                raise ValueError("Incomplete E3 target pairs")
            if {k[1] for k in keys}!={r["query_id"] for r in cohort["approved"]}:
                raise ValueError("E3 evaluator query grid changed")
        return
    doc=read_json(path)
    if kind=="json":
        if doc.get("schema")!=rule["schema"] or ("eligible" in rule and doc.get("eligible")!=rule["eligible"]):
            raise ValueError("Unexpected output schema or approval")
    elif kind=="generation":
        predictions=path.parent/"predictions.jsonl"
        rows=read_jsonl(predictions)
        values=unique(rows,["model_id","query_id","arm"])
        traces=read_jsonl(path.parent/"input_traces.jsonl")
        if any(r.get("ok") is not True or not r.get("caption","").strip() or
               r.get("binding_sha256")!=doc["binding_sha256"] for r in rows):
            raise ValueError("Invalid generation binding")
        count=(read_json(rule["dynamic_cohort"])["n_approved"]*rule["arms"]
               if "dynamic_cohort" in rule else rule["count"])
        if [r["input_trace"] for r in rows]!=traces:
            raise ValueError("Trace file differs from prediction traces")
        if rule.get("smoke"):
            if doc.get("accepted_for_request") is not True or len(values)<count:
                raise ValueError("Incomplete smoke")
        elif count!=len(values) or doc.get("n_expected")!=count:
            raise ValueError("Incomplete generation")
        if "dynamic_cohort" in rule:
            qids={r["query_id"] for r in read_json(rule["dynamic_cohort"])["approved"]}
            arm_names={"E3_BASE","E3_SUP","E3_CON","E3_NEU"}
        elif "query_manifest" in rule:
            plan=Path(rule["query_manifest"])
            if plan.suffix==".jsonl":
                inputs=[r["query_id"] for r in read_jsonl(plan)]
                if plan.name=="formal_candidates.jsonl":
                    inputs.sort()
            else:
                inputs=read_json(plan)["query_ids"]
            qids=set(inputs[:rule["queries"]])
            arm_names=set(rule["arm_names"])
            if rule.get("smoke"):
                active=set(inputs[:2])
                present={(q,arm) for _,q,arm in values}
                if not {(q,arm) for q in active for arm in arm_names} <= present:
                    raise ValueError("Smoke must cover the fixed first two queries and all arms")
        else:
            qids=set()
            arm_names=set()
        if qids:
            target={(q,arm) for q in qids for arm in arm_names}
            actual={(q,arm) for _,q,arm in values}
            if len(actual)!=len(values) or not actual<=target or (not rule.get("smoke") and actual!=target):
                raise ValueError("Generation rows do not match the declared query × arm grid")
        if doc.get("predictions_sha256") and sha256_file(predictions)!=doc["predictions_sha256"]:
            raise ValueError("Prediction hash differs")
    elif kind=="cross":
        from importlib.util import spec_from_file_location,module_from_spec
        mod=ROOT/"bn_carebench_pilot_a01/overnight_20261126/tasks/a2_cross_judge.py"
        spec=spec_from_file_location("cross",mod)
        cross=module_from_spec(spec)
        spec.loader.exec_module(cross)
        expected={(r["query_id"],cross.norm(c["claim"])) for r in read_jsonl(rule["eval"])
                  if r["parse_status"]=="OK" for c in r["claims"]}
        observed=read_jsonl(path.parent/"A2_CROSS_JUDGE_LABELS.jsonl")
        got=unique(observed,["query_id","norm_key"])
        if expected!=got or len(got)!=doc.get("n_evaluated") or any(r["llama_label"] not in ("entailed","contradicted","unmentioned") for r in observed):
            raise ValueError("Incomplete/invalid cross-judge labels")
    elif kind=="metric":
        if doc["n_predictions"]!=rule["count"]:
            raise ValueError("Metric grid incomplete")
        for name,asset in doc["artifacts"].items():
            item=path.parent/name
            if item.stat().st_size!=asset["bytes"] or sha256_file(item)!=asset["sha256"]:
                raise ValueError("Metric output receipt does not match")
    else:
        raise ValueError(f"Unknown acceptance type {kind}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract",type=Path,required=True)
    p.add_argument("--task-id",required=True)
    a=p.parse_args()
    tasks=read_json(a.contract)["tasks"]
    matches=[t for t in tasks if t["id"]==a.task_id]
    if len(matches)!=1:
        p.exit(3,"HARD_BLOCK task key is absent or duplicated\n")
    task=matches[0]
    by_id={t["id"]:t for t in tasks}
    try:
        for dep in task["depends_on"]:
            contract=by_id[dep]
            accept(contract["acceptance"],Path(contract["output"]))
        for name in task["inputs"]:
            if not Path(name).exists():
                raise ValueError("Missing declared input "+name)
        if (sha256_file(ROOT/task["script"])!=task["script_sha256"]
                or primitive_hash(task["argv"])!=task["argv_sha256"]):
            raise ValueError("Deployment script or argv fingerprint changed")
    except (ValueError,OSError,KeyError) as error:
        p.exit(3,f"HARD_BLOCK {error}\n")
    if task["requires_gpu"]:
        import fcntl
        from uarcap_q2.common.gpu_lock import query_nvidia_smi_compute_pids
        deadline=time.monotonic()+min(1800,task["timeout_sec"]//3)
        while True:
            ok,active,reason=query_nvidia_smi_compute_pids(UUID)
            if not ok:
                p.exit(3,f"HARD_BLOCK cannot verify GPU occupancy: {reason}\n")
            held=True
            with open(GPU,"a+b") as lock:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    held=False
                except BlockingIOError:
                    pass
            if not active and not held:
                break
            if time.monotonic()>=deadline:
                print("GPU unavailable during cooperative wait",file=sys.stderr)
                return 2
            time.sleep(30)
    completed=subprocess.run(task["argv"],cwd=task["cwd"])
    if completed.returncode:
        return completed.returncode
    try:
        accept(task["acceptance"],Path(task["output"]))
    except (ValueError,OSError,KeyError) as error:
        print(f"INCOMPLETE_OUTPUT {error}",file=sys.stderr)
        return 2
    return 0


if __name__=="__main__":
    raise SystemExit(main())
