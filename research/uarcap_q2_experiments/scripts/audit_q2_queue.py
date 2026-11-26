"""Read all contracts and exercise real argparse help locally without running jobs."""

import argparse
from collections import Counter
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"deployment"))
from q2_laptop_queue import CPU, METRIC, VLLM, validate
from uarcap_q2.common.io_utils import atomic_write_json, primitive_hash, read_json, sha256_file


def audit(path, archived=False):
    doc=read_json(path)
    tasks=doc["tasks"]
    validate(tasks)
    traces={}
    errors=[]
    for t in tasks:
        script=ROOT/t["script"]
        if not script.is_file():
            errors.append([t["id"],"SCRIPT_MISSING"])
            continue
        if t["python"] not in (CPU,METRIC,VLLM) or t["argv"][0]!=t["python"]:
            errors.append([t["id"],"UNKNOWN_RUNTIME"])
        if not archived and (sha256_file(script)!=t["script_sha256"] or primitive_hash(t["argv"])!=t["argv_sha256"]):
            errors.append([t["id"],"FINGERPRINT_DRIFT"])
        if "/home/bn/uarcap_q2_carebench" in script.read_text(encoding="utf-8"):
            errors.append([t["id"],"LEGACY_IMPORT_PATH_REQUIRES_REVIEW"])
        sub=t["argv"][3:4]
        if not sub or str(sub[0]).startswith("-"):
            sub=[]
        key=(script,tuple(sub))
        if key not in traces:
            completed=subprocess.run([sys.executable,str(script),*sub,"--help"],cwd=ROOT,
                                     capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=30)
            traces[key]=(completed.returncode, set(re.findall(r"--[a-z][a-z0-9-]*",completed.stdout)),completed.stderr)
        retval, flags, stderr=traces[key]
        expected={f for f in t["argv"] if f.startswith("--")}
        if retval or expected-flags:
            errors.append([t["id"],"CLI_MISMATCH",sorted(expected-flags),stderr[-400:]])
        target=PurePosixPath(t["output"])
        if not target.is_absolute() or not t["cwd"].endswith(f"{doc['version']}/research/uarcap_q2_experiments"):
            errors.append([t["id"],"VERSIONED_OUTPUT_OR_CWD_INVALID"])
    if len({t["output"] for t in tasks})!=len(tasks):
        errors.append(["queue","DUPLICATE_PRIMARY_OUTPUT"])
    return {"schema":"uarcap_q2.local_queue_audit.v2","queue_sha256":sha256_file(path),"version":doc["version"],
            "n_tasks":len(tasks),"n_gpu_wrapper_tasks":sum(t["requires_gpu"] for t in tasks),
            "runtime_counts":dict(Counter(t["python"] for t in tasks)),"n_actual_argparse_helps":len(traces),
            "checks":["unique IDs and primary outputs","acyclic dependencies","local scripts and applicable help flags",
                      "source fingerprints for V2", "declared interpreter class", "versioned working directories"],
            "archived_v1_hashes_not_checked_against_v2_files":archived,"errors":errors,
            "runtime_scope":"CPU inspection only; BN input assets and model calls have not been exercised"}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--archived-v1",action="store_true")
    a=p.parse_args()
    result=audit(a.queue,a.archived_v1)
    atomic_write_json(a.out,result)
    raise SystemExit(2 if result["errors"] else 0)
