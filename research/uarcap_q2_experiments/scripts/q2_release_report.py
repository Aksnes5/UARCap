"""Inventory accepted and partial outputs and fill only supported Q2 paper slots."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"deployment"))
from q2_task_entry import accept
from uarcap_q2.common.io_utils import atomic_write_json, atomic_write_text, read_json, sha256_file

HANDOFF = ROOT / "laptop_final_handoff_20261126"
ARCH = ROOT / "bn_carebench_pilot_a01/overnight_20261126_artifacts"


def collect(queue_path, state_path, out, extras=()):
    queue = read_json(queue_path)
    state = read_json(state_path)["tasks"] if state_path and state_path.is_file() else {}
    records = []
    artifacts = {}
    for task in queue["tasks"]:
        tid, path = task["id"], Path(task["output"])
        st = state.get(tid,{})
        if tid.endswith("_final_snapshot"):
            continue
        r = {"task":tid,"scheduler_status":st.get("status","UNVERIFIED"),"output":str(path),
             "status":"MISSING","science_gate":task["science_gate"],"elapsed_seconds":st.get("elapsed_sec")}
        blocked = [dep for dep in task.get("depends_on",[]) if state.get(dep,{}).get("status")=="BLOCKED"]
        if blocked:
            r["direct_blockers"]=blocked
        if path.is_file():
            r["sha256"] = sha256_file(path)
            try:
                accept(task["acceptance"], path)
            except (ValueError,OSError,KeyError) as exc:
                r.update(status="PARTIAL_OR_REJECTED",reason=str(exc))
                if path.suffix==".json":
                    try:
                        doc=read_json(path)
                        r["partial_counts"]={k:v for k,v in doc.items() if k in ("status","n_approved","minimum","n_parse_failures")}
                    except (ValueError,OSError):
                        pass
            else:
                r["status"]="ARTIFACT_ACCEPTED"
                if path.suffix==".json":
                    doc=read_json(path)
                    r["counts"]={k:v for k,v in doc.items() if k.startswith("n_") or k in ("parsed_by_judge","reference_summary","per_arm","spice_status","n_candidates","query_count")}
                    if any(k in doc for k in ("estimates","contrasts","results","arm_means","paired_sensitivity")):
                        r["results"]={k:doc[k] for k in ("estimates","contrasts","results","arm_means","paired_sensitivity","judge_conflict_delta_difference") if k in doc}
                    r["input_sha256"]=doc.get("input_sha256",doc.get("inputs_sha256"))
                    r["artifact_scope"]=doc.get("status")
                    children=doc.get("artifacts", {})
                    for name, asset in (children.items() if isinstance(children,dict) else ()):
                        companion=path.parent/name
                        if isinstance(asset,dict) and companion.is_file() and asset.get("sha256")==sha256_file(companion):
                            artifacts[str(companion)]={"sha256":asset["sha256"],"bytes":companion.stat().st_size}
            artifacts[str(path)]={"sha256":r["sha256"],"bytes":path.stat().st_size}
        records.append(r)
    for arg in extras:
        p=Path(arg)
        if not p.is_file():
            continue
        artifacts[str(p)]={"sha256":sha256_file(p),"bytes":p.stat().st_size}
    output_root = out.parent
    if output_root.is_dir():
        for p in output_root.rglob("*"):
            if p.is_file() and p.suffix in (".svg",".pdf",".png") and out not in p.parents and "frames" not in p.parts:
                artifacts[str(p)]={"sha256":sha256_file(p),"bytes":p.stat().st_size}
    legacy=ROOT.parents[1]/"reports/v07_final"
    archive_refs={}
    for name,path in {
        "v08_manuscript":ROOT.parents[1]/"paper/UARCap_paper_v0.8.md",
        "v08_followup_A_frozen_statistics":legacy/"FOLLOWUP_A_FORMAL_RESULTS.json",
        "v08_followup_B_frozen_statistics":legacy/"FOLLOWUP_B_FORMAL_RESULTS.json",
        "repaired_metric_bridge":HANDOFF/"metric_bridge_repaired/E2_METRIC_BRIDGE_RECEIPT.json",
        "previous_budget":HANDOFF/"current_budget/Q2_BUDGET_REPORT.json",
        "formal_machine_baseline":ARCH/"final_analysis/stats.json",
        "source_overlap":HANDOFF/"SOURCE_AUDIT_RECEIPT.json",
    }.items():
        if path.is_file():
            archive_refs[name]={"path":str(path),"sha256":sha256_file(path)}
            if path.suffix==".json":
                archive_refs[name]["record"]=read_json(path)
    from run_final_analysis import gates
    matrix={"schema":"uarcap_q2.release_matrix.v2","version":queue["version"],"utc":datetime.now(timezone.utc).isoformat(),
            "queue_sha256":sha256_file(queue_path),"state_sha256":sha256_file(state_path) if state_path and state_path.is_file() else None,
            "tasks":records,"archived_evidence":archive_refs,"explicit_scientific_gates":gates(),
            "scope":"Post-hoc modules remain exploratory. Acceptance here verifies engineering or descriptive artifacts.",
            "allowed":["Within-system paired associations with identified condition, actual denominators and uncertainty.",
                       "Agreement on a shared claim set, restricted to the evaluated sample.",
                       "Written-reference conflict, separated from full-video truth."],
            "unsupported":["Calibrated factuality without the required independent review and gate receipts.",
                           "Causal attribution from natural retrieval correspondence.",
                           "Architectural attribution across separately configured models.",
                           "Contamination-free dataset claims from anonymous source identifiers."]}
    atomic_write_json(out/"FINAL_EXPERIMENT_MATRIX.json",matrix)
    atomic_write_json(out/"ARCHIVE_MANIFEST.json",{"schema":"uarcap_q2.archive_manifest.v2","files":artifacts})
    md=["# Q2 evidence for manuscript", "", matrix["scope"], "",
        "| Module | Artifact status | Observed denominators |", "| --- | --- | --- |"]
    for r in records:
        md.append(f"| {r['task']} | {r['status']} | "+json.dumps(r.get("counts",r.get("partial_counts",{})),ensure_ascii=False).replace("|","\\|")+" |")
    md.extend(("", "The versioned JSON contains available per-model and per-condition estimates. Missing rows remain blank.",
               "", "## Unsupported interpretations", "", *["- "+x for x in matrix["unsupported"]], ""))
    atomic_write_text(out/"MANUSCRIPT_EVIDENCE.md","\n".join(md))
    return matrix


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue",type=Path,required=True)
    parser.add_argument("--state",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--extra",type=Path,action="append",default=[])
    args=parser.parse_args()
    collect(args.queue,args.state,args.out,args.extra)
