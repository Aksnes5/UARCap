"""Windows CPU verification and scoring. Run from a checkout containing all pinned commits."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import platform
import shutil
import statistics
import subprocess
import sys
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEP = ROOT / "research/uarcap_v06_followup_deployment"
DESIGN = DEP.parent / "uarcap_v06_followup_experiments"
AUDIT = ROOT / "reports/local_independent_review"
DELIVERY = "8ec143b9dd0a3b6f2a03fdd95b3d8ce3016230c0"
BASE = "52b5ed3a69e4b84a8e16ed747f141eefb5b1e2ec"
PROTOCOL = "ee7d3b9876dea94a1ee9eeb2626b92102c85d458"
VALIDATOR = "94f5529f8a6e4424d2af207df42482cf87c7ce1b"
INDEPENDENT = "d80b0876c8e8ef403e7528be02f75a56c378d40d"
sys.path[:0] = [str(DEP), str(DESIGN), str(AUDIT)]
from audit_archives import INPUTS, PRED_BASENAME, read, verify
from prepare_inputs import primitive_hash
from audit_statistics import holm, table


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def blob(rev, name):
    return subprocess.check_output(["git", "-C", str(ROOT), "cat-file", "blob", f"{rev}:{name}"])


def store(path, obj):
    path.write_bytes((json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def run(args):
    cmd = [sys.executable, "-B", *map(str, args)]
    subprocess.run(cmd, check=True)
    return cmd


def save_witness(out, work, data):
    store(work/"LOCAL_INTEGRITY_AND_COMMANDS.json", data)
    public = json.loads(json.dumps(data))
    for cmd in public["commands"]:
        cmd[0] = "python"
    for gate in public["local_replays"].values():
        gate["runtime"].pop("gpu_uuid", None)
    store(out/"INTEGRITY_AND_COMMANDS.json", public)


def verify_code(path, rev):
    src = blob(rev, path.relative_to(ROOT).as_posix())
    verify(path.read_bytes().replace(b"\r\n", b"\n") == src.replace(b"\r\n", b"\n"), f"Modified {path.name}")
    return sha256(src).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("work", "prepared", "metadata", "jar", "java", "out"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    work, out = args.work.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    scratch = work / "inputs"
    scratch.mkdir(parents=True, exist_ok=True)
    data = DEP / "bn_v06_full_ab"
    prior = DEP / "bn_v06_smoke"
    manifest = read(data / "FULL_AB_ARTIFACT_MANIFEST.json")
    full = read(data / "FULL_AB_SCORER_MANIFEST.json")
    receipt = read(data / "FULL_AB_RUNTIME_RECEIPTS.json")["full_arms"]
    witness = {"delivery_commit": DELIVERY, "zip_and_seal": {}, "local_replays": {}, "commands": [], "scoring_files": {}}
    verify_code(data / "FULL_AB_ARTIFACT_MANIFEST.json", DELIVERY)
    for r in manifest["files"]:
        path = data / r["path"]
        verify(path.stat().st_size == r["bytes"] and digest(path) == r["sha256"], "Delivery byte mismatch: " + r["path"])
    witness["delivery_manifest_files_verified"] = len(manifest["files"])
    pre = read(DESIGN / "CPU_PREFLIGHT_SUMMARY.json")
    for key in ("cap", "sims", "full_ids", "test_ids"):
        dst = scratch / Path(INPUTS[key]).name
        if not dst.exists():
            dst.write_bytes(blob(BASE, INPUTS[key]))
        verify(digest(dst) == pre["source_sha256"][key], f"Archive mismatch: {key}")
    refs = scratch / "reference_witness.json"
    if not refs.exists():
        rows = json.loads(blob(BASE, f"evcap_external/outputs/eval_test2990_D1/{PRED_BASENAME}"))
        store(refs, [r["captions"] for r in rows])
    prepared = scratch / "prepared"
    prepared.mkdir(exist_ok=True)
    for key, h in pre["inputs_fingerprints"].items():
        target = prepared / (key + ".json")
        if not target.exists():
            shutil.copyfile(args.prepared / target.name, target)
        verify(primitive_hash(read(target)) == h, "Prepared input mismatch: " + key)
    eligibility = prepared / "B_eligible_queries.json"
    eligible = read(eligibility)
    bw = full["b_eligible_queries"]
    verify(len(eligible) == len(set(eligible)) == bw["count"] == 2618 and digest(eligibility) == bw["sha256"]
           and primitive_hash(eligible) == bw["canonical_sha256"], "Eligibility binding")
    witness["eligibility"] = bw
    witness["scoring_files"]["validator"] = verify_code(DEP / "validate_bundle.py", VALIDATOR)
    for key in ("evaluate_followup.py", "prepare_inputs.py", "launch_pinned.py"):
        witness["scoring_files"][key] = verify_code(DESIGN / key, PROTOCOL)
    witness["scoring_files"]["audit_statistics.py"] = verify_code(AUDIT / "audit_statistics.py", INDEPENDENT)
    expected_smoke = read(data / "gates/A_SMOKE_ACCEPTED_GATE.json")
    verify(expected_smoke == read(prior / "A_SMOKE_INDEPENDENT_RECHECK_GATE.json") == read(prior / "A_SMOKE_BN_GATE.json"), "Accepted smoke lineage")
    for arm, group in expected_smoke["validated_files_sha256"].items():
        folder = prior / "raw" / arm
        seal = read(folder / "arm_completed.json")
        for role, name, hkey in (("predictions",seal["predictions"],"prediction_sha256"), ("trace","input_trace.jsonl","trace_sha256"), ("receipt","run_receipt.json","receipt_sha256")):
            # Earlier smoke text files may have CRLF in the Windows checkout.
            actual = sha256(blob(DELIVERY, (folder / name).relative_to(ROOT).as_posix())).hexdigest()
            verify(actual == group[role] == seal[hkey] and seal["rows"] == 8, arm + ": prior smoke evidence")
    witness["accepted_A_smoke_gate_sha256"] = digest(data / "gates/A_SMOKE_ACCEPTED_GATE.json")
    events = [json.loads(s) for s in (data / "runner_events.jsonl").read_text().splitlines()]
    extracted = work / "full"
    chosen = set(eligible)
    lengths = {}
    for arm, row in full["arms"].items():
        archive = data / row["zip"]
        verify(digest(archive) == row["zip_sha256"] and archive.stat().st_size == row["zip_bytes"], "ZIP mismatch")
        with zipfile.ZipFile(archive) as z:
            verify(z.namelist() == receipt[arm]["zip"]["members"] and
                   all(PurePosixPath(n).parts == (arm, PurePosixPath(n).name) and "\\" not in n for n in z.namelist()), "ZIP members")
            for n in z.namelist():
                if not (extracted / n).is_file():
                    z.extract(n, extracted)
        seal = read(extracted / arm / "arm_completed.json")
        paths = {}
        old = read(data / f"SCORER_MANIFEST_{arm[0]}.json")[arm]
        for role, hkey in (("predictions","prediction_sha256"), ("trace","trace_sha256"), ("receipt","receipt_sha256")):
            path = extracted / row[role]
            verify(PurePosixPath(old[role]).name == path.name and digest(path) == seal[hkey] == row[hkey] == receipt[arm][hkey], arm + ": " + role)
            paths[role] = path
        rec = read(paths["receipt"])
        config = read(data / "configs" / ("deployment_uarcapv06_full_authorized.json" if arm[0]=="A" else "deployment_uarcapv06_B_full_authorized.json"))
        verify(all(rec["common"][k] == config[k] for k in rec["common"] if k != "run_id"), arm + ": configuration")
        starts = [e for e in events if e["step"] == "full_" + arm]
        verify([e["status"] for e in starts] == ["START","END"] and starts[-1]["rc"] == 0 and rec["attempt"] == 1, "Attempts: " + arm)
        witness["zip_and_seal"][arm] = {"zip_bytes":row["zip_bytes"], "zip_sha256": row["zip_sha256"],
             "completion_seal_sha256": digest(extracted/arm/"arm_completed.json"), "rows":seal["rows"], "attempt":rec["attempt"]}
        if arm[0] == "B":
            rows = [json.loads(s) for s in paths["trace"].read_bytes().splitlines()]
            lengths[arm] = [r["prompt_tokens"] for r in rows if r["query"] in chosen]
    for family, mode in (("A","full"), ("B","smoke"), ("B","full")):
        folder = extracted if mode == "full" else data / "b_smoke_raw"
        tag = f"{family}_{mode.upper()}"
        dirs = work / f"{tag}_directories.json"
        store(dirs, {arm:str(folder/arm) for arm in full["families"][family]})
        target = work / f"{tag}_gate.json"
        call = [DEP/"validate_bundle.py", "--family", family, "--mode", mode, "--arm-directories",dirs, "--archive-scratch",scratch,"--gate-out",target]
        if mode == "full":
            call += ["--scorer-manifest",work / f"SCORER_MANIFEST_{family}_WINDOWS.json"]
        witness["commands"].append(run(call))
        verify(read(target) == read(data / "gates" / f"{tag}_VALIDATED_GATE.json"), "Gate mismatch: " + tag)
        witness["local_replays"][tag] = read(target)
    # Gate all inputs before scoring either family.
    verify(digest(args.jar) == "2fcb91bb7a111f93d71e264f4ee0e3afd19ba0dde6d21b38605088df9e940399" and
           digest(args.metadata) == "3540d91f9fd5f9a56d75b00c6576f30814fd19395d027791a6d438e9dc297014", "PTB or source metadata")
    vendor = work / "vendor"
    for name in ("__init__.py","cider.py","cider_scorer.py"):
        rel = "pycocoevalcap/cider/" + name
        path = vendor / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob(BASE,"evcap_external/evaluation/" + rel))
        witness["scoring_files"][rel] = digest(path)
    witness.update(python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
            java=subprocess.run([str(args.java),"-version"], capture_output=True,text=True,check=True).stderr.strip(),
            jar_sha256=digest(args.jar), metadata_sha256=digest(args.metadata), source_hashes=pre["source_sha256"],
            reference_lists_canonical_sha256=primitive_hash(read(refs)))
    save_witness(out, work, witness)
    primary = []
    for family, n, nc in (("A",2690,2004),("B",2618,1969)):
        target = work / ("score_" + family)
        runtime = work / f"SCORER_MANIFEST_{family}_WINDOWS.json"
        shutil.copyfile(runtime, out/runtime.name)
        call = [DESIGN/"evaluate_followup.py","--family",family,"--runtime-manifest",runtime,"--full-ids",scratch/Path(INPUTS["full_ids"]).name,
                "--test-ids",scratch/Path(INPUTS["test_ids"]).name,"--audit-dir",AUDIT,"--vendor",vendor,"--java",args.java,
                "--jar",args.jar,"--metadata",args.metadata,"--out",target]
        if family == "B":
            call += ["--eligible",eligibility]
        witness["commands"].append(run(call))
        metrics = read(target/"metrics.json")
        verify(metrics["sample_count"] == n and metrics["clusters"]["n_clusters"] == nc, "Cohort size")
        with (target/"paired_ptb.csv").open(newline="") as f:
            tests = [{k:(v if k=="test" else int(v) if k=="n" else float(v)) for k,v in r.items()} for r in csv.DictReader(f)]
        metrics.update(family=family, unit="CIDEr points x100", seed=20261125, resamples=10000, paired_results=tests)
        if family == "B":
            metrics.update(matching=pre["B_matching_metrics"], prompt_tokens={
                arm:{"mean":statistics.mean(xs),"median":statistics.median(xs),"min":min(xs),"max":max(xs)} for arm,xs in lengths.items()})
        for r in tests:
            primary.append({"family":family,"contrast":r["test"],"queries":n,"sources":nc,"effect":r["diff"],
                            "ci_low":r["cluster_lo"],"ci_high":r["cluster_hi"],"p_raw":r["cluster_p"],"p_family_holm":r["cluster_holm"]})
        store(out/f"FOLLOWUP_{family}_FORMAL_RESULTS.json",metrics)
    table(out/"FOLLOWUP_PRIMARY_CONTRASTS.csv",primary)
    for row, p in zip(primary, holm([r["p_raw"] for r in primary]), strict=True):
        row["p_five_holm"] = float(p)
        row["clear_practical_effect"] = bool(p < .05 and (row["ci_low"] > 1 or row["ci_high"] < -1))
    table(out/"FOLLOWUP_FIVE_TEST_HOLM_RESULTS.csv",primary)
    save_witness(out, work, witness)


if __name__ == "__main__":
    main()
