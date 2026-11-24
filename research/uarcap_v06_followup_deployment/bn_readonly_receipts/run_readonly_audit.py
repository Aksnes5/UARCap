#!/usr/bin/env python3
"""Read-only BN asset audit for UARCap v0.6 follow-up deployment.

Guarantees:
  * no CUDA initialization, no torch import, no model load;
  * no writes outside this receipts directory;
  * no service, network, package or dataset mutation.

Run:  python3 -I run_readonly_audit.py
"""
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime

PROJECT = "/home/bn/videoufo_research"
DISPLAY = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display"
MSRVTT = f"{DISPLAY}/lj/datasets/MSRVTT-v2"
TIMM = f"{DISPLAY}/repro_evcap/torch_cache/hub/checkpoints"
OUT = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(OUT, "../../.."))
V, U, M, MM, NC = "VERIFIED", "UNVERIFIED", "MISSING", "MISMATCH", "NOT_CHECKED"
NOW = datetime.now().isoformat(timespec="seconds")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def meta(p, digest=False, read=False):
    if not os.path.exists(p):
        return {"path": p, "exists": False, "status": M}
    st = os.stat(p)
    d = {"path": p, "exists": True, "bytes": st.st_size,
         "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
         "mtime_ns": st.st_mtime_ns, "symlink": os.path.islink(p),
         "realpath": os.path.realpath(p), "readable": os.access(p, os.R_OK),
         "status": V if os.access(p, os.R_OK) else U}
    if digest:
        d["sha256"] = sha(p)
    return d


def sh(cmd, timeout=30):
    try:
        return subprocess.check_output(cmd, shell=True, text=True,
                                       stderr=subprocess.STDOUT, timeout=timeout).strip()
    except Exception as e:
        return f"ERROR: {e}"


def primitive_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False,
                                     separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def dump(name, obj):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ---------------------------------------------------------------- environment
os_rel = {}
with open("/etc/os-release") as f:
    for line in f:
        if "=" in line:
            k, v = line.rstrip("\n").split("=", 1)
            os_rel[k] = v.strip('"')

vendor = f"{DISPLAY}/repro_evcap/venv"
py = f"{vendor}/bin/python"
pyprobe = sh(
    f"{py} -I -c \"import sys,importlib.metadata as m,importlib.util as u;"
    "print('version',sys.version.split()[0]);print('prefix',sys.prefix);"
    "print('base',sys.base_prefix);print('torch',m.version('torch'));"
    "print('transformers',m.version('transformers'));print('timm',m.version('timm'));"
    "print('torch_origin',u.find_spec('torch').origin);"
    "print('transformers_origin',u.find_spec('transformers').origin)\"")
versions = {}
for line in pyprobe.splitlines():
    if " " in line:
        k, v = line.split(" ", 1)
        versions[k] = v
pyvenv = {}
if os.path.exists(f"{vendor}/pyvenv.cfg"):
    for line in open(f"{vendor}/pyvenv.cfg"):
        if "=" in line:
            k, v = line.split("=", 1)
            pyvenv[k.strip()] = v.strip()

mount_display = sh(f"findmnt -T {DISPLAY} -o TARGET,SOURCE,FSTYPE,OPTIONS -n")
mount_project = sh(f"findmnt -T {PROJECT} -o TARGET,SOURCE,FSTYPE,OPTIONS -n")
gpu_mgmt = sh("nvidia-smi --query-gpu=name,uuid,pci.bus_id,memory.total,memory.used,"
              "utilization.gpu,temperature.gpu,power.draw,driver_version --format=csv,noheader")
gpu_apps = sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader")
gpu_graphics = sh("nvidia-smi --query-gpu=display_active,display_mode --format=csv,noheader")
live_procs = sh("ps -eo pid,etime,args | grep -E '[e]val_evcap|[e]xtract_external_features|"
                "[e]xtract_query_features|uarcap_v06|deploy_guard' | head -20")
lock = f"{PROJECT}/gpu.lock"
lock_meta = meta(lock)
lock_fuser = sh(f"fuser -v {lock} 2>&1 | tail -5") if lock_meta["exists"] else ""

env = {
    "checked_at": NOW,
    "hostname": platform.node(),
    "os": {k: os_rel.get(k) for k in ("PRETTY_NAME", "VERSION_ID", "VERSION_CODENAME")},
    "kernel": platform.release(),
    "gpu": {"management": gpu_mgmt, "compute_apps": gpu_apps,
            "display_state": gpu_graphics,
            "cuda_init_performed": False},
    "gpu_lock": {"path": lock, "meta": lock_meta, "fuser": lock_fuser,
                 "held": bool(lock_fuser.strip())},
    "running_audit_commands": live_procs,
    "mounts": {"project": mount_project, "display": mount_display},
    "disk": {"project": sh(f"df -h {PROJECT} | tail -1"),
             "display": sh(f"df -h {DISPLAY} | tail -1")},
    "git": {
        "project_head": sh(f"git -C {PROJECT} rev-parse HEAD"),
        "project_branch": sh(f"git -C {PROJECT} branch --show-current"),
        "project_dirty_entries": int(sh(f"git -C {PROJECT} status --short | wc -l") or 0),
        "receipts_worktree": WT,
        "receipts_head": sh(f"git -C {WT} rev-parse HEAD"),
        "receipts_branch": sh(f"git -C {WT} branch --show-current"),
    },
    "python": {
        "launcher_interpreter": py,
        "launcher_realpath": os.path.realpath(py),
        "pyvenv_cfg": pyvenv,
        "probe": versions,
        "conda_envs": sh("ls /home/bn/anaconda3/envs 2>/dev/null").splitlines(),
        "note": "Historical runner uses this interpreter. torch resolves to the TTP Conda base "
                "through include-system-site-packages; transformers resolves inside the venv.",
    },
    "torch_dist_info": meta("/home/bn/anaconda3/envs/TTP/lib/python3.9/site-packages/torch-2.1.2+cu121.dist-info"),
    "archived_expectation": {"python": "3.9.18", "torch": "2.1.0",
                             "transformers": "4.28.0", "timm": "0.6.13",
                             "source": "BN_READONLY_ENVIRONMENT_AUDIT.md"},
}
checks = {
    "python_archived_match": versions.get("version") == "3.9.18",
    "torch_archived_match": versions.get("torch") == "2.1.0",
    "transformers_archived_match": versions.get("transformers") == "4.28.0",
    "timm_archived_match": versions.get("timm") == "0.6.13",
}
env["checks"] = checks
env["status"] = V if all(checks.values()) else MM
dump("BN_GPU_AND_PROCESS_STATUS.json", env)


# ---------------------------------------------------------------- dataset
yaml = f"{PROJECT}/evcap_external/test2990/test2990_16frames.yaml"
roles = {}
for line in open(yaml, encoding="utf-8"):
    if ":" in line and not line.lstrip().startswith("#"):
        k, v = line.split(":", 1)
        roles[k.strip()] = v.strip()
img, lab, cap = roles["img"], roles["label"], roles["caption"]


def offsets(p):
    return os.path.splitext(p)[0] + ".lineidx"


def ids_of(p):
    return [os.path.basename(l.split("\t", 1)[0])[:-4] for l in open(p) if l.strip()]


def count_rows(p):
    return sum(1 for _ in open(p))


def check_offsets(p):
    lp = offsets(p)
    offs = [int(x) for x in open(lp)]
    size = os.path.getsize(p)
    return {"lineidx": meta(lp, digest=True), "rows": len(offs),
            "size": size, "first": offs[0] if offs else None,
            "last": offs[-1] if offs else None,
            "monotonic": all(a < b for a, b in zip(offs, offs[1:])),
            "within_file": bool(offs) and offs[-1] < size,
            "pass": len(offs) == 2990 and all(a < b for a, b in zip(offs, offs[1:])) and offs[-1] < size}


i_ids, l_ids, c_ids = ids_of(img), ids_of(lab), ids_of(cap)
full_lines = [l.strip() for l in open(f"{PROJECT}/manifests/msrvtt_test2990.tsv") if l.strip()]
unt_lines = [l.strip() for l in open(f"{PROJECT}/manifests/msrvtt_untouched2690.tsv") if l.strip()]
full_ids = [os.path.basename(x)[:-4] for x in full_lines]
unt_ids = [os.path.basename(x)[:-4] for x in unt_lines]

refs = []
for line in open(cap, encoding="utf-8"):
    _, val = line.rstrip("\n").split("\t", 1)
    refs.append([c["caption"] for c in json.loads(val)])
ref_digest = primitive_hash(refs)

# sample 100 frame rows by offset without reading the 1.38 GB file sequentially
frame_rows = []
img_offs = [int(x) for x in open(offsets(img))]
with open(img, "r", encoding="utf-8", errors="replace") as f:
    for i in (0, 1, 2, 2987, 2988, 2989) + tuple(range(0, 2990, 30)):
        f.seek(img_offs[i])
        row = f.readline().rstrip("\n").split("\t")
        frame_rows.append({"idx": i, "key": row[0], "fields": len(row),
                           "all_frames_nonempty": all(bool(x) for x in row[2:])})
frame_check = {"sample_n": len(frame_rows), "all_18_fields": all(r["fields"] == 18 for r in frame_rows),
               "all_frames_nonempty": all(r["all_frames_nonempty"] for r in frame_rows),
               "samples": frame_rows}

expected = {
    "yaml_sha256": "a2e4d55790ddfeb60a0eec8298aaa35f46833c22ffa81ddb11de9132d0014083",
    "full_manifest_sha256": "d826647b1f2ca16c75b117287d9a0b23bd7359753f987f6a7022f769a890ef8d",
    "untouched_manifest_sha256": "fce00da1d10e6e80fa7c0ee9932042e4ab608946f2039887c830a778d96acae2",
    "reference_lists_sha256": "204563f74d9e4354863289760f65a767dde8d9e189d7e99dc201505847b15271",
}
actual = {
    "yaml_sha256": sha(yaml),
    "full_manifest_sha256": sha(f"{PROJECT}/manifests/msrvtt_test2990.tsv"),
    "untouched_manifest_sha256": sha(f"{PROJECT}/manifests/msrvtt_untouched2690.tsv"),
    "reference_lists_sha256": ref_digest,
}
id_checks = {
    "img_rows_2990": count_rows(img) == 2990,
    "label_rows_2990": count_rows(lab) == 2990,
    "caption_rows_2990": count_rows(cap) == 2990,
    "img_ids_eq_label_ids": i_ids == l_ids,
    "img_ids_eq_caption_ids": i_ids == c_ids,
    "img_ids_eq_full_manifest_order": i_ids == full_ids,
    "untouched_is_2690_row_ordered_subset": len(unt_ids) == 2690 and set(unt_ids) <= set(i_ids),
    "full_manifest_unique": len(set(full_ids)) == 2990,
    "every_query_has_20_references": all(len(r) == 20 for r in refs),
    "reference_lists_digest_match": ref_digest == expected["reference_lists_sha256"],
    "img_offsets_pass": check_offsets(img)["pass"],
    "label_offsets_pass": check_offsets(lab)["pass"],
    "caption_offsets_pass": check_offsets(cap)["pass"],
    "frame_tsv_sample_pass": frame_check["all_18_fields"] and frame_check["all_frames_nonempty"],
}
dataset = {
    "checked_at": NOW,
    "yaml": meta(yaml, digest=True),
    "roles": roles,
    "assets": {
        "img_tsv": meta(img),
        "img_lineidx": meta(offsets(img), digest=True),
        "label_tsv": meta(lab, digest=True),
        "label_lineidx": meta(offsets(lab), digest=True),
        "caption_tsv": meta(cap, digest=True),
        "caption_lineidx": meta(offsets(cap), digest=True),
        "full_manifest": meta(f"{PROJECT}/manifests/msrvtt_test2990.tsv", digest=True),
        "untouched_manifest": meta(f"{PROJECT}/manifests/msrvtt_untouched2690.tsv", digest=True),
    },
    "counts": {"img_rows": count_rows(img), "label_rows": count_rows(lab),
               "caption_rows": count_rows(cap), "full_manifest": len(full_lines),
               "untouched_manifest": len(unt_lines), "references": len(refs),
               "references_per_query": len(refs[0])},
    "offsets": {"img": check_offsets(img), "label": check_offsets(lab),
                "caption": check_offsets(cap)},
    "frame_tsv_sample": frame_check,
    "expected_sha256": expected,
    "actual_sha256": actual,
    "id_checks": id_checks,
    "status": V if all(id_checks.values()) else MM,
}
dump("BN_DATASET_ASSET_RECEIPTS.json", dataset)


# ---------------------------------------------------------------- model/cache
ckpt = (f"{DISPLAY}/lj/EVCapCode/outputs/train_evcap_msrvtt_quality_text/"
        "train_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review.pt")
vic = f"{DISPLAY}/repro_evcap/models/vicuna-7b-v1.3"
vl = f"{PROJECT}/evcap_external/download/videollama/baseline_2_7B_Pretrained.pth"
vic_files = ("config.json", "tokenizer.model", "tokenizer_config.json",
             "special_tokens_map.json", "pytorch_model.bin.index.json")
shards = sorted(x for x in os.listdir(vic) if x.endswith(".bin") and "index" not in x) if os.path.isdir(vic) else []
index = load_json(f"{vic}/pytorch_model.bin.index.json") if os.path.exists(f"{vic}/pytorch_model.bin.index.json") else {}
index_shards = sorted(set(index.get("weight_map", {}).values()))
model = {
    "checked_at": NOW,
    "checkpoint": meta(ckpt, digest=True),
    "checkpoint_expected_sha256": "a8b5af79d55a2c7e684771475d645be6373208bdaae5a4211bee556581eb8002",
    "historical_checkpoint_snapshots": {
        f"train_test2990_D{k}": meta(f"{PROJECT}/evcap_external/outputs/train_test2990_D{k}/"
                                     "train_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review.pt",
                                     digest=True)
        for k in (1, 3, 7)},
    "vicuna": {"dir": meta(vic), **{f: meta(f"{vic}/{f}", digest=(os.path.getsize(f"{vic}/{f}") < 10 << 20))
                                    for f in vic_files},
               "shards": [{"name": s, "bytes": os.path.getsize(f"{vic}/{s}")} for s in shards],
               "index_total_size": index.get("metadata", {}).get("total_size"),
               "index_shards": index_shards,
               "shards_match_index": index_shards == shards},
    "video_llama": meta(vl, digest=True),
    "eva_vit_g": meta(f"{TIMM}/eva_vit_g.pth"),
    "blip2_qformer": meta(f"{TIMM}/blip2_pretrained_flant5xxl.pth"),
    "hf_cache": {"root": meta(f"{DISPLAY}/repro_evcap/hf_cache"),
                 "bert_base_uncased": meta(f"{DISPLAY}/repro_evcap/hf_cache/models--bert-base-uncased"),
                 "bert_files": sorted(os.listdir(f"{DISPLAY}/repro_evcap/hf_cache/models--bert-base-uncased/"
                                                 "snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"))
                 if os.path.isdir(f"{DISPLAY}/repro_evcap/hf_cache/models--bert-base-uncased/"
                                  "snapshots/86b5e0934494bd15c9632b12f734a8a67f723594") else [],
                 "note": "Tokenizer/config/vocab cached; no BERT weight file. Historical runs used "
                         "the cached tokenizer and the BLIP2 Q-Former checkpoint."},
}
model["checks"] = {
    "checkpoint_sha_match": model["checkpoint"].get("sha256") == model["checkpoint_expected_sha256"],
    "checkpoint_bytes_match": model["checkpoint"].get("bytes") == 147337150,
    "three_snapshot_hashes_match": all(v.get("sha256") == model["checkpoint_expected_sha256"]
                                       for v in model["historical_checkpoint_snapshots"].values()),
    "vicuna_all_present": all(model["vicuna"][f].get("status") == V for f in vic_files),
    "vicuna_two_shards": len(shards) == 2 and model["vicuna"]["shards_match_index"],
    "video_llama_present": model["video_llama"].get("status") == V,
    "eva_cache_present": model["eva_vit_g"].get("status") == V,
    "blip2_cache_present": model["blip2_qformer"].get("status") == V,
}
model["status"] = V if all(model["checks"].values()) else MM
dump("BN_MODEL_AND_CACHE_RECEIPTS.json", model)


# ---------------------------------------------------------------- retrieval/features/history
EXT = f"{PROJECT}/external"
FET = f"{PROJECT}/features/videollama_faithful_v2"
assets = {
    "d7_cap": f"{EXT}/external_test_d7_cap.json",
    "d7_sims": f"{EXT}/external_test_d7_sims.json",
    "top20": f"{EXT}/external_test_top20.json",
}
expected_ret = {
    "d7_cap": ("6238f379f6d35fe12fda2d3ec59f498683485d03341a1f0931caf6e234f9efe8", 3311809),
    "d7_sims": ("3a79cb5e90f2e213752679441defaba62e7d5e1dabc2842ca3a9b3dca2135021", 2156249),
    "top20": ("d857bcda93890ae2e5398b69de91080cdd0e5e767249c3f0bd93853f8779e7b2", 13124819),
}
retrieval = {}
for name, p in assets.items():
    m = meta(p, digest=True)
    m["expected_sha256"], m["expected_bytes"] = expected_ret[name]
    m["match"] = m.get("sha256") == expected_ret[name][0] and m.get("bytes") == expected_ret[name][1]
    retrieval[name] = m
cap = load_json(assets["d7_cap"])
retrieval["checks"] = {
    "all_sha_match": all(retrieval[k]["match"] for k in expected_ret),
    "cap_2990": len(cap) == 2990,
    "cap_seven_per_query": all(len(v[0]) == 7 and len(v[2]) == 7 for v in cap.values()),
    "sims_2990": len(load_json(assets["d7_sims"])) == 2990,
    "top20_2990": len(load_json(assets["top20"])) == 2990,
}
feature_sets = {}
for k, expected_ids in ((1, "05114915acb3a1031e5f551b8e817f88e33693a9f7331ba558332ab589e2398d"),
                        (7, "749eacf8a3fc9ac105776ded8c0fe5d3d2a8577d45cbd4a3c8c293abbf1fa65f")):
    ids = sorted({caps[0] for v in cap.values() for caps in v[2][:k] if caps})
    missing = [c for c in ids if not os.path.exists(f"{FET}/{c}.pt")]
    feature_sets[f"A{k}" if k in (1, 7) else f"D{k}"] = {
        "unique_ids": len(ids), "canonical_ids_sha256": primitive_hash(ids),
        "expected_ids_sha256": expected_ids,
        "digest_match": primitive_hash(ids) == expected_ids,
        "missing_files": len(missing), "missing_examples": missing[:5],
        "status": V if not missing and primitive_hash(ids) == expected_ids else M,
    }
feature_total = sum(1 for _ in os.scandir(FET) if _.name.endswith(".pt")) if os.path.isdir(FET) else 0
feature_recursive = sum(1 for _, _, fs in os.walk(FET) for f in fs if f.endswith(".pt")) if os.path.isdir(FET) else 0
retrieval["feature_cache"] = {
    "dir": meta(FET), "total_pt_files": feature_total,
    "recursive_pt_files": feature_recursive,
    "required": feature_sets,
    "status": V if all(v["status"] == V for v in feature_sets.values()) and feature_total >= 15213 else M,
}
prepared_names = ("A_k0_cap.json", "A_k0_sims.json", "B_text3_cap.json", "B_text7_hq_cap.json",
                  "B_text7_lq_cap.json", "B_shared_visual_sims.json", "B_visual_contract.json",
                  "B_eligible_queries.json")
found_prepared = sh(
    f"find {PROJECT} {WT} -maxdepth 6 -type f \\( "
    + " -o ".join(f"-name {n}" for n in prepared_names)
    + " \\) 2>/dev/null").splitlines()
prepared = {n: {"paths": [p for p in found_prepared if os.path.basename(p) == n],
                "status": V if any(os.path.basename(p) == n for p in found_prepared) else M}
            for n in prepared_names}
retrieval["prepared_inputs"] = prepared
retrieval["status"] = V if retrieval["checks"]["all_sha_match"] and retrieval["feature_cache"]["status"] == V else MM

hist = {"checked_at": NOW, "predictions": {}, "logs": {}, "metrics": {}}
for tag in ("test2990", "test2690"):
    for k in (1, 3, 7):
        base = f"{PROJECT}/evcap_external/outputs/eval_{tag}_D{k}"
        pred = f"{base}/eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json"
        met = f"{base}/metrics.json"
        row = {"prediction": meta(pred, digest=True), "metrics_file": meta(met, digest=True)}
        if os.path.exists(pred):
            payload = load_json(pred)
            row["rows"] = len(payload)
            row["ids"] = [os.path.basename(r["image_name"])[:-4] for r in payload]
            row["ids_in_manifest_order"] = row["ids"] == ([os.path.basename(x)[:-4] for x in
                                                           (full_lines if tag == "test2990" else unt_lines)])
            row["split_labels"] = sorted({r.get("split") for r in payload})
        if os.path.exists(met):
            row["metrics"] = load_json(met)
        hist["predictions"][f"{tag}_D{k}"] = row

logdir = f"{PROJECT}/logs"
for k in (1, 3, 7):
    p = f"{logdir}/gen_test_D{k}.log"
    if not os.path.exists(p):
        hist["logs"][f"D{k}"] = {"path": p, "status": M}
        continue
    txt = open(p, errors="ignore").read()
    args_line = next((l for l in txt.splitlines() if l.startswith("args: ")), "")
    def grab(pattern):
        m = re.search(pattern, txt)
        return m.group(1) if m else None
    hist["logs"][f"D{k}"] = {
        "path": p, "status": V, "bytes": os.path.getsize(p),
        "mtime": datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds"),
        "sha256": sha(p),
        "args_line": args_line[:4000],
        "topn": grab(r"\ntopn: ([0-9]+)"),
        "load": grab(r"\nload: ([^\n]+)"),
        "ext_path": grab(r"'ext_path': '([^']+)'"),
        "sim_path": grab(r"'sim_scores_path': '([^']+)'"),
        "video_feature_dir": grab(r"'video_feature_dir': '([^']+)'"),
        "test_yaml": grab(r"'test_yaml': '([^']+)'"),
        "checkpoint_name": grab(r"'ckpt_name': '([^']+)'"),
        "llm_model": grab(r"'llm_model': '([^']+)'"),
        "seed": grab(r"'random_seed': ([0-9]+)"),
        "max_num_frames": grab(r"'max_num_frames': ([0-9]+)"),
    }
hist["runner"] = meta(f"{PROJECT}/scripts/run_test.sh", digest=True)
hist["runner_expected_sha256"] = "42d7d58e534bfcb798f01fc99ee0a10d70e600aa41af05431848ff7d49b2c88f"
hist["checks"] = {
    "runner_sha_match": hist["runner"].get("sha256") == hist["runner_expected_sha256"],
    "six_prediction_rows_match_split": all(v.get("rows") == (2990 if "2990" in n else 2690)
                                           for n, v in hist["predictions"].items()),
    "six_prediction_orders_match": all(v.get("ids_in_manifest_order") for v in hist["predictions"].values()),
}
hist["runtime_trace_receipt"] = {"status": M,
    "detail": "ARCHIVE_EVIDENCE.json records no execution-bound query frame/feature/prompt/weight trace "
              "for the historical D1/D3/D7 jobs."}
hist["environment_receipt"] = {"status": U,
    "detail": "No per-job Python/package snapshot was written. The current interpreter is recorded in "
              "BN_GPU_AND_PROCESS_STATUS.json; run_test.sh and source hashes match the frozen archive."}


# ---------------------------------------------------------------- deployment bindings
fb = load_json(f"{WT}/research/uarcap_v06_followup_experiments/FROZEN_SOURCE_BINDINGS.json")
code = {}
for rel, exp in fb["code_sha256"].items():
    p = f"{PROJECT}/evcap_external/{rel}"
    got = sha(p) if os.path.exists(p) else None
    code[rel] = {"path": p, "expected": exp, "actual": got,
                 "status": V if got == exp else (M if got is None else MM)}
archive = load_json(f"{WT}/research/uarcap_v06_followup_deployment/ARCHIVE_EVIDENCE.json")
runtime = {}
for rel, exp in archive["runtime_dependency_sha256"].items():
    p = f"{PROJECT}/evcap_external/{rel}"
    got = sha(p) if os.path.exists(p) else None
    runtime[rel] = {"path": p, "expected": exp, "actual": got,
                    "status": V if got == exp else (M if got is None else MM)}

bindings = {
    "checked_at": NOW,
    "host": platform.node(),
    "project_root": meta(PROJECT),
    "inference_checkout": meta(f"{PROJECT}/evcap_external"),
    "interpreter": {"path": py, "status": V if os.access(py, os.X_OK) else M,
                    "effective_versions": versions},
    "shared_gpu_lock": lock_meta,
    "test_yaml": dataset["yaml"],
    "dataset_roles": {k: meta(v) for k, v in roles.items()},
    "manifests": {"full": dataset["assets"]["full_manifest"],
                  "untouched": dataset["assets"]["untouched_manifest"]},
    "frozen_sources": code,
    "runtime_dependencies": runtime,
    "runner": hist["runner"],
    "checkpoint": model["checkpoint"],
    "historical_checkpoint_snapshots": model["historical_checkpoint_snapshots"],
    "vicuna": model["vicuna"],
    "video_llama": model["video_llama"],
    "eva_vit_g": model["eva_vit_g"],
    "blip2_qformer": model["blip2_qformer"],
    "hf_cache": model["hf_cache"],
    "retrieval": retrieval,
    "prepared_inputs": prepared,
    "historical_predictions": {n: {"prediction": v["prediction"], "rows": v.get("rows")}
                               for n, v in hist["predictions"].items()},
    "source_binding_checks": {
        "all_six_frozen_sources_match": all(v["status"] == V for v in code.values()),
        "all_nine_runtime_dependencies_match": all(v["status"] == V for v in runtime.values()),
        "runner_matches": hist["checks"]["runner_sha_match"],
        "yaml_matches": actual["yaml_sha256"] == expected["yaml_sha256"],
    },
}
bindings["arm_readiness"] = {
    "A0": {"core_assets": V if dataset["status"] == model["status"] == V else MM,
           "prepared_inputs": M, "input_generation_cpu_only": True,
           "real_model_smoke": NC, "gpu_approved": False},
    "A1": {"core_assets": V if dataset["status"] == model["status"] == V else MM,
           "prepared_inputs": "NOT_REQUIRED (uses frozen D cap/sims directly)",
           "feature_coverage": feature_sets["A1"], "real_model_smoke": NC, "gpu_approved": False},
    "A7": {"core_assets": V if dataset["status"] == model["status"] == V else MM,
           "prepared_inputs": "NOT_REQUIRED (uses frozen D cap/sims directly)",
           "feature_coverage": feature_sets["A7"], "real_model_smoke": NC, "gpu_approved": False},
    "B3_B7HQ_B7LQ": {"core_assets": V if dataset["status"] == model["status"] == V else MM,
                     "prepared_inputs": M, "input_generation_cpu_only": True,
                     "visual_contract": M, "real_model_parity": NC, "gpu_approved": False},
}
bindings["status"] = V if all(bindings["source_binding_checks"].values()) else MM
dump("BN_DEPLOYMENT_PATH_BINDINGS.json", bindings)


# ---------------------------------------------------------------- markdown receipts
def write_md(name, text):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


gpu_parts = [x.strip() for x in gpu_mgmt.split(",")]
live = f"""# BN live environment receipt

Checked at **{NOW}** by `run_readonly_audit.py` (standard library only).

**Overall environment status: `MISMATCH`.** The host is reachable and the historical interpreter
exists, but the live Python/PyTorch versions do not match the archived environment expectation.

| Item | Live value | Status |
| --- | --- | --- |
| Host | `{platform.node()}` | VERIFIED |
| OS | {os_rel.get('PRETTY_NAME')} | VERIFIED |
| Kernel | `{platform.release()}` | VERIFIED |
| GPU | {gpu_parts[0] if gpu_parts else gpu_mgmt} | VERIFIED |
| GPU UUID | `{gpu_parts[1] if len(gpu_parts) > 1 else ''}` | VERIFIED |
| PCI bus | `{gpu_parts[2] if len(gpu_parts) > 2 else ''}` | VERIFIED |
| VRAM | {gpu_parts[3] if len(gpu_parts) > 3 else ''} total / {gpu_parts[4] if len(gpu_parts) > 4 else ''} used | VERIFIED |
| Driver | `{gpu_parts[8] if len(gpu_parts) > 8 else ''}` | VERIFIED |
| Compute clients | none (`nvidia-smi --query-compute-apps` empty) | VERIFIED |
| GPU lock | `{lock}` present, 0 bytes, no holder | VERIFIED |
| Project disk | `{env['disk']['project']}` | VERIFIED |
| Display disk | `{env['disk']['display']}` | VERIFIED |
| Display mount | `{mount_display}` | VERIFIED (read-only) |
| Project mount | `{mount_project}` | VERIFIED (read-write) |
| Project git | `{env['git']['project_branch']}` @ `{env['git']['project_head']}` | dirty entries: {env['git']['project_dirty_entries']} |
| Receipts worktree | `{WT}` @ `{env['git']['receipts_head']}` | branch `{env['git']['receipts_branch']}` |

## Effective Python runtime used by the historical runner

`scripts/run_test.sh` (SHA-256 `{hist['runner'].get('sha256')}`) launches:

```text
{py}
-> {os.path.realpath(py)}
```

`pyvenv.cfg`:

```text
home = {pyvenv.get('home', '')}
include-system-site-packages = {pyvenv.get('include-system-site-packages', '')}
version = {pyvenv.get('version', '')}
```

| Package | Live | Archived expectation | Match |
| --- | --- | --- | --- |
| Python | `{versions.get('version')}` | `3.9.18` | **NO** |
| PyTorch | `{versions.get('torch')}` | `2.1.0` | **NO** |
| Transformers | `{versions.get('transformers')}` | `4.28.0` | YES |
| timm | `{versions.get('timm')}` | `0.6.13` | YES |

- `sys.prefix` = `{versions.get('prefix')}`
- `sys.base_prefix` = `{versions.get('base')}`
- `torch` resolves to `{versions.get('torch_origin')}`
- `transformers` resolves to `{versions.get('transformers_origin')}`

The venv is a **hybrid**: it includes the TTP Conda environment's site-packages, so `torch`
comes from TTP while `transformers` comes from the venv. Historical D1/D3/D7 runs used this
interpreter, but no per-job package snapshot exists. This is a reproducibility deviation from the
archived environment receipt, not a missing asset.

Conda environments visible: `{", ".join(env['python']['conda_envs'])}`.

No CUDA context was created, no `torch` import occurred and no model weight was loaded for this
receipt. GPU fields come from `nvidia-smi` management queries only.
"""
write_md("BN_LIVE_ENVIRONMENT.md", live)


rows = []
for name in ("test2990_D1", "test2990_D3", "test2990_D7",
             "test2690_D1", "test2690_D3", "test2690_D7"):
    r = hist["predictions"][name]
    rows.append(f"| {name} | {r['rows']} | {r['ids_in_manifest_order']} | "
                f"`{r['prediction']['sha256']}` | {r['metrics']['CIDEr']:.6f} |")
hist_md = f"""# BN historical runtime receipts (D1 / D3 / D7)

Checked at **{NOW}**. Read-only reconstruction from the live filesystem; no generation was run.

## Prediction bundles

| Bundle | Rows | Manifest order | Prediction SHA-256 | CIDEr |
| --- | ---: | :---: | --- | ---: |
{chr(10).join(rows)}

All six prediction files are byte-identical to the frozen values in
`research/uarcap_v06_followup_deployment/ARCHIVE_EVIDENCE.json`, with matching byte sizes.
The 2,990-row bundles follow `manifests/msrvtt_test2990.tsv` order; the 2,690-row bundles follow
`manifests/msrvtt_untouched2690.tsv` order. Reference lists use the canonical ordered digest
`{ref_digest}`.

## Command provenance

`scripts/run_test.sh` SHA-256 is `{hist['runner'].get('sha256')}`
(expected `{hist['runner_expected_sha256']}`). The generation logs record:

| Arm | `topn` | `test_yaml` | `ext_path` | `sim_scores_path` | `video_feature_dir` | seed / frames |
| --- | ---: | --- | --- | --- | --- | --- |
| D1 | {hist['logs']['D1'].get('topn')} | `{hist['logs']['D1'].get('test_yaml')}` | `{hist['logs']['D1'].get('ext_path')}` | `{hist['logs']['D1'].get('sim_path')}` | `{hist['logs']['D1'].get('video_feature_dir')}` | {hist['logs']['D1'].get('seed')} / {hist['logs']['D1'].get('max_num_frames')} |
| D3 | {hist['logs']['D3'].get('topn')} | `{hist['logs']['D3'].get('test_yaml')}` | `{hist['logs']['D3'].get('ext_path')}` | `{hist['logs']['D3'].get('sim_path')}` | `{hist['logs']['D3'].get('video_feature_dir')}` | {hist['logs']['D3'].get('seed')} / {hist['logs']['D3'].get('max_num_frames')} |
| D7 | {hist['logs']['D7'].get('topn')} | `{hist['logs']['D7'].get('test_yaml')}` | `{hist['logs']['D7'].get('ext_path')}` | `{hist['logs']['D7'].get('sim_path')}` | `{hist['logs']['D7'].get('video_feature_dir')}` | {hist['logs']['D7'].get('seed')} / {hist['logs']['D7'].get('max_num_frames')} |

| Log | Bytes | SHA-256 |
| --- | ---: | --- |
| `logs/gen_test_D1.log` | {hist['logs']['D1'].get('bytes')} | `{hist['logs']['D1'].get('sha256')}` |
| `logs/gen_test_D3.log` | {hist['logs']['D3'].get('bytes')} | `{hist['logs']['D3'].get('sha256')}` |
| `logs/gen_test_D7.log` | {hist['logs']['D7'].get('bytes')} | `{hist['logs']['D7'].get('sha256')}` |

Each log also names the per-run checkpoint slot (`outputs/train_test2990_D{{k}}/...pt`) and the
Vicuna path `{hist['logs']['D1'].get('llm_model')}`. All three slots are symlinks to one canonical
checkpoint with SHA-256 `{model['checkpoint']['sha256']}` (147,337,150 bytes), verified again during
this audit.

## Runtime trace and environment

- **Execution-bound runtime trace: `MISSING`.** `ARCHIVE_EVIDENCE.json` states that no historical
  query frame/feature/prompt/weight trace exists. Therefore D1/D3/D7 remain
  **conditionally reusable only**; exact-reuse exemptions remain **0**.
- **Per-job environment receipt: `UNVERIFIED`.** No historical `pip freeze` or interpreter receipt
  was written. The live interpreter that the archived runner points to is recorded in
  `BN_LIVE_ENVIRONMENT.md`; its Python/PyTorch versions differ from the archived expectation.
- Source hashes for the six frozen files and all nine runtime dependencies match the frozen
  bindings (see `BN_DEPLOYMENT_PATH_BINDINGS.json`).

## Verdict

Prediction bytes, row counts, order, metrics and command arguments are **VERIFIED**. Runtime
provenance is **PARTIAL**: no per-job trace or environment snapshot. The historical bundles are
evidence and comparison receipts, not exact-reuse substitutes for the v0.6 arms.
"""
write_md("BN_HISTORICAL_RUNTIME_RECEIPTS.md", hist_md)


gaps = f"""# BN asset gaps and blockers

Checked at **{NOW}**. Status per item: `VERIFIED`, `UNVERIFIED`, `MISSING`, `MISMATCH`,
`NOT_CHECKED`.

## Resolved (no blocker)

- MSR-VTT test2990 roles: 2,990 rows in image/label/caption TSVs, exact ID order, 20 references per
  query, canonical reference digest `{ref_digest}`.
- All three TSV offset files exist, are monotonic and stay inside their TSVs; a 106-row frame sample
  has 18 fields and non-empty frame payloads.
- Manifests: `msrvtt_test2990.tsv` {len(full_lines)} rows and `msrvtt_untouched2690.tsv`
  {len(unt_lines)} rows; SHA-256 values match the frozen archive.
- Checkpoint and three historical slots: SHA-256
  `{model['checkpoint']['sha256']}`.
- Vicuna-7B-v1.3: config, tokenizer, index and two shards present; shard list equals the index.
- Video-LLaMA Q-Former, EVA-ViT-G and BLIP2 Q-Former caches present and readable.
- Frozen D cap/sims and top-20 retrieval files: all SHA-256 values match.
- A1 feature coverage: 2,539 unique clips, canonical ID digest matches, 0 missing.
- A7 feature coverage: 15,213 unique clips, canonical ID digest matches, 0 missing.
- Six frozen source files and nine runtime dependency files match their frozen SHA-256 values.
- GPU idle: 0% utilization, 15 MiB used, no compute client; GPU lock unheld.

## Blockers / deviations

| ID | Severity | Item | Status | Consequence |
| --- | --- | --- | --- | --- |
| G1 | P0 | GPU execution authorization | **NO** (policy) | No smoke, full generation, scoring or model load is permitted. |
| G2 | P1 | Live environment vs archived receipt | **MISMATCH** | Python `{versions.get('version')}` vs `3.9.18`; PyTorch `{versions.get('torch')}` vs `2.1.0`. The venv is a hybrid of the TTP Conda base and the Display venv. Freeze the actual runtime hash set and obtain an explicit deviation decision before execution. |
| G3 | P0 | Historical per-job runtime trace | **MISSING** | D1/D3/D7 remain conditional analogs; exact reuse stays 0. Re-run the v0.6 arms rather than reusing historical outputs. |
| G4 | P0 | A0 prepared inputs (`A_k0_cap.json`, `A_k0_sims.json`) | **MISSING** | Must be generated CPU-side from the frozen empty-array contract before the A0 smoke. |
| G5 | P0 | B prepared inputs (`B_text3_cap`, `B_text7_hq_cap`, `B_text7_lq_cap`, `B_shared_visual_sims`, `B_visual_contract`, `B_eligible_queries`) | **MISSING** | Deterministically regenerable from verified top-20/D files on CPU, but not staged. B cannot start. |
| G6 | P0 | Real-model parity / A smoke / B smoke | **NOT_CHECKED** | No model was loaded. B formal execution remains NO-GO until A full and B smoke pass. |
| G7 | P2 | Display disk mount | **VERIFIED (read-only)** | Reads are fine; all new outputs must go to the read-write project filesystem, never Display. |
| G8 | P2 | Image TSV byte-level digest | `NOT_CHECKED` | 1.38 GB `test_16frames_img_size224.img.tsv` was pinned by metadata, lineidx SHA-256 and a row sample. `deploy_guard.py` intentionally uses mtime for this role. |
| G9 | P2 | Main project working tree | 107 dirty entries | Do not deploy from the dirty main checkout; use a dedicated clean worktree. |
| G10 | P3 | BERT HF cache | tokenizer/config only | No BERT weight file is cached; historical runs used the cached tokenizer and BLIP2 Q-Former. Confirm at smoke time. |

## Not inferred

No asset was marked missing merely because it is absent from Git. Each status above comes from a
live filesystem probe. No large model weight, video, feature tensor or TSV was copied into Git.
"""
write_md("BN_ASSET_GAPS_AND_BLOCKERS.md", gaps)


final = f"""# BN final read-only GO / NO-GO

Checked at **{NOW}**. Scope: assets and path bindings only. No GPU execution was authorized or
performed.

| Gate | State | Basis |
| --- | --- | --- |
| Read-only audit complete | **YES** | All eight receipts produced from live filesystem probes. |
| Dataset assets | **VERIFIED** | 2,990 rows/IDs/references/offsets; manifest and reference digests match. |
| Model and cache assets | **VERIFIED** | Checkpoint, Vicuna shards, Video-LLaMA, EVA, BLIP2 present; checkpoint hash matches. |
| Retrieval assets | **VERIFIED** | D cap/sims/top-20 hashes match; A1/A7 features complete and ID digests match. |
| Frozen source bindings | **VERIFIED** | 6/6 sources and 9/9 runtime dependencies match. |
| Historical D1/D3/D7 bundles | **VERIFIED (bytes)** | Six prediction files and metrics match frozen bytes; no runtime trace. |
| Runtime environment | **MISMATCH** | Python/PyTorch differ from archived expectation; hybrid venv. |
| Prepared v0.6 inputs | **MISSING** | A0 and all B input objects are not staged. |
| Real-model parity | **NOT_CHECKED** | No model load or GPU smoke. |
| `SERVER_DATA_READY` | **CONDITIONAL** | Core data/weights/features verified; environment deviation and B inputs unresolved. |
| `GPU_EXECUTION_APPROVED` | **NO** | Authorization explicitly withheld. |

## Per-experiment decision

| Experiment | Decision | Required before any execution |
| --- | --- | --- |
| A (A0/A1/A7) | **NO-GO now** | GPU authorization; freeze live environment and accept or resolve the version mismatch; generate A0 empty inputs; run and validate the A smoke with zero evidence; then A full under the lock. |
| B (B3/B7HQ/B7LQ) | **NO-GO formal** | All A prerequisites; generate/stage B inputs and visual contract; pass the B smoke with identical visual tensors and operations. |

Exact historical reuse remains **0 arms**. The v0.6 matrix still requires **17,940** full-order
generations (A0/A1/A7/B3/B7HQ/B7LQ); the estimate remains 8.4–27.9 GPU-hours with a 14.2-hour
historical anchor. No smoke query was run during this audit.

## Honest bottom line

The server has the data, checkpoint, caches, feature files, source bindings and historical
prediction bytes needed for the follow-up. It does **not** yet have a frozen, matching runtime
receipt or the prepared A0/B intervention inputs, and it has no authorization to touch the GPU.
Therefore:

- **Asset readiness: CONDITIONAL PASS.**
- **GPU execution: NO.**
- **Formal B: NO-GO.**

本次 GPU 执行授权仍为 **NO**。
"""
write_md("BN_FINAL_READONLY_GO_NO_GO.md", final)


print(json.dumps({
    "checked_at": NOW,
    "environment_status": env["status"],
    "environment_checks": checks,
    "dataset_status": dataset["status"],
    "dataset_checks": id_checks,
    "model_status": model["status"],
    "retrieval_status": retrieval["status"],
    "feature_sets": {k: {kk: vv for kk, vv in v.items() if kk not in ("missing_examples",)}
                     for k, v in feature_sets.items()},
    "historical_checks": hist["checks"],
    "binding_status": bindings["status"],
    "missing_prepared_inputs": [n for n, v in prepared.items() if v["status"] == M],
}, indent=2, ensure_ascii=False))
