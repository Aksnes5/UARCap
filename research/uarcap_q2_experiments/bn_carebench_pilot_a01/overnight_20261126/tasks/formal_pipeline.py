#!/usr/bin/env python3
"""Formal CaReBench candidate pipeline (manifest / retrieval / features).

Pre-frozen, output-independent candidate order. Generation is marked
FORMAL_CANDIDATE_GENERATION_UNVALIDATED and never performs confirmatory statistics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(os.environ.get("OVERNIGHT_BASE", "/home/bn/uarcap_q2_runtime/overnight/20261126"))
TASK_ID = os.environ.get("OVERNIGHT_TASK_ID", "formal")
PROGRESS = BASE / "progress" / f"{TASK_ID}.json"
REPO = "/home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments"
sys.path.insert(0, REPO)
Q2 = Path("/home/bn/uarcap_q2_runtime")
OUT = Q2 / "outputs" / "formal_candidate_unvalidated"
FROZEN_PY = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python"
HF_CACHE = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/hf_cache"
TORCH_CACHE = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/torch_cache"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []


def append_jsonl(p: Path, row: dict) -> None:
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_write(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def progress(phase: str, done: int, total: int, **extra) -> None:
    atomic_write(PROGRESS, {"task": TASK_ID, "phase": phase, "completed": done, "total": total,
                            "updated_utc": utcnow(), **extra})


def sha_file(p: Path) -> str:
    h = sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def stage_manifest() -> int:
    import yaml
    from uarcap_q2.datasets.carebench import load_carebench_records
    from uarcap_q2.datasets.formal_sampling import FormalSamplingConfig, build_formal_e2_manifest
    cfg_paths = yaml.safe_load(Path(f"{REPO}/configs/bn_paths.local.yaml").read_text(encoding="utf-8"))
    root = Path(cfg_paths["q2_writable_workspace"]["data_root"])
    recs = load_carebench_records(root / "carebench/json/metadata.json", root / "carebench/videos", require_video_exists=False)
    excl = json.loads(Path("/home/bn/uarcap_q2_runtime/outputs/pilot_a1/pilot_a1_carebench_v1/pilot_a_excluded_carebench_vids.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = FormalSamplingConfig(e2_target_n=600, e2_validation_target_n=60, seed=20261125)
    selected, manifest = build_formal_e2_manifest(
        all_records=recs,
        pilot_excluded_ids=set(excl["excluded_query_ids"]),
        cfg=cfg,
        out_manifest_path=OUT / "FORMAL_CAREBENCH_ANALYSIS_MANIFEST.json",
        pilot_excluded_parent_ids=set(excl["excluded_parent_video_ids"]),
        extra_debug_query_ids=set(excl.get("extra_debug_query_ids", [])),
        extra_debug_parent_ids=set(excl.get("extra_debug_parent_video_ids", [])),
    )
    val_ids = set(manifest.get("e2_validation_set", {}).get("validation_query_ids", []))
    rows = []
    for r in selected:
        md = r.metadata or {}
        rows.append({"query_id": r.query_id, "video_path": r.video_path, "parent_video_id": md.get("parent_video_id"),
                     "category": r.domain, "subcategory": md.get("subcategory"), "reference_caption": r.reference_description,
                     "spatial_caption": md.get("spatial_caption"), "temporal_caption": md.get("temporal_caption"),
                     "core_event_candidate": (r.core_events[0] if r.core_events else None),
                     "in_e2_validation": r.query_id in val_ids})
    atomic_write(OUT / "formal_candidates.jsonl", {"dummy": True}) if False else None
    tmp = (OUT / "formal_candidates.jsonl").with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, OUT / "formal_candidates.jsonl")
    lock_entries = [{"query_id": r["query_id"], "reference_caption": r["reference_caption"],
                     "spatial_caption": r["spatial_caption"], "temporal_caption": r["temporal_caption"],
                     "core_event_candidate": r["core_event_candidate"]} for r in rows]
    lock = {"schema": "uarcap_q2.formal_core_lock.v1", "created_before_any_generation": True, "utc": utcnow(),
            "manifest_sha256": manifest.get("manifest_sha256"), "n_queries": len(rows), "entries": lock_entries}
    lock["lock_sha256"] = sha256(json.dumps({k: v for k, v in lock.items() if k != "lock_sha256"}, sort_keys=True).encode()).hexdigest()
    atomic_write(OUT / "formal_core_lock.json", lock)
    receipt = {"schema": "uarcap_q2.formal_manifest_receipt.v1", "status": "FORMAL_CANDIDATE_MANIFEST_FROZEN_UNVALIDATED",
               "utc": utcnow(), "n_selected": len(rows), "n_validation": len(val_ids),
               "manifest_sha256": manifest.get("manifest_sha256"), "core_lock_sha256": lock["lock_sha256"],
               "parent_isolation_verified": manifest.get("parent_isolation_verified"),
               "n_eligible_pool": manifest.get("n_eligible_pool"),
               "note": "Generation-only; no confirmatory statistics; sample size pre-frozen at 600 within protocol range."}
    atomic_write(OUT / "FORMAL_MANIFEST_RECEIPT.json", receipt)
    progress("manifest", len(rows), len(rows))
    print(json.dumps(receipt, indent=2)[:1500])
    return 0 if rows else 2


def stage_retrieval() -> int:
    import yaml
    from uarcap_q2.common.gpu_lock import GPULockConfig, acquire_cooperative_gpu_lock
    from uarcap_q2.common.schemas import QueryRecord
    from uarcap_q2.retrieval.reranker import rerank_d_consensus
    from uarcap_q2.retrieval.videoufo_retriever import VideoUFOIndexConfig, VideoUFOXCLIPRetriever
    cfgp = yaml.safe_load(Path(f"{REPO}/configs/bn_paths.local.yaml").read_text(encoding="utf-8"))
    hist = cfgp["historical_frozen_assets"]
    cands = read_jsonl(OUT / "formal_candidates.jsonl")
    out = OUT / "formal_retrieval.jsonl"
    done = {r["query_id"] for r in read_jsonl(out)}
    pending = [c for c in cands if c["query_id"] not in done]
    vcfg = VideoUFOIndexConfig(videoufo_csv_path=hist["videoufo_csv_path"], clips_full_tsv_path=hist["clips_full_tsv_path"],
                               brief_text_emb_npy_path=hist["brief_text_emb_npy_path"], xclip_repo_dir=hist["xclip_repo_dir"],
                               xclip_ckpt_path=hist["xclip_ckpt_path"], video_feature_dir=hist["video_feature_dir"])
    retriever = VideoUFOXCLIPRetriever(vcfg, verify_feature_files_exist=False)
    lock = GPULockConfig(lock_path=Path("/home/bn/videoufo_research/gpu.lock"), run_id=TASK_ID,
                         gpu_uuid="GPU-66d83fcc-2695-33a7-33aa-af30b7e3d26d", require_nvidia_smi_idle=True, sidecar_dir=OUT)
    with acquire_cooperative_gpu_lock(lock):
        for i, c in enumerate(pending):
            q = QueryRecord(query_id=c["query_id"], dataset_id="CaReBench", source_id=None, source_id_status="UNKNOWN_OPAQUE",
                            video_path=c["video_path"], video_sha256=None, duration=None, domain=c["category"],
                            reference_description=c["reference_caption"], reference_events=[], core_events=[c["core_event_candidate"]] if c["core_event_candidate"] else [],
                            split="formal_candidate_unvalidated", eligibility_status="CANDIDATE_PENDING_VERIFICATION", metadata={})
            top = retriever.retrieve_top20(q)
            rer = rerank_d_consensus(top)
            append_jsonl(out, {"query_id": c["query_id"], "original_top20": [x.to_dict() for x in top], "reranked_d": [x.to_dict() for x in rer]})
            if i % 10 == 0:
                progress("retrieval", len(done)+i+1, len(cands), last_qid=c["query_id"])
    progress("retrieval_done", len(cands), len(cands))
    return 0


def stage_features() -> int:
    from uarcap_q2.common.gpu_lock import GPULockConfig, acquire_cooperative_gpu_lock
    rows = read_jsonl(OUT / "formal_retrieval.jsonl")
    feat_root = Path("/home/bn/videoufo_research/features/videollama_faithful_v2")
    missing = set()
    for r in rows:
        for c in (r.get("reranked_d") or [])[:7]:
            rep = (c.get("clip_ids") or [None])[0]
            if rep and not (feat_root / f"{rep}.pt").is_file():
                missing.add(rep)
    miss_path = OUT / "formal_missing_features.txt"
    miss_path.write_text("\n".join(sorted(missing)) + ("\n" if missing else ""), encoding="utf-8")
    progress("features", 0, len(missing), n_missing=len(missing))
    if missing:
        cmd = (f"HUGGINGFACE_HUB_CACHE={HF_CACHE} HF_HOME={HF_CACHE} TORCH_HOME={TORCH_CACHE} "
               f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 {FROZEN_PY} -u "
               f"/home/bn/videoufo_research/scripts/extract_external_features_v2.py {miss_path}")
        lock = GPULockConfig(lock_path=Path("/home/bn/videoufo_research/gpu.lock"), run_id=TASK_ID,
                             gpu_uuid="GPU-66d83fcc-2695-33a7-33aa-af30b7e3d26d", require_nvidia_smi_idle=True, sidecar_dir=OUT)
        with acquire_cooperative_gpu_lock(lock):
            proc = subprocess.run(["/bin/bash", "-lc", cmd], capture_output=True, text=True)
        (OUT / "formal_features_extract.log").write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
        if proc.returncode != 0:
            print("HARD_BLOCK: feature extraction failed", flush=True); return 3
    still = [p for p in missing if not (feat_root / f"{p}.pt").is_file()]
    receipt = {"status": "FEATURES_READY" if not still else "FEATURES_PARTIAL", "utc": utcnow(),
               "n_requested": len(missing), "n_still_missing": len(still), "sample": still[:20]}
    atomic_write(OUT / "FORMAL_FEATURES_RECEIPT.json", receipt)
    progress("features_done", len(missing)-len(still), len(missing))
    print(json.dumps(receipt)[:1000])
    return 0


def stage_frames() -> int:
    from uarcap_q2.datasets.video_decoder import decode_and_sample_video
    cands = read_jsonl(OUT / "formal_candidates.jsonl")
    froot = OUT / "formal_frames"
    idx_path = OUT / "formal_frame_indices.jsonl"
    done = {r["query_id"] for r in read_jsonl(idx_path)}
    pending = [c for c in cands if c["query_id"] not in done]
    for i, c in enumerate(pending):
        d = froot / c["query_id"]
        d.mkdir(parents=True, exist_ok=True)
        try:
            frames, rep = decode_and_sample_video(c["video_path"], num_frames=16)
            paths = []
            for j, im in enumerate(frames):
                p = d / f"frame_{j:02d}.jpg"
                im.save(p, "JPEG", quality=92)
                paths.append(str(p))
            append_jsonl(idx_path, {"query_id": c["query_id"], "frame_indices": list(rep.sampled_indices), "frames": paths})
        except Exception as e:
            append_jsonl(idx_path, {"query_id": c["query_id"], "error": str(e)[:500]})
        if i % 20 == 0:
            progress("frames", len(done)+i+1, len(cands))
    progress("frames_done", len(cands), len(cands))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["manifest","retrieval","features","frames"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    return {"manifest":stage_manifest,"retrieval":stage_retrieval,"features":stage_features,"frames":stage_frames}[args.stage]()


if __name__ == "__main__":
    raise SystemExit(main())
