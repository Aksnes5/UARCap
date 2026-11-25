#!/usr/bin/env python3
"""Discover Pilot A1 top-20 retrieval candidates and report missing cached visual features.

This does NOT change retrieval semantics: it runs the exact same X-CLIP text search as
the pilot, but with feature-existence verification disabled, so we can extract the
external visual features (Video-LLaMA faithful_v2) for the retrieved representative
clips before the real run.

Usage (inside the frozen Video-LLaMA/X-CLIP environment, GPU):
  python scripts/discover_pilot_a1_candidates.py \
    --manifest bn_carebench_pilot_a01/PILOT_A1_FROZEN_SAMPLING_MANIFEST.json \
    --metadata /home/bn/uarcap_q2_runtime/data/carebench/json/metadata.json \
    --video-root /home/bn/uarcap_q2_runtime/data/carebench/videos \
    --paths-config configs/bn_paths.local.yaml \
    --out-json /home/bn/uarcap_q2_runtime/outputs/pilot_a1/pilot_a1_carebench_v1/pilot_a1_retrieval_discovery.json \
    --missing-out /home/bn/uarcap_q2_runtime/outputs/pilot_a1/pilot_a1_carebench_v1/pilot_a1_missing_features.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uarcap_q2.common.gpu_lock import GPULockConfig, acquire_cooperative_gpu_lock  # noqa: E402
from uarcap_q2.datasets.carebench import load_carebench_records  # noqa: E402
from uarcap_q2.retrieval.videoufo_retriever import (  # noqa: E402
    VideoUFOIndexConfig,
    VideoUFOXCLIPRetriever,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--paths-config", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--missing-out", type=Path, required=True)
    ap.add_argument("--top-m", type=int, default=20)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.paths_config.read_text(encoding="utf-8"))
    hist = cfg["historical_frozen_assets"]
    host = cfg.get("host", {})
    frozen = json.loads(args.manifest.read_text(encoding="utf-8"))
    wanted = [e["query_id"] for e in frozen["selected"]]

    records = load_carebench_records(args.metadata, args.video_root, require_video_exists=True)
    by_id = {r.query_id: r for r in records}
    missing_queries = [q for q in wanted if q not in by_id]
    if missing_queries:
        raise SystemExit(f"Frozen manifest queries missing from metadata: {missing_queries[:5]}")
    queries = [by_id[q] for q in wanted]

    vcfg = VideoUFOIndexConfig(
        videoufo_csv_path=hist["videoufo_csv_path"],
        clips_full_tsv_path=hist["clips_full_tsv_path"],
        brief_text_emb_npy_path=hist["brief_text_emb_npy_path"],
        xclip_repo_dir=hist["xclip_repo_dir"],
        xclip_ckpt_path=hist["xclip_ckpt_path"],
        video_feature_dir=hist["video_feature_dir"],
        topk=args.top_m,
    )
    retriever = VideoUFOXCLIPRetriever(vcfg, verify_feature_files_exist=False)
    feat_root = Path(hist["video_feature_dir"])

    lock = GPULockConfig(
        lock_path=Path(host["shared_gpu_lock"]),
        run_id="pilot_a1_candidate_discovery",
        gpu_uuid=host.get("gpu_uuid"),
        require_nvidia_smi_idle=True,
        sidecar_dir=args.out_json.parent,
    )
    payload = {"manifest_sha256": frozen["manifest_sha256"], "top_m": args.top_m, "queries": {}}
    missing: set[str] = set()
    with acquire_cooperative_gpu_lock(lock):
        for q in queries:
            cands = retriever.retrieve_top20(q)
            rows = []
            for c in cands:
                rep = c.clip_ids[0]
                present = (feat_root / f"{rep}.pt").is_file()
                if not present:
                    missing.add(rep)
                rows.append(
                    {
                        "caption": c.caption,
                        "score": c.score,
                        "clip_ids": c.clip_ids,
                        "source_ids": c.source_ids,
                        "rep_clip": rep,
                        "feature_present": present,
                    }
                )
            payload["queries"][q.query_id] = rows
        payload["retriever_manifest"] = retriever.get_asset_manifest()

    payload["n_missing_features"] = len(missing)
    payload["missing_features"] = sorted(missing)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.missing_out.write_text("\n".join(sorted(missing)) + ("\n" if missing else ""), encoding="utf-8")
    print(json.dumps({"n_queries": len(queries), "n_missing_features": len(missing), "out": str(args.out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
