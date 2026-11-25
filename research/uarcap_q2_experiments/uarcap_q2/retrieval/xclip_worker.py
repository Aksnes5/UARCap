"""Isolated X-CLIP retrieval worker (runs inside the existing frozen BN X-CLIP/Video-LLaMA Python env).

Allows `uarcap_q2_cpu` (no CUDA) to execute full 1M VideoUFO X-CLIP query-conditioned retrieval in a single
batch subprocess so X-CLIP weights are loaded ONCE and VRAM is cleanly released on worker exit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

EXP_ROOT = Path(__file__).resolve().parents[2]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from uarcap_q2.common.gpu_runtime import release_gpu_memory
from uarcap_q2.common.io_utils import atomic_write_json, read_json
from uarcap_q2.common.schemas import QueryRecord
from uarcap_q2.retrieval.videoufo_retriever import VideoUFOIndexConfig, VideoUFOXCLIPRetriever


def run_xclip_batch_job(request_json: Path | str, response_json: Path | str) -> dict[str, Any]:
    req = read_json(request_json)
    cfg = VideoUFOIndexConfig(**req["index_config"])
    retriever = VideoUFOXCLIPRetriever(
        cfg=cfg,
        verify_feature_files_exist=bool(req.get("verify_feature_files_exist", True)),
    )
    results: dict[str, list[dict[str, Any]]] = {}
    try:
        for qd in req["queries"]:
            q = QueryRecord(**qd)
            cands = retriever.retrieve_top20(q)
            results[q.query_id] = [c.to_dict() for c in cands]
        payload = {
            "status": "OK",
            "manifest": retriever.get_asset_manifest(),
            "results": results,
        }
        atomic_write_json(response_json, payload)
        return payload
    finally:
        release_gpu_memory()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Isolated BN X-CLIP VideoUFO retrieval worker")
    ap.add_argument("--request-json", type=Path, required=True)
    ap.add_argument("--response-json", type=Path, required=True)
    args = ap.parse_args(argv)
    run_xclip_batch_job(args.request_json, args.response_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
