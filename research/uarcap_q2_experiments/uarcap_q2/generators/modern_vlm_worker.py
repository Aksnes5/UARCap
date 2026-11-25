"""Isolated Modern VLM worker (runs inside `uarcap_q2_modern_vlm` conda environment).

Loads the Modern VLM once per arm batch, reads exact lossless PNG frames + ArmInputPackets from disk,
executes Qwen25VLAdapter.generate_one with full processor-observed telemetry, writes verified JSONL outputs,
and releases GPU memory cleanly on worker exit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

from PIL import Image

EXP_ROOT = Path(__file__).resolve().parents[2]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from uarcap_q2.common.gpu_runtime import release_gpu_memory
from uarcap_q2.common.io_utils import atomic_write_jsonl, read_json
from uarcap_q2.common.schemas import PilotStage, QueryRecord
from uarcap_q2.generators.modern_vlm_adapter import ModernVLMConfig, Qwen25VLAdapter
from uarcap_q2.retrieval.arm_builder import ArmInputPacket


def run_modern_vlm_batch_job(request_json: Path | str, response_jsonl: Path | str) -> list[dict[str, Any]]:
    req = read_json(request_json)
    cfg = ModernVLMConfig(**req["model_config"])
    stage = PilotStage(req["stage"])
    adapter = Qwen25VLAdapter(cfg)
    rows: list[dict[str, Any]] = []
    try:
        for item in req["items"]:
            q = QueryRecord(**item["query"])
            pkt = ArmInputPacket(**item["packet"])
            frames = [Image.open(p).convert("RGB") for p in item["frame_png_paths"]]
            out = adapter.generate_one(
                query=q,
                packet=pkt,
                stage=stage,
                visual_input_sha256=str(item["visual_input_sha256"]),
                frame_indices=list(item["frame_indices"]),
                frame_images=frames,
            )
            d = asdict(out)
            d["input_trace"] = out.input_trace.to_dict()
            rows.append(d)
        atomic_write_jsonl(response_jsonl, rows)
        return rows
    finally:
        release_gpu_memory()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Isolated Modern VLM batch worker")
    ap.add_argument("--request-json", type=Path, required=True)
    ap.add_argument("--response-jsonl", type=Path, required=True)
    args = ap.parse_args(argv)
    run_modern_vlm_batch_job(args.request_json, args.response_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
