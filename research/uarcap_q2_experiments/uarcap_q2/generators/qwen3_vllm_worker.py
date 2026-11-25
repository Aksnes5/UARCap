"""Isolated Qwen3.8-27B-FP8 / Qwen3.5-27B vLLM worker (runs inside existing BN vLLM 0.29.0 env).

Loads Qwen3.8-27B-FP8 (or Qwen3.5-27B) ONCE per arm batch, reads exact lossless 16 PNG frames,
invokes Qwen3VLLMMultimodalAdapter.generate_batch with thinking disabled by default, writes verified
JSONL outputs, and releases VRAM cleanly on exit.
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
from uarcap_q2.generators.base import BatchItem
from uarcap_q2.generators.qwen3_vllm_adapter import Qwen3VLLMConfig, Qwen3VLLMMultimodalAdapter
from uarcap_q2.retrieval.arm_builder import ArmInputPacket


def run_qwen3_vllm_batch_job(request_json: Path | str, response_jsonl: Path | str) -> list[dict[str, Any]]:
    req = read_json(request_json)
    cfg = Qwen3VLLMConfig(**req["model_config"])
    stage = PilotStage(req["stage"])
    adapter = Qwen3VLLMMultimodalAdapter(cfg)
    batch_items: list[BatchItem] = []
    for item in req["items"]:
        q = QueryRecord(**item["query"])
        pkt = ArmInputPacket(**item["packet"])
        frames = [Image.open(p).convert("RGB") for p in item["frame_png_paths"]]
        batch_items.append(
            BatchItem(
                query=q,
                packet=pkt,
                visual_input_sha256=str(item["visual_input_sha256"]),
                frame_indices=list(item["frame_indices"]),
                frame_images=frames,
            )
        )
    try:
        outs = adapter.generate_batch(batch_items, stage=stage)
        rows: list[dict[str, Any]] = []
        for out in outs:
            d = asdict(out)
            d["input_trace"] = out.input_trace.to_dict()
            rows.append(d)
        atomic_write_jsonl(response_jsonl, rows)
        return rows
    finally:
        adapter.close()
        release_gpu_memory()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Isolated BN Qwen3 vLLM multimodal video batch worker")
    ap.add_argument("--request-json", type=Path, required=True)
    ap.add_argument("--response-jsonl", type=Path, required=True)
    args = ap.parse_args(argv)
    run_qwen3_vllm_batch_job(args.request_json, args.response_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
