"""Frozen BN Video-LLaMA / EVCap adapter reusing verified v0.6/v0.8 checkpoint, code, and interpreter.

Never modifies historical frozen files or upgrades the legacy Transformers 4.28.0 venv!
Builds exact EVCap argv, verifies canonical checkpoint SHA-256, writes per-arm cap/sims JSONs,
and executes inside an isolated workdir via subprocess on Linux BN when authorized.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

from ..common.io_utils import atomic_write_json, primitive_hash, sha256_file
from ..common.schemas import ExpArm, InputTrace, PilotStage, QueryRecord
from ..retrieval.arm_builder import ArmInputPacket
from .base import GenerationOutput, GeneratorAdapter, GeneratorOOMError, TraceVerificationError, verify_actual_input_trace

FROZEN_CHECKPOINT_NAME = (
    "train_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review.pt"
)
FROZEN_CHECKPOINT_SHA256 = "a8b5af79d55a2c7e684771475d645be6373208bdaae5a4211bee556581eb8002"
FROZEN_EVALUATOR_SCRIPT = "eval_evcap_fix_video_llama_xclip_msrvtt_fusion_gate_prompt5.py"
FROZEN_PROMPT_REL = "prompts/prompt_evcap_msrvtt_detail_smallcap_prompt5.txt"
FROZEN_BASE_ARGS: dict[str, Any] = {
    "bs": 1,
    "random_seed": 42,
    "max_num_frames": 16,
    "num_video_query_token": 32,
    "beam_width": 5,
    "freeze_video_qformer": True,
    "vit_precision": "fp16",
    "freeze_llm": True,
    "freeze_vit": True,
    "freeze_qformer": True,
    "load_video_feature": True,
    "share_query_tokens": True,
    "retriever": True,
    "deterministic": True,
    "use_fusion": True,
    "no_retrieval": False,
}


@dataclass
class FrozenVideoLLaMAConfig:
    python_executable: str
    evcap_work_root: str
    checkpoint_path: str
    llm_model_dir: str
    video_feature_dir: str
    torch_home: str
    hf_hub_cache: str
    device: str = "cuda:0"
    expected_checkpoint_sha256: str = FROZEN_CHECKPOINT_SHA256
    timeout_sec: int = 1800


def build_evcap_batch_inputs(
    packets: list[ArmInputPacket],
    out_dir: Path | str,
    arm: str,
) -> tuple[Path, Path]:
    """Serialize arm packets into exact EVCap cap.json and sims.json wire format."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap_map = {p.query_id: p.evcap_cap_entry for p in packets}
    sims_map = {p.query_id: p.evcap_sims_entry for p in packets}
    if arm == ExpArm.ARM_A0.value:
        for qid, entry in cap_map.items():
            if entry != [[], [], []] or sims_map[qid] != [[], []]:
                raise TraceVerificationError(f"ARM_A0 non-empty evidence refused for {qid}")
    cap_path = atomic_write_json(out / f"{arm}_cap.json", cap_map)
    sims_path = atomic_write_json(out / f"{arm}_sims.json", sims_map)
    return cap_path, sims_path


def build_evcap_cli_argv(
    cfg: FrozenVideoLLaMAConfig,
    test_yaml_path: Path | str,
    cap_json_path: Path | str,
    sims_json_path: Path | str,
    out_rel_path: str,
    text_topn: int,
    visual_topn: int,
) -> list[str]:
    """Construct exact command-line arguments matching frozen BN runner without altering frozen code."""
    topn = max(text_topn, visual_topn)
    raw = {
        "device": cfg.device,
        "test_yaml": str(test_yaml_path),
        "out_path": out_rel_path,
        "ckpt_name": FROZEN_CHECKPOINT_NAME,
        "prompt_path": FROZEN_PROMPT_REL,
        "llm_model": cfg.llm_model_dir,
        "ext_path": str(cap_json_path),
        "sim_scores_path": str(sims_json_path),
        "video_feature_dir": cfg.video_feature_dir,
        "topn": topn,
        "text_topn": text_topn,
        "visual_topn": visual_topn,
        **FROZEN_BASE_ARGS,
    }
    argv = [cfg.python_executable, str(Path(cfg.evcap_work_root) / FROZEN_EVALUATOR_SCRIPT)]
    for k, v in raw.items():
        argv.extend([f"--{k}", str(v)])
    return argv


class FrozenVideoLLaMAAdapter(GeneratorAdapter):
    """Adapter wrapping the frozen BN Video-LLaMA/EVCap evaluator for batch or single-query runs."""

    model_id = "videollama_evcap_frozen_v08"

    def __init__(self, cfg: FrozenVideoLLaMAConfig) -> None:
        self.cfg = cfg
        self.expected_checkpoint_sha256 = cfg.expected_checkpoint_sha256

    def verify_checkpoint_integrity(self) -> str:
        ckpt = Path(self.cfg.checkpoint_path)
        if not ckpt.is_file():
            raise FileNotFoundError(f"Frozen Video-LLaMA checkpoint missing: {ckpt}")
        actual = sha256_file(ckpt)
        if actual != self.expected_checkpoint_sha256:
            raise TraceVerificationError(
                f"Frozen checkpoint SHA-256 mismatch! Got {actual}, expected {self.expected_checkpoint_sha256}"
            )
        return actual

    def run_batch_subprocess(
        self,
        test_yaml_path: Path | str,
        packets: list[ArmInputPacket],
        arm: str,
        work_out_dir: Path | str,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Prepare verified arm inputs and execute frozen evaluator in its isolated legacy venv."""
        out_dir = Path(work_out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cap_path, sims_path = build_evcap_batch_inputs(packets, out_dir / "inputs", arm)
        text_topn = packets[0].text_topn if packets else 0
        visual_topn = packets[0].visual_topn if packets else 0
        out_rel = str(out_dir / f"evcap_out_{arm}")
        argv = build_evcap_cli_argv(
            self.cfg,
            test_yaml_path=test_yaml_path,
            cap_json_path=cap_path,
            sims_json_path=sims_path,
            out_rel_path=out_rel,
            text_topn=text_topn,
            visual_topn=visual_topn,
        )
        if dry_run:
            return {
                "dry_run": True,
                "arm": arm,
                "argv": argv,
                "cap_path": str(cap_path),
                "sims_path": str(sims_path),
                "expected_checkpoint_sha256": self.expected_checkpoint_sha256,
            }

        if not sys.platform.startswith("linux"):
            raise RuntimeError("Frozen BN Video-LLaMA GPU execution is supported only on authorized Linux BN server.")

        ckpt_sha = self.verify_checkpoint_integrity()
        env = {
            "TORCH_HOME": self.cfg.torch_home,
            "HUGGINGFACE_HUB_CACHE": self.cfg.hf_hub_cache,
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "USE_TF": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
        t0 = perf_counter()
        proc = subprocess.run(
            argv,
            cwd=self.cfg.evcap_work_root,
            env={**dict(__import__("os").environ), **env},
            capture_output=True,
            text=True,
            timeout=self.cfg.timeout_sec,
        )
        elapsed = perf_counter() - t0
        if "CUDA out of memory" in (proc.stderr or "") or "OutOfMemoryError" in (proc.stderr or ""):
            raise GeneratorOOMError(f"CUDA OOM in frozen Video-LLaMA arm {arm}: {proc.stderr[-500:]}")
        if proc.returncode != 0:
            raise RuntimeError(f"Frozen Video-LLaMA exited rc={proc.returncode}: {proc.stderr[-800:]}")

        pred_files = list(Path(out_rel).glob("*generated_captions.json"))
        if len(pred_files) != 1:
            raise RuntimeError(f"Expected 1 generated_captions.json in {out_rel}, found {len(pred_files)}")
        preds = json.loads(pred_files[0].read_text(encoding="utf-8"))
        return {
            "dry_run": False,
            "arm": arm,
            "checkpoint_sha256": ckpt_sha,
            "wall_seconds": round(elapsed, 3),
            "prediction_file": str(pred_files[0]),
            "predictions": preds,
        }

    def generate_one(
        self,
        query: QueryRecord,
        packet: ArmInputPacket,
        stage: PilotStage,
        visual_input_sha256: str,
        frame_indices: list[int],
    ) -> GenerationOutput:
        raise NotImplementedError(
            "Use run_batch_subprocess() on BN Linux for frozen Video-LLaMA, or MockGeneratorAdapter for CPU tests."
        )
