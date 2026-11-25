"""Isolated Modern Video VLM adapter (primary candidate: Qwen2.5-VL-7B-Instruct).

Designed strictly for an isolated Conda environment (env_modern_vlm.yaml, Transformers >= 4.49),
NEVER sharing or upgrading the frozen Video-LLaMA Transformers 4.28.0 environment!

Note: Real GPU inference has NOT been executed on RTX A6000 during this offline Windows phase;
timed smoke verification on BN is required upon server access.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from ..common.io_utils import primitive_hash
from ..common.schemas import ExpArm, InputTrace, PilotStage, QueryRecord
from ..retrieval.arm_builder import ArmInputPacket
from .base import GenerationOutput, GeneratorAdapter, GeneratorOOMError, verify_actual_input_trace

SHORT_CAPTION_SYSTEM_PROMPT = (
    "You are a precise, visually grounded video captioning assistant. "
    "Write ONE concise factual sentence of 8 to 18 words describing only what is visibly happening in the video. "
    "Do not add speculation, lists, or extra commentary."
)

DETAILED_SENSITIVITY_SYSTEM_PROMPT = (
    "Describe the video clip clearly and factually in 2 to 4 concise sentences, "
    "focusing on visible subjects and actions."
)


@dataclass
class ModernVLMConfig:
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    local_model_path: str = ""
    pinned_revision: str = "PENDING_PIN_AT_DOWNLOAD"
    offline_only: bool = True
    num_frames: int = 16
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 384 * 28 * 28  # ~256-384 visual tokens/frame -> ~4K-6K visual tokens for 16 frames on A6000 48GB
    prompt_regime: str = "short_aligned"  # short_aligned (primary) | detailed_sensitivity (secondary)
    max_new_tokens: int = 48
    do_sample: bool = False
    temperature: float = 0.0
    random_seed: int = 42
    dtype: str = "bfloat16"
    device: str = "cuda:0"

    def validate(self) -> None:
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if self.prompt_regime not in ("short_aligned", "detailed_sensitivity"):
            raise ValueError(f"Unsupported prompt_regime: {self.prompt_regime}")
        if self.prompt_regime == "short_aligned" and self.max_new_tokens > 64:
            raise ValueError("short_aligned regime must cap max_new_tokens <= 64 to prevent length drift")
        if self.offline_only and not self.local_model_path:
            raise ValueError("offline_only=True requires an explicit local_model_path")


def render_modern_vlm_prompt(packet: ArmInputPacket, prompt_regime: str = "short_aligned") -> dict[str, str]:
    """Deterministically render system + user prompt for zero evidence (A0/Base), k=1/3/7, or E3 Sup/Con/Neu."""
    sys_prompt = (
        SHORT_CAPTION_SYSTEM_PROMPT
        if prompt_regime == "short_aligned"
        else DETAILED_SENSITIVITY_SYSTEM_PROMPT
    )
    if not packet.text_evidence:
        user_prompt = (
            "Watch the video frames carefully and describe the main visible event in one concise sentence (8-18 words)."
            if prompt_regime == "short_aligned"
            else "Watch the video frames carefully and describe the visible events accurately."
        )
    else:
        bullets = "\n".join(f"- {cap}" for cap in packet.text_evidence)
        user_prompt = (
            "Retrieved reference descriptions from related videos (may contain useful context or inaccuracies):\n"
            f"{bullets}\n\n"
            "Using the video frames as visual input, describe the main visible event in this video "
            + (
                "in one concise sentence (8-18 words)."
                if prompt_regime == "short_aligned"
                else "accurately in 2-4 sentences."
            )
        )
    return {"system": sys_prompt, "user": user_prompt}


class Qwen25VLAdapter(GeneratorAdapter):
    """Offline-safe Qwen2.5-VL-7B-Instruct adapter (lazy GPU import; CPU prompt/config testable)."""

    def __init__(self, cfg: ModernVLMConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        self.model_id = cfg.model_id
        self.expected_checkpoint_sha256 = f"revision:{cfg.pinned_revision}"
        self._model = None
        self._processor = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        local_dir = Path(self.cfg.local_model_path)
        if self.cfg.offline_only and not local_dir.is_dir():
            raise FileNotFoundError(f"Offline Modern VLM local_model_path missing: {local_dir}")
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        torch_dtype = torch.bfloat16 if self.cfg.dtype == "bfloat16" else torch.float16
        self._processor = AutoProcessor.from_pretrained(
            str(local_dir),
            local_files_only=self.cfg.offline_only,
            min_pixels=self.cfg.min_pixels,
            max_pixels=self.cfg.max_pixels,
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(local_dir),
            torch_dtype=torch_dtype,
            device_map=self.cfg.device,
            local_files_only=self.cfg.offline_only,
        ).eval()

    def build_messages_and_trace(
        self,
        query: QueryRecord,
        packet: ArmInputPacket,
        stage: PilotStage,
        visual_input_sha256: str,
        frame_indices: list[int],
        frame_images: list[Any] | None = None,
    ) -> tuple[list[dict[str, Any]], InputTrace]:
        rendered = render_modern_vlm_prompt(packet, self.cfg.prompt_regime)
        user_content: list[dict[str, Any]] = []
        if frame_images is not None:
            user_content.append({"type": "video", "video": frame_images})
        else:
            user_content.append({"type": "video", "video": query.video_path})
        user_content.append({"type": "text", "text": rendered["user"]})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": rendered["system"]}]},
            {"role": "user", "content": user_content},
        ]
        trace = InputTrace(
            query_id=query.query_id,
            dataset_id=query.dataset_id,
            stage=stage.value,
            arm=packet.arm,
            model_id=self.model_id,
            checkpoint_sha256=self.expected_checkpoint_sha256,
            visual_input_sha256=visual_input_sha256,
            frame_indices=list(frame_indices),
            text_evidence=list(packet.text_evidence),
            visual_evidence_ids=list(packet.visual_clip_ids),
            visual_evidence_scores=list(packet.visual_scores),
            prompt_rendered_sha256=primitive_hash(rendered),
            rng_seed=self.cfg.random_seed,
            decoding_config={
                "prompt_regime": self.cfg.prompt_regime,
                "max_new_tokens": self.cfg.max_new_tokens,
                "do_sample": self.cfg.do_sample,
                "temperature": self.cfg.temperature,
                "num_frames": self.cfg.num_frames,
                "min_pixels": self.cfg.min_pixels,
                "max_pixels": self.cfg.max_pixels,
            },
            base_visual_mode=packet.base_visual_mode,
        )
        verify_actual_input_trace(trace, packet, self.expected_checkpoint_sha256, visual_input_sha256)
        return messages, trace

    def generate_one(
        self,
        query: QueryRecord,
        packet: ArmInputPacket,
        stage: PilotStage,
        visual_input_sha256: str,
        frame_indices: list[int],
        frame_images: list[Any] | None = None,
    ) -> GenerationOutput:
        messages, trace = self.build_messages_and_trace(
            query, packet, stage, visual_input_sha256, frame_indices, frame_images=frame_images
        )
        self._lazy_load()
        import torch

        t0 = perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            text_in = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(
                text=[text_in],
                videos=[frame_images] if frame_images is not None else None,
                padding=True,
                return_tensors="pt",
            ).to(self.cfg.device)
            with torch.inference_mode():
                gen_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=self.cfg.do_sample,
                )
            trimmed = gen_ids[:, inputs.input_ids.shape[1] :]
            caption = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
            peak_vram = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise GeneratorOOMError(f"CUDA OOM in Modern VLM for {query.query_id}: {exc}") from exc
            raise
        elapsed = perf_counter() - t0
        return GenerationOutput(
            query_id=query.query_id,
            dataset_id=query.dataset_id,
            stage=stage.value,
            arm=packet.arm,
            model_id=self.model_id,
            caption=caption,
            input_trace=trace,
            wall_seconds=round(elapsed, 4),
            peak_vram_bytes=peak_vram,
            ok=bool(caption),
            error="" if caption else "EMPTY_GENERATION",
        )
