"""Isolated Modern Video VLM adapter (P0-4 remediation: real frame handoff & processor-observed Trace).

Key fixes vs. earlier draft:
  1. REQUIRES actual decoded `frame_images` (`len == cfg.num_frames`); NEVER silently calls processor with videos=None!
  2. Explicitly marks `injection_interface="PROMPT_TEXT_ONLY"` and logs `visual_evidence_ids=[]`,
     `visual_evidence_scores=[]` (NEVER recording external visual clip IDs that a text-prompt VLM never saw!).
  3. Actually applies deterministic RNG seeding (Python/NumPy/PyTorch) before generation!
  4. Binds & verifies `pinned_revision` against local model directory metadata!
  5. Builds `InputTrace` directly from actual processor inputs (`actual_frames_sha256`) and actual tensors
     returned by processor (`pixel_values_shape`, `pixel_values_sha256`, `input_ids_len`, `input_ids_sha256`).
  6. Supports injecting CPU `fake_processor` + `fake_model` so contract tests exercise the REAL adapter code!
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from time import perf_counter
from typing import Any

import numpy as np

from ..common.io_utils import primitive_hash, sha256_bytes
from ..common.schemas import InputTrace, PilotStage, QueryRecord
from ..retrieval.arm_builder import ArmInputPacket
from .base import GenerationOutput, GeneratorAdapter, GeneratorOOMError, TraceVerificationError, verify_actual_input_trace

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
    max_pixels: int = 384 * 28 * 28
    prompt_regime: str = "short_aligned"
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
    sys_prompt = SHORT_CAPTION_SYSTEM_PROMPT if prompt_regime == "short_aligned" else DETAILED_SENSITIVITY_SYSTEM_PROMPT
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


def hash_pil_frames(frame_images: list[Any]) -> str:
    """Compute canonical combined SHA-256 directly from the exact PIL.Image frames passed to the processor."""
    per_frame = [sha256_bytes(np.asarray(im, dtype=np.uint8).tobytes()) for im in frame_images]
    return sha256_bytes("|".join(per_frame).encode("utf-8"))


def _tensor_fingerprint(val: Any) -> dict[str, Any]:
    if val is None:
        return {"present": False, "shape": None, "sha256": ""}
    if hasattr(val, "detach"):
        arr = val.detach().cpu().numpy()
    else:
        arr = np.asarray(val)
    return {"present": True, "shape": list(arr.shape), "dtype": str(arr.dtype), "sha256": sha256_bytes(arr.tobytes())}


def _verify_local_model_revision(local_dir: Path, pinned_revision: str) -> str:
    """Verify local offline model directory contains matching revision receipt if pinned."""
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Offline Modern VLM local_model_path missing: {local_dir}")
    rev_file = local_dir / "model_revision.json"
    if rev_file.is_file():
        meta = json.loads(rev_file.read_text(encoding="utf-8"))
        actual = str(meta.get("revision", "")).strip()
        if pinned_revision and pinned_revision != "PENDING_PIN_AT_DOWNLOAD" and actual != pinned_revision:
            raise TraceVerificationError(
                f"Modern VLM revision mismatch: local={actual!r}, pinned={pinned_revision!r}"
            )
        return actual or pinned_revision
    return pinned_revision


def _apply_deterministic_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class Qwen25VLAdapter(GeneratorAdapter):
    """Offline-safe Qwen2.5-VL-7B-Instruct adapter with real processor/model trace capture & CPU injection."""

    injection_interface = "PROMPT_TEXT_ONLY"

    def __init__(
        self,
        cfg: ModernVLMConfig,
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.model_id = cfg.model_id
        self._processor = processor
        self._model = model
        self._bound_revision = cfg.pinned_revision
        self.expected_checkpoint_sha256 = f"revision:{cfg.pinned_revision}"

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            if self.cfg.local_model_path and Path(self.cfg.local_model_path).is_dir():
                self._bound_revision = _verify_local_model_revision(Path(self.cfg.local_model_path), self.cfg.pinned_revision)
                self.expected_checkpoint_sha256 = f"revision:{self._bound_revision}"
            return
        local_dir = Path(self.cfg.local_model_path)
        self._bound_revision = _verify_local_model_revision(local_dir, self.cfg.pinned_revision)
        self.expected_checkpoint_sha256 = f"revision:{self._bound_revision}"
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

    def build_messages(self, packet: ArmInputPacket, frame_images: list[Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
        if not frame_images or len(frame_images) != self.cfg.num_frames:
            raise TraceVerificationError(
                f"Modern VLM requires exactly {self.cfg.num_frames} actual decoded PIL frames, got {len(frame_images) if frame_images else 0}"
            )
        rendered = render_modern_vlm_prompt(packet, self.cfg.prompt_regime)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": rendered["system"]}]},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frame_images},
                    {"type": "text", "text": rendered["user"]},
                ],
            },
        ]
        return messages, rendered

    def generate_one(
        self,
        query: QueryRecord,
        packet: ArmInputPacket,
        stage: PilotStage,
        visual_input_sha256: str,
        frame_indices: list[int],
        frame_images: list[Any] | None = None,
    ) -> GenerationOutput:
        if not frame_images:
            raise TraceVerificationError(f"Modern VLM received NO actual video frames for {query.query_id}!")
        messages, rendered = self.build_messages(packet, frame_images)
        actual_frames_sha = hash_pil_frames(frame_images)
        if visual_input_sha256 and actual_frames_sha != visual_input_sha256:
            raise TraceVerificationError(
                f"Decoded frame images hash ({actual_frames_sha}) != expected visual_input_sha256 ({visual_input_sha256})"
            )

        self._ensure_loaded()
        _apply_deterministic_rng(self.cfg.random_seed)

        t0 = perf_counter()
        peak_vram = 0
        try:
            import torch

            has_cuda = torch.cuda.is_available() and str(self.cfg.device).startswith("cuda")
            if has_cuda:
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            has_cuda = False

        try:
            text_in = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(
                text=[text_in],
                videos=[frame_images],
                padding=True,
                return_tensors="pt",
            )
            if hasattr(inputs, "to") and has_cuda:
                inputs = inputs.to(self.cfg.device)

            # Extract REAL model-observed processor tensors BEFORE generate()
            in_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
            pv = (
                inputs.get("pixel_values_videos", inputs.get("pixel_values"))
                if isinstance(inputs, dict)
                else getattr(inputs, "pixel_values_videos", getattr(inputs, "pixel_values", None))
            )
            in_ids_fp = _tensor_fingerprint(in_ids)
            pv_fp = _tensor_fingerprint(pv)
            if not pv_fp["present"]:
                raise TraceVerificationError(
                    f"Processor returned no visual tensor (pixel_values/pixel_values_videos) for {query.query_id}!"
                )

            gen_kwargs = {
                "max_new_tokens": self.cfg.max_new_tokens,
                "do_sample": self.cfg.do_sample,
            }
            if hasattr(self._model, "generate"):
                gen_ids = self._model.generate(**inputs, **gen_kwargs)
            else:
                raise RuntimeError("Loaded Modern VLM lacks .generate()")

            prompt_len = int(in_ids_fp["shape"][-1])
            trimmed = gen_ids[:, prompt_len:]
            caption = str(self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]).strip()
            if has_cuda:
                import torch

                peak_vram = int(torch.cuda.max_memory_allocated())
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise GeneratorOOMError(f"CUDA OOM in Modern VLM for {query.query_id}: {exc}") from exc
            raise

        elapsed = perf_counter() - t0
        actual_model_cfg = getattr(self._model, "config", None)
        actual_cfg_dict = (
            actual_model_cfg.to_dict() if hasattr(actual_model_cfg, "to_dict") else (dict(actual_model_cfg) if isinstance(actual_model_cfg, dict) else {})
        )

        # Build InputTrace strictly from ACTUAL processor/model observations; visual_evidence_ids=[] (PROMPT_TEXT_ONLY!)
        trace = InputTrace(
            query_id=query.query_id,
            dataset_id=query.dataset_id,
            stage=stage.value,
            arm=packet.arm,
            model_id=self.model_id,
            checkpoint_sha256=self.expected_checkpoint_sha256,
            visual_input_sha256=actual_frames_sha,
            frame_indices=list(frame_indices),
            text_evidence=list(packet.text_evidence),
            visual_evidence_ids=[],  # Modern VLM does NOT receive retrieved external video clips!
            visual_evidence_scores=[],
            prompt_rendered_sha256=primitive_hash({"rendered": rendered, "chat_template_text": text_in}),
            rng_seed=self.cfg.random_seed,
            decoding_config={
                "prompt_regime": self.cfg.prompt_regime,
                "max_new_tokens": self.cfg.max_new_tokens,
                "do_sample": self.cfg.do_sample,
                "temperature": self.cfg.temperature,
                "num_frames": self.cfg.num_frames,
                "min_pixels": self.cfg.min_pixels,
                "max_pixels": self.cfg.max_pixels,
                "bound_revision": self._bound_revision,
            },
            base_visual_mode=packet.base_visual_mode,
            injection_interface=self.injection_interface,
            actual_processor_telemetry={
                "actual_frames_received": len(frame_images),
                "actual_frames_sha256": actual_frames_sha,
                "pixel_values_shape": pv_fp["shape"],
                "pixel_values_sha256": pv_fp["sha256"],
                "input_ids_len": prompt_len,
                "input_ids_sha256": in_ids_fp["sha256"],
                "actual_model_config_sha256": primitive_hash(actual_cfg_dict),
            },
        )
        verify_actual_input_trace(trace, packet, self.expected_checkpoint_sha256, actual_frames_sha)

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


class SubprocessModernVLMAdapter(GeneratorAdapter):
    """Cross-env Modern VLM adapter invoking `modern_vlm_worker.py` in isolated `uarcap_q2_modern_vlm` env.

    Loads the Modern VLM ONCE per arm batch inside the isolated env, passes exact decoded 16 PIL frames
    via lossless PNGs, enforces shared GPU lock, verifies returned processor-observed InputTrace on the
    parent side, and guarantees full VRAM release upon worker process exit.
    """

    injection_interface = "PROMPT_TEXT_ONLY"

    def __init__(
        self,
        cfg: ModernVLMConfig,
        python_executable: str | None = None,
        conda_env: str | None = "uarcap_q2_modern_vlm",
        gpu_lock_path: Path | None = None,
        gpu_uuid: str | None = None,
        require_nvidia_smi_idle: bool = False,
        subprocess_runner: Any | None = None,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.model_id = cfg.model_id
        self.expected_checkpoint_sha256 = f"revision:{cfg.pinned_revision}"
        self.python_executable = python_executable
        self.conda_env = conda_env
        self.gpu_lock_path = gpu_lock_path
        self.gpu_uuid = gpu_uuid
        self.require_nvidia_smi_idle = require_nvidia_smi_idle
        self.subprocess_runner = subprocess_runner

    def generate_one(
        self,
        query: QueryRecord,
        packet: ArmInputPacket,
        stage: PilotStage,
        visual_input_sha256: str,
        frame_indices: list[int],
        frame_images: list[Any] | None = None,
    ) -> GenerationOutput:
        from .base import BatchItem

        outs = self.generate_batch(
            [BatchItem(query, packet, visual_input_sha256, frame_indices, frame_images)],
            stage=stage,
        )
        return outs[0]

    def generate_batch(
        self,
        items: list[Any],
        stage: PilotStage,
        work_dir: Path | None = None,
        run_id: str = "",
    ) -> list[GenerationOutput]:
        from dataclasses import asdict
        import tempfile
        from ..common.gpu_lock import GPULockConfig
        from ..common.gpu_runtime import run_isolated_gpu_subprocess
        from ..common.io_utils import atomic_write_json, read_jsonl

        if not items:
            return []

        worker_script = Path(__file__).resolve().parent / "modern_vlm_worker.py"
        lock_cfg = (
            GPULockConfig(
                lock_path=self.gpu_lock_path,
                run_id=f"modern_vlm_{items[0].packet.arm}",
                gpu_uuid=self.gpu_uuid,
                require_nvidia_smi_idle=self.require_nvidia_smi_idle,
            )
            if self.gpu_lock_path
            else None
        )

        with tempfile.TemporaryDirectory(prefix="uarcap_vlm_") as td:
            td_p = Path(td)
            req_items: list[dict[str, Any]] = []
            for idx, it in enumerate(items):
                if not it.frame_images or len(it.frame_images) != self.cfg.num_frames:
                    raise TraceVerificationError(
                        f"SubprocessModernVLMAdapter requires {self.cfg.num_frames} decoded PIL frames for {it.query.query_id}"
                    )
                fdir = td_p / f"frames_{idx:04d}"
                fdir.mkdir(parents=True, exist_ok=True)
                fpaths: list[str] = []
                for fi, im in enumerate(it.frame_images):
                    fp = fdir / f"f{fi:02d}.png"
                    im.save(fp, format="PNG")
                    fpaths.append(str(fp))
                req_items.append(
                    {
                        "query": it.query.to_dict(),
                        "packet": it.packet.to_dict(),
                        "visual_input_sha256": it.visual_input_sha256,
                        "frame_indices": list(it.frame_indices),
                        "frame_png_paths": fpaths,
                    }
                )

            req_p = td_p / "vlm_request.json"
            resp_p = td_p / "vlm_response.jsonl"
            atomic_write_json(
                req_p,
                {
                    "model_config": asdict(self.cfg),
                    "stage": stage.value,
                    "items": req_items,
                },
            )
            proc = run_isolated_gpu_subprocess(
                script_or_module=str(worker_script),
                script_args=["--request-json", str(req_p), "--response-jsonl", str(resp_p)],
                python_executable=self.python_executable,
                conda_env=self.conda_env,
                gpu_lock_cfg=lock_cfg,
                subprocess_runner=self.subprocess_runner,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Modern VLM worker failed (rc={proc.returncode}): {proc.stderr[-1000:]}")
            if not resp_p.is_file():
                raise RuntimeError(f"Modern VLM worker produced no response JSONL: {resp_p}")
            rows = read_jsonl(resp_p)
            if len(rows) != len(items):
                raise RuntimeError(f"Modern VLM worker returned {len(rows)} rows for {len(items)} items")

            outputs: list[GenerationOutput] = []
            for it, r in zip(items, rows):
                tr = InputTrace(**r["input_trace"])
                verify_actual_input_trace(tr, it.packet, self.expected_checkpoint_sha256, it.visual_input_sha256)
                outputs.append(
                    GenerationOutput(
                        query_id=str(r["query_id"]),
                        dataset_id=str(r["dataset_id"]),
                        stage=str(r["stage"]),
                        arm=str(r["arm"]),
                        model_id=str(r["model_id"]),
                        caption=str(r["caption"]),
                        input_trace=tr,
                        wall_seconds=float(r.get("wall_seconds", 0.0)),
                        peak_vram_bytes=int(r.get("peak_vram_bytes", 0)),
                        ok=bool(r.get("ok", False)),
                        error=str(r.get("error", "")),
                    )
                )
            return outputs
