"""Qwen3.8-27B-FP8 / Qwen3.5-27B Multimodal Video Adapter reusing existing BN vLLM 0.29.0 environment.

Design principles:
  - Prioritizes reusing already-deployed BN models (primary: Qwen/Qwen3.8-27B-FP8, backup: Qwen/Qwen3.5-27B)
    on RTX A6000 48GB with vLLM 0.29.0 / PyTorch 2.13.0+cu130 / CUDA 13.0 (FP8 block-128 / Marlin W8A16).
  - ZERO new model downloads by default (`offline_only=True`).
  - Supports 16 actual decoded PIL frames (`frame_images`), short aligned captions, retrieved text injection
    (`PROMPT_TEXT_ONLY`, `visual_evidence_ids=[]`), explicit thinking control (`enable_thinking=False` default),
    and full processor/engine-observed InputTrace verification.
  - Supports three runtime modes:
      1. `vllm_offline`: Direct `vllm.LLM` batch inference (or CPU-injectable fake engine).
      2. `vllm_openai_http`: Local offline vLLM OpenAI-compatible HTTP server (`127.0.0.1`, or injectable fake HTTP).
      3. `vllm_subprocess`: Spawns `qwen3_vllm_worker.py` inside the existing BN vLLM Python env with shared
         GPU lock enforcement and guaranteed VRAM release upon worker exit (or injectable fake subprocess).
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
import re
import tempfile
from time import perf_counter
from typing import Any, Callable
import urllib.request

import numpy as np

from ..common.gpu_lock import GPULockConfig
from ..common.gpu_runtime import release_gpu_memory, run_isolated_gpu_subprocess
from ..common.io_utils import atomic_write_json, primitive_hash, read_jsonl, sha256_bytes
from ..common.schemas import InputTrace, PilotStage, QueryRecord
from ..retrieval.arm_builder import ArmInputPacket
from .base import (
    BatchItem,
    GenerationOutput,
    GeneratorAdapter,
    GeneratorOOMError,
    TraceVerificationError,
    verify_actual_input_trace,
)
from .modern_vlm_adapter import (
    _apply_deterministic_rng,
    hash_pil_frames,
    render_modern_vlm_prompt,
)

DEFAULT_BN_QWEN38_PATH = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/models/Qwen3.8-27B-FP8"
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.S | re.I)
THINK_UNCLOSED_RE = re.compile(r"<think>.*$", re.S | re.I)


@dataclass
class Qwen3VLLMConfig:
    model_id: str = "Qwen/Qwen3.8-27B-FP8"
    local_model_path: str = DEFAULT_BN_QWEN38_PATH
    backend_mode: str = "vllm_offline"  # vllm_offline | vllm_openai_http | vllm_subprocess
    api_base_url: str = ""
    vllm_python_executable: str = ""
    vllm_conda_env: str = ""
    pinned_revision: str = "LOCAL_BN_QWEN38_FP8_66SHARDS"
    offline_only: bool = True
    num_frames: int = 16
    multimodal_input_key: str = "video"  # video | image (multi-frame fallback)
    prompt_regime: str = "short_aligned"
    max_new_tokens: int = 48
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    enable_thinking: bool = False
    strip_thinking_tags: bool = True
    quantization: str = "fp8"
    gpu_memory_utilization: float = 0.88
    max_model_len: int = 8192
    enforce_eager: bool = False
    random_seed: int = 42
    device: str = "cuda:0"
    expected_num_shards: int | None = None
    timeout_sec: int = 1800

    def validate(self) -> None:
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if self.backend_mode not in ("vllm_offline", "vllm_openai_http", "vllm_subprocess"):
            raise ValueError(f"Unsupported backend_mode: {self.backend_mode}")
        if self.multimodal_input_key not in ("video", "image"):
            raise ValueError(f"Unsupported multimodal_input_key: {self.multimodal_input_key}")
        if self.prompt_regime not in ("short_aligned", "detailed_sensitivity"):
            raise ValueError(f"Unsupported prompt_regime: {self.prompt_regime}")
        if self.prompt_regime == "short_aligned" and self.max_new_tokens > 64:
            raise ValueError("short_aligned regime must cap max_new_tokens <= 64 to prevent length drift")
        if self.backend_mode == "vllm_openai_http" and not self.api_base_url:
            raise ValueError("vllm_openai_http backend requires api_base_url (e.g. http://127.0.0.1:8000/v1)")
        if self.offline_only and self.backend_mode != "vllm_openai_http" and not self.local_model_path:
            raise ValueError("offline_only=True requires explicit local_model_path to existing BN model")


def strip_thinking_block(raw_text: str) -> tuple[str, bool, int]:
    """Strip any accidental `<think>...</think>` reasoning block and report whether stripping occurred."""
    s = str(raw_text or "")
    cleaned = THINK_BLOCK_RE.sub("", s)
    cleaned = THINK_UNCLOSED_RE.sub("", cleaned)
    stripped_chars = len(s) - len(cleaned)
    return cleaned.strip(), (stripped_chars > 0), max(0, stripped_chars)


def verify_local_vllm_model_dir(
    local_dir: Path,
    pinned_revision: str,
    expected_num_shards: int | None = None,
) -> dict[str, Any]:
    """Inspect existing local BN model directory without loading weights (config + shard index check)."""
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Offline Qwen3 local_model_path missing: {local_dir}")

    cfg_path = local_dir / "config.json"
    cfg_data: dict[str, Any] = {}
    if cfg_path.is_file():
        cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))

    rev_file = local_dir / "model_revision.json"
    bound_rev = pinned_revision
    if rev_file.is_file():
        meta = json.loads(rev_file.read_text(encoding="utf-8"))
        actual = str(meta.get("revision", "")).strip()
        if pinned_revision and pinned_revision not in ("PENDING_PIN_AT_DOWNLOAD", "LOCAL_BN_QWEN38_FP8_66SHARDS") and actual != pinned_revision:
            raise TraceVerificationError(f"Qwen3 model revision mismatch: local={actual!r}, pinned={pinned_revision!r}")
        if actual:
            bound_rev = actual

    shard_info: dict[str, Any] = {"index_present": False, "n_shards_listed": 0, "n_shards_present": 0, "missing_shards": []}
    idx_path = local_dir / "model.safetensors.index.json"
    if idx_path.is_file():
        idx_obj = json.loads(idx_path.read_text(encoding="utf-8"))
        wmap = idx_obj.get("weight_map", {})
        unique_shards = sorted(set(wmap.values()))
        missing = [s for s in unique_shards if not (local_dir / s).is_file()]
        shard_info = {
            "index_present": True,
            "n_shards_listed": len(unique_shards),
            "n_shards_present": len(unique_shards) - len(missing),
            "missing_shards": missing[:10],
        }
        if missing:
            raise FileNotFoundError(
                f"Incomplete local model shards in {local_dir}: {len(missing)}/{len(unique_shards)} missing (e.g. {missing[:3]})"
            )
        if expected_num_shards is not None and len(unique_shards) != expected_num_shards:
            raise TraceVerificationError(
                f"Expected {expected_num_shards} shards in {local_dir}, found {len(unique_shards)}"
            )

    manifest_summary = {
        "local_model_path": str(local_dir),
        "bound_revision": bound_rev,
        "architectures": cfg_data.get("architectures", []),
        "model_type": cfg_data.get("model_type", ""),
        "has_vision_config": bool("vision_config" in cfg_data or "visual" in cfg_data),
        "shards": shard_info,
    }
    manifest_summary["local_model_manifest_sha256"] = primitive_hash(manifest_summary)
    return manifest_summary


def encode_pil_frames_to_jpeg_data_urls(frame_images: list[Any], quality: int = 90) -> list[str]:
    """Encode 16 PIL RGB frames into deterministic JPEG base64 data URLs for local vLLM HTTP chat API."""
    urls: list[str] = []
    for im in frame_images:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        urls.append(f"data:image/jpeg;base64,{b64}")
    return urls


class Qwen3VLLMMultimodalAdapter(GeneratorAdapter):
    """Primary Modern Video VLM adapter for Qwen3.8-27B-FP8 (primary) / Qwen3.5-27B (backup) via vLLM."""

    injection_interface = "PROMPT_TEXT_ONLY"

    def __init__(
        self,
        cfg: Qwen3VLLMConfig,
        vllm_engine: Any | None = None,
        http_requester: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        subprocess_runner: Any | None = None,
        gpu_lock_path: Path | None = None,
        gpu_uuid: str | None = None,
        require_nvidia_smi_idle: bool = False,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.model_id = cfg.model_id
        self._engine = vllm_engine
        self._http_requester = http_requester
        self._subprocess_runner = subprocess_runner
        self.gpu_lock_path = gpu_lock_path
        self.gpu_uuid = gpu_uuid
        self.require_nvidia_smi_idle = require_nvidia_smi_idle
        self._manifest: dict[str, Any] = {"bound_revision": cfg.pinned_revision, "local_model_manifest_sha256": ""}
        self.expected_checkpoint_sha256 = f"revision:{cfg.pinned_revision}"

    def _ensure_manifest(self) -> dict[str, Any]:
        if self.cfg.local_model_path and Path(self.cfg.local_model_path).is_dir():
            self._manifest = verify_local_vllm_model_dir(
                Path(self.cfg.local_model_path),
                self.cfg.pinned_revision,
                expected_num_shards=self.cfg.expected_num_shards,
            )
            self.expected_checkpoint_sha256 = f"revision:{self._manifest['bound_revision']}"
        elif self.cfg.offline_only and self.cfg.backend_mode == "vllm_offline" and self._engine is None:
            raise FileNotFoundError(f"Offline Qwen3 local_model_path missing: {self.cfg.local_model_path}")
        return self._manifest

    def _ensure_vllm_engine(self) -> Any:
        self._ensure_manifest()
        if self._engine is not None:
            return self._engine
        from vllm import LLM  # type: ignore[import-not-found]

        llm_kwargs: dict[str, Any] = {
            "model": str(self.cfg.local_model_path),
            "gpu_memory_utilization": self.cfg.gpu_memory_utilization,
            "max_model_len": self.cfg.max_model_len,
            "enforce_eager": self.cfg.enforce_eager,
            "seed": self.cfg.random_seed,
            "limit_mm_per_prompt": {"video": 1} if self.cfg.multimodal_input_key == "video" else {"image": self.cfg.num_frames},
        }
        if self.cfg.quantization:
            llm_kwargs["quantization"] = self.cfg.quantization
        self._engine = LLM(**llm_kwargs)
        return self._engine

    def build_multimodal_messages(
        self,
        packet: ArmInputPacket,
        frame_images: list[Any],
        use_data_urls: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, str], str]:
        """Construct Qwen3 multimodal video messages and compute deterministic visual payload SHA-256."""
        if not frame_images or len(frame_images) != self.cfg.num_frames:
            raise TraceVerificationError(
                f"Qwen3 VLM requires exactly {self.cfg.num_frames} decoded PIL frames, got {len(frame_images) if frame_images else 0}"
            )
        rendered = render_modern_vlm_prompt(packet, self.cfg.prompt_regime)
        if use_data_urls:
            urls = encode_pil_frames_to_jpeg_data_urls(frame_images)
            visual_payload_sha = sha256_bytes("|".join(urls).encode("utf-8"))
            user_content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
            user_content.append({"type": "text", "text": rendered["user"]})
        else:
            frame_arrays = np.stack([np.asarray(im.convert("RGB"), dtype=np.uint8) for im in frame_images], axis=0)
            visual_payload_sha = sha256_bytes(frame_arrays.tobytes())
            if self.cfg.multimodal_input_key == "video":
                user_content = [
                    {"type": "video", "video": frame_arrays},
                    {"type": "text", "text": rendered["user"]},
                ]
            else:
                user_content = [{"type": "image", "image": im.convert("RGB")} for im in frame_images]
                user_content.append({"type": "text", "text": rendered["user"]})

        messages = [
            {"role": "system", "content": rendered["system"]},
            {"role": "user", "content": user_content},
        ]
        return messages, rendered, visual_payload_sha

    def _call_local_http(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        url = self.cfg.api_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.cfg.model_id,
            "messages": messages,
            "max_tokens": self.cfg.max_new_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "seed": self.cfg.random_seed,
            "chat_template_kwargs": {"enable_thinking": self.cfg.enable_thinking},
        }
        if self._http_requester is not None:
            return self._http_requester(url, payload)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _finalize_output(
        self,
        query: QueryRecord,
        packet: ArmInputPacket,
        stage: PilotStage,
        frame_images: list[Any],
        frame_indices: list[int],
        actual_frames_sha: str,
        visual_payload_sha: str,
        rendered: dict[str, str],
        raw_text: str,
        prompt_tokens: int,
        completion_tokens: int,
        elapsed: float,
        peak_vram: int,
    ) -> GenerationOutput:
        if self.cfg.strip_thinking_tags:
            caption, had_think, stripped_chars = strip_thinking_block(raw_text)
        else:
            caption, had_think, stripped_chars = str(raw_text or "").strip(), False, 0

        resolutions = [list(im.size) for im in frame_images]
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
            visual_evidence_ids=[],  # PROMPT_TEXT_ONLY: never injects external visual clip IDs!
            visual_evidence_scores=[],
            prompt_rendered_sha256=primitive_hash({"rendered": rendered, "enable_thinking": self.cfg.enable_thinking}),
            rng_seed=self.cfg.random_seed,
            decoding_config={
                "backend_mode": self.cfg.backend_mode,
                "prompt_regime": self.cfg.prompt_regime,
                "max_new_tokens": self.cfg.max_new_tokens,
                "do_sample": self.cfg.do_sample,
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "enable_thinking": self.cfg.enable_thinking,
                "quantization": self.cfg.quantization,
                "num_frames": self.cfg.num_frames,
                "bound_revision": self._manifest.get("bound_revision", self.cfg.pinned_revision),
            },
            base_visual_mode=packet.base_visual_mode,
            injection_interface=self.injection_interface,
            actual_processor_telemetry={
                "backend_mode": self.cfg.backend_mode,
                "actual_frames_received": len(frame_images),
                "actual_frames_sha256": actual_frames_sha,
                "frame_resolutions_wh": resolutions[0] if resolutions else None,
                "multimodal_input_key": self.cfg.multimodal_input_key,
                "visual_payload_sha256": visual_payload_sha,
                "enable_thinking": self.cfg.enable_thinking,
                "thinking_tags_stripped": had_think,
                "thinking_chars_stripped": stripped_chars,
                "prompt_token_count": int(prompt_tokens),
                "input_ids_len": int(prompt_tokens),
                "completion_token_count": int(completion_tokens),
                "local_model_manifest_sha256": self._manifest.get("local_model_manifest_sha256", ""),
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

    def generate_one(
        self,
        query: QueryRecord,
        packet: ArmInputPacket,
        stage: PilotStage,
        visual_input_sha256: str,
        frame_indices: list[int],
        frame_images: list[Any] | None = None,
    ) -> GenerationOutput:
        outs = self.generate_batch(
            [BatchItem(query, packet, visual_input_sha256, frame_indices, frame_images)],
            stage=stage,
        )
        return outs[0]

    def generate_batch(
        self,
        items: list[BatchItem],
        stage: PilotStage,
        work_dir: Path | None = None,
        run_id: str = "",
    ) -> list[GenerationOutput]:
        if not items:
            return []

        # Validate all decoded frames upfront before touching GPU / subprocess!
        verified_hashes: list[str] = []
        for it in items:
            if not it.frame_images or len(it.frame_images) != self.cfg.num_frames:
                raise TraceVerificationError(
                    f"Qwen3 VLM received NO/invalid decoded video frames for {it.query.query_id} "
                    f"(expected {self.cfg.num_frames}, got {len(it.frame_images) if it.frame_images else 0})"
                )
            fsha = hash_pil_frames(it.frame_images)
            if it.visual_input_sha256 and fsha != it.visual_input_sha256:
                raise TraceVerificationError(
                    f"Decoded frame images hash ({fsha}) != expected visual_input_sha256 ({it.visual_input_sha256})"
                )
            verified_hashes.append(fsha)

        if self.cfg.backend_mode == "vllm_subprocess":
            return self._generate_batch_subprocess(items, stage, work_dir=work_dir, run_id=run_id)

        self._ensure_manifest()
        _apply_deterministic_rng(self.cfg.random_seed)

        if self.cfg.backend_mode == "vllm_openai_http":
            outputs: list[GenerationOutput] = []
            for it, fsha in zip(items, verified_hashes):
                t0 = perf_counter()
                msgs, rendered, vsha = self.build_multimodal_messages(it.packet, it.frame_images or [], use_data_urls=True)
                resp = self._call_local_http(msgs)
                raw_txt = str(resp["choices"][0]["message"]["content"])
                usage = resp.get("usage", {})
                outputs.append(
                    self._finalize_output(
                        it.query,
                        it.packet,
                        stage,
                        it.frame_images or [],
                        it.frame_indices,
                        fsha,
                        vsha,
                        rendered,
                        raw_txt,
                        int(usage.get("prompt_tokens", 0)),
                        int(usage.get("completion_tokens", 0)),
                        perf_counter() - t0,
                        0,
                    )
                )
            return outputs

        # Default: vllm_offline (loads model ONCE per batch, runs vllm.LLM.chat with chat_template_kwargs!)
        engine = self._ensure_vllm_engine()
        batch_msgs: list[list[dict[str, Any]]] = []
        batch_rendered: list[dict[str, str]] = []
        batch_vsha: list[str] = []
        for it in items:
            msgs, rendered, vsha = self.build_multimodal_messages(it.packet, it.frame_images or [], use_data_urls=False)
            batch_msgs.append(msgs)
            batch_rendered.append(rendered)
            batch_vsha.append(vsha)

        t0 = perf_counter()
        try:
            try:
                from vllm import SamplingParams  # type: ignore[import-not-found]

                sp: Any = SamplingParams(
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    max_tokens=self.cfg.max_new_tokens,
                    seed=self.cfg.random_seed,
                )
            except ImportError:
                sp = {
                    "temperature": self.cfg.temperature,
                    "top_p": self.cfg.top_p,
                    "max_tokens": self.cfg.max_new_tokens,
                    "seed": self.cfg.random_seed,
                }
            engine_outputs = engine.chat(
                batch_msgs,
                sampling_params=sp,
                chat_template_kwargs={"enable_thinking": self.cfg.enable_thinking},
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise GeneratorOOMError(f"CUDA OOM in Qwen3 vLLM batch: {exc}") from exc
            raise
        elapsed_each = (perf_counter() - t0) / max(len(items), 1)

        peak_vram = 0
        try:
            import torch

            if torch.cuda.is_available():
                peak_vram = int(torch.cuda.max_memory_allocated())
        except ImportError:
            pass

        outputs = []
        for it, fsha, rendered, vsha, eo in zip(items, verified_hashes, batch_rendered, batch_vsha, engine_outputs):
            if isinstance(eo, dict):
                raw_txt = str(eo.get("text", ""))
                ptoks = int(eo.get("prompt_tokens", 0))
                ctoks = int(eo.get("completion_tokens", 0))
            else:
                first = eo.outputs[0]
                raw_txt = str(first.text)
                ptoks = len(getattr(eo, "prompt_token_ids", []) or [])
                ctoks = len(getattr(first, "token_ids", []) or [])
            outputs.append(
                self._finalize_output(
                    it.query,
                    it.packet,
                    stage,
                    it.frame_images or [],
                    it.frame_indices,
                    fsha,
                    vsha,
                    rendered,
                    raw_txt,
                    ptoks,
                    ctoks,
                    elapsed_each,
                    peak_vram,
                )
            )
        return outputs

    def _generate_batch_subprocess(
        self,
        items: list[BatchItem],
        stage: PilotStage,
        work_dir: Path | None = None,
        run_id: str = "",
    ) -> list[GenerationOutput]:
        worker_script = Path(__file__).resolve().parent / "qwen3_vllm_worker.py"
        lock_cfg = (
            GPULockConfig(
                lock_path=self.gpu_lock_path,
                run_id=f"{run_id or 'qwen3_vllm'}_{items[0].packet.arm}",
                gpu_uuid=self.gpu_uuid,
                require_nvidia_smi_idle=self.require_nvidia_smi_idle,
            )
            if self.gpu_lock_path
            else None
        )
        worker_cfg = asdict(self.cfg)
        worker_cfg["backend_mode"] = "vllm_offline"

        with tempfile.TemporaryDirectory(prefix="uarcap_qwen3_", dir=str(work_dir) if work_dir else None) as td:
            td_p = Path(td)
            req_items: list[dict[str, Any]] = []
            for idx, it in enumerate(items):
                fdir = td_p / f"frames_{idx:04d}"
                fdir.mkdir(parents=True, exist_ok=True)
                fpaths: list[str] = []
                for fi, im in enumerate(it.frame_images or []):
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
            req_p = td_p / "qwen3_request.json"
            resp_p = td_p / "qwen3_response.jsonl"
            atomic_write_json(req_p, {"model_config": worker_cfg, "stage": stage.value, "items": req_items})
            proc = run_isolated_gpu_subprocess(
                script_or_module=str(worker_script),
                script_args=["--request-json", str(req_p), "--response-jsonl", str(resp_p)],
                python_executable=self.cfg.vllm_python_executable or None,
                conda_env=self.cfg.vllm_conda_env or None,
                gpu_lock_cfg=lock_cfg,
                timeout_sec=self.cfg.timeout_sec,
                subprocess_runner=self._subprocess_runner,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Qwen3 vLLM worker failed (rc={proc.returncode}): {proc.stderr[-1000:]}")
            if not resp_p.is_file():
                raise RuntimeError(f"Qwen3 vLLM worker wrote no response JSONL: {resp_p}")
            rows = read_jsonl(resp_p)
            if len(rows) != len(items):
                raise RuntimeError(f"Qwen3 vLLM worker returned {len(rows)} rows for {len(items)} items")

            outputs: list[GenerationOutput] = []
            for it, r in zip(items, rows):
                tr = InputTrace(**r["input_trace"])
                expected_sha = tr.checkpoint_sha256 if self.cfg.pinned_revision == "LOCAL_BN_QWEN38_FP8_66SHARDS" else self.expected_checkpoint_sha256
                verify_actual_input_trace(tr, it.packet, expected_sha, it.visual_input_sha256)
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

    def close(self) -> None:
        self._engine = None
        release_gpu_memory()
