"""Abstract generator interface, BatchItem carrying real decoded frames, and strict actual-input trace verifier."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..common.schemas import ExpArm, InputTrace, PilotStage, QueryRecord
from ..retrieval.arm_builder import ArmInputPacket


class GeneratorOOMError(MemoryError):
    """Raised on CUDA OOM so the runner halts safely without corrupting outputs."""


class TraceVerificationError(ValueError):
    """Raised when actual model inputs deviate from the declared arm packet, visual contract, or checkpoint."""


@dataclass
class BatchItem:
    query: QueryRecord
    packet: ArmInputPacket
    visual_input_sha256: str
    frame_indices: list[int]
    frame_images: list[Any] = field(default_factory=list)  # Actual decoded PIL.Image frames
    external_frame_images: list[Any] = field(default_factory=list)
    external_frame_indices: list[int] = field(default_factory=list)
    external_video_sha256: str = ""


@dataclass
class GenerationOutput:
    query_id: str
    dataset_id: str
    stage: str
    arm: str
    model_id: str
    caption: str
    input_trace: InputTrace
    wall_seconds: float
    peak_vram_bytes: int
    ok: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["input_trace"] = self.input_trace.to_dict()
        return d


def verify_actual_input_trace(
    trace: InputTrace,
    packet: ArmInputPacket,
    expected_checkpoint_sha256: str,
    expected_visual_sha256: str,
) -> None:
    """Verify actual observed model inputs rather than trusting declarative config alone."""
    if trace.query_id != packet.query_id:
        raise TraceVerificationError(f"Query mismatch in trace: {trace.query_id} != {packet.query_id}")
    if trace.arm != packet.arm:
        raise TraceVerificationError(f"Arm mismatch in trace: {trace.arm} != {packet.arm}")
    if expected_checkpoint_sha256 and trace.checkpoint_sha256 != expected_checkpoint_sha256:
        raise TraceVerificationError(
            f"Checkpoint SHA-256 mismatch: got {trace.checkpoint_sha256}, expected {expected_checkpoint_sha256}"
        )
    if expected_visual_sha256 and trace.visual_input_sha256 != expected_visual_sha256:
        raise TraceVerificationError(
            f"Visual input hash drift detected for {trace.query_id} ({trace.arm}): "
            f"{trace.visual_input_sha256} != {expected_visual_sha256}"
        )
    if trace.arm == ExpArm.ARM_A0.value:
        if trace.text_evidence or trace.visual_evidence_ids or trace.visual_evidence_scores:
            raise TraceVerificationError(f"ARM_A0 trace violation for {trace.query_id}: non-empty external evidence")
    if trace.text_evidence != packet.text_evidence:
        raise TraceVerificationError(f"Actual text_evidence in trace differs from arm packet for {trace.query_id}")

    if trace.injection_interface == "PROMPT_TEXT_ONLY":
        # Modern VLM uses text-only prompt injection: MUST NOT falsely log external visual features it never received!
        if trace.visual_evidence_ids or trace.visual_evidence_scores:
            raise TraceVerificationError(
                f"PROMPT_TEXT_ONLY model {trace.model_id} falsely logged external visual evidence for {trace.query_id}"
            )
        tel = trace.actual_processor_telemetry or {}
        if not tel.get("actual_frames_received") or tel.get("input_ids_len", 0) <= 0:
            raise TraceVerificationError(
                f"PROMPT_TEXT_ONLY trace missing real processor telemetry (frames/tokens) for {trace.query_id}"
            )
        if tel.get("actual_frames_sha256") != expected_visual_sha256:
            raise TraceVerificationError(
                f"Actual frames passed to processor differ from decoded visual hash for {trace.query_id}"
            )
    else:
        if trace.visual_evidence_ids != packet.visual_clip_ids:
            raise TraceVerificationError(
                f"Actual visual_evidence_ids in trace differ from arm packet for {trace.query_id}"
            )


class GeneratorAdapter(ABC):
    model_id: str
    expected_checkpoint_sha256: str
    injection_interface: str = "BIMODAL_CROSS_GATING"

    def generate_one(
        self,
        query: QueryRecord,
        packet: ArmInputPacket,
        stage: PilotStage,
        visual_input_sha256: str,
        frame_indices: list[int],
        frame_images: list[Any] | None = None,
    ) -> GenerationOutput:
        raise NotImplementedError("Override generate_one or generate_batch")

    def generate_batch(
        self,
        items: list[BatchItem],
        stage: PilotStage,
        work_dir: Path | str,
        run_id: str = "run",
    ) -> list[GenerationOutput]:
        """Default batch loop; batch-native 7B adapters (FrozenVideoLLaMAAdapter) override to load once per arm."""
        return [
            self.generate_one(
                it.query,
                it.packet,
                stage,
                it.visual_input_sha256,
                it.frame_indices,
                frame_images=it.frame_images,
            )
            for it in items
        ]
