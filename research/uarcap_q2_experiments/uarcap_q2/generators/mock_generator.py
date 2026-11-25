"""Deterministic CPU Mock Generator with controllable fault-injection modes for offline E2E testing."""

from __future__ import annotations

from time import perf_counter

from ..common.io_utils import primitive_hash
from ..common.schemas import ExpArm, InputTrace, PilotStage, QueryRecord
from ..retrieval.arm_builder import ArmInputPacket
from .base import GenerationOutput, GeneratorAdapter, GeneratorOOMError, verify_actual_input_trace


class MockGeneratorAdapter(GeneratorAdapter):
    """CPU-only deterministic mock generator. NEVER produces real scientific findings."""

    def __init__(
        self,
        model_id: str = "mock_videollama_cpu",
        expected_checkpoint_sha256: str = "mock_ckpt_sha256_0001",
        fault_mode: str | None = None,
        fault_on_query_id: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.expected_checkpoint_sha256 = expected_checkpoint_sha256
        self.fault_mode = fault_mode  # None | wrong_checkpoint | non_empty_a0 | visual_drift | empty_output | oom | timeout
        self.fault_on_query_id = fault_on_query_id

    def _should_fault(self, query_id: str) -> bool:
        return self.fault_mode is not None and (self.fault_on_query_id is None or self.fault_on_query_id == query_id)

    def generate_one(
        self,
        query: QueryRecord,
        packet: ArmInputPacket,
        stage: PilotStage,
        visual_input_sha256: str,
        frame_indices: list[int],
    ) -> GenerationOutput:
        t0 = perf_counter()
        fault = self.fault_mode if self._should_fault(query.query_id) else None
        if fault == "oom":
            raise GeneratorOOMError(f"Simulated CUDA OOM on {query.query_id}")
        if fault == "timeout":
            raise TimeoutError(f"Simulated generation timeout on {query.query_id}")

        ckpt_sha = "CORRUPTED_WRONG_CKPT_SHA" if fault == "wrong_checkpoint" else self.expected_checkpoint_sha256
        vis_sha = "DRIFTED_VISUAL_SHA" if fault == "visual_drift" else visual_input_sha256
        text_ev = (
            ["INJECTED_ILLEGAL_A0_CAPTION"]
            if (fault == "non_empty_a0" and packet.arm == ExpArm.ARM_A0.value)
            else list(packet.text_evidence)
        )

        trace = InputTrace(
            query_id=query.query_id,
            dataset_id=query.dataset_id,
            stage=stage.value,
            arm=packet.arm,
            model_id=self.model_id,
            checkpoint_sha256=ckpt_sha,
            visual_input_sha256=vis_sha,
            frame_indices=list(frame_indices),
            text_evidence=text_ev,
            visual_evidence_ids=list(packet.visual_clip_ids),
            visual_evidence_scores=list(packet.visual_scores),
            prompt_rendered_sha256=primitive_hash({"arm": packet.arm, "text": text_ev}),
            rng_seed=42,
            decoding_config={"mock": True, "max_new_tokens": 32},
            base_visual_mode=packet.base_visual_mode,
        )
        # Always verify actual trace before emitting output!
        verify_actual_input_trace(trace, packet, self.expected_checkpoint_sha256, visual_input_sha256)

        if fault == "empty_output":
            caption = ""
        elif packet.arm == ExpArm.E3_CON.value and query.candidate_conflict_fact:
            # Deterministic synthetic behavior for E3 pipeline testing only
            caption = f"A person is {query.candidate_conflict_fact} outside."
        elif packet.arm == ExpArm.E3_SUP.value and query.target_true_fact:
            caption = f"A person is {query.target_true_fact} outdoors."
        elif packet.arm in (ExpArm.E3_BASE.value, ExpArm.E3_NEU.value) and query.target_true_fact:
            caption = f"A person is {query.target_true_fact} in the yard."
        elif packet.arm == ExpArm.ARM_A7.value and packet.text_evidence:
            base_evt = (query.core_events[0] if query.core_events else query.reference_description).rstrip(".")
            tail_hint = packet.text_evidence[-1].split()[-1].strip(".")
            caption = f"{base_evt} near a {tail_hint}."
        else:
            base_evt = (query.core_events[0] if query.core_events else query.reference_description).rstrip(".")
            caption = f"{base_evt}."

        elapsed = perf_counter() - t0
        return GenerationOutput(
            query_id=query.query_id,
            dataset_id=query.dataset_id,
            stage=stage.value,
            arm=packet.arm,
            model_id=self.model_id,
            caption=caption,
            input_trace=trace,
            wall_seconds=round(elapsed, 6),
            peak_vram_bytes=0,
            ok=bool(caption),
            error="" if caption else "EMPTY_GENERATION",
        )
