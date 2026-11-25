"""Unified scientific & engineering data schemas.

Explicitly separates:
  - PilotStage (workflow stage: PILOT_A0, PILOT_A1, PILOT_A2, PILOT_A3, FORMAL_B)
  - ExpArm (experimental condition: ARM_A0, ARM_A1, ARM_A3, ARM_A7, E3_BASE, E3_SUP, E3_CON, E3_NEU)
  - Reference-fact proxy labels vs. Human video ground-truth labels
  - E3 independent dual binary endpoints (Y_true, Y_conflict) allowing co-occurrence
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class PilotStage(str, Enum):
    PILOT_A0 = "PILOT_A0"
    PILOT_A1 = "PILOT_A1"
    PILOT_A2 = "PILOT_A2"
    PILOT_A3 = "PILOT_A3"
    FORMAL_B = "FORMAL_B"


class ExpArm(str, Enum):
    ARM_A0 = "ARM_A0"  # k=0 strictly zero external evidence
    ARM_A1 = "ARM_A1"  # k=1 top-1 D text + visual evidence
    ARM_A3 = "ARM_A3"  # k=3 optional auxiliary D text + visual evidence
    ARM_A7 = "ARM_A7"  # k=7 top-7 D text + visual evidence
    E3_BASE = "E3_BASE"  # Controlled E3 base (Base=A0 or Base=B3)
    E3_SUP = "E3_SUP"  # Controlled E3 supporting target fact
    E3_CON = "E3_CON"  # Controlled E3 conflicting target fact
    E3_NEU = "E3_NEU"  # Controlled E3 target-neutral scene-compatible text


class SourceIdStatus(str, Enum):
    VERIFIED_EXACT = "VERIFIED_EXACT"  # e.g., 11-char YouTube ID parsed from VidHalluc
    UNKNOWN_OPAQUE = "UNKNOWN_OPAQUE"  # e.g., DREAM-1K integer vid 1..1000 (NEVER fabricate source_id!)
    NEAR_DUP_FLAGGED = "NEAR_DUP_FLAGGED"  # Flagged by X-CLIP / pHash / text near-dup screening


class EligibilityStatus(str, Enum):
    CANDIDATE_PENDING_VERIFICATION = "CANDIDATE_PENDING_VERIFICATION"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXCLUDED_PILOT = "EXCLUDED_PILOT"
    EXCLUDED_OVERLAP = "EXCLUDED_OVERLAP"
    CORRUPT_VIDEO = "CORRUPT_VIDEO"


class RefRelation(str, Enum):
    ENTAILED = "entailed"
    CONTRADICTED = "contradicted"
    UNMENTIONED = "unmentioned"


class VideoTruthLabel(str, Enum):
    VIDEO_SUPPORTED = "Video-Supported"
    VIDEO_CONTRADICTED = "Video-Contradicted"
    VIDEO_UNVERIFIABLE = "Video-Unverifiable"
    NON_FACTUAL = "Non-factual"


class VerdictStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class StageRunStatus(str, Enum):
    """Unified stage execution state machine (strictly separates MOCK_PASS vs REAL_RUNTIME_PASS)."""

    MOCK_PASS = "MOCK_PASS"
    REAL_RUNTIME_PASS = "REAL_RUNTIME_PASS"
    HUMAN_VALIDATION_PENDING = "HUMAN_VALIDATION_PENDING"
    INCONCLUSIVE = "INCONCLUSIVE"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    SCIENTIFIC_PROTOCOL_PENDING = "SCIENTIFIC_PROTOCOL_PENDING"


class FailureCategory(str, Enum):
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    MODEL_INCOMPATIBLE = "MODEL_INCOMPATIBLE"
    EVALUATOR_UNRELIABLE = "EVALUATOR_UNRELIABLE"
    SCIENTIFIC_NULL_EFFECT = "SCIENTIFIC_NULL_EFFECT"
    RESOURCE_UNACCEPTABLE = "RESOURCE_UNACCEPTABLE"


@dataclass
class QueryRecord:
    dataset_id: str
    query_id: str
    source_id: str | None  # None when unknown; NEVER fabricate
    source_id_status: str  # SourceIdStatus value
    video_path: str
    video_sha256: str | None
    duration: float | None
    domain: str
    reference_description: str
    reference_events: list[str] = field(default_factory=list)
    core_events: list[str] = field(default_factory=list)
    target_true_fact: str | None = None
    candidate_conflict_fact: str | None = None
    split: str = "unassigned"  # pilot_a | formal_b | excluded_overlap | unassigned
    eligibility_status: str = EligibilityStatus.CANDIDATE_PENDING_VERIFICATION.value
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.dataset_id or not self.query_id:
            raise ValueError("dataset_id and query_id are required")
        valid_src = {e.value for e in SourceIdStatus}
        if self.source_id_status not in valid_src:
            raise ValueError(f"Invalid source_id_status: {self.source_id_status}")
        if self.source_id_status == SourceIdStatus.UNKNOWN_OPAQUE.value and self.source_id is not None:
            raise ValueError("source_id must be None when source_id_status is UNKNOWN_OPAQUE (never fabricate source ID)")
        if self.source_id_status == SourceIdStatus.VERIFIED_EXACT.value and not self.source_id:
            raise ValueError("VERIFIED_EXACT requires a non-empty source_id")
        valid_elig = {e.value for e in EligibilityStatus}
        if self.eligibility_status not in valid_elig:
            raise ValueError(f"Invalid eligibility_status: {self.eligibility_status}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QueryRecord":
        obj = cls(**d)
        obj.validate()
        return obj


@dataclass
class RetrievalCandidate:
    caption: str
    score: float
    clip_ids: list[str]
    source_ids: list[str] = field(default_factory=list)
    rerank_score: float | None = None
    original_rank: int | None = None
    rerank_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InputTrace:
    query_id: str
    dataset_id: str
    stage: str  # PilotStage value
    arm: str  # ExpArm value
    model_id: str
    checkpoint_sha256: str
    visual_input_sha256: str
    frame_indices: list[int]
    text_evidence: list[str]
    visual_evidence_ids: list[str]
    visual_evidence_scores: list[float]
    prompt_rendered_sha256: str
    rng_seed: int
    decoding_config: dict[str, Any]
    base_visual_mode: str = "NONE"  # NONE | BASE_A0 | BASE_B3
    injection_interface: str = "BIMODAL_CROSS_GATING"  # BIMODAL_CROSS_GATING | PROMPT_TEXT_ONLY
    actual_processor_telemetry: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class E3TargetAssessment:
    query_id: str
    arm: str
    model_id: str
    y_true: int  # 0 or 1 (independent)
    y_conflict: int  # 0 or 1 (independent; co-occurrence (1,1) explicitly allowed!)
    negated_true: bool = False
    negated_conflict: bool = False
    hedged_true: bool = False
    hedged_conflict: bool = False
    generic_only: bool = False
    evidence_span_true: str = ""
    evidence_span_conflict: str = ""
    evaluator_id: str = ""

    def validate(self) -> None:
        if self.y_true not in (0, 1) or self.y_conflict not in (0, 1):
            raise ValueError("y_true and y_conflict must be binary (0 or 1)")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
