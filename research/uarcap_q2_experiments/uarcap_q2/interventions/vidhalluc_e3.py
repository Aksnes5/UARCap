"""VidHalluc E3 controlled intervention candidate builder, surface-balance checker, and human applicability workflow.

Strict scientific rules:
  - Every candidate starts in CANDIDATE_PENDING_VERIFICATION; MCQ distractors NEVER auto-qualify as true conflicts.
  - Checks word count, tokenizer subword length balance (dual tokenizer), duplicate text, and non-target edit purity.
  - Requires explicit 5-item human applicability sign-off before entering runnable E3 manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Callable

from ..common.schemas import EligibilityStatus, QueryRecord


@dataclass
class E3ApplicabilityChecklist:
    distractor_strictly_absent: bool = False  # 1. f^- never co-occurs in clip
    target_in_sampled_window: bool = False  # 2. f^+ occurs inside sampled temporal window
    action_clear_in_sampled_frames: bool = False  # 3. 16 uniform frames preserve action cues
    neu_strictly_neutral_and_compatible: bool = False  # 4. Neu neutral to f^+/f^- & scene-compatible
    surface_balance_acceptable: bool = False  # 5. Balanced length/syntax/naturalness
    cooccurrence_risk_flagged: bool = False
    reviewer_id: str = ""
    notes: str = ""

    @property
    def all_passed(self) -> bool:
        return (
            self.distractor_strictly_absent
            and self.target_in_sampled_window
            and self.action_clear_in_sampled_frames
            and self.neu_strictly_neutral_and_compatible
            and self.surface_balance_acceptable
            and not self.cooccurrence_risk_flagged
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["all_passed"] = self.all_passed
        return d


@dataclass
class E3InterventionCandidate:
    query_id: str
    source_id: str | None
    dataset_id: str
    target_true_fact: str
    candidate_conflict_fact: str
    sup_text: str
    con_text: str
    neu_text: str
    provenance: str  # e.g., vidhalluc_ach_mcq_minimal_frame | dream1k_natural_videoufo_bridge
    eligibility_status: str = EligibilityStatus.CANDIDATE_PENDING_VERIFICATION.value
    surface_audit: dict[str, Any] = field(default_factory=dict)
    checklist: dict[str, Any] = field(default_factory=lambda: E3ApplicabilityChecklist().to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_subword_length(text: str) -> int:
    """Deterministic subword proxy when external model tokenizers are unavailable on CPU."""
    pieces = re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", text.strip())
    total = 0
    for p in pieces:
        total += 1 + max(0, (len(p) - 1) // 6)
    return total


def audit_intervention_surface_balance(
    sup_text: str,
    con_text: str,
    neu_text: str,
    vicuna_len_fn: Callable[[str], int] | None = None,
    modern_vlm_len_fn: Callable[[str], int] | None = None,
    max_token_diff: int = 3,
) -> dict[str, Any]:
    """Check word counts, dual-tokenizer subword balance, duplicate guard, and shared syntactic frame."""
    v_fn = vicuna_len_fn or default_subword_length
    m_fn = modern_vlm_len_fn or default_subword_length

    texts = {"sup": sup_text.strip(), "con": con_text.strip(), "neu": neu_text.strip()}
    duplicates = len({t.lower() for t in texts.values()}) < 3

    word_counts = {k: len(re.findall(r"[A-Za-z0-9']+", v)) for k, v in texts.items()}
    vicuna_lens = {k: int(v_fn(v)) for k, v in texts.items()}
    vlm_lens = {k: int(m_fn(v)) for k, v in texts.items()}

    vicuna_span = max(vicuna_lens.values()) - min(vicuna_lens.values())
    vlm_span = max(vlm_lens.values()) - min(vlm_lens.values())

    # Check non-target prefix stability between Sup and Con (minimal semantic edit check)
    sup_toks = re.findall(r"[a-z0-9']+", texts["sup"].lower())
    con_toks = re.findall(r"[a-z0-9']+", texts["con"].lower())
    shared_prefix = 0
    for a, b in zip(sup_toks, con_toks):
        if a == b:
            shared_prefix += 1
        else:
            break

    balanced = (not duplicates) and (vicuna_span <= max_token_diff) and (vlm_span <= max_token_diff)
    return {
        "duplicate_text_detected": duplicates,
        "word_counts": word_counts,
        "vicuna_subword_lengths": vicuna_lens,
        "modern_vlm_subword_lengths": vlm_lens,
        "vicuna_token_span": vicuna_span,
        "modern_vlm_token_span": vlm_span,
        "max_token_diff_allowed": max_token_diff,
        "sup_con_shared_prefix_words": shared_prefix,
        "surface_balanced": balanced,
    }


def build_vidhalluc_e3_candidate(
    record: QueryRecord,
    neu_text: str | None = None,
    frame_prefix: str = "In this video clip, a person is",
    vicuna_len_fn: Callable[[str], int] | None = None,
    modern_vlm_len_fn: Callable[[str], int] | None = None,
) -> E3InterventionCandidate:
    """Construct minimal-edit Sup/Con/Neu candidate kept strictly in CANDIDATE_PENDING_VERIFICATION."""
    if not record.target_true_fact or not record.candidate_conflict_fact:
        raise ValueError(f"Record {record.query_id} missing target_true_fact or candidate_conflict_fact")

    sup = f"{frame_prefix} {record.target_true_fact.strip().rstrip('.')}."
    con = f"{frame_prefix} {record.candidate_conflict_fact.strip().rstrip('.')}."
    neu = neu_text.strip() if neu_text else f"{frame_prefix} clearly visible in the main scene."

    surface = audit_intervention_surface_balance(
        sup, con, neu, vicuna_len_fn=vicuna_len_fn, modern_vlm_len_fn=modern_vlm_len_fn
    )
    return E3InterventionCandidate(
        query_id=record.query_id,
        source_id=record.source_id,
        dataset_id=record.dataset_id,
        target_true_fact=record.target_true_fact.strip(),
        candidate_conflict_fact=record.candidate_conflict_fact.strip(),
        sup_text=sup,
        con_text=con,
        neu_text=neu,
        provenance="vidhalluc_ach_mcq_minimal_frame",
        eligibility_status=EligibilityStatus.CANDIDATE_PENDING_VERIFICATION.value,
        surface_audit=surface,
        checklist=E3ApplicabilityChecklist(surface_balance_acceptable=surface["surface_balanced"]).to_dict(),
    )


def apply_human_applicability_review(
    candidate: E3InterventionCandidate,
    checklist: E3ApplicabilityChecklist,
    revised_neu_text: str | None = None,
) -> E3InterventionCandidate:
    """Transition candidate from CANDIDATE_PENDING_VERIFICATION to APPROVED or REJECTED after human check."""
    neu = revised_neu_text.strip() if revised_neu_text else candidate.neu_text
    surface = audit_intervention_surface_balance(candidate.sup_text, candidate.con_text, neu)
    status = EligibilityStatus.APPROVED.value if checklist.all_passed else EligibilityStatus.REJECTED.value
    return E3InterventionCandidate(
        query_id=candidate.query_id,
        source_id=candidate.source_id,
        dataset_id=candidate.dataset_id,
        target_true_fact=candidate.target_true_fact,
        candidate_conflict_fact=candidate.candidate_conflict_fact,
        sup_text=candidate.sup_text,
        con_text=candidate.con_text,
        neu_text=neu,
        provenance=candidate.provenance,
        eligibility_status=status,
        surface_audit=surface,
        checklist=checklist.to_dict(),
    )


def filter_runnable_e3_candidates(candidates: list[E3InterventionCandidate]) -> list[E3InterventionCandidate]:
    """Strictly allow ONLY human-APPROVED candidates into real E3 runs."""
    return [c for c in candidates if c.eligibility_status == EligibilityStatus.APPROVED.value]
