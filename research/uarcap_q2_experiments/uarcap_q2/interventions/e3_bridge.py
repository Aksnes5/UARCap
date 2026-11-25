"""E3-Bridge: Low-cost same-cohort controlled intervention subset on DREAM-1K natural VideoUFO conflicts.

Scientific rules from v1.2-final review (Section 5 P0):
  - Reuses already human-verified DREAM-1K natural VideoUFO conflict videos (planned 50-60 videos).
  - Selects clean single-target visually verifiable conflict pairs (target 30-40 clips).
  - Enforces minimum threshold (>= 25 clean single-target pairs); if unmet, gracefully disables E3-Bridge
    and emits explicit claim-narrowing directive (never extrapolate VidHalluc interventions as proven full
    causal explanation of DREAM-1K natural errors!).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..common.schemas import EligibilityStatus
from .vidhalluc_e3 import (
    E3ApplicabilityChecklist,
    E3InterventionCandidate,
    audit_intervention_surface_balance,
)

E3_BRIDGE_MIN_CLEAN_PAIRS = 25


@dataclass
class NaturalConflictObservation:
    query_id: str
    domain: str
    verified_true_fact: str
    natural_videoufo_caption: str
    natural_conflict_fact: str
    single_target_clean: bool
    conflict_strictly_absent_in_video: bool
    proposed_neu_text: str
    reviewer_id: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_e3_bridge_cohort(
    observations: list[NaturalConflictObservation],
    min_clean_pairs: int = E3_BRIDGE_MIN_CLEAN_PAIRS,
    max_pairs: int = 40,
    source_dataset_id: str = "CaReBench",
) -> dict[str, Any]:
    """Filter verified natural E2 (CaReBench / DREAM-1K) x VideoUFO conflicts into an E3-Bridge cohort or disable."""
    approved: list[E3InterventionCandidate] = []
    rejected_ids: list[str] = []
    ds_slug = source_dataset_id.lower().replace("-", "")

    for obs in sorted(observations, key=lambda o: o.query_id):
        if not (obs.single_target_clean and obs.conflict_strictly_absent_in_video):
            rejected_ids.append(obs.query_id)
            continue
        sup = f"In the video, {obs.verified_true_fact.strip().rstrip('.')}."
        con = f"In the video, {obs.natural_conflict_fact.strip().rstrip('.')}."
        neu = obs.proposed_neu_text.strip() or "In the video, the main scene continues steadily."
        surface = audit_intervention_surface_balance(sup, con, neu)
        chk = E3ApplicabilityChecklist(
            distractor_strictly_absent=obs.conflict_strictly_absent_in_video,
            target_in_sampled_window=True,
            action_clear_in_sampled_frames=True,
            neu_strictly_neutral_and_compatible=bool(obs.proposed_neu_text.strip()),
            surface_balance_acceptable=surface["surface_balanced"],
            reviewer_id=obs.reviewer_id,
            notes=obs.notes,
        )
        cand = E3InterventionCandidate(
            query_id=obs.query_id,
            source_id=None,
            dataset_id=f"{source_dataset_id}-E3-Bridge",
            target_true_fact=obs.verified_true_fact.strip(),
            candidate_conflict_fact=obs.natural_conflict_fact.strip(),
            sup_text=sup,
            con_text=con,
            neu_text=neu,
            provenance=f"{ds_slug}_natural_videoufo_bridge",
            eligibility_status=EligibilityStatus.APPROVED.value if chk.all_passed else EligibilityStatus.REJECTED.value,
            surface_audit=surface,
            checklist=chk.to_dict(),
        )
        if cand.eligibility_status == EligibilityStatus.APPROVED.value:
            approved.append(cand)
        else:
            rejected_ids.append(obs.query_id)

    enabled = len(approved) >= min_clean_pairs
    selected = approved[:max_pairs] if enabled else []
    return {
        "module": "E3-Bridge",
        "source_dataset_id": source_dataset_id,
        "enabled": enabled,
        "status": "ENABLED_READY" if enabled else "DISABLED_INSUFFICIENT_CLEAN_PAIRS",
        "total_natural_conflicts_reviewed": len(observations),
        "clean_single_target_count": len(approved),
        "min_required_pairs": min_clean_pairs,
        "selected_count": len(selected),
        "rejected_query_ids": rejected_ids,
        "claim_scope_directive": (
            f"E3-Bridge supports within-cohort behavioral verification that real {source_dataset_id} x VideoUFO conflict patterns "
            "can induce corresponding errors under fixed visuals, without claiming full-cohort mediation."
            if enabled
            else f"E3-Bridge disabled (< {min_clean_pairs} clean pairs). Narrow mechanistic claims: report {source_dataset_id} observational "
            "association and VidHalluc action-focused controlled intervention separately without claiming VidHalluc causally "
            f"explains all {source_dataset_id} static/dynamic natural errors."
        ),
        "candidates": [c.to_dict() for c in selected],
    }
