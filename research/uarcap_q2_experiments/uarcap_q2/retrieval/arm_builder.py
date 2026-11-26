"""Verifiable multi-arm input constructor for natural retrieval (A0/A1/A3/A7) and E3 (Base/Sup/Con/Neu).

Enforces strict scientific invariants:
  - ARM_A0: strictly empty external text and empty external visual evidence (cap=[[],[],[]], sims=[[],[]]).
  - ARM_A1 / ARM_A3 / ARM_A7: exact prefix consistency (A1 is prefix of A3 is prefix of A7) on the same
    reranked D candidate list!
  - E3 (Base / Sup / Con / Neu): locks visual evidence identically across all 4 arms under either
    BASE_A0 (zero external visual features) or BASE_B3 (3 locked visual heads with shared visual contract SHA-256).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..common.io_utils import primitive_hash
from ..common.schemas import ExpArm, RetrievalCandidate


@dataclass
class ArmInputPacket:
    query_id: str
    arm: str
    text_topn: int
    visual_topn: int
    text_evidence: list[str]
    visual_clip_ids: list[str]
    visual_scores: list[float]
    evcap_cap_entry: list[Any]  # Exact EVCap wire format: [captions, dummy_or_rerank, clip_groups]
    evcap_sims_entry: list[Any]  # Exact EVCap wire format: [captions, scores]
    visual_contract_sha256: str
    base_visual_mode: str = "NONE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _visual_contract_digest(clip_ids: list[str], scores: list[float], mode: str) -> str:
    return primitive_hash({"mode": mode, "clip_ids": clip_ids, "scores": [round(s, 6) for s in scores]})


def build_natural_retrieval_arm(
    query_id: str,
    arm: ExpArm | str,
    reranked_candidates: list[RetrievalCandidate],
) -> ArmInputPacket:
    arm_enum = ExpArm(arm) if not isinstance(arm, ExpArm) else arm
    k_map = {
        ExpArm.ARM_A0: 0,
        ExpArm.ARM_A1: 1,
        ExpArm.ARM_A3: 3,
        ExpArm.ARM_A5: 5,
        ExpArm.ARM_A7: 7,
    }
    if arm_enum not in k_map:
        raise ValueError(f"Arm {arm_enum.value} is not a natural retrieval arm")
    k = k_map[arm_enum]

    if k == 0:
        cap_entry: list[Any] = [[], [], []]
        sims_entry: list[Any] = [[], []]
        return ArmInputPacket(
            query_id=query_id,
            arm=arm_enum.value,
            text_topn=0,
            visual_topn=0,
            text_evidence=[],
            visual_clip_ids=[],
            visual_scores=[],
            evcap_cap_entry=cap_entry,
            evcap_sims_entry=sims_entry,
            visual_contract_sha256=_visual_contract_digest([], [], "NATURAL_K0"),
            base_visual_mode="BASE_A0",
        )

    if len(reranked_candidates) < k:
        raise ValueError(f"Query {query_id} has only {len(reranked_candidates)} candidates, need k={k}")

    prefix = reranked_candidates[:k]
    texts = [c.caption for c in prefix]
    scores = [float(c.score) for c in prefix]
    rerank_scores = [float(c.rerank_score if c.rerank_score is not None else c.score) for c in prefix]
    clip_groups = [list(c.clip_ids) for c in prefix]
    if any(not grp for grp in clip_groups):
        raise ValueError(f"Empty clip_ids in candidate prefix for {query_id}")
    if len(set(texts)) != len(texts):
        raise ValueError(f"Duplicate captions in candidate prefix for {query_id}")
    reps = [grp[0] for grp in clip_groups]

    return ArmInputPacket(
        query_id=query_id,
        arm=arm_enum.value,
        text_topn=k,
        visual_topn=k,
        text_evidence=texts,
        visual_clip_ids=reps,
        visual_scores=scores,
        evcap_cap_entry=[texts, rerank_scores, clip_groups],
        evcap_sims_entry=[texts, scores],
        visual_contract_sha256=_visual_contract_digest(reps, scores, f"NATURAL_K{k}"),
        base_visual_mode="NONE",
    )


def verify_natural_prefix_consistency(packets_by_arm: dict[str, ArmInputPacket]) -> None:
    """Assert exact A0 empty invariant and A1 ⊂ A3 ⊂ A7 prefix identity."""
    if ExpArm.ARM_A0.value in packets_by_arm:
        a0 = packets_by_arm[ExpArm.ARM_A0.value]
        if a0.text_evidence or a0.visual_clip_ids or a0.evcap_cap_entry != [[], [], []] or a0.evcap_sims_entry != [[], []]:
            raise ValueError(f"ARM_A0 violation for {a0.query_id}: external evidence must be strictly empty")

    ordered = [
        (1, packets_by_arm.get(ExpArm.ARM_A1.value)),
        (3, packets_by_arm.get(ExpArm.ARM_A3.value)),
        (5, packets_by_arm.get(ExpArm.ARM_A5.value)),
        (7, packets_by_arm.get(ExpArm.ARM_A7.value)),
    ]
    present = [(k, p) for k, p in ordered if p is not None]
    for i in range(len(present) - 1):
        k_small, p_small = present[i]
        k_large, p_large = present[i + 1]
        if p_large.text_evidence[:k_small] != p_small.text_evidence:
            raise ValueError(
                f"Prefix consistency violation ({p_small.arm} vs {p_large.arm}) for {p_small.query_id}: text mismatch"
            )
        if p_large.visual_clip_ids[:k_small] != p_small.visual_clip_ids:
            raise ValueError(
                f"Prefix consistency violation ({p_small.arm} vs {p_large.arm}) for {p_small.query_id}: visual clip mismatch"
            )
        if p_large.visual_scores[:k_small] != p_small.visual_scores:
            raise ValueError(
                f"Prefix consistency violation ({p_small.arm} vs {p_large.arm}) for {p_small.query_id}: score mismatch"
            )


def build_e3_intervention_arms(
    query_id: str,
    sup_text: str,
    con_text: str,
    neu_text: str,
    base_visual_mode: str = "BASE_A0",
    b3_visual_prefix: list[RetrievalCandidate] | None = None,
) -> dict[str, ArmInputPacket]:
    """Build 4 visual-locked E3 arms (Base / Sup / Con / Neu) sharing identical visual contract."""
    if base_visual_mode not in ("BASE_A0", "BASE_B3"):
        raise ValueError(f"Unsupported base_visual_mode: {base_visual_mode}")

    if base_visual_mode == "BASE_A0":
        base_texts: list[str] = []
        vis_clips: list[str] = []
        vis_scores: list[float] = []
        vis_groups: list[list[str]] = []
        visual_topn = 0
    else:
        if not b3_visual_prefix or len(b3_visual_prefix) < 3:
            raise ValueError(f"BASE_B3 requires 3 visual head candidates for {query_id}")
        head3 = b3_visual_prefix[:3]
        base_texts = [c.caption for c in head3]
        vis_clips = [c.clip_ids[0] for c in head3]
        vis_scores = [float(c.score) for c in head3]
        vis_groups = [list(c.clip_ids) for c in head3]
        visual_topn = 3

    shared_vis_sha = _visual_contract_digest(vis_clips, vis_scores, base_visual_mode)
    arm_extra_text = {
        ExpArm.E3_BASE.value: [],
        ExpArm.E3_SUP.value: [sup_text.strip()],
        ExpArm.E3_CON.value: [con_text.strip()],
        ExpArm.E3_NEU.value: [neu_text.strip()],
    }

    packets: dict[str, ArmInputPacket] = {}
    for arm_name, extra in arm_extra_text.items():
        if extra and not extra[0]:
            raise ValueError(f"Empty intervention text for {query_id} {arm_name}")
        full_texts = base_texts + extra
        if base_visual_mode == "BASE_A0":
            # In Base=A0, text is injected while visual external evidence stays strictly empty (visual_topn=0)
            cap_entry = [full_texts, [1.0] * len(full_texts), [[] for _ in full_texts]] if full_texts else [[], [], []]
            sims_entry = [full_texts, [1.0] * len(full_texts)] if full_texts else [[], []]
        else:
            cap_entry = [full_texts, vis_scores + [1.0] * len(extra), vis_groups + [[] for _ in extra]]
            sims_entry = [full_texts, vis_scores + [1.0] * len(extra)]

        packets[arm_name] = ArmInputPacket(
            query_id=query_id,
            arm=arm_name,
            text_topn=len(full_texts),
            visual_topn=visual_topn,
            text_evidence=full_texts,
            visual_clip_ids=list(vis_clips),
            visual_scores=list(vis_scores),
            evcap_cap_entry=cap_entry,
            evcap_sims_entry=sims_entry,
            visual_contract_sha256=shared_vis_sha,
            base_visual_mode=base_visual_mode,
        )
    return packets


def verify_e3_visual_lock(e3_packets: dict[str, ArmInputPacket]) -> None:
    """Assert that Base / Sup / Con / Neu carry identical visual contract and visual clips."""
    required = {ExpArm.E3_BASE.value, ExpArm.E3_SUP.value, ExpArm.E3_CON.value, ExpArm.E3_NEU.value}
    missing = required - set(e3_packets)
    if missing:
        raise ValueError(f"Missing E3 arms: {sorted(missing)}")
    digests = {e3_packets[a].visual_contract_sha256 for a in required}
    if len(digests) != 1:
        raise ValueError(f"E3 visual lock violation: visual contracts differ across arms: {digests}")
    clip_lists = {tuple(e3_packets[a].visual_clip_ids) for a in required}
    if len(clip_lists) != 1:
        raise ValueError("E3 visual lock violation: visual_clip_ids differ across arms")
