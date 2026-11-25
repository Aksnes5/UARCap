"""Blind independent dual-endpoint E3 semantic evaluator (Y_true, Y_conflict allowing co-occurrence).

Safeguards:
  - NEVER uses string matching as primary evaluation.
  - Runs TWO separate, randomized-order single-target blind queries per caption so the judge NEVER sees
    the arm label (Base/Sup/Con/Neu), intervention template, MCQ choices, or which fact is true/false!
  - Handles explicit negation (negated -> y=0), hedging (counted in primary y with hedged=True flag for
    strict sensitivity analysis), and generic hypernyms (generic_only -> y=0).
  - Enforces strict separation between intervention construction model and evaluation model.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from ..common.schemas import E3TargetAssessment
from .backends import JudgeBackend
from .reference_fact_eval import JudgeOutputValidationError, REFUSAL_RE, _strip_code_fence


def _parse_single_target_response(raw: str) -> dict[str, Any]:
    if not raw or not raw.strip():
        raise JudgeOutputValidationError("Empty E3 judge response")
    if REFUSAL_RE.search(raw):
        raise JudgeOutputValidationError("E3 judge refused evaluation")
    cleaned = _strip_code_fence(raw)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise JudgeOutputValidationError(f"Invalid E3 judge JSON: {exc}") from exc
    for key in ("expressed", "negated", "hedged", "generic_only", "evidence_span"):
        if key not in obj:
            raise JudgeOutputValidationError(f"Missing required E3 field '{key}'")
    return {
        "expressed": bool(obj["expressed"]),
        "negated": bool(obj["negated"]),
        "hedged": bool(obj["hedged"]),
        "generic_only": bool(obj["generic_only"]),
        "evidence_span": str(obj["evidence_span"]).strip(),
    }


class BlindE3SemanticEvaluator:
    """Evaluates Y_true and Y_conflict via two blind, randomized single-target queries."""

    def __init__(
        self,
        backend: JudgeBackend,
        intervention_builder_model_id: str | None = None,
        seed: int = 20261125,
    ) -> None:
        if intervention_builder_model_id and backend.cfg.model_id == intervention_builder_model_id:
            raise ValueError(
                "Circular evaluation blocked: E3 semantic evaluator model must differ from intervention builder model."
            )
        self.backend = backend
        self.rng = random.Random(seed)

    def _eval_single_target(self, caption: str, target_proposition: str) -> dict[str, Any]:
        sys_p = (
            "Determine whether CAPTION semantically expresses TARGET_PROPOSITION. "
            "Explicit negations count as expressed=false, negated=true. "
            "Overly generic hypernyms count as expressed=false, generic_only=true. "
            "Output strict JSON: {expressed, negated, hedged, generic_only, evidence_span}."
        )
        usr_p = f"[TASK:E3_SINGLE_TARGET]\nCAPTION: {caption.strip()}\nTARGET_PROPOSITION: {target_proposition.strip()}"
        raw = self.backend.complete(sys_p, usr_p)
        return _parse_single_target_response(raw)

    def evaluate_caption(
        self,
        query_id: str,
        arm: str,
        model_id: str,
        caption: str,
        target_true_fact: str,
        target_conflict_fact: str,
    ) -> E3TargetAssessment:
        # Randomize call order to eliminate ordering/position bias
        order = [("true", target_true_fact), ("conflict", target_conflict_fact)]
        self.rng.shuffle(order)

        results: dict[str, dict[str, Any]] = {}
        for role, prop in order:
            results[role] = self._eval_single_target(caption, prop)

        res_t = results["true"]
        res_c = results["conflict"]

        # Apply explicit negation & generic-only protocol rules
        y_true = 1 if (res_t["expressed"] and not res_t["negated"] and not res_t["generic_only"]) else 0
        y_conflict = 1 if (res_c["expressed"] and not res_c["negated"] and not res_c["generic_only"]) else 0

        assessment = E3TargetAssessment(
            query_id=query_id,
            arm=arm,
            model_id=model_id,
            y_true=y_true,
            y_conflict=y_conflict,
            negated_true=res_t["negated"],
            negated_conflict=res_c["negated"],
            hedged_true=res_t["hedged"],
            hedged_conflict=res_c["hedged"],
            generic_only=bool(res_t["generic_only"] and res_c["generic_only"]),
            evidence_span_true=res_t["evidence_span"],
            evidence_span_conflict=res_c["evidence_span"],
            evaluator_id=self.backend.cfg.model_id,
        )
        assessment.validate()
        return assessment
