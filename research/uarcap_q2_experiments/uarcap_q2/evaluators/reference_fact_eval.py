"""UARCap Custom 3-Way Reference-Fact Diagnostic Protocol (Layer 3 — NEVER called 'fixed AutoDQ').

Strict validation safeguards:
  - Rejects invalid JSON, missing fields, illegal labels, duplicate claims, refusals, truncation,
    empty outputs, definite labels without evidence, and claim text mismatches.
  - NEVER defaults parse failures to entailed or contradicted!
  - Includes atomic proposition extraction audit (recall, false-claim omission rate, duplicates, granularity).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any

from ..common.schemas import RefRelation
from .backends import JudgeBackend

PROTOCOL_LAYER = "LAYER3_UARCAP_3WAY_REF_PROXY"
VALID_REF_LABELS = {e.value for e in RefRelation}
REFUSAL_RE = re.compile(r"\b(as an ai|i cannot|i'm unable to|cannot comply|policy prevents)\b", re.I)


class JudgeOutputValidationError(ValueError):
    """Raised when structured judge output fails strict schema/fidelity validation."""


@dataclass
class ClaimVerdict:
    claim: str
    label: str  # entailed | contradicted | unmentioned
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaptionRefFactEval:
    query_id: str
    arm: str
    model_id: str
    judge_model_id: str
    protocol_layer: str
    caption: str
    parse_status: str  # OK | PARSE_FAILED
    parse_error: str
    claims: list[ClaimVerdict] = field(default_factory=list)
    n_claims: int = 0
    n_contra_ref: int | None = None
    n_entail_ref: int | None = None
    n_unment_ref: int | None = None
    p_unment_ref: float | None = None
    r_core_ref: float | None = None
    core_event_verdicts: list[ClaimVerdict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["claims"] = [c.to_dict() for c in self.claims]
        d["core_event_verdicts"] = [c.to_dict() for c in self.core_event_verdicts]
        return d


def _strip_code_fence(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def parse_and_validate_verdicts(raw_response: str, expected_claims: list[str]) -> list[ClaimVerdict]:
    """Strictly parse and validate 3-way judge JSON output. NEVER silently defaults on error."""
    if not raw_response or not raw_response.strip():
        raise JudgeOutputValidationError("Empty judge output")
    if REFUSAL_RE.search(raw_response):
        raise JudgeOutputValidationError("Judge refused to evaluate input")
    cleaned = _strip_code_fence(raw_response)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise JudgeOutputValidationError(f"Invalid or truncated JSON: {exc}") from exc

    if not isinstance(obj, dict) or "claims" not in obj or not isinstance(obj["claims"], list):
        raise JudgeOutputValidationError("Missing top-level 'claims' list in JSON")

    items = obj["claims"]
    if len(items) != len(expected_claims):
        raise JudgeOutputValidationError(
            f"Claim count mismatch: expected {len(expected_claims)}, got {len(items)}"
        )

    verdicts: list[ClaimVerdict] = []
    seen_norm: set[str] = set()
    for idx, (entry, exp_claim) in enumerate(zip(items, expected_claims)):
        if not isinstance(entry, dict):
            raise JudgeOutputValidationError(f"Claim entry #{idx} is not an object")
        for req_key in ("claim", "label", "evidence"):
            if req_key not in entry:
                raise JudgeOutputValidationError(f"Missing required field '{req_key}' in claim #{idx}")
        claim_txt = str(entry["claim"]).strip()
        label = str(entry["label"]).strip().lower()
        evidence = str(entry["evidence"]).strip()

        if claim_txt.lower() != exp_claim.strip().lower():
            raise JudgeOutputValidationError(
                f"Returned claim mismatch at #{idx}: got '{claim_txt}', expected '{exp_claim}'"
            )
        norm_key = re.sub(r"\s+", " ", claim_txt.lower())
        if norm_key in seen_norm:
            raise JudgeOutputValidationError(f"Duplicate proposition in judge output: '{claim_txt}'")
        seen_norm.add(norm_key)

        if label not in VALID_REF_LABELS:
            raise JudgeOutputValidationError(f"Illegal NLI label '{label}' (allowed: {sorted(VALID_REF_LABELS)})")
        if label in (RefRelation.ENTAILED.value, RefRelation.CONTRADICTED.value) and not evidence:
            raise JudgeOutputValidationError(
                f"Definite label '{label}' provided without supporting evidence span/reason at #{idx}"
            )
        verdicts.append(ClaimVerdict(claim=claim_txt, label=label, evidence=evidence))

    return verdicts


def parse_extracted_claims(raw_response: str) -> list[str]:
    if not raw_response or not raw_response.strip():
        raise JudgeOutputValidationError("Empty claim extraction output")
    if REFUSAL_RE.search(raw_response):
        raise JudgeOutputValidationError("Judge refused claim extraction")
    cleaned = _strip_code_fence(raw_response)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise JudgeOutputValidationError(f"Invalid extraction JSON: {exc}") from exc
    if not isinstance(obj, dict) or not isinstance(obj.get("claims"), list):
        raise JudgeOutputValidationError("Extraction output missing 'claims' list")
    deduped: list[str] = []
    seen: set[str] = set()
    for item in obj["claims"]:
        s = str(item).strip()
        if not s:
            continue
        norm = re.sub(r"\s+", " ", s.lower())
        if norm not in seen:
            seen.add(norm)
            deduped.append(s)
    return deduped


class ThreeWayReferenceFactEvaluator:
    """Layer 3 UARCap 3-Way Reference-Fact Diagnostic Evaluator."""

    def __init__(self, backend: JudgeBackend, generator_model_family: str | None = None, enforce_cross_family: bool = True) -> None:
        if enforce_cross_family and generator_model_family:
            if backend.cfg.model_family.lower() == generator_model_family.lower():
                raise ValueError(
                    f"Cross-family judge policy violated: judge family '{backend.cfg.model_family}' "
                    f"matches generator family '{generator_model_family}'"
                )
        self.backend = backend

    def extract_claims(self, caption: str) -> list[str]:
        if not caption.strip():
            return []
        sys_p = "Extract atomic factual propositions from the caption. Output JSON {'claims': [...]} only."
        usr_p = f"[TASK:EXTRACT_CLAIMS]\nCAPTION: {caption.strip()}"
        raw = self.backend.complete(sys_p, usr_p)
        return parse_extracted_claims(raw)

    def evaluate_caption(
        self,
        query_id: str,
        arm: str,
        model_id: str,
        caption: str,
        reference_description: str,
        core_events: list[str],
    ) -> CaptionRefFactEval:
        try:
            claims = self.extract_claims(caption)
            if claims:
                sys_p = (
                    "Compare each claim against REFERENCE and classify into entailed, contradicted, or unmentioned. "
                    "Output strict JSON with keys claims:[{claim,label,evidence}]."
                )
                usr_p = (
                    f"[TASK:3WAY_REF_NLI]\nREFERENCE: {reference_description.strip()}\n"
                    f"CLAIMS_JSON: {json.dumps(claims, ensure_ascii=False)}"
                )
                raw_nli = self.backend.complete(sys_p, usr_p)
                verdicts = parse_and_validate_verdicts(raw_nli, claims)
            else:
                verdicts = []

            core_clean = [c.strip() for c in core_events if c and c.strip()]
            if core_clean and caption.strip():
                sys_c = (
                    "Check whether CAPTION entails each core reference event. Output strict JSON "
                    "with keys claims:[{claim,label,evidence}]."
                )
                usr_c = (
                    f"[TASK:3WAY_REF_NLI]\nREFERENCE: {caption.strip()}\n"
                    f"CLAIMS_JSON: {json.dumps(core_clean, ensure_ascii=False)}"
                )
                raw_core = self.backend.complete(sys_c, usr_c)
                core_verdicts = parse_and_validate_verdicts(raw_core, core_clean)
                r_core = sum(1 for v in core_verdicts if v.label == RefRelation.ENTAILED.value) / len(core_clean)
            elif core_clean:
                core_verdicts = []
                r_core = 0.0
            else:
                core_verdicts = []
                r_core = None

            n_contra = sum(1 for v in verdicts if v.label == RefRelation.CONTRADICTED.value)
            n_entail = sum(1 for v in verdicts if v.label == RefRelation.ENTAILED.value)
            n_unment = sum(1 for v in verdicts if v.label == RefRelation.UNMENTIONED.value)
            p_unment = (n_unment / len(verdicts)) if verdicts else 0.0

            return CaptionRefFactEval(
                query_id=query_id,
                arm=arm,
                model_id=model_id,
                judge_model_id=self.backend.cfg.model_id,
                protocol_layer=PROTOCOL_LAYER,
                caption=caption,
                parse_status="OK",
                parse_error="",
                claims=verdicts,
                n_claims=len(verdicts),
                n_contra_ref=n_contra,
                n_entail_ref=n_entail,
                n_unment_ref=n_unment,
                p_unment_ref=round(p_unment, 6),
                r_core_ref=round(r_core, 6) if r_core is not None else None,
                core_event_verdicts=core_verdicts,
            )
        except Exception as exc:
            # NEVER default parse failures to 0 errors or 100% entailed!
            return CaptionRefFactEval(
                query_id=query_id,
                arm=arm,
                model_id=model_id,
                judge_model_id=self.backend.cfg.model_id,
                protocol_layer=PROTOCOL_LAYER,
                caption=caption,
                parse_status="PARSE_FAILED",
                parse_error=str(exc),
            )


def _jaccard_tokens(a: str, b: str) -> float:
    sa = set(re.findall(r"[a-z0-9']+", a.lower()))
    sb = set(re.findall(r"[a-z0-9']+", b.lower()))
    return (len(sa & sb) / len(sa | sb)) if (sa or sb) else 0.0


def audit_atomic_extraction(
    extracted_claims_by_item: dict[str, list[str]],
    human_claims_by_item: dict[str, list[dict[str, Any]]],
    match_jaccard_threshold: float = 0.45,
) -> dict[str, Any]:
    """Audit atomic claim extractor quality, explicitly tracking false-claim omission rate."""
    total_human = 0
    matched_human = 0
    total_human_errors = 0
    omitted_human_errors = 0
    total_extracted = 0
    duplicate_extracted = 0
    granularity_anomalies = 0  # Overly long (>18 words) or trivial (<2 words)

    for item_id, h_claims in human_claims_by_item.items():
        ext = extracted_claims_by_item.get(item_id, [])
        total_extracted += len(ext)
        seen_ext: set[str] = set()
        for c in ext:
            norm = re.sub(r"\s+", " ", c.lower().strip())
            if norm in seen_ext:
                duplicate_extracted += 1
            seen_ext.add(norm)
            wlen = len(c.split())
            if wlen < 2 or wlen > 18:
                granularity_anomalies += 1

        for hc in h_claims:
            total_human += 1
            htext = str(hc["claim"])
            is_err = bool(hc.get("is_video_error", False))
            if is_err:
                total_human_errors += 1
            hit = any(_jaccard_tokens(htext, ec) >= match_jaccard_threshold for ec in ext)
            if hit:
                matched_human += 1
            elif is_err:
                omitted_human_errors += 1

    return {
        "total_human_claims": total_human,
        "total_extracted_claims": total_extracted,
        "extraction_recall": round(matched_human / total_human, 4) if total_human else None,
        "total_human_error_claims": total_human_errors,
        "false_claim_omission_rate": (
            round(omitted_human_errors / total_human_errors, 4) if total_human_errors else None
        ),
        "duplicate_claim_rate": round(duplicate_extracted / total_extracted, 4) if total_extracted else 0.0,
        "granularity_anomaly_rate": round(granularity_anomalies / total_extracted, 4) if total_extracted else 0.0,
    }
