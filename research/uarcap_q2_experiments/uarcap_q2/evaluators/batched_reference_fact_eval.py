"""Batched Layer-3 reference-fact evaluation with strict parsing, retries, and raw-response audit.

Why this exists: `ThreeWayReferenceFactEvaluator.evaluate_caption` calls `backend.complete`
once per prompt. With a subprocess-per-batch backend that would reload the 24B judge for
every prompt (~270 loads for Pilot A2). This module keeps the *exact same prompts and
strict parsers* but evaluates in three batched rounds (extract -> NLI -> core event),
deduplicates identical captions within a video (same input -> same verdicts), and records
every raw judge response, attempt, and parse status.

Scientific safeguards preserved:
  - identical caption + reference => identical evaluation input and verdicts across arms;
    only the deterministic per-arm claim_id differs, and the original arm mapping is kept.
  - no silent defaults on parse failure; failures are explicit and retried.
  - CaReBench `events`/`objects` are never used as Gold evidence.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import re
import time
from typing import Any, Callable

from ..common.schemas import RefRelation
from .backends import JudgeBackend
from .calibration import make_stable_claim_id
from .reference_fact_eval import (
    CaptionRefFactEval,
    PROTOCOL_LAYER,
    ClaimVerdict,
    classify_claim_fact_dimension,
    parse_and_validate_verdicts,
    parse_core_event_verdicts,
    parse_extracted_claims,
)

EXTRACT_SYS = "Extract atomic factual propositions as natural-language strings from the caption. Output JSON {'claims': ['...']} only."
NLI_SYS = (
    "Compare each claim against REFERENCE and classify into entailed, contradicted, or unmentioned. "
    "Copy each claim string exactly as provided. Output strict JSON with keys claims:[{claim,label,evidence}]."
)
CORE_SYS = (
    "You check whether a CAPTION entails each CORE EVENT. Output strict JSON "
    '{"verdicts": [{"label": "entailed|contradicted|unmentioned", "evidence": "..."}]} '
    "with exactly one verdict per core event, in the same order. Do not repeat the event text."
)
CORE_SINGLE_SYS = (
    "You check whether a CAPTION entails one CORE EVENT. Output strict JSON "
    '{"label": "entailed|contradicted|unmentioned", "evidence": "..."} only. '
    "Do not output a list, do not repeat the event text."
)
RETRY_SUFFIX = (
    "\n\nIMPORTANT: Return ONLY the JSON object. No prose, no markdown fences. "
    "Labels must be exactly one of entailed / contradicted / unmentioned. "
    "For entailed or contradicted, evidence must be a non-empty reference span or reason; "
    "return the claims in the same order and with the same text as provided."
)


def _sha(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _run_round(
    backend: JudgeBackend,
    stage: str,
    prompts: dict[Any, tuple[str, str]],
    parse_fn: Callable[[str], Any],
    *,
    max_retries: int,
    chunk_size: int,
    raw_calls: list[dict[str, Any]],
    context: dict[Any, dict[str, Any]],
) -> tuple[dict[Any, Any], dict[Any, str], int]:
    """Run one prompt per key with parse validation and retries. Returns (parsed, errors, n_retries)."""
    pending = list(prompts.keys())
    parsed: dict[Any, Any] = {}
    errors: dict[Any, str] = {}
    n_retries = 0
    for attempt in range(max_retries + 1):
        if not pending:
            break
        if attempt > 0:
            n_retries += len(pending)
        for chunk in _chunks(pending, chunk_size):
            batch_prompts: list[tuple[str, str]] = []
            for key in chunk:
                sp, up = prompts[key]
                if attempt > 0:
                    up = up + RETRY_SUFFIX
                batch_prompts.append((sp, up))
            t0 = time.time()
            responses = backend.complete_batch(batch_prompts)
            elapsed = round(time.time() - t0, 4)
            for pos, (key, raw) in enumerate(zip(chunk, responses)):
                ok = False
                err = ""
                try:
                    value = parse_fn(raw, key)
                    ok = True
                    parsed[key] = value
                    errors.pop(key, None)
                except Exception as exc:  # noqa: BLE001 - recorded, never defaulted
                    err = str(exc)
                    errors[key] = err
                    value = None
                ctx = context.get(key, {})
                sp_used, up_used = batch_prompts[pos]
                raw_calls.append(
                    {
                        "stage": stage,
                        "key_sha256": _sha(str(key)),
                        "query_id": ctx.get("query_id"),
                        "caption_sha256": ctx.get("caption_sha256"),
                        "attempt": attempt,
                        "system_prompt_sha256": _sha(sp_used),
                        "user_prompt_sha256": _sha(up_used),
                        "system_prompt": sp_used,
                        "user_prompt": up_used,
                        "raw_response": raw,
                        "parsed_ok": ok,
                        "parse_error": err,
                        "batch_elapsed_sec": elapsed,
                    }
                )
        pending = [k for k in pending if k not in parsed]
    return parsed, errors, n_retries


def evaluate_captions_batched(
    backend: JudgeBackend,
    items: list[dict[str, Any]],
    *,
    max_retries: int = 2,
    chunk_size: int = 24,
) -> dict[str, Any]:
    """Evaluate caption items with three batched rounds and exact-input duplicate reuse.

    Each item: {query_id, arm, model_id, caption, reference_description, core_events}
    Returns {"rows", "raw_calls", "unique_captions", "n_duplicate_reuses", "n_retries", "parse_failures"}.
    """
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    for it in items:
        key = (str(it["query_id"]), str(it["caption"]))
        if key not in unique:
            unique[key] = {
                "query_id": key[0],
                "caption": key[1],
                "reference_description": str(it["reference_description"]),
                "reference_fields": list(it.get("reference_fields", [])),
                "core_events": [c for c in it.get("core_events", []) if str(c).strip()],
                "arms": [],
            }
            order.append(key)
        unique[key]["arms"].append(str(it["arm"]))

    context = {
        key: {"query_id": key[0], "caption_sha256": _sha(key[1])}
        for key in order
    }
    raw_calls: list[dict[str, Any]] = []
    n_retries = 0

    # Round 1: atomic claim extraction
    extract_prompts = {
        key: (EXTRACT_SYS, f"[TASK:EXTRACT_CLAIMS]\nCAPTION: {unique[key]['caption'].strip()}")
        for key in order
        if unique[key]["caption"].strip()
    }
    claims_by_key, extract_err, r1 = _run_round(
        backend,
        "EXTRACT_CLAIMS",
        extract_prompts,
        lambda raw, key: parse_extracted_claims(raw),
        max_retries=max_retries,
        chunk_size=chunk_size,
        raw_calls=raw_calls,
        context=context,
    )
    n_retries += r1

    # Round 2: three-way NLI against the human reference
    nli_prompts: dict[tuple[str, str], tuple[str, str]] = {}
    for key in order:
        claims = claims_by_key.get(key)
        if claims:
            nli_prompts[key] = (
                NLI_SYS,
                f"[TASK:3WAY_REF_NLI]\nREFERENCE: {unique[key]['reference_description'].strip()}\n"
                f"CLAIMS_JSON: {json.dumps(claims, ensure_ascii=False)}",
            )
    verdicts_by_key, nli_err, r2 = _run_round(
        backend,
        "3WAY_REF_NLI",
        nli_prompts,
        lambda raw, key: parse_and_validate_verdicts(raw, claims_by_key[key]),
        max_retries=max_retries,
        chunk_size=chunk_size,
        raw_calls=raw_calls,
        context=context,
    )
    n_retries += r2

    # Round 3: core-event coverage (REFERENCE = generated caption; claims = pre-locked core events)
    core_prompts: dict[tuple[str, str], tuple[str, str]] = {}
    for key in order:
        core = unique[key]["core_events"]
        if core and unique[key]["caption"].strip():
            if len(core) == 1:
                core_prompts[key] = (
                    CORE_SINGLE_SYS,
                    f"[TASK:CORE_EVENT_CHECK_SINGLE]\nCAPTION: {unique[key]['caption'].strip()}\n"
                    f"CORE_EVENT: {core[0]}\n"
                    'Return exactly one JSON object with "label" and "evidence" (no list).',
                )
            else:
                core_prompts[key] = (
                    CORE_SYS,
                    f"[TASK:CORE_EVENT_CHECK]\nCAPTION: {unique[key]['caption'].strip()}\n"
                    f"CORE_EVENTS_JSON: {json.dumps(core, ensure_ascii=False)}\n"
                    "Return one verdict per core event, in the same order.",
                )
    core_by_key, core_err, r3 = _run_round(
        backend,
        "CORE_EVENT_NLI",
        core_prompts,
        lambda raw, key: parse_core_event_verdicts(raw, unique[key]["core_events"]),
        max_retries=max_retries,
        chunk_size=chunk_size,
        raw_calls=raw_calls,
        context=context,
    )
    n_retries += r3

    rows: list[dict[str, Any]] = []
    parse_failures = 0
    for it in items:
        key = (str(it["query_id"]), str(it["caption"]))
        u = unique[key]
        qid, arm, mid = str(it["query_id"]), str(it["arm"]), str(it["model_id"])
        caption = str(it["caption"])

        fail_stage = ""
        fail_error = ""
        verdicts: list[ClaimVerdict] = []
        core_verdicts: list[ClaimVerdict] = []
        if not caption.strip():
            pass
        elif key in extract_err:
            fail_stage, fail_error = "EXTRACT_CLAIMS", extract_err[key]
        elif key in nli_err:
            fail_stage, fail_error = "3WAY_REF_NLI", nli_err[key]
        elif key in core_err:
            fail_stage, fail_error = "CORE_EVENT_NLI", core_err[key]
        else:
            verdicts = verdicts_by_key.get(key, [])
            core_verdicts = core_by_key.get(key, [])

        if fail_stage:
            parse_failures += 1
            row = CaptionRefFactEval(
                query_id=qid,
                arm=arm,
                model_id=mid,
                judge_model_id=backend.cfg.model_id,
                protocol_layer=PROTOCOL_LAYER,
                caption=caption,
                parse_status="PARSE_FAILED",
                parse_error=f"[{fail_stage}] {fail_error}",
            )
        else:
            n_contra = sum(1 for v in verdicts if v.label == RefRelation.CONTRADICTED.value)
            n_entail = sum(1 for v in verdicts if v.label == RefRelation.ENTAILED.value)
            n_unment = sum(1 for v in verdicts if v.label == RefRelation.UNMENTIONED.value)
            denom = len(verdicts)
            dim_breakdown: dict[str, dict[str, int]] = {}
            for dim in ("static_entity_attribute", "dynamic_action_temporal", "mixed", "unclassified"):
                dv = [v for v in verdicts if v.fact_dimension_aux == dim]
                dim_breakdown[dim] = {
                    "total": len(dv),
                    "entailed": sum(1 for v in dv if v.label == RefRelation.ENTAILED.value),
                    "contradicted": sum(1 for v in dv if v.label == RefRelation.CONTRADICTED.value),
                    "unmentioned": sum(1 for v in dv if v.label == RefRelation.UNMENTIONED.value),
                }
            detail_breakdown: dict[str, dict[str, int]] = {}
            for dim in ("object_entity", "attribute", "action_event", "relation_spatial", "mixed", "unclassified"):
                dv = [v for v in verdicts if getattr(v, "fact_dimension_detail", "unclassified") == dim]
                detail_breakdown[dim] = {
                    "total": len(dv),
                    "entailed": sum(1 for v in dv if v.label == RefRelation.ENTAILED.value),
                    "contradicted": sum(1 for v in dv if v.label == RefRelation.CONTRADICTED.value),
                    "unmentioned": sum(1 for v in dv if v.label == RefRelation.UNMENTIONED.value),
                }
            core_clean = u["core_events"]
            r_core = (
                (sum(1 for v in core_verdicts if v.label == RefRelation.ENTAILED.value) / len(core_clean))
                if core_clean and caption.strip()
                else (0.0 if core_clean else None)
            )
            row = CaptionRefFactEval(
                query_id=qid,
                arm=arm,
                model_id=mid,
                judge_model_id=backend.cfg.model_id,
                protocol_layer=PROTOCOL_LAYER,
                caption=caption,
                parse_status="OK",
                parse_error="",
                claims=verdicts,
                n_claims=denom,
                n_contra_ref=n_contra,
                n_entail_ref=n_entail,
                n_unment_ref=n_unment,
                p_entail_ref=round(n_entail / denom, 6) if denom else 0.0,
                p_contra_ref=round(n_contra / denom, 6) if denom else 0.0,
                p_unment_ref=round(n_unment / denom, 6) if denom else 0.0,
                r_core_ref=round(r_core, 6) if r_core is not None else None,
                core_event_verdicts=core_verdicts,
                auxiliary_dimension_breakdown=dim_breakdown,
                auxiliary_dimension_detail_breakdown=detail_breakdown,
            )

        d = row.to_dict()
        d["claims"] = [
            {**c, "claim_id": make_stable_claim_id(qid, arm, mid, idx)}
            for idx, c in enumerate(d["claims"])
        ]
        d["caption_sha256"] = _sha(caption)
        d["evaluated_unique_caption_key"] = f"{qid}::{_sha(caption)[:16]}"
        d["reuse_group_size"] = len(u["arms"])
        d["reuse_arms"] = sorted(u["arms"])
        d["reference_fields_used"] = list(u.get("reference_fields", []))
        d["reference_description_sha256"] = _sha(u["reference_description"])
        d["fact_dimension_classifier"] = "rule_based_lexical_aux_v1"
        rows.append(d)

    return {
        "rows": rows,
        "raw_calls": raw_calls,
        "unique_captions": len(order),
        "n_duplicate_reuses": len(items) - len(order),
        "n_retries": n_retries,
        "parse_failures": parse_failures,
    }
