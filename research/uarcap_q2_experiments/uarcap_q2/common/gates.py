"""Authorization gates, pilot-vs-formal isolation guards, and credential/license leak scanners."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .io_utils import read_json
from .schemas import PilotStage, VerdictStatus


class AuthorizationError(PermissionError):
    """Raised whenever an unapproved stage or Formal B run is attempted."""


class ProtocolIsolationError(ValueError):
    """Raised when pilot/formal isolation, visual locking, or license rules are violated."""


# Patterns for accidental secret/token leakage in configs, logs, or repo files
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("HF_TOKEN", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("OPENAI_KEY", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")),
    ("GITHUB_PAT", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("AWS_ACCESS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GENERIC_BEARER", re.compile(r"Authorization:\s*Bearer\s+[A-Za-z0-9._\-]{20,}", re.I)),
]

# Restricted data file patterns that must NEVER be committed to GitHub
FORBIDDEN_PUBLIC_RELEASE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r".*\.(mp4|avi|mkv|mov|webm|zip|tar|gz|pt|pth|bin|safetensors)$", re.I),
    re.compile(r".*ach_(mcq|binaryqa)\.json$", re.I),
    re.compile(r".*video\.zip$", re.I),
    re.compile(r".*ACH_videos\.zip$", re.I),
]


def scan_text_for_secrets(text: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for label, pat in SECRET_PATTERNS:
        for m in pat.finditer(text):
            hits.append({"kind": label, "start": m.start(), "length": len(m.group(0))})
    return hits


def scan_directory_for_secrets(root: Path | str, exclude_dirs: set[str] | None = None) -> list[dict[str, Any]]:
    excl = exclude_dirs or {".git", "__pycache__", ".pytest_cache", "node_modules"}
    root_p = Path(root)
    findings: list[dict[str, Any]] = []
    for p in root_p.rglob("*"):
        if any(part in excl for part in p.parts) or not p.is_file():
            continue
        if p.stat().st_size > 5 * 1024 * 1024:
            continue
        try:
            txt = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for hit in scan_text_for_secrets(txt):
            findings.append({"file": str(p), **hit})
    return findings


def check_public_release_compliance(file_paths: list[Path | str], allow_fixture_dir: str = "fixtures") -> list[str]:
    """Ensure restricted raw videos, model weights, or CC-BY-NC-ND JSON copies are not staged for release."""
    violations: list[str] = []
    for raw in file_paths:
        p = Path(raw)
        posix = p.as_posix()
        # Tiny synthetic test fixtures under tests/ or fixtures/ are exempt from raw-video rule if < 512 KiB
        if f"/{allow_fixture_dir}/" in f"/{posix}" or posix.startswith(f"{allow_fixture_dir}/"):
            if p.exists() and p.stat().st_size < 512 * 1024 and not re.search(r"ach_(mcq|binaryqa)\.json$", posix, re.I):
                continue
        for pat in FORBIDDEN_PUBLIC_RELEASE_PATTERNS:
            if pat.match(posix):
                violations.append(f"Forbidden public release asset (license/binary policy): {posix}")
                break
    return violations


def verify_stage_authorization(
    stage: PilotStage | str,
    auth_receipt_path: Path | str | None,
    prior_receipts: dict[str, Path | str] | None = None,
    allow_cpu_mock: bool = False,
) -> dict[str, Any]:
    """Enforce strict stage authorization and prerequisite receipts.

    - FORMAL_B is ALWAYS blocked unless:
        1. auth_receipt_path exists and explicitly authorizes FORMAL_B with non-empty approval_id,
        2. frozen_pilot_params_confirmed is True,
        3. prior receipts for PILOT_A1, PILOT_A2, PILOT_A3 all exist with verdict == PASS.
    - Real GPU execution of PILOT_A1/A2/A3 requires explicit stage approval receipt unless allow_cpu_mock=True.
    """
    stg = PilotStage(stage) if not isinstance(stage, PilotStage) else stage

    if stg == PilotStage.PILOT_A0 and allow_cpu_mock:
        return {"stage": stg.value, "authorized": True, "mode": "cpu_mock_or_preflight"}

    if stg == PilotStage.FORMAL_B:
        if not auth_receipt_path or not Path(auth_receipt_path).is_file():
            raise AuthorizationError(
                "FORMAL_B execution is HARD-BLOCKED by default. "
                "Provide a signed authorization receipt after completing Pilot A0-A3."
            )
        receipt = read_json(auth_receipt_path)
        if receipt.get("stage") != PilotStage.FORMAL_B.value:
            raise AuthorizationError(f"Authorization receipt stage mismatch: expected FORMAL_B, got {receipt.get('stage')}")
        if not receipt.get("approved") or not receipt.get("approval_id"):
            raise AuthorizationError("FORMAL_B receipt missing approved=true or non-empty approval_id")
        if receipt.get("frozen_pilot_params_confirmed") is not True:
            raise AuthorizationError(
                "FORMAL_B requires frozen_pilot_params_confirmed=true after Pilot A variance/judge freezing."
            )
        required_priors = [PilotStage.PILOT_A1.value, PilotStage.PILOT_A2.value, PilotStage.PILOT_A3.value]
        priors = prior_receipts or {}
        for req in required_priors:
            p = priors.get(req)
            if not p or not Path(p).is_file():
                raise AuthorizationError(f"FORMAL_B blocked: missing prerequisite receipt for {req}")
            pr = read_json(p)
            if pr.get("verdict") != VerdictStatus.PASS.value:
                raise AuthorizationError(f"FORMAL_B blocked: prerequisite {req} verdict is {pr.get('verdict')} (need PASS)")
        return {"stage": stg.value, "authorized": True, "approval_id": receipt["approval_id"], "mode": "formal"}

    if allow_cpu_mock:
        return {"stage": stg.value, "authorized": True, "mode": "cpu_mock"}

    if not auth_receipt_path or not Path(auth_receipt_path).is_file():
        raise AuthorizationError(
            f"Real execution of {stg.value} requires an explicit authorization receipt (--auth-receipt)."
        )
    receipt = read_json(auth_receipt_path)
    allowed_stages = receipt.get("authorized_stages", [receipt.get("stage")])
    if stg.value not in allowed_stages or not receipt.get("approved") or not receipt.get("approval_id"):
        raise AuthorizationError(f"Stage {stg.value} is not approved in {auth_receipt_path}")
    return {"stage": stg.value, "authorized": True, "approval_id": receipt["approval_id"], "mode": "authorized"}


def enforce_pilot_formal_exclusion(
    formal_query_ids: list[str],
    pilot_excluded_ids: set[str],
    formal_source_ids: list[str | None] | None = None,
    pilot_excluded_sources: set[str] | None = None,
) -> None:
    """Hard guard ensuring zero Pilot A queries or sources ever leak into Formal B."""
    q_overlap = sorted(set(formal_query_ids) & pilot_excluded_ids)
    if q_overlap:
        raise ProtocolIsolationError(
            f"Pilot-to-Formal contamination blocked! {len(q_overlap)} pilot query_ids in formal set: {q_overlap[:5]}"
        )
    if formal_source_ids and pilot_excluded_sources:
        s_overlap = sorted({s for s in formal_source_ids if s} & pilot_excluded_sources)
        if s_overlap:
            raise ProtocolIsolationError(
                f"Pilot-to-Formal source contamination blocked! {len(s_overlap)} sources overlap: {s_overlap[:5]}"
            )
