"""Unified authorization state machine, mock-vs-real receipt isolation, and secret/license leak scanners.

Strict state-machine rules (P0-5 remediation):
  - Strictly distinguishes StageRunStatus:
      MOCK_PASS | REAL_RUNTIME_PASS | HUMAN_VALIDATION_PENDING | INCONCLUSIVE |
      FAILED | BLOCKED | SCIENTIFIC_PROTOCOL_PENDING
  - MOCK_PASS can NEVER unlock ANY real subsequent stage (PILOT_A2, PILOT_A3, FORMAL_B) or dataset download!
  - Real PILOT_A2 requires PILOT_A1 receipt with status == REAL_RUNTIME_PASS.
  - Real PILOT_A3 requires PILOT_A2 receipt with status == REAL_RUNTIME_PASS AND human_calibration_completed == True.
  - Real FORMAL_B requires:
      1. PILOT_A1, PILOT_A2, PILOT_A3 all having status == REAL_RUNTIME_PASS (never merely existing or MOCK_PASS),
      2. Signed FORMAL_B authorization receipt binding:
           approved=True, approval_id, frozen_pilot_params_confirmed=True,
           frozen_formal_config_sha256 (verified against actual formal config file if provided),
           evaluator_version, asset_manifest_sha256, pilot_exclusion_manifest_sha256.
  - Asset downloads actively enforce stage_unlock authorization receipts!
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .io_utils import read_json, sha256_file
from .schemas import PilotStage, StageRunStatus


class AuthorizationError(PermissionError):
    """Raised whenever an unapproved stage, missing real prerequisite, or Formal B run is attempted."""


class ProtocolIsolationError(ValueError):
    """Raised when pilot/formal isolation, visual locking, or license rules are violated."""


SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("HF_TOKEN", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("OPENAI_KEY", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")),
    ("GITHUB_PAT", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("AWS_ACCESS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GENERIC_BEARER", re.compile(r"Authorization:\s*Bearer\s+[A-Za-z0-9._\-]{20,}", re.I)),
]

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
    violations: list[str] = []
    for raw in file_paths:
        p = Path(raw)
        posix = p.as_posix()
        if f"/{allow_fixture_dir}/" in f"/{posix}" or posix.startswith(f"{allow_fixture_dir}/"):
            if p.exists() and p.stat().st_size < 512 * 1024 and not re.search(r"ach_(mcq|binaryqa)\.json$", posix, re.I):
                continue
        for pat in FORBIDDEN_PUBLIC_RELEASE_PATTERNS:
            if pat.match(posix):
                violations.append(f"Forbidden public release asset (license/binary policy): {posix}")
                break
    return violations


def _verify_real_prior_receipt(stage_name: str, receipt_path: Path | str | None, require_human_calibration: bool = False) -> dict[str, Any]:
    if not receipt_path or not Path(receipt_path).is_file():
        raise AuthorizationError(f"Blocked: missing prerequisite receipt for {stage_name}.")
    pr = read_json(receipt_path)
    status = pr.get("status")
    exec_mode = pr.get("execution_mode", "")
    # Explicitly reject MOCK_PASS or legacy verdict-only mock receipts!
    if status == StageRunStatus.MOCK_PASS.value or exec_mode == "cpu_mock":
        raise AuthorizationError(
            f"Blocked: prerequisite {stage_name} receipt is MOCK_PASS (execution_mode={exec_mode}). "
            "Mock receipts can NEVER unlock real experimental stages!"
        )
    if status != StageRunStatus.REAL_RUNTIME_PASS.value:
        raise AuthorizationError(
            f"Blocked: prerequisite {stage_name} has status={status!r} (expected {StageRunStatus.REAL_RUNTIME_PASS.value!r})."
        )
    if require_human_calibration and pr.get("human_calibration_completed") is not True:
        raise AuthorizationError(
            f"Blocked: prerequisite {stage_name} has not completed real human calibration (human_calibration_completed!=True)."
        )
    return pr


def verify_stage_authorization(
    stage: PilotStage | str,
    auth_receipt_path: Path | str | None,
    prior_receipts: dict[str, Path | str] | None = None,
    allow_cpu_mock: bool = False,
    formal_config_path: Path | str | None = None,
    asset_manifest_path: Path | str | None = None,
    pilot_exclusion_path: Path | str | None = None,
) -> dict[str, Any]:
    """Enforce unified state-machine authorization & prerequisite receipt checks."""
    stg = PilotStage(stage) if not isinstance(stage, PilotStage) else stage
    priors = prior_receipts or {}

    if stg == PilotStage.PILOT_A0:
        return {
            "stage": stg.value,
            "authorized": True,
            "mode": "cpu_mock" if allow_cpu_mock else "real_preflight",
        }

    # FORMAL_B is ALWAYS hard-blocked even if allow_cpu_mock=True!
    if stg == PilotStage.FORMAL_B:
        if not auth_receipt_path or not Path(auth_receipt_path).is_file():
            raise AuthorizationError(
                "FORMAL_B execution is HARD-BLOCKED by default. "
                "Provide a signed authorization receipt after completing real Pilot A0-A3."
            )
        receipt = read_json(auth_receipt_path)
        if receipt.get("stage") != PilotStage.FORMAL_B.value:
            raise AuthorizationError(f"Authorization receipt stage mismatch: expected FORMAL_B, got {receipt.get('stage')}")
        if not receipt.get("approved") or not receipt.get("approval_id"):
            raise AuthorizationError("FORMAL_B receipt missing approved=true or non-empty approval_id")
        if receipt.get("frozen_pilot_params_confirmed") is not True:
            raise AuthorizationError("FORMAL_B requires frozen_pilot_params_confirmed=true after Pilot A freezing.")

        # Mandatory scientific & asset bindings on Formal B
        required_bindings = (
            "frozen_formal_config_sha256",
            "evaluator_version",
            "asset_manifest_sha256",
            "pilot_exclusion_manifest_sha256",
        )
        missing_bindings = [k for k in required_bindings if not str(receipt.get(k, "")).strip()]
        if missing_bindings:
            raise AuthorizationError(
                f"FORMAL_B receipt missing mandatory frozen bindings: {', '.join(missing_bindings)}"
            )

        # Cross-verify hashes against actual files when supplied
        if formal_config_path and Path(formal_config_path).is_file():
            actual_cfg_sha = sha256_file(formal_config_path)
            if actual_cfg_sha != receipt["frozen_formal_config_sha256"]:
                raise AuthorizationError(
                    f"FORMAL_B frozen config SHA-256 mismatch: file={actual_cfg_sha}, auth={receipt['frozen_formal_config_sha256']}"
                )
        if asset_manifest_path and Path(asset_manifest_path).is_file():
            if sha256_file(asset_manifest_path) != receipt["asset_manifest_sha256"]:
                raise AuthorizationError("FORMAL_B asset manifest SHA-256 mismatch against authorization receipt.")
        if pilot_exclusion_path and Path(pilot_exclusion_path).is_file():
            if sha256_file(pilot_exclusion_path) != receipt["pilot_exclusion_manifest_sha256"]:
                raise AuthorizationError("FORMAL_B pilot exclusion manifest SHA-256 mismatch against authorization receipt.")

        # Verify ALL 3 real Pilot stage receipts (NEVER accepting MOCK_PASS!)
        _verify_real_prior_receipt(PilotStage.PILOT_A1.value, priors.get(PilotStage.PILOT_A1.value))
        _verify_real_prior_receipt(
            PilotStage.PILOT_A2.value, priors.get(PilotStage.PILOT_A2.value), require_human_calibration=True
        )
        _verify_real_prior_receipt(PilotStage.PILOT_A3.value, priors.get(PilotStage.PILOT_A3.value))

        return {
            "stage": stg.value,
            "authorized": True,
            "approval_id": receipt["approval_id"],
            "evaluator_version": receipt["evaluator_version"],
            "mode": "real_formal",
        }

    # CPU mock mode is permitted ONLY for PILOT_A1/A2/A3 (never FORMAL_B) and stamps execution_mode='cpu_mock'
    if allow_cpu_mock:
        return {"stage": stg.value, "authorized": True, "mode": "cpu_mock"}

    # Real mode for PILOT_A1 / PILOT_A2 / PILOT_A3 requires explicit approval receipt
    if not auth_receipt_path or not Path(auth_receipt_path).is_file():
        raise AuthorizationError(
            f"Real execution of {stg.value} requires an explicit authorization receipt (--auth-receipt)."
        )
    receipt = read_json(auth_receipt_path)
    allowed_stages = receipt.get("authorized_stages", [receipt.get("stage")])
    if stg.value not in allowed_stages or not receipt.get("approved") or not receipt.get("approval_id"):
        raise AuthorizationError(f"Stage {stg.value} is not approved in {auth_receipt_path}")

    # Enforce sequential real prerequisites: A1 -> A2 -> A3
    if stg == PilotStage.PILOT_A2:
        _verify_real_prior_receipt(PilotStage.PILOT_A1.value, priors.get(PilotStage.PILOT_A1.value))
    elif stg == PilotStage.PILOT_A3:
        _verify_real_prior_receipt(
            PilotStage.PILOT_A2.value, priors.get(PilotStage.PILOT_A2.value), require_human_calibration=True
        )

    return {
        "stage": stg.value,
        "authorized": True,
        "approval_id": receipt["approval_id"],
        "mode": "real_authorized",
    }


def verify_asset_download_authorization(
    asset_id: str,
    stage_unlock: str,
    auth_receipt_path: Path | str | None,
    prior_receipts: dict[str, Path | str] | None = None,
) -> dict[str, Any]:
    """Actively enforce stage_unlock before allowing any large dataset/model download (P0-5)."""
    stg = PilotStage(stage_unlock)
    if stg == PilotStage.PILOT_A0:
        return {"asset_id": asset_id, "stage_unlock": stg.value, "authorized": True}
    return verify_stage_authorization(
        stage=stg,
        auth_receipt_path=auth_receipt_path,
        prior_receipts=prior_receipts,
        allow_cpu_mock=False,
    )


def enforce_pilot_formal_exclusion(
    formal_query_ids: list[str],
    pilot_excluded_ids: set[str],
    formal_source_ids: list[str | None] | None = None,
    pilot_excluded_sources: set[str] | None = None,
) -> None:
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
