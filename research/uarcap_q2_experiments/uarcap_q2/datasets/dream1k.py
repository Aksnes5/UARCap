"""DREAM-1K dataset adapter: metadata normalization, core-event pre-extraction, and stratified sampling.

Primary-source facts pinned in protocol v1.2-final:
  - HF repo: omni-research/DREAM-1K (gated: auto, video/video.zip = 3,849,677,473 bytes / 3.59 GiB)
  - License notice: HF card lists apache-2.0, while extra_gated_prompt explicitly states video copyrights
    belong to original creators/platforms and are for academic research use only (NEVER redistribute videos).
  - Source ID reality: public vid is opaque integer 1..1000 across 5 domains (200 clips each):
      1..200 movie_animation, 201..400 movie_live_action, 401..600 vertical_video,
      601..800 videvo, 801..1000 youtube.
    Therefore source_id MUST remain None with SourceIdStatus.UNKNOWN_OPAQUE unless a verified source URL/ID exists.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import re
from typing import Any
import urllib.error
import urllib.request

from ..common.io_utils import atomic_write_json, read_json
from ..common.schemas import EligibilityStatus, QueryRecord, SourceIdStatus

DREAM1K_HF_REPO = "omni-research/DREAM-1K"
DREAM1K_HF_URL = "https://huggingface.co/datasets/omni-research/DREAM-1K"
DREAM1K_EXPECTED_VIDEO_ZIP_BYTES = 3849677473  # 3.59 GiB / 3.85 GB
DREAM1K_DOMAINS = (
    "movie_animation",
    "movie_live_action",
    "vertical_video",
    "videvo",
    "youtube",
)


def infer_dream1k_domain(vid_int: int, raw_source: str | None = None) -> str:
    if raw_source:
        s = raw_source.strip().lower()
        for dom in DREAM1K_DOMAINS:
            if dom in s:
                return dom
        if "live" in s:
            return "movie_live_action"
        if "anim" in s:
            return "movie_animation"
        if "short" in s or "tiktok" in s or "vertical" in s:
            return "vertical_video"
        if "stock" in s or "videvo" in s:
            return "videvo"
        if "youtube" in s:
            return "youtube"
    if 1 <= vid_int <= 200:
        return "movie_animation"
    if 201 <= vid_int <= 400:
        return "movie_live_action"
    if 401 <= vid_int <= 600:
        return "vertical_video"
    if 601 <= vid_int <= 800:
        return "videvo"
    if 801 <= vid_int <= 1000:
        return "youtube"
    return "unknown"


def extract_core_events_pre_model(reference_description: str, reference_events: list[str], max_core: int = 2) -> list[str]:
    """Deterministic pre-model extraction of 1-2 main core events for R_core_ref (prevents short-caption floor).

    Rules (applied strictly BEFORE running any captioning model):
      1. If pre-existing reference_events are available, filter out trivial/meta camera/atmosphere clauses
         (e.g., 'A clear day presents', 'The scene fades out') and select up to max_core events that
         carry concrete subject-action structure, favoring earliest prominent events and shared subjects.
      2. Otherwise split reference_description into sentences and pick the first max_core content sentences.
    """
    camera_meta = re.compile(
        r"\b(scene turns|scene cuts|scene fades|camera pans|camera zooms|clear day presents|shot switches)\b", re.I
    )
    cleaned = [e.strip() for e in reference_events if e and e.strip()]
    if cleaned:
        substantive = [e for e in cleaned if not camera_meta.search(e) and len(e.split()) >= 3]
        pool = substantive if substantive else cleaned
        return pool[:max_core]
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", reference_description.strip()) if s.strip()]
    return sents[:max_core]


def load_dream1k_records(
    annotation_path: Path | str,
    video_root: Path | str,
    require_video_exists: bool = False,
) -> list[QueryRecord]:
    """Load DREAM-1K metadata from either HF json/metadata.json or Tarsier2 DREAM-1k.jsonl."""
    apath = Path(annotation_path)
    vroot = Path(video_root)
    raw_text = apath.read_text(encoding="utf-8").strip()
    if raw_text.startswith("["):
        rows = json.loads(raw_text)
    elif raw_text.startswith("{"):
        parsed = json.loads(raw_text)
        rows = parsed["data"] if isinstance(parsed, dict) and "data" in parsed else [parsed]
    else:
        rows = [json.loads(line) for line in raw_text.splitlines() if line.strip()]

    records: list[QueryRecord] = []
    for row in rows:
        # Support both HF metadata.json and Tarsier2 DREAM-1k.jsonl schemas
        vid_raw = row.get("vid", row.get("idx"))
        if vid_raw is None:
            raise ValueError(f"DREAM-1K record missing vid/idx: {row}")
        vid_str = str(vid_raw).strip()
        m_num = re.search(r"(\d+)", vid_str)
        vid_int = int(m_num.group(1)) if m_num else -1
        query_id = f"dream1k_{vid_int:04d}" if vid_int >= 0 else f"dream1k_{vid_str}"

        # Extract reference description & events
        ref_desc = row.get("description", "")
        if not ref_desc and "messages" in row:
            for msg in row["messages"]:
                if msg.get("role") == "assistant":
                    for part in msg.get("content", []):
                        if part.get("reference"):
                            ref_desc = str(part["reference"]).strip()
                            break
        extra = row.get("extra_info", {}) if isinstance(row.get("extra_info"), dict) else {}
        events_raw = row.get("events", extra.get("events", []))
        if isinstance(events_raw, str):
            try:
                events_list = json.loads(events_raw)
            except json.JSONDecodeError:
                events_list = [e.strip() for e in events_raw.splitlines() if e.strip()]
        else:
            events_list = [str(e).strip() for e in (events_raw or []) if str(e).strip()]

        domain = infer_dream1k_domain(vid_int, row.get("source") or row.get("dataset"))
        duration_val = row.get("duration", extra.get("duration"))
        duration = float(duration_val) if duration_val not in (None, "") else None

        # Resolve video path
        vfile = row.get("video_file") or f"{vid_raw}.mp4"
        candidate_paths = [
            vroot / vfile,
            vroot / f"{vid_raw}.mp4",
            vroot / f"{vid_int}.mp4",
            vroot / f"{vid_int:04d}.mp4",
        ]
        resolved_vpath = next((p for p in candidate_paths if p.is_file()), candidate_paths[0])
        if require_video_exists and not resolved_vpath.is_file():
            raise FileNotFoundError(f"DREAM-1K video missing for {query_id}: {resolved_vpath}")

        # NEVER fabricate a source_id from opaque integer vid!
        explicit_url = row.get("original_url") or row.get("youtube_id")
        if explicit_url and isinstance(explicit_url, str) and explicit_url.strip():
            src_id: str | None = explicit_url.strip()
            src_status = SourceIdStatus.VERIFIED_EXACT.value
        else:
            src_id = None
            src_status = SourceIdStatus.UNKNOWN_OPAQUE.value

        rec = QueryRecord(
            dataset_id="DREAM-1K",
            query_id=query_id,
            source_id=src_id,
            source_id_status=src_status,
            video_path=resolved_vpath.as_posix(),
            video_sha256=None,
            duration=duration,
            domain=domain,
            reference_description=ref_desc,
            reference_events=events_list,
            core_events=extract_core_events_pre_model(ref_desc, events_list, max_core=2),
            split="unassigned",
            eligibility_status=EligibilityStatus.APPROVED.value if resolved_vpath.is_file() else EligibilityStatus.CANDIDATE_PENDING_VERIFICATION.value,
            metadata={
                "raw_vid": vid_raw,
                "n_subjects": row.get("n_subjects", extra.get("n_subjects")),
                "n_shots": row.get("n_shots", extra.get("n_shots")),
                "n_events": row.get("n_events", extra.get("n_events", len(events_list))),
            },
        )
        rec.validate()
        records.append(rec)

    records.sort(key=lambda r: r.query_id)
    return records


def sample_dream1k_stratified_pilot(
    records: list[QueryRecord],
    per_domain: int = 6,
    seed: int = 20261125,
    exclude_query_ids: set[str] | None = None,
) -> tuple[list[QueryRecord], list[QueryRecord]]:
    """Deterministically sample per_domain clips from each of the 5 DREAM-1K domains (default 5x6=30).

    Returns (pilot_records, remaining_records) with split fields cleanly assigned and zero overlap.
    """
    excl = exclude_query_ids or set()
    by_dom: dict[str, list[QueryRecord]] = {d: [] for d in DREAM1K_DOMAINS}
    for r in records:
        if r.query_id in excl:
            continue
        if r.domain in by_dom:
            by_dom[r.domain].append(r)

    rng = random.Random(seed)
    pilot_ids: set[str] = set()
    pilot_list: list[QueryRecord] = []
    for dom in DREAM1K_DOMAINS:
        pool = sorted(by_dom[dom], key=lambda x: x.query_id)
        if len(pool) < per_domain:
            raise ValueError(f"Insufficient DREAM-1K clips in domain '{dom}': need {per_domain}, have {len(pool)}")
        chosen = sorted(rng.sample(pool, per_domain), key=lambda x: x.query_id)
        for rec in chosen:
            pilot_ids.add(rec.query_id)
            rec_copy = QueryRecord.from_dict({**rec.to_dict(), "split": "pilot_a"})
            pilot_list.append(rec_copy)

    remaining: list[QueryRecord] = []
    for r in records:
        if r.query_id in pilot_ids or r.query_id in excl:
            continue
        remaining.append(QueryRecord.from_dict({**r.to_dict(), "split": "formal_b_pool"}))

    return pilot_list, remaining


def write_pilot_exclusion_manifest(pilot_records: list[QueryRecord], out_path: Path | str, seed: int) -> Path:
    payload = {
        "dataset_id": "DREAM-1K",
        "purpose": "Permanent exclusion list of Pilot A queries from Formal B confirmatory analysis",
        "sampling_seed": seed,
        "count": len(pilot_records),
        "by_domain": {d: sum(1 for r in pilot_records if r.domain == d) for d in DREAM1K_DOMAINS},
        "excluded_query_ids": sorted(r.query_id for r in pilot_records),
    }
    return atomic_write_json(out_path, payload)


def check_dream1k_hf_access(token_env_var: str = "HF_TOKEN", timeout_sec: int = 10) -> dict[str, Any]:
    """Read-only gated permission probe. Reads token strictly from environment and NEVER logs it."""
    token = os.environ.get(token_env_var, "").strip()
    url = f"https://huggingface.co/datasets/{DREAM1K_HF_REPO}/resolve/main/json/metadata.json"
    headers = {"User-Agent": "UARCap-Q2-Preflight/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return {
                "repo": DREAM1K_HF_REPO,
                "url": DREAM1K_HF_URL,
                "token_present": bool(token),
                "accessible": resp.status == 200,
                "http_status": resp.status,
                "action_required": None,
            }
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return {
                "repo": DREAM1K_HF_REPO,
                "url": DREAM1K_HF_URL,
                "token_present": bool(token),
                "accessible": False,
                "http_status": exc.code,
                "action_required": (
                    f"Accept gated license agreement at {DREAM1K_HF_URL} and export {token_env_var} "
                    "in your shell environment (NEVER commit tokens to Git)."
                ),
            }
        return {
            "repo": DREAM1K_HF_REPO,
            "url": DREAM1K_HF_URL,
            "token_present": bool(token),
            "accessible": False,
            "http_status": exc.code,
            "action_required": f"HTTP {exc.code} while checking DREAM-1K access.",
        }
    except Exception as exc:
        return {
            "repo": DREAM1K_HF_REPO,
            "url": DREAM1K_HF_URL,
            "token_present": bool(token),
            "accessible": False,
            "http_status": None,
            "action_required": f"Network unreachable or blocked ({type(exc).__name__}); use Route 2 offline staging.",
        }
