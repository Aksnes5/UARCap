"""CaReBench dataset adapter: real metadata normalization, FineAction parent-video dedup, and stratified sampling.

Primary-source facts pinned from direct inspection of MCG-NJU/CaReBench (arXiv:2501.00513v2):
  - HF repo: MCG-NJU/CaReBench (gated: False, license: MIT for code/metadata; upstream FineAction video copyrights apply)
  - Official GitHub: https://github.com/MCG-NJU/CaReBench
  - Verified metadata file: json/metadata.json (1,000 clips, ~3.70 MB)
  - Verified video archive: videos/videos.zip = 3,918,994,506 bytes (3.65 GiB / 3.92 GB)
  - Clip naming: `v_XXXXXXXX_N.mp4` (1,000 clips across 913 distinct FineAction parent IDs `v_XXXXXXXX`; 68 parents have 2-6 clips).
  - Categories (4): Socializing, Relaxing (340 clips / 311 parents), Sports, Excercise (260 / 239),
    Personal Care (210 / 186), Household Activities (190 / 177), spanning 82 subcategories.

CRITICAL SCIENTIFIC SAFEGUARDS:
  1. `caption`, `spatial_caption`, and `temporal_caption` are human-authored (2 annotators + expert review/merge) and serve
     as the primary human reference facts.
  2. `events` and `objects` in public `metadata.json` are GPT-4 extracted from the human captions (via `utils/dream_gpt.py`)
     and MUST NEVER be treated as independent human-certified Gold facts.
  3. The opening General Overview sentence (~94% of clips) is stored strictly as an unverified core-event CANDIDATE
     whose completeness must be checked during Pilot A2.
  4. Deduplicating to 913 FineAction parent IDs eliminates known within-FineAction multi-clip clustering,
     but does NOT prove 913 independent original YouTube uploads and does NOT enable exact YouTube-ID overlap proof
     against VideoUFO (`source_id` MUST remain None with `SourceIdStatus.UNKNOWN_OPAQUE`).
  5. Missing metadata fields (e.g., `duration` absent in public `metadata.json`) remain explicitly None; NEVER fabricate.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
import re
from typing import Any
import urllib.error
import urllib.request

from ..common.io_utils import atomic_write_json
from ..common.schemas import EligibilityStatus, QueryRecord, SourceIdStatus

CAREBENCH_HF_REPO = "MCG-NJU/CaReBench"
CAREBENCH_HF_URL = "https://huggingface.co/datasets/MCG-NJU/CaReBench"
CAREBENCH_GITHUB_URL = "https://github.com/MCG-NJU/CaReBench"
CAREBENCH_EXPECTED_VIDEO_ZIP_BYTES = 3918994506  # 3.65 GiB / 3.92 GB (verified via HF Tree API)
CAREBENCH_CATEGORIES = (
    "Socializing, Relaxing",
    "Sports, Excercise",  # Exact upstream spelling in CaReBench/json/metadata.json
    "Personal Care",
    "Household Activities",
)

FINEACTION_CLIP_RE = re.compile(r"^(v_\d+)_(\d+)(?:\.[A-Za-z0-9]+)?$")
OVERVIEW_OPENING_RE = re.compile(
    r"^(?:this\s+video|the\s+video|in\s+this\s+video|the\s+clip|this\s+clip)\b",
    re.I,
)


def parse_fineaction_parent_id(video_name: str) -> tuple[str | None, int | None]:
    """Parse FineAction parent video ID (`v_00000030`) and clip index (`5`) from `v_00000030_5.mp4`."""
    base = Path(str(video_name).strip()).name
    m = FINEACTION_CLIP_RE.match(base)
    if m:
        return m.group(1), int(m.group(2))
    stem = Path(base).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[0].startswith("v_") and parts[1].isdigit():
        return parts[0], int(parts[1])
    return None, None


def extract_opening_overview_candidate(caption: str) -> tuple[str | None, bool]:
    """Extract the opening sentence as an UNVERIFIED core-event candidate (reliability checked in Pilot A2).

    Returns (opening_sentence_or_None, matches_overview_template).
    NEVER fabricates a synthetic fallback when caption is empty.
    """
    text = (caption or "").strip()
    if not text:
        return None, False
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sents:
        return None, False
    first = sents[0]
    matches_tpl = bool(OVERVIEW_OPENING_RE.match(first))
    return first, matches_tpl


def _clean_str_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass
        return [ln.strip() for ln in s.splitlines() if ln.strip()]
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


def load_carebench_records(
    annotation_path: Path | str,
    video_root: Path | str,
    require_video_exists: bool = False,
) -> list[QueryRecord]:
    """Load real CaReBench `json/metadata.json` (or JSONL) into unified QueryRecords.

    Missing optional fields (e.g., `duration`) are preserved explicitly as None; pseudo-Gold facts are NEVER synthesized.
    """
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
        video_field = row.get("video") or row.get("video_file") or row.get("video_id")
        if not video_field:
            raise ValueError(f"CaReBench record missing 'video' field: {row}")
        raw_video_name = Path(str(video_field).strip()).name
        stem = Path(raw_video_name).stem
        video_filename = raw_video_name if "." in raw_video_name else f"{stem}.mp4"
        query_id = f"carebench_{stem}"

        parent_id, clip_idx = parse_fineaction_parent_id(video_filename)
        category = str(row.get("category", "")).strip() or "unknown"
        subcategory = str(row.get("subcategory", "")).strip() or "unknown"

        caption = str(row.get("caption", "")).strip()
        spatial_caption = str(row.get("spatial_caption", "")).strip()
        temporal_caption = str(row.get("temporal_caption", "")).strip()
        opening_cand, opening_is_tpl = extract_opening_overview_candidate(caption)

        # Explicitly preserve LLM-extracted events & objects with non-Gold provenance!
        llm_events = _clean_str_list(row.get("events"))
        llm_objects = _clean_str_list(row.get("objects"))

        # Preserve actual duration if present in metadata; otherwise keep None (NEVER fabricate).
        dur_raw = row.get("duration", row.get("video_duration"))
        duration = float(dur_raw) if dur_raw not in (None, "") else None

        # Locate video file cleanly across flat or nested extracted layouts
        candidate_paths = [
            vroot / video_filename,
            vroot / "videos" / video_filename,
            vroot / f"{stem}.mp4",
        ]
        resolved_vpath = next((p for p in candidate_paths if p.is_file()), candidate_paths[0])
        if require_video_exists and not resolved_vpath.is_file():
            raise FileNotFoundError(f"CaReBench video missing for {query_id}: {resolved_vpath}")

        # FineAction anonymous parent IDs (v_XXXXXXXX) are NOT proven original YouTube IDs.
        explicit_yt = row.get("youtube_id") or row.get("original_url")
        if explicit_yt and isinstance(explicit_yt, str) and explicit_yt.strip():
            src_id: str | None = explicit_yt.strip()
            src_status = SourceIdStatus.VERIFIED_EXACT.value
        else:
            src_id = None
            src_status = SourceIdStatus.UNKNOWN_OPAQUE.value

        rec = QueryRecord(
            dataset_id="CaReBench",
            query_id=query_id,
            source_id=src_id,
            source_id_status=src_status,
            video_path=resolved_vpath.as_posix(),
            video_sha256=None,
            duration=duration,
            domain=category,
            reference_description=caption,
            # Store LLM-extracted events for auxiliary inspection ONLY; provenance explicitly flags NON-GOLD!
            reference_events=llm_events,
            # Opening overview sentence is an unverified candidate only (checked in Pilot A2)
            core_events=[opening_cand] if opening_cand else [],
            split="unassigned",
            eligibility_status=(
                EligibilityStatus.APPROVED.value
                if resolved_vpath.is_file()
                else EligibilityStatus.CANDIDATE_PENDING_VERIFICATION.value
            ),
            metadata={
                "raw_video_id": stem,
                "video_filename": video_filename,
                "parent_video_id": parent_id,
                "clip_index": clip_idx,
                "category": category,
                "subcategory": subcategory,
                "spatial_caption": spatial_caption or None,
                "temporal_caption": temporal_caption or None,
                "opening_overview_candidate": opening_cand,
                "opening_overview_matches_template": opening_is_tpl,
                "core_event_candidate_status": "UNVERIFIED_CANDIDATE_REQUIRES_PILOT_A2_AUDIT",
                "human_reference_fields": [
                    f for f, v in (("caption", caption), ("spatial_caption", spatial_caption), ("temporal_caption", temporal_caption)) if v
                ],
                "llm_extracted_events": llm_events,
                "llm_extracted_objects": llm_objects,
                "llm_extracted_facts_provenance": "GPT4_EXTRACTED_NOT_INDEPENDENT_HUMAN_GOLD",
                "missing_fields": [
                    name
                    for name, val in (
                        ("duration", duration),
                        ("parent_video_id", parent_id),
                        ("spatial_caption", spatial_caption or None),
                        ("temporal_caption", temporal_caption or None),
                        ("opening_overview_candidate", opening_cand),
                    )
                    if val is None
                ],
            },
        )
        rec.validate()
        records.append(rec)

    records.sort(key=lambda r: r.query_id)
    return records


def get_carebench_parent_id(rec: QueryRecord) -> str:
    """Return canonical FineAction parent_video_id for clustering/isolation, falling back to query_id if absent."""
    pid = rec.metadata.get("parent_video_id")
    if pid:
        return str(pid)
    parsed, _ = parse_fineaction_parent_id(Path(rec.video_path).name)
    return parsed or rec.query_id


def deduplicate_carebench_by_parent_video(
    records: list[QueryRecord],
) -> tuple[list[QueryRecord], dict[str, Any]]:
    """Keep deterministically 1 clip per FineAction parent video ID (`v_XXXXXXXX`).

    Selection rule within each parent cluster: lowest integer `clip_index` (tie-break by `query_id`).
    NOTE: Eliminates known FineAction multi-clip clustering; does NOT prove independent original YouTube uploads.
    """
    by_parent: dict[str, list[QueryRecord]] = defaultdict(list)
    for r in records:
        by_parent[get_carebench_parent_id(r)].append(r)

    kept: list[QueryRecord] = []
    dropped_siblings: list[dict[str, Any]] = []
    multi_clip_parents = 0

    for pid in sorted(by_parent.keys()):
        group = sorted(
            by_parent[pid],
            key=lambda r: (
                int(r.metadata["clip_index"]) if r.metadata.get("clip_index") is not None else 10**9,
                r.query_id,
            ),
        )
        kept.append(group[0])
        if len(group) > 1:
            multi_clip_parents += 1
            for dup in group[1:]:
                dropped_siblings.append(
                    {
                        "query_id": dup.query_id,
                        "parent_video_id": pid,
                        "kept_query_id": group[0].query_id,
                        "reason": "FINEACTION_SAME_PARENT_VIDEO_DEDUP",
                    }
                )

    kept.sort(key=lambda r: r.query_id)
    audit = {
        "dataset_id": "CaReBench",
        "n_raw_clips": len(records),
        "n_unique_fineaction_parents": len(kept),
        "n_multi_clip_parents": multi_clip_parents,
        "n_dropped_same_parent_siblings": len(dropped_siblings),
        "dropped_siblings": dropped_siblings,
        "parent_video_dedup_does_not_prove_independent_youtube_sources": True,
        "residual_source_overlap_notice": (
            "Deduplicating to distinct FineAction parent IDs removes known within-FineAction multi-clip clustering, "
            "but does not prove independent original YouTube sources or exact YouTube-ID zero-overlap with VideoUFO."
        ),
    }
    return kept, audit


def _allocate_proportional_counts(counts_by_stratum: dict[str, int], target_n: int) -> dict[str, int]:
    """Deterministic proportional allocation across non-empty strata (min 1 per stratum when target_n >= n_strata)."""
    total = sum(counts_by_stratum.values())
    if target_n >= total:
        return dict(counts_by_stratum)
    strata = sorted(counts_by_stratum.keys())
    if not strata or target_n <= 0:
        return {s: 0 for s in strata}

    raw = {s: (counts_by_stratum[s] * target_n) / total for s in strata}
    alloc = {s: min(counts_by_stratum[s], int(raw[s])) for s in strata}
    if target_n >= len(strata):
        for s in strata:
            if alloc[s] == 0 and counts_by_stratum[s] > 0:
                alloc[s] = 1

    diff = target_n - sum(alloc.values())
    if diff > 0:
        for s in sorted(strata, key=lambda k: (-(raw[k] - int(raw[k])), k)):
            if diff == 0:
                break
            if alloc[s] < counts_by_stratum[s]:
                alloc[s] += 1
                diff -= 1
    elif diff < 0:
        for s in sorted(strata, key=lambda k: ((raw[k] - int(raw[k])), k)):
            if diff == 0:
                break
            if alloc[s] > (1 if target_n >= len(strata) else 0):
                alloc[s] -= 1
                diff += 1
    return alloc


def _sample_with_subcategory_spread(
    pool: list[QueryRecord],
    n_select: int,
    rng: random.Random,
) -> list[QueryRecord]:
    """Draw `n_select` records from one category while spreading picks across subcategories reproducibly."""
    if n_select <= 0:
        return []
    if n_select >= len(pool):
        return sorted(pool, key=lambda r: r.query_id)

    by_sub: dict[str, list[QueryRecord]] = defaultdict(list)
    for r in sorted(pool, key=lambda x: x.query_id):
        sub = str(r.metadata.get("subcategory") or "unknown")
        by_sub[sub].append(r)

    sub_keys = sorted(by_sub.keys())
    rng.shuffle(sub_keys)
    for sk in sub_keys:
        rng.shuffle(by_sub[sk])

    chosen: list[QueryRecord] = []
    # Round-robin across shuffled subcategories to maximize subcategory coverage without biasing category shares
    while len(chosen) < n_select:
        progressed = False
        for sk in sub_keys:
            if by_sub[sk] and len(chosen) < n_select:
                chosen.append(by_sub[sk].pop(0))
                progressed = True
        if not progressed:
            break
    return sorted(chosen, key=lambda r: r.query_id)


def sample_carebench_stratified_pilot(
    records: list[QueryRecord],
    total_queries: int = 30,
    seed: int = 20261125,
    exclude_query_ids: set[str] | None = None,
    exclude_parent_ids: set[str] | None = None,
) -> tuple[list[QueryRecord], list[QueryRecord], dict[str, Any]]:
    """Deduplicate by FineAction parent video, then draw reproducible proportional stratified Pilot A queries.

    Guarantees:
      - Zero parent-video overlap within Pilot A.
      - Zero parent-video overlap between Pilot A and the returned `remaining` Formal B pool
        (ALL sibling clips sharing a Pilot parent video ID are permanently excluded from `remaining`).
      - Proportional category allocation across CaReBench categories with round-robin subcategory spread.
    """
    excl_q = set(exclude_query_ids or set())
    excl_p = set(exclude_parent_ids or set())

    deduped, dedup_audit = deduplicate_carebench_by_parent_video(records)
    eligible = [
        r
        for r in deduped
        if r.query_id not in excl_q and get_carebench_parent_id(r) not in excl_p
    ]
    if len(eligible) < total_queries:
        raise ValueError(
            f"Insufficient CaReBench unique parent videos for Pilot A: need {total_queries}, have {len(eligible)}"
        )

    by_cat: dict[str, list[QueryRecord]] = defaultdict(list)
    for r in eligible:
        by_cat[r.domain or "unknown"].append(r)
    cat_counts = {c: len(v) for c, v in sorted(by_cat.items())}
    alloc = _allocate_proportional_counts(cat_counts, total_queries)

    rng = random.Random(seed)
    pilot_list: list[QueryRecord] = []
    pilot_qids: set[str] = set()
    pilot_pids: set[str] = set()

    for cat in sorted(by_cat.keys()):
        chosen = _sample_with_subcategory_spread(by_cat[cat], alloc[cat], rng)
        for rec in chosen:
            pilot_qids.add(rec.query_id)
            pilot_pids.add(get_carebench_parent_id(rec))
            pilot_list.append(QueryRecord.from_dict({**rec.to_dict(), "split": "pilot_a"}))

    pilot_list.sort(key=lambda r: r.query_id)

    # Remaining pool excludes BOTH selected query_ids AND any clip sharing a pilot parent_video_id!
    remaining: list[QueryRecord] = []
    for r in deduped:
        pid = get_carebench_parent_id(r)
        if r.query_id in pilot_qids or pid in pilot_pids or r.query_id in excl_q or pid in excl_p:
            continue
        remaining.append(QueryRecord.from_dict({**r.to_dict(), "split": "formal_b_pool"}))

    audit = {
        "dataset_id": "CaReBench",
        "strategy": "parent_video_dedup_then_proportional_category_stratified_with_subcategory_spread",
        "seed": seed,
        "target_pilot_n": total_queries,
        "selected_pilot_n": len(pilot_list),
        "n_unique_pilot_parents": len(pilot_pids),
        "n_unique_subcategories_in_pilot": len({r.metadata.get("subcategory") for r in pilot_list}),
        "category_pool_counts": cat_counts,
        "category_pilot_allocation": alloc,
        "remaining_formal_parent_pool_n": len(remaining),
        "parent_video_dedup_audit": dedup_audit,
    }
    return pilot_list, remaining, audit


def write_carebench_pilot_exclusion_manifest(
    pilot_records: list[QueryRecord],
    out_path: Path | str,
    seed: int,
    extra_debug_query_ids: set[str] | None = None,
    extra_debug_parent_ids: set[str] | None = None,
) -> Path:
    """Persist permanent Pilot A + debug exclusion manifest tracking BOTH query_ids and FineAction parent_video_ids."""
    excl_q = sorted(set(r.query_id for r in pilot_records) | set(extra_debug_query_ids or set()))
    excl_p = sorted(set(get_carebench_parent_id(r) for r in pilot_records) | set(extra_debug_parent_ids or set()))
    by_cat: dict[str, int] = defaultdict(int)
    for r in pilot_records:
        by_cat[r.domain or "unknown"] += 1
    payload = {
        "dataset_id": "CaReBench",
        "purpose": "Permanent exclusion list of Pilot A (and extra debug) CaReBench clips and FineAction parent videos from Formal B",
        "sampling_seed": seed,
        "count": len(pilot_records),
        "by_category": dict(sorted(by_cat.items())),
        "excluded_query_ids": excl_q,
        "excluded_parent_video_ids": excl_p,
        "extra_debug_query_ids": sorted(extra_debug_query_ids or set()),
        "extra_debug_parent_video_ids": sorted(extra_debug_parent_ids or set()),
    }
    return atomic_write_json(out_path, payload)


def check_carebench_hf_access(timeout_sec: int = 10, hf_endpoint: str | None = None) -> dict[str, Any]:
    """Read-only public availability probe for MCG-NJU/CaReBench (ungated, MIT, no HF token required)."""
    base = (hf_endpoint or "https://huggingface.co").rstrip("/")
    url = f"{base}/datasets/{CAREBENCH_HF_REPO}/resolve/main/json/metadata.json"
    req = urllib.request.Request(url, headers={"User-Agent": "UARCap-Q2-Preflight/0.2"}, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return {
                "repo": CAREBENCH_HF_REPO,
                "url": CAREBENCH_HF_URL,
                "gated": False,
                "token_required": False,
                "accessible": resp.status == 200,
                "http_status": resp.status,
                "action_required": None,
            }
    except urllib.error.HTTPError as exc:
        return {
            "repo": CAREBENCH_HF_REPO,
            "url": CAREBENCH_HF_URL,
            "gated": False,
            "token_required": False,
            "accessible": False,
            "http_status": exc.code,
            "action_required": f"HTTP {exc.code} while checking public CaReBench endpoint ({url}); try HF mirror or Route 2 offline staging.",
        }
    except Exception as exc:
        return {
            "repo": CAREBENCH_HF_REPO,
            "url": CAREBENCH_HF_URL,
            "gated": False,
            "token_required": False,
            "accessible": False,
            "http_status": None,
            "action_required": f"Network unreachable ({type(exc).__name__}); use HF mirror or Route 2 offline chunk transfer.",
        }
