"""Formal Study B Sample Isolation, Stratified Probability Sampling, and Analysis Manifest Freezing.

All target sample sizes are 100% parameterized via config (NEVER pre-frozen before Pilot A0-A3 verdict!).

Guarantees:
  - Supports CaReBench (default E2 primary) and DREAM-1K (backup E2).
  - CaReBench FineAction parent-video deduplication (1 clip per parent ID) and parent-ID isolation across
    Pilot A, extra debug clips, suspected near-duplicates, independent E2 human validation set, and formal analysis pool.
  - Permanent Pilot A E2 exclusion (query_id AND parent_video_id for CaReBench).
  - Suspected near-duplicate exclusion (external manifest + deterministic normalized text similarity audit).
  - Category/domain-stratified probability sampling with exact recorded inclusion probabilities (n_h / N_h).
  - Independent E2 human validation set drawn strictly disjoint from Pilot A, with both full formal cohort
    and explicit main-excluding-validation IDs recorded.
  - VidHalluc same-source isolation (1 clip per known YouTube source_id) + known-ID VideoUFO overlap exclusion
    + VidHalluc Pilot source exclusion.
  - Deterministic frozen manifest output (`formal_<dataset>_analysis_manifest.json`) with SHA-256 checksum.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import random
import re
from typing import Any

from ..common.io_utils import atomic_write_json, sha256_file
from ..common.schemas import QueryRecord
from .carebench import deduplicate_carebench_by_parent_video, get_carebench_parent_id
from .vidhalluc import deduplicate_and_filter_vidhalluc


@dataclass
class FormalSamplingConfig:
    e2_target_n: int | None = None  # Preferred generic name (CaReBench planned range 600-750; NEVER pre-promised!)
    dream1k_target_n: int | None = None  # Backward-compatible alias for e2_target_n
    e2_validation_target_n: int | None = None  # Planned 50-80 independent validation subsample (drawn inside formal cohort)
    vidhalluc_target_n: int | None = None  # None = use all eligible non-pilot non-overlap VidHalluc records
    near_duplicate_jaccard_threshold: float = 0.92
    enable_text_near_duplicate_filter: bool = True
    seed: int = 20261125

    def resolved_e2_target_n(self) -> int | None:
        return self.e2_target_n if self.e2_target_n is not None else self.dream1k_target_n

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _norm_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", (text or "").lower()))


def detect_e2_near_duplicates(
    records: list[QueryRecord],
    jaccard_threshold: float = 0.92,
) -> tuple[list[QueryRecord], list[dict[str, Any]]]:
    """Deterministically detect & exclude suspected near-duplicate E2 clips within the same category/domain."""
    ordered = sorted(records, key=lambda r: r.query_id)
    kept: list[QueryRecord] = []
    kept_tokens_by_domain: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
    excluded: list[dict[str, Any]] = []

    for r in ordered:
        toks = _norm_tokens(r.reference_description)
        dom = r.domain or "unknown"
        dup_of: str | None = None
        best_jac = 0.0
        if len(toks) >= 4:
            for prev_qid, prev_toks in kept_tokens_by_domain[dom]:
                union = len(toks | prev_toks)
                jac = (len(toks & prev_toks) / union) if union else 0.0
                if jac >= jaccard_threshold:
                    dup_of = prev_qid
                    best_jac = jac
                    break
        if dup_of is not None:
            excluded.append(
                {
                    "query_id": r.query_id,
                    "domain": dom,
                    "duplicate_of": dup_of,
                    "token_jaccard": round(best_jac, 4),
                    "reason": "SUSPECTED_NEAR_DUPLICATE_REFERENCE",
                }
            )
        else:
            kept.append(r)
            kept_tokens_by_domain[dom].append((r.query_id, toks))
    return kept, excluded


# Backward-compatible alias
detect_dream1k_near_duplicates = detect_e2_near_duplicates


def _allocate_proportional_strata(counts_by_stratum: dict[str, int], target_n: int) -> dict[str, int]:
    """Allocate `target_n` proportionally across non-empty strata (min 1 per stratum when target_n >= n_strata)."""
    total = sum(counts_by_stratum.values())
    if target_n >= total:
        return dict(counts_by_stratum)
    strata = sorted(counts_by_stratum.keys())
    if not strata or target_n <= 0:
        return {s: 0 for s in strata}

    raw = {s: (counts_by_stratum[s] * target_n) / total for s in strata}
    alloc = {s: min(counts_by_stratum[s], int(raw[s])) for s in strata}
    # Ensure every stratum gets >= 1 when target_n >= len(strata)
    if target_n >= len(strata):
        for s in strata:
            if alloc[s] == 0 and counts_by_stratum[s] > 0:
                alloc[s] = 1

    # Adjust rounding remainder deterministically by largest fractional part
    diff = target_n - sum(alloc.values())
    if diff > 0:
        order = sorted(strata, key=lambda s: (-(raw[s] - int(raw[s])), s))
        for s in order:
            if diff == 0:
                break
            if alloc[s] < counts_by_stratum[s]:
                alloc[s] += 1
                diff -= 1
    elif diff < 0:
        order = sorted(strata, key=lambda s: ((raw[s] - int(raw[s])), s))
        for s in order:
            if diff == 0:
                break
            if alloc[s] > (1 if target_n >= len(strata) else 0):
                alloc[s] -= 1
                diff += 1
    return alloc


def build_formal_e2_manifest(
    all_records: list[QueryRecord],
    pilot_excluded_ids: set[str],
    cfg: FormalSamplingConfig,
    explicit_near_dup_excluded_ids: set[str] | None = None,
    out_manifest_path: Path | str | None = None,
    pilot_excluded_parent_ids: set[str] | None = None,
    extra_debug_query_ids: set[str] | None = None,
    extra_debug_parent_ids: set[str] | None = None,
) -> tuple[list[QueryRecord], dict[str, Any]]:
    """Apply parent-video dedup (CaReBench) + Pilot/debug isolation + near-dup filter + stratified sampling + E2 val draw."""
    dataset_id = all_records[0].dataset_id if all_records else "CaReBench"
    is_carebench = dataset_id.lower() == "carebench"

    parent_dedup_audit: dict[str, Any] | None = None
    working_records = list(all_records)
    if is_carebench:
        working_records, parent_dedup_audit = deduplicate_carebench_by_parent_video(working_records)

    pilot_q = set(pilot_excluded_ids)
    debug_q = set(extra_debug_query_ids or set())
    expl_dup = set(explicit_near_dup_excluded_ids or set())

    # Derive excluded parent IDs from both explicit parent list AND any matching raw record in all_records
    pilot_p = set(pilot_excluded_parent_ids or set())
    debug_p = set(extra_debug_parent_ids or set())
    if is_carebench:
        for r in all_records:
            if r.query_id in pilot_q:
                pilot_p.add(get_carebench_parent_id(r))
            if r.query_id in debug_q:
                debug_p.add(get_carebench_parent_id(r))

    pilot_excl_applied: list[str] = []
    debug_excl_applied: list[str] = []
    explicit_dup_applied: list[str] = []
    after_isolation: list[QueryRecord] = []

    for r in working_records:
        pid = get_carebench_parent_id(r) if is_carebench else None
        if r.query_id in pilot_q or (pid is not None and pid in pilot_p):
            pilot_excl_applied.append(r.query_id)
            continue
        if r.query_id in debug_q or (pid is not None and pid in debug_p):
            debug_excl_applied.append(r.query_id)
            continue
        if r.query_id in expl_dup:
            explicit_dup_applied.append(r.query_id)
            continue
        after_isolation.append(r)

    pilot_excl_applied.sort()
    debug_excl_applied.sort()
    explicit_dup_applied.sort()

    if cfg.enable_text_near_duplicate_filter:
        eligible, auto_near_dups = detect_e2_near_duplicates(
            after_isolation, jaccard_threshold=cfg.near_duplicate_jaccard_threshold
        )
    else:
        eligible, auto_near_dups = sorted(after_isolation, key=lambda r: r.query_id), []

    by_stratum: dict[str, list[QueryRecord]] = defaultdict(list)
    for r in eligible:
        by_stratum[r.domain or "unknown"].append(r)

    counts = {s: len(v) for s, v in sorted(by_stratum.items())}
    total_eligible = len(eligible)
    cfg_target = cfg.resolved_e2_target_n()
    target_n = total_eligible if cfg_target is None else min(cfg_target, total_eligible)
    alloc = _allocate_proportional_strata(counts, target_n)

    rng = random.Random(cfg.seed)
    selected: list[QueryRecord] = []
    inclusion_records: dict[str, dict[str, Any]] = {}
    strata_report: dict[str, dict[str, Any]] = {}

    for s in sorted(by_stratum.keys()):
        pool = sorted(by_stratum[s], key=lambda r: r.query_id)
        rng.shuffle(pool)
        n_h = alloc[s]
        N_h = len(pool)
        pi_h = (n_h / N_h) if N_h > 0 else 0.0
        chosen = sorted(pool[:n_h], key=lambda r: r.query_id)
        selected.extend(chosen)
        strata_report[s] = {"N_h": N_h, "n_h": n_h, "sample_inclusion_prob": round(pi_h, 6)}
        for r in chosen:
            inclusion_records[r.query_id] = {
                "query_id": r.query_id,
                "parent_video_id": get_carebench_parent_id(r) if is_carebench else None,
                "stratum": s,
                "stratum_N": N_h,
                "stratum_n": n_h,
                "sample_inclusion_prob": round(pi_h, 6),
                "in_e2_human_validation": False,
                "val_inclusion_prob": 0.0,
            }

    # Draw independent E2 human validation subset (Phase-2 stratified probability sample inside formal cohort!)
    val_selected_ids: list[str] = []
    val_strata_report: dict[str, dict[str, Any]] = {}
    if cfg.e2_validation_target_n and selected:
        sel_by_s: dict[str, list[QueryRecord]] = defaultdict(list)
        for r in selected:
            sel_by_s[r.domain or "unknown"].append(r)
        val_counts = {s: len(v) for s, v in sorted(sel_by_s.items())}
        val_target = min(cfg.e2_validation_target_n, len(selected))
        val_alloc = _allocate_proportional_strata(val_counts, val_target)
        rng_val = random.Random(int(sha256(f"e2_val:{cfg.seed}".encode()).hexdigest()[:12], 16))
        for s in sorted(sel_by_s.keys()):
            pool = sorted(sel_by_s[s], key=lambda r: r.query_id)
            rng_val.shuffle(pool)
            nv = val_alloc[s]
            Nv = len(pool)
            piv = (nv / Nv) if Nv > 0 else 0.0
            chosen_v = sorted(pool[:nv], key=lambda r: r.query_id)
            val_strata_report[s] = {"formal_N_h": Nv, "val_n_h": nv, "val_inclusion_prob": round(piv, 6)}
            for r in chosen_v:
                assert r.query_id not in pilot_q, "Pilot A calibration ID leaked into E2 validation set!"
                val_selected_ids.append(r.query_id)
                inclusion_records[r.query_id]["in_e2_human_validation"] = True
                inclusion_records[r.query_id]["val_inclusion_prob"] = round(piv, 6)

    val_id_set = set(val_selected_ids)
    main_excluding_val_ids = sorted(r.query_id for r in selected if r.query_id not in val_id_set)
    selected_parent_ids = sorted({get_carebench_parent_id(r) for r in selected}) if is_carebench else []
    val_parent_ids = sorted({get_carebench_parent_id(r) for r in selected if r.query_id in val_id_set}) if is_carebench else []
    main_parent_ids = sorted({get_carebench_parent_id(r) for r in selected if r.query_id not in val_id_set}) if is_carebench else []

    slug = "carebench" if is_carebench else "dream1k"
    manifest: dict[str, Any] = {
        "schema_version": f"uarcap-q2-formal-{slug}-manifest-v2",
        "dataset_id": dataset_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sampling_config": cfg.to_dict(),
        "n_raw_total": len(all_records),
        "n_after_parent_dedup": len(working_records),
        "parent_video_dedup_audit": parent_dedup_audit,
        "n_pilot_excluded": len(pilot_excl_applied),
        "pilot_excluded_query_ids": pilot_excl_applied,
        "pilot_excluded_parent_video_ids": sorted(pilot_p) if is_carebench else [],
        "n_extra_debug_excluded": len(debug_excl_applied),
        "extra_debug_excluded_query_ids": debug_excl_applied,
        "n_explicit_near_dup_excluded": len(explicit_dup_applied),
        "explicit_near_dup_excluded_query_ids": explicit_dup_applied,
        "n_auto_near_dup_excluded": len(auto_near_dups),
        "auto_near_duplicates": auto_near_dups,
        "n_eligible_pool": total_eligible,
        "n_selected_formal": len(selected),
        "selected_query_ids": [r.query_id for r in selected],
        "formal_main_excluding_validation_ids": main_excluding_val_ids,
        "n_formal_main_excluding_validation": len(main_excluding_val_ids),
        "strata": strata_report,
        "parent_isolation_verified": (
            (len(selected_parent_ids) == len(selected))
            and (len(set(selected_parent_ids) & pilot_p) == 0)
            and (len(set(selected_parent_ids) & debug_p) == 0)
            and (len(set(val_parent_ids) & set(main_parent_ids)) == 0)
        )
        if is_carebench
        else None,
        "e2_validation_set": {
            "n_validation_selected": len(val_selected_ids),
            "validation_query_ids": sorted(val_selected_ids),
            "validation_parent_video_ids": val_parent_ids,
            "disjoint_from_pilot_verified": len(val_id_set & pilot_q) == 0
            and (len(set(val_parent_ids) & pilot_p) == 0 if is_carebench else True),
            "disjoint_from_formal_main_verified": len(val_id_set & set(main_excluding_val_ids)) == 0,
            "strata": val_strata_report,
        },
        "units": inclusion_records,
    }
    if out_manifest_path is not None:
        mp = atomic_write_json(out_manifest_path, manifest)
        manifest["manifest_path"] = str(mp)
        manifest["manifest_sha256"] = sha256_file(mp)
    return selected, manifest


def build_formal_dream1k_manifest(
    all_records: list[QueryRecord],
    pilot_excluded_ids: set[str],
    cfg: FormalSamplingConfig,
    explicit_near_dup_excluded_ids: set[str] | None = None,
    out_manifest_path: Path | str | None = None,
) -> tuple[list[QueryRecord], dict[str, Any]]:
    """Backward-compatible wrapper around `build_formal_e2_manifest`."""
    return build_formal_e2_manifest(
        all_records=all_records,
        pilot_excluded_ids=pilot_excluded_ids,
        cfg=cfg,
        explicit_near_dup_excluded_ids=explicit_near_dup_excluded_ids,
        out_manifest_path=out_manifest_path,
    )


def build_formal_vidhalluc_manifest(
    raw_records: list[QueryRecord],
    videoufo_source_ids: set[str],
    pilot_excluded_query_ids: set[str],
    pilot_excluded_source_ids: set[str],
    cfg: FormalSamplingConfig,
    out_manifest_path: Path | str | None = None,
) -> tuple[list[QueryRecord], dict[str, Any]]:
    """Enforce VidHalluc same-source isolation + VideoUFO overlap + Pilot source exclusion + probability sampling."""
    clean_records, overlap_audit = deduplicate_and_filter_vidhalluc(
        raw_records, videoufo_source_ids=videoufo_source_ids
    )
    pilot_q_excl = set(pilot_excluded_query_ids)
    pilot_s_excl = set(pilot_excluded_source_ids)

    excluded_pilot_links: list[dict[str, Any]] = []
    eligible: list[QueryRecord] = []
    for r in sorted(clean_records, key=lambda x: x.query_id):
        if r.query_id in pilot_q_excl or (r.source_id and r.source_id in pilot_s_excl):
            excluded_pilot_links.append({"query_id": r.query_id, "source_id": r.source_id})
        else:
            eligible.append(r)

    N = len(eligible)
    target_n = N if cfg.vidhalluc_target_n is None else min(cfg.vidhalluc_target_n, N)
    rng = random.Random(int(sha256(f"vh_formal:{cfg.seed}".encode()).hexdigest()[:12], 16))
    pool = list(eligible)
    rng.shuffle(pool)
    selected = sorted(pool[:target_n], key=lambda x: x.query_id)
    pi = (len(selected) / N) if N > 0 else 0.0

    manifest: dict[str, Any] = {
        "schema_version": "uarcap-q2-formal-vidhalluc-manifest-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sampling_config": cfg.to_dict(),
        "videoufo_overlap_and_same_source_audit": overlap_audit,
        "n_excluded_pilot_query_or_source": len(excluded_pilot_links),
        "excluded_pilot_links": excluded_pilot_links,
        "n_eligible_formal_pool": N,
        "n_selected_formal": len(selected),
        "inclusion_prob": round(pi, 6),
        "selected_query_ids": [r.query_id for r in selected],
        "selected_source_ids": sorted({r.source_id for r in selected if r.source_id}),
        "same_source_isolated": len({r.source_id for r in selected if r.source_id})
        == sum(1 for r in selected if r.source_id),
    }
    if out_manifest_path is not None:
        mp = atomic_write_json(out_manifest_path, manifest)
        manifest["manifest_path"] = str(mp)
        manifest["manifest_sha256"] = sha256_file(mp)
    return selected, manifest
