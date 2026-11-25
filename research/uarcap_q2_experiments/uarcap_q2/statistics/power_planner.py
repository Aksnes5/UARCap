"""Pilot-to-Formal sample size planner & RTX A6000 resource extrapolator."""

from __future__ import annotations

import math
from typing import Any

from scipy.stats import norm


def plan_formal_dream1k_sample_size(
    pilot_delta_sd: float,
    min_detectable_effect: float = 0.08,
    holm_alpha: float = 0.0125,
    power: float = 0.80,
    min_floor: int = 600,
    max_cap: int = 800,
    available_eligible_pool: int = 900,
) -> dict[str, Any]:
    """Compute required Formal B DREAM-1K sample size from Pilot A2 empirical paired-delta SD."""
    if min_detectable_effect <= 0:
        raise ValueError("min_detectable_effect must be positive")
    if pilot_delta_sd < 0:
        raise ValueError("pilot_delta_sd must be non-negative")
    z_a = float(norm.ppf(1.0 - holm_alpha / 2.0))
    z_b = float(norm.ppf(power))
    n_raw = math.ceil(((z_a + z_b) * pilot_delta_sd / min_detectable_effect) ** 2) if pilot_delta_sd > 0 else min_floor
    recommended = max(min_floor, min(max_cap, n_raw))
    exceeds_cap = n_raw > max_cap
    exceeds_pool = n_raw > available_eligible_pool
    return {
        "pilot_delta_sd": round(pilot_delta_sd, 5),
        "min_detectable_effect": min_detectable_effect,
        "holm_alpha": holm_alpha,
        "target_power": power,
        "n_required_raw": n_raw,
        "recommended_formal_n": min(recommended, available_eligible_pool),
        "exceeds_800_cap": exceeds_cap,
        "exceeds_eligible_pool": exceeds_pool,
    }


def extrapolate_formal_b_resources(
    pilot_a6000_sec_per_video_vl: float,
    pilot_a6000_sec_per_video_vlm: float,
    pilot_a6000_sec_per_caption_judge: float,
    pilot_two_annotator_min_per_video: float,
    formal_dream1k_n: int = 700,
    formal_vidhalluc_n: int = 170,
    e2_val_n: int = 50,
    gpu_ceiling_hours: float = 45.0,
    human_ceiling_person_hours: float = 70.0,
) -> dict[str, Any]:
    """Extrapolate Formal B GPU hours & 2-annotator person-hours from measured Pilot A throughput."""
    vl_gens = formal_dream1k_n * 3 + formal_vidhalluc_n * 4
    vlm_gens = formal_dream1k_n * 3 + formal_vidhalluc_n * 4
    judge_caps = formal_dream1k_n * 3 * 2 + formal_vidhalluc_n * 4 * 2

    gpu_hours = (
        vl_gens * pilot_a6000_sec_per_video_vl
        + vlm_gens * pilot_a6000_sec_per_video_vlm
        + judge_caps * pilot_a6000_sec_per_caption_judge
    ) / 3600.0 + 2.5  # +2.5h buffer for X-CLIP extraction, cold starts, and E3-Bridge

    # Human total person-hours: E2-Val + MSR-VTT bridge (55) + VidHalluc screening/audit (half rate per clip)
    human_person_hours = (
        (e2_val_n + 55) * pilot_two_annotator_min_per_video + formal_vidhalluc_n * (pilot_two_annotator_min_per_video * 0.45)
    ) / 60.0

    within_gpu = gpu_hours <= gpu_ceiling_hours
    within_human = human_person_hours <= human_ceiling_person_hours
    return {
        "projected_a6000_gpu_hours": round(gpu_hours, 2),
        "gpu_ceiling_hours": gpu_ceiling_hours,
        "within_gpu_ceiling": within_gpu,
        "projected_two_annotator_person_hours": round(human_person_hours, 2),
        "human_ceiling_person_hours": human_ceiling_person_hours,
        "within_human_ceiling": within_human,
        "resource_gate_pass": bool(within_gpu and within_human),
    }
