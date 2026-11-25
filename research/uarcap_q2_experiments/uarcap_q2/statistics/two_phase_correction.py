"""Two-phase design-based error-correction estimator (generalized difference / Horvitz-Thompson IPW).

Combines full-cohort automatic proxy paired deltas with a stratified probability human-audited subsample
(supporting simple random, domain-stratified, AND enriched probability designs) with proper variance
propagation and strict calibration-vs-validation isolation!
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass
class TwoPhaseUnit:
    query_id: str
    stratum: str
    auto_delta: float  # Full-cohort automatic proxy paired delta
    in_human_sample: bool
    inclusion_prob: float  # pi_i in (0, 1] for stratum h (n_h / N_h)
    human_delta: float | None = None  # Observed only when in_human_sample is True


def estimate_two_phase_corrected_delta(
    units: list[TwoPhaseUnit],
    calibration_excluded_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Compute design-unbiased two-phase difference estimator with stratified finite-population variance.

    Formula:
        Delta_2P = (1/N) * sum_{i=1..N} auto_i + (1/N) * sum_{i in S} (human_i - auto_i) / pi_i
    """
    cal_excl = calibration_excluded_ids or set()
    leak = sorted({u.query_id for u in units if u.in_human_sample and u.query_id in cal_excl})
    if leak:
        raise ValueError(
            f"Calibration-to-Validation leakage blocked in two-phase estimator! {len(leak)} IDs overlap: {leak[:5]}"
        )

    N = len(units)
    if N == 0:
        return {"status": "EMPTY", "n_total": 0}

    by_stratum: dict[str, list[TwoPhaseUnit]] = defaultdict(list)
    for u in units:
        if u.in_human_sample:
            if not (0.0 < u.inclusion_prob <= 1.0):
                raise ValueError(f"Invalid inclusion_prob={u.inclusion_prob} for {u.query_id}")
            if u.human_delta is None:
                raise ValueError(f"Missing human_delta for sampled unit {u.query_id}")
        by_stratum[u.stratum].append(u)

    auto_mean = float(np.mean([u.auto_delta for u in units]))
    correction_sum = 0.0
    var_phase2 = 0.0
    n_sampled_total = 0
    stratum_summary: dict[str, Any] = {}

    for h, s_units in sorted(by_stratum.items()):
        N_h = len(s_units)
        sampled = [u for u in s_units if u.in_human_sample]
        n_h = len(sampled)
        n_sampled_total += n_h
        if n_h == 0:
            return {
                "status": "INCONCLUSIVE_EMPTY_STRATUM",
                "n_total": N,
                "n_sampled": n_sampled_total,
                "empty_stratum": h,
                "auto_uncorrected_mean": round(auto_mean, 6),
                "corrected_mean": None,
                "se": None,
                "ci95": (None, None),
            }
        residuals = np.array([float(u.human_delta) - u.auto_delta for u in sampled], dtype=np.float64)
        inv_pi = np.array([1.0 / u.inclusion_prob for u in sampled], dtype=np.float64)
        correction_sum += float(np.sum(residuals * inv_pi))

        # Finite-population-corrected stratified variance of the Phase-2 residual estimator
        fpc = max(0.0, 1.0 - (n_h / N_h))
        s2_h = float(np.var(residuals, ddof=1)) if n_h > 1 else 0.0
        var_h = ((N_h / N) ** 2) * fpc * (s2_h / n_h)
        var_phase2 += var_h
        stratum_summary[h] = {
            "N_h": N_h,
            "n_h": n_h,
            "mean_inclusion_prob": round(float(np.mean([u.inclusion_prob for u in sampled])), 4),
            "mean_residual": round(float(np.mean(residuals)), 5),
        }

    # Phase-1 superpopulation sampling variance of full-cohort auto proxy
    var_phase1 = float(np.var([u.auto_delta for u in units], ddof=1)) / N if N > 1 else 0.0
    total_var = var_phase1 + var_phase2
    se = math.sqrt(max(total_var, 0.0))
    corrected_mean = auto_mean + (correction_sum / N)
    ci95 = (round(corrected_mean - 1.96 * se, 6), round(corrected_mean + 1.96 * se, 6))

    return {
        "status": "OK",
        "n_total": N,
        "n_sampled": n_sampled_total,
        "auto_uncorrected_mean": round(auto_mean, 6),
        "bias_correction_term": round(correction_sum / N, 6),
        "corrected_mean": round(corrected_mean, 6),
        "se_total": round(se, 6),
        "se_phase1_auto": round(math.sqrt(var_phase1), 6),
        "se_phase2_human_residual": round(math.sqrt(var_phase2), 6),
        "ci95": ci95,
        "strata": stratum_summary,
    }
