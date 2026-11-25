"""E1 retrospective CPU explorer on frozen per-query CIDEr scores (0 GPU)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def analyze_e1_selection_heterogeneity(
    per_query_scores_ref_a: dict[str, dict[str, float]],
    per_query_scores_ref_b: dict[str, dict[str, float]],
    source_id_by_query: dict[str, str],
    baseline_arm: str = "ARM_A1",
    n_boot: int = 1000,
    seed: int = 20261125,
) -> dict[str, Any]:
    """Compute same-ref oracle gain, cross-ref selection gain, decoupled random-arm baseline, and source-clustered CI."""
    common_qids = sorted(set(per_query_scores_ref_a) & set(per_query_scores_ref_b))
    if not common_qids:
        return {"n_queries": 0, "status": "EMPTY"}

    arms = sorted({a for q in common_qids for a in per_query_scores_ref_a[q]})
    rng = np.random.default_rng(seed)

    same_ref_gains: dict[str, float] = {}
    cross_ref_gains: dict[str, float] = {}
    random_arm_gains: dict[str, float] = {}
    missing_arms_count = 0

    for q in common_qids:
        sa = per_query_scores_ref_a[q]
        sb = per_query_scores_ref_b[q]
        if baseline_arm not in sa or baseline_arm not in sb:
            missing_arms_count += 1
            continue
        avail = [a for a in arms if a in sa and a in sb]
        if not avail:
            missing_arms_count += 1
            continue
        best_a = max(avail, key=lambda a: (sa[a], a))
        same_ref_gains[q] = sa[best_a] - sa[baseline_arm]
        cross_ref_gains[q] = sb[best_a] - sb[baseline_arm]
        rand_arm = avail[int(rng.integers(0, len(avail)))]
        random_arm_gains[q] = sb[rand_arm] - sb[baseline_arm]

    scored_qids = sorted(cross_ref_gains.keys())
    if not scored_qids:
        return {"n_queries": 0, "missing_arms_count": missing_arms_count, "status": "NO_SCORED_QUERIES"}

    by_source: dict[str, list[str]] = defaultdict(list)
    for q in scored_qids:
        by_source[source_id_by_query.get(q, q)].append(q)
    sources = sorted(by_source.keys())

    def cluster_ci(val_map: dict[str, float]) -> tuple[float, float]:
        if len(sources) < 2:
            v = float(np.mean(list(val_map.values())))
            return round(v, 5), round(v, 5)
        boot_means = []
        n_s = len(sources)
        for _ in range(n_boot):
            draw = rng.integers(0, n_s, size=n_s)
            vals = [val_map[q] for idx in draw for q in by_source[sources[int(idx)]]]
            boot_means.append(float(np.mean(vals)))
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        return round(float(lo), 5), round(float(hi), 5)

    return {
        "n_queries": len(scored_qids),
        "n_unique_sources": len(sources),
        "missing_arms_count": missing_arms_count,
        "baseline_arm": baseline_arm,
        "same_ref_oracle_mean_gain": round(float(np.mean(list(same_ref_gains.values()))), 5),
        "same_ref_ci95": cluster_ci(same_ref_gains),
        "cross_ref_selection_mean_gain": round(float(np.mean(list(cross_ref_gains.values()))), 5),
        "cross_ref_ci95": cluster_ci(cross_ref_gains),
        "decoupled_random_arm_mean_gain": round(float(np.mean(list(random_arm_gains.values()))), 5),
        "decoupled_random_arm_ci95": cluster_ci(random_arm_gains),
    }
