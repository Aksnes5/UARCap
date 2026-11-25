"""Paired & source-clustered statistical inference: bootstrap CI, sign-flip permutation, Holm, McNemar, and DiD."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import binomtest


def holm_bonferroni(p_values: dict[str, float | None], alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    """Apply Holm-Bonferroni step-down family-wise correction preserving None p-values transparently."""
    valid = [(k, float(p)) for k, p in p_values.items() if p is not None]
    valid.sort(key=lambda t: (t[1], t[0]))
    m = len(valid)
    results: dict[str, dict[str, Any]] = {}
    running_max = 0.0
    rejected_so_far = True

    for rank, (name, p) in enumerate(valid):
        adj = min(1.0, p * (m - rank))
        running_max = max(running_max, adj)
        sig = bool(rejected_so_far and running_max < alpha)
        if not sig:
            rejected_so_far = False
        results[name] = {
            "raw_p": round(p, 6),
            "holm_adj_p": round(running_max, 6),
            "significant": sig,
            "family_size_m": m,
        }

    for k, p in p_values.items():
        if p is None:
            results[k] = {"raw_p": None, "holm_adj_p": None, "significant": False, "family_size_m": m}
    return results


def cluster_paired_effect(
    diffs_by_query: dict[str, float],
    cluster_id_by_query: dict[str, str] | None = None,
    n_boot: int = 2000,
    n_perm: int = 4000,
    seed: int = 20261125,
) -> dict[str, Any]:
    """Compute paired mean diff, cluster bootstrap 95% CI, and cluster-level Monte Carlo sign-flip p-value."""
    qids = sorted(diffs_by_query.keys())
    if not qids:
        return {"n_queries": 0, "n_clusters": 0, "mean_diff": None, "ci95": (None, None), "p_value": None}

    cmap = cluster_id_by_query or {}
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for q in qids:
        by_cluster[cmap.get(q, q)].append(float(diffs_by_query[q]))

    clusters = sorted(by_cluster.keys())
    cluster_means = np.array([float(np.mean(by_cluster[c])) for c in clusters], dtype=np.float64)
    all_vals = np.array([float(diffs_by_query[q]) for q in qids], dtype=np.float64)
    obs_mean = float(np.mean(all_vals))
    n_c = len(clusters)

    # All-zero differences -> exact p=1.0, CI=(0.0, 0.0)
    if np.all(np.abs(all_vals) < 1e-12):
        return {
            "n_queries": len(qids),
            "n_clusters": n_c,
            "mean_diff": 0.0,
            "sd_diff": 0.0,
            "ci95": (0.0, 0.0),
            "p_value": 1.0,
            "note": "ALL_ZERO_DIFFERENCES",
        }

    rng = np.random.default_rng(seed)
    # Cluster bootstrap
    if n_c >= 2:
        boot_vals = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            idx = rng.integers(0, n_c, size=n_c)
            drawn = [v for i in idx for v in by_cluster[clusters[int(i)]]]
            boot_vals[b] = float(np.mean(drawn))
        lo, hi = np.percentile(boot_vals, [2.5, 97.5])
        ci95 = (round(float(lo), 6), round(float(hi), 6))
    else:
        ci95 = (round(obs_mean, 6), round(obs_mean, 6))

    # Cluster-level sign-flip permutation test (two-sided)
    obs_cluster_mean = float(np.mean(cluster_means))
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n_c))
    perm_means = np.abs(signs @ cluster_means / n_c)
    p_val = float((np.sum(perm_means >= abs(obs_cluster_mean) - 1e-12) + 1) / (n_perm + 1))

    return {
        "n_queries": len(qids),
        "n_clusters": n_c,
        "mean_diff": round(obs_mean, 6),
        "sd_diff": round(float(np.std(all_vals, ddof=1)) if len(all_vals) > 1 else 0.0, 6),
        "ci95": ci95,
        "p_value": round(p_val, 6),
    }


def paired_binary_mcnemar_effect(
    arm_a_binary: dict[str, int],
    arm_b_binary: dict[str, int],
    n_boot: int = 2000,
    seed: int = 20261125,
) -> dict[str, Any]:
    """Paired binary endpoint analysis (arm_a - arm_b): exact McNemar p, risk diff, discordant proportion, bootstrap CI."""
    common = sorted(set(arm_a_binary) & set(arm_b_binary))
    missing = len(set(arm_a_binary) ^ set(arm_b_binary))
    n = len(common)
    if n == 0:
        return {"n_paired": 0, "missing_unpaired": missing, "risk_diff": None, "mcnemar_p": None, "discordant_prop": None}

    n11 = n10 = n01 = n00 = 0
    diffs = np.empty(n, dtype=np.float64)
    for idx, q in enumerate(common):
        a = int(arm_a_binary[q])
        b = int(arm_b_binary[q])
        diffs[idx] = float(a - b)
        if a == 1 and b == 1:
            n11 += 1
        elif a == 1 and b == 0:
            n10 += 1
        elif a == 0 and b == 1:
            n01 += 1
        else:
            n00 += 1

    risk_a = (n11 + n10) / n
    risk_b = (n11 + n01) / n
    risk_diff = risk_a - risk_b
    disc = n10 + n01
    disc_prop = disc / n

    if disc == 0:
        p_val = 1.0
    else:
        p_val = float(binomtest(n10, n=disc, p=0.5, alternative="two-sided").pvalue)

    rng = np.random.default_rng(seed)
    if n >= 2 and np.any(diffs != 0):
        idx_mat = rng.integers(0, n, size=(n_boot, n))
        boot_rd = diffs[idx_mat].mean(axis=1)
        lo, hi = np.percentile(boot_rd, [2.5, 97.5])
        ci95 = (round(float(lo), 6), round(float(hi), 6))
    else:
        ci95 = (round(risk_diff, 6), round(risk_diff, 6))

    return {
        "n_paired": n,
        "missing_unpaired": missing,
        "contingency": {"n11": n11, "n10": n10, "n01": n01, "n00": n00},
        "rate_arm_a": round(risk_a, 6),
        "rate_arm_b": round(risk_b, 6),
        "risk_diff": round(risk_diff, 6),
        "ci95": ci95,
        "discordant_count": disc,
        "discordant_prop": round(disc_prop, 6),
        "mcnemar_p": round(p_val, 6),
    }


def classify_e3_joint_pattern(
    delta_y_conflict: float | None,
    sig_y_conflict: bool,
    delta_y_true: float | None,
    sig_y_true: bool,
) -> str:
    """Protocol Section 7.3 4-quadrant joint interpretation of (Delta Y^-, Delta Y^+) on Con vs Neu."""
    c_pos = bool(sig_y_conflict and delta_y_conflict is not None and delta_y_conflict > 0)
    t_neg = bool(sig_y_true and delta_y_true is not None and delta_y_true < 0)
    if c_pos and t_neg:
        return "SUBSTITUTION_CONTAMINATION"
    if c_pos and not t_neg:
        return "ADDITIVE_OR_HEDGED_CONTAMINATION"
    if (not c_pos) and t_neg:
        return "SUPPRESSION_OR_DISTRACTION"
    return "VISUAL_ROBUST_OR_NULL_EFFECT"


def cross_model_did_effect(
    model_a_deltas: dict[str, float],
    model_b_deltas: dict[str, float],
    cluster_id_by_query: dict[str, str] | None = None,
    n_boot: int = 2000,
    n_perm: int = 4000,
    seed: int = 20261125,
) -> dict[str, Any]:
    """Difference-in-Differences on common queries: D_i = Delta_model_a(i) - Delta_model_b(i)."""
    common = sorted(set(model_a_deltas) & set(model_b_deltas))
    did_map = {q: float(model_a_deltas[q] - model_b_deltas[q]) for q in common}
    res = cluster_paired_effect(did_map, cluster_id_by_query=cluster_id_by_query, n_boot=n_boot, n_perm=n_perm, seed=seed)
    res["missing_non_common_queries"] = len(set(model_a_deltas) ^ set(model_b_deltas))
    return res
