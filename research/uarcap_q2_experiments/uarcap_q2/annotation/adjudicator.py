"""Dual-annotator agreement (Cohen's kappa, Gwet's AC1), disagreement extraction, and 2-annotator time ledger."""

from __future__ import annotations

from collections import Counter
from typing import Any


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    if len(labels_a) != len(labels_b) or not labels_a:
        return None
    n = len(labels_a)
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    ca = Counter(labels_a)
    cb = Counter(labels_b)
    cats = set(ca) | set(cb)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if abs(1.0 - pe) < 1e-12:
        return None  # Undefined when single constant category
    return round((po - pe) / (1.0 - pe), 4)


def gwets_ac1(labels_a: list[str], labels_b: list[str], categories: list[str] | None = None) -> float | None:
    """Gwet's AC1 handles imbalanced prevalence (e.g., low error base rate) stably where kappa collapses."""
    if len(labels_a) != len(labels_b) or not labels_a:
        return None
    n = len(labels_a)
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    cats = sorted(set(categories) if categories else (set(labels_a) | set(labels_b)))
    q = len(cats)
    if q < 2:
        return None
    pi = {c: (labels_a.count(c) + labels_b.count(c)) / (2.0 * n) for c in cats}
    pe = sum(p * (1.0 - p) for p in pi.values()) / (q - 1)
    if abs(1.0 - pe) < 1e-12:
        return None
    return round((po - pe) / (1.0 - pe), 4)


def extract_disagreements_and_time_summary(
    ann1_map: dict[str, str],
    ann2_map: dict[str, str],
    ann1_elapsed_sec: float,
    ann2_elapsed_sec: float,
    adjudication_joint_sec: float = 0.0,
) -> dict[str, Any]:
    """Compare two independent annotators, list disagreements for adjudication, and report TOTAL 2-annotator person-hours."""
    common_keys = sorted(set(ann1_map) & set(ann2_map))
    la = [ann1_map[k] for k in common_keys]
    lb = [ann2_map[k] for k in common_keys]
    disagreements = [
        {"item_key": k, "annotator_1": ann1_map[k], "annotator_2": ann2_map[k]}
        for k in common_keys
        if ann1_map[k] != ann2_map[k]
    ]
    # Joint adjudication involves BOTH annotators -> counts 2x wall seconds in total person-hours!
    total_person_hours = (ann1_elapsed_sec + ann2_elapsed_sec + 2.0 * adjudication_joint_sec) / 3600.0
    return {
        "n_common_items": len(common_keys),
        "n_disagreements": len(disagreements),
        "raw_agreement": round((len(common_keys) - len(disagreements)) / len(common_keys), 4) if common_keys else None,
        "cohens_kappa": cohens_kappa(la, lb),
        "gwets_ac1": gwets_ac1(la, lb),
        "disagreements": disagreements,
        "time_ledger": {
            "annotator_1_hours": round(ann1_elapsed_sec / 3600.0, 4),
            "annotator_2_hours": round(ann2_elapsed_sec / 3600.0, 4),
            "joint_adjudication_wall_hours": round(adjudication_joint_sec / 3600.0, 4),
            "total_two_annotator_person_hours": round(total_person_hours, 4),
        },
    }
