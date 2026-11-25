"""3x4 Cross-Evaluation Matrix, Differential Measurement Bias Audit, and PASS/FAIL/INCONCLUSIVE Verdict.

Implements UARCap_Q2_EVALUATOR_VALIDATION_PROTOCOL.md strictly:
  - Computes 3 proxy classes x 4 human video truth classes overall, per arm (A0/A1/A7), and per model.
  - Computes all 6 mandatory quantities + video-clustered bootstrap 95% CIs.
  - Evaluates video-level paired-delta differential bias B_i = Delta_ref(A7-A0) - Delta_human(A7-A0).
  - Returns INCONCLUSIVE (NEVER fake PASS!) when denominators are zero, positive error count is sparse,
    or correlation is undefined.
  - Returns FAIL if A7 systematic omission bias triggers even when overall atomic kappa is high!
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from ..common.schemas import ExpArm, RefRelation, VerdictStatus, VideoTruthLabel

REF_ORDER = [RefRelation.ENTAILED.value, RefRelation.CONTRADICTED.value, RefRelation.UNMENTIONED.value]
TRUTH_ORDER = [
    VideoTruthLabel.VIDEO_SUPPORTED.value,
    VideoTruthLabel.VIDEO_CONTRADICTED.value,
    VideoTruthLabel.VIDEO_UNVERIFIABLE.value,
    VideoTruthLabel.NON_FACTUAL.value,
]


@dataclass
class AlignedAtomRecord:
    video_id: str
    arm: str
    model_id: str
    ref_label: str  # RefRelation value
    human_label: str  # VideoTruthLabel value


def _safe_div(num: float, den: float) -> float | None:
    return (num / den) if den > 0 else None


def compute_3x4_matrix_metrics(records: list[AlignedAtomRecord]) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {r: {t: 0 for t in TRUTH_ORDER} for r in REF_ORDER}
    for rec in records:
        if rec.ref_label in counts and rec.human_label in counts[rec.ref_label]:
            counts[rec.ref_label][rec.human_label] += 1

    total = sum(sum(row.values()) for row in counts.values())
    n_contra_row = sum(counts["contradicted"].values())
    n_unment_row = sum(counts["unmentioned"].values())
    n_err_col = sum(counts[r][VideoTruthLabel.VIDEO_CONTRADICTED.value] for r in REF_ORDER)
    n_unv_col = sum(counts[r][VideoTruthLabel.VIDEO_UNVERIFIABLE.value] for r in REF_ORDER)

    c_err = counts["contradicted"][VideoTruthLabel.VIDEO_CONTRADICTED.value]
    u_err = counts["unmentioned"][VideoTruthLabel.VIDEO_CONTRADICTED.value]
    u_sup = counts["unmentioned"][VideoTruthLabel.VIDEO_SUPPORTED.value]

    return {
        "total_atoms": total,
        "matrix_3x4": counts,
        "n_ref_contradicted": n_contra_row,
        "n_true_video_errors": n_err_col,
        "n_ref_unmentioned": n_unment_row,
        "prec_contra_to_err": _safe_div(c_err, n_contra_row),
        "rec_err_to_contra": _safe_div(c_err, n_err_col),
        "leak_unment_to_err": _safe_div(u_err, n_unment_row),
        "true_unrec_unment_to_sup": _safe_div(u_sup, n_unment_row),
        "p_unverifiable": _safe_div(n_unv_col, total),
    }


def _video_cluster_bootstrap_ci(
    video_ids: list[str],
    records_by_video: dict[str, list[AlignedAtomRecord]],
    metric_key: str,
    n_boot: int = 1000,
    seed: int = 20261125,
) -> tuple[float | None, float | None]:
    if len(video_ids) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    n = len(video_ids)
    for _ in range(n_boot):
        draw = rng.integers(0, n, size=n)
        sample_recs: list[AlignedAtomRecord] = []
        for idx in draw:
            sample_recs.extend(records_by_video[video_ids[int(idx)]])
        v = compute_3x4_matrix_metrics(sample_recs)[metric_key]
        if v is not None:
            vals.append(v)
    if len(vals) < max(50, n_boot // 10):
        return None, None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def audit_evaluator_calibration(
    records: list[AlignedAtomRecord],
    n_boot: int = 1000,
    min_error_videos_for_conclusive: int = 5,
    prec_min_threshold: float = 0.65,
    rho_min_threshold: float = 0.45,
    max_reversal_rate: float = 0.20,
    max_a7_recall_drop: float = 0.15,
    max_a7_leak_increase: float = 0.10,
    max_cross_model_recall_gap: float = 0.20,
) -> dict[str, Any]:
    """Full evaluator calibration audit producing PASS / FAIL / INCONCLUSIVE with explicit reasons."""
    overall = compute_3x4_matrix_metrics(records)
    by_video: dict[str, list[AlignedAtomRecord]] = defaultdict(list)
    by_arm: dict[str, list[AlignedAtomRecord]] = defaultdict(list)
    by_model: dict[str, list[AlignedAtomRecord]] = defaultdict(list)

    for r in records:
        by_video[r.video_id].append(r)
        by_arm[r.arm].append(r)
        by_model[r.model_id].append(r)

    video_ids = sorted(by_video.keys())
    arm_metrics = {arm: compute_3x4_matrix_metrics(recs) for arm, recs in sorted(by_arm.items())}
    model_metrics = {m: compute_3x4_matrix_metrics(recs) for m, recs in sorted(by_model.items())}

    # Video-level paired deltas (A7 - A0) per (video_id, model_id)
    per_vm_arm: dict[tuple[str, str], dict[str, tuple[int, int]]] = defaultdict(dict)
    grouped_vma: dict[tuple[str, str, str], list[AlignedAtomRecord]] = defaultdict(list)
    for r in records:
        grouped_vma[(r.video_id, r.model_id, r.arm)].append(r)
    for (vid, mid, arm), recs in grouped_vma.items():
        n_ref_c = sum(1 for x in recs if x.ref_label == RefRelation.CONTRADICTED.value)
        n_hum_c = sum(1 for x in recs if x.human_label == VideoTruthLabel.VIDEO_CONTRADICTED.value)
        per_vm_arm[(vid, mid)][arm] = (n_ref_c, n_hum_c)

    delta_ref: list[float] = []
    delta_hum: list[float] = []
    bias_vals: list[float] = []
    paired_vids: list[str] = []
    for (vid, _mid), amap in sorted(per_vm_arm.items()):
        if ExpArm.ARM_A7.value in amap and ExpArm.ARM_A0.value in amap:
            dr = float(amap[ExpArm.ARM_A7.value][0] - amap[ExpArm.ARM_A0.value][0])
            dh = float(amap[ExpArm.ARM_A7.value][1] - amap[ExpArm.ARM_A0.value][1])
            delta_ref.append(dr)
            delta_hum.append(dh)
            bias_vals.append(dr - dh)
            paired_vids.append(vid)

    # Paired-delta diagnostics
    rho_val: float | None = None
    if len(delta_ref) >= 3 and np.std(delta_ref) > 0 and np.std(delta_hum) > 0:
        res = spearmanr(delta_ref, delta_hum)
        if not math.isnan(float(res.statistic)):
            rho_val = round(float(res.statistic), 4)

    nonzero_pairs = [(r, h) for r, h in zip(delta_ref, delta_hum) if h != 0]
    same_sign = sum(1 for r, h in nonzero_pairs if (r > 0 and h > 0) or (r < 0 and h < 0))
    opposite_sign = sum(1 for r, h in nonzero_pairs if (r > 0 and h < 0) or (r < 0 and h > 0))
    directional_concordance = _safe_div(same_sign, len(nonzero_pairs))
    reversal_rate = _safe_div(opposite_sign, len(nonzero_pairs))

    mean_bias = round(float(np.mean(bias_vals)), 4) if bias_vals else None
    mean_hum_delta = round(float(np.mean(delta_hum)), 4) if delta_hum else None

    # Differential arm bias: A7 vs A0
    a7_m = arm_metrics.get(ExpArm.ARM_A7.value)
    a0_m = arm_metrics.get(ExpArm.ARM_A0.value)
    a7_vs_a0_rec_diff = None
    a7_vs_a0_leak_diff = None
    if a7_m and a0_m:
        if a7_m["rec_err_to_contra"] is not None and a0_m["rec_err_to_contra"] is not None:
            a7_vs_a0_rec_diff = round(a7_m["rec_err_to_contra"] - a0_m["rec_err_to_contra"], 4)
        if a7_m["leak_unment_to_err"] is not None and a0_m["leak_unment_to_err"] is not None:
            a7_vs_a0_leak_diff = round(a7_m["leak_unment_to_err"] - a0_m["leak_unment_to_err"], 4)

    # Cross-model capture gap
    model_recalls = [m["rec_err_to_contra"] for m in model_metrics.values() if m["rec_err_to_contra"] is not None]
    cross_model_rec_gap = round(max(model_recalls) - min(model_recalls), 4) if len(model_recalls) >= 2 else None

    # Count distinct videos containing at least one true video error
    error_videos = sum(
        1 for vid, recs in by_video.items() if any(x.human_label == VideoTruthLabel.VIDEO_CONTRADICTED.value for x in recs)
    )

    fail_reasons: list[str] = []
    inconclusive_reasons: list[str] = []

    if overall["prec_contra_to_err"] is None:
        inconclusive_reasons.append("Zero Ref-Contradicted claims (precision denominator=0).")
    elif overall["prec_contra_to_err"] < prec_min_threshold:
        fail_reasons.append(f"Prec(Contra->Err)={overall['prec_contra_to_err']:.3f} < {prec_min_threshold}")

    if error_videos < min_error_videos_for_conclusive:
        inconclusive_reasons.append(
            f"Only {error_videos} videos contain true video errors (< {min_error_videos_for_conclusive}); insufficient power."
        )

    if rho_val is None:
        inconclusive_reasons.append("Paired-delta Spearman rho undefined (constant deltas or < 3 paired videos).")
    elif rho_val < rho_min_threshold:
        fail_reasons.append(f"Paired-delta Spearman rho={rho_val:.3f} < {rho_min_threshold}")

    if reversal_rate is not None and reversal_rate > max_reversal_rate:
        fail_reasons.append(f"Directional reversal rate={reversal_rate:.3f} > {max_reversal_rate}")

    if a7_vs_a0_rec_diff is not None and a7_vs_a0_rec_diff < -max_a7_recall_drop:
        fail_reasons.append(
            f"A7 systematic omission bias: Rec(A7)-Rec(A0)={a7_vs_a0_rec_diff:.3f} < -{max_a7_recall_drop}"
        )

    if (
        a7_vs_a0_leak_diff is not None
        and a7_vs_a0_leak_diff > max_a7_leak_increase
        and mean_hum_delta is not None
        and mean_hum_delta > 0
        and mean_bias is not None
        and (-mean_bias / mean_hum_delta) > 0.40
    ):
        fail_reasons.append(
            f"A7 Unmentioned leakage bias={a7_vs_a0_leak_diff:.3f} causes >40% underestimation of A7-A0 risk delta"
        )

    if cross_model_rec_gap is not None and cross_model_rec_gap > max_cross_model_recall_gap:
        fail_reasons.append(f"Cross-model capture gap={cross_model_rec_gap:.3f} > {max_cross_model_recall_gap}")

    if fail_reasons:
        verdict = VerdictStatus.FAIL.value
    elif inconclusive_reasons:
        verdict = VerdictStatus.INCONCLUSIVE.value
    else:
        verdict = VerdictStatus.PASS.value

    cis = {
        "prec_contra_to_err_ci95": _video_cluster_bootstrap_ci(video_ids, by_video, "prec_contra_to_err", n_boot=n_boot),
        "rec_err_to_contra_ci95": _video_cluster_bootstrap_ci(video_ids, by_video, "rec_err_to_contra", n_boot=n_boot),
        "leak_unment_to_err_ci95": _video_cluster_bootstrap_ci(video_ids, by_video, "leak_unment_to_err", n_boot=n_boot),
    }

    return {
        "verdict": verdict,
        "fail_reasons": fail_reasons,
        "inconclusive_reasons": inconclusive_reasons,
        "n_videos": len(video_ids),
        "n_error_videos": error_videos,
        "overall_3x4": overall,
        "video_cluster_ci95": cis,
        "by_arm": arm_metrics,
        "by_model": model_metrics,
        "paired_delta_a7_minus_a0": {
            "n_paired_video_models": len(delta_ref),
            "mean_delta_ref": round(float(np.mean(delta_ref)), 4) if delta_ref else None,
            "mean_delta_human": mean_hum_delta,
            "mean_differential_bias": mean_bias,
            "spearman_rho": rho_val,
            "nonzero_human_delta_count": len(nonzero_pairs),
            "directional_concordance": round(directional_concordance, 4) if directional_concordance is not None else None,
            "directional_reversal_rate": round(reversal_rate, 4) if reversal_rate is not None else None,
            "a7_minus_a0_recall_diff": a7_vs_a0_rec_diff,
            "a7_minus_a0_unment_leak_diff": a7_vs_a0_leak_diff,
            "cross_model_recall_gap": cross_model_rec_gap,
        },
    }
