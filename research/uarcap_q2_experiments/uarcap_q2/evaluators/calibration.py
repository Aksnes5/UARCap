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

from ..annotation.adjudicator import cohens_kappa, gwets_ac1
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


def make_stable_claim_id(query_id: str, arm: str, model_id: str, claim_index: int) -> str:
    return f"{query_id}::{arm}::{model_id}::C{claim_index + 1:02d}"


def reconcile_a2_human_and_auto(
    eval_rows: list[dict[str, Any]],
    human_payload: dict[str, Any] | list[dict[str, Any]],
    expected_package_sha256: str | None = None,
) -> dict[str, Any]:
    """Strict claim-ID reconciliation between A2 auto 3-way eval and human verification (NEVER silent zip!).

    Checks:
      - Bundle/package SHA-256 version binding.
      - Duplicate caption records and duplicate claim IDs.
      - Missing claim labels and unknown claim IDs.
      - Explicit support for human-added omitted claims (`added_claims`) and split/merge `claim_revisions`.
    """
    if isinstance(human_payload, dict):
        pkg_sha = str(human_payload.get("package_sha256", "") or "")
        if expected_package_sha256 and pkg_sha and pkg_sha != expected_package_sha256:
            raise ValueError(
                f"A2 bundle version mismatch: human package_sha256={pkg_sha!r} != expected={expected_package_sha256!r}"
            )
        human_records = list(human_payload.get("records", []))
    else:
        pkg_sha = ""
        human_records = list(human_payload)

    valid_truth = set(TRUTH_ORDER)
    auto_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in eval_rows:
        auto_by_key[(str(r["query_id"]), str(r["arm"]), str(r["model_id"]))] = r

    seen_caption_keys: set[tuple[str, str, str]] = set()
    aligned: list[AlignedAtomRecord] = []
    n_direct_matched = 0
    n_splits = 0
    n_merges = 0
    n_added_claims = 0
    n_added_video_errors = 0
    n_parse_failed_skipped = 0
    covered_auto_keys: set[tuple[str, str, str]] = set()

    for h in human_records:
        h_pkg = str(h.get("package_sha256", "") or "")
        if expected_package_sha256 and h_pkg and h_pkg != expected_package_sha256:
            raise ValueError(
                f"A2 record bundle mismatch: {h_pkg!r} != expected={expected_package_sha256!r}"
            )
        key = (str(h["query_id"]), str(h["arm"]), str(h["model_id"]))
        if key in seen_caption_keys:
            raise ValueError(f"Duplicate human annotation record for caption key {key}")
        seen_caption_keys.add(key)

        auto_r = auto_by_key.get(key)
        if auto_r is None:
            raise KeyError(f"Human annotation references unknown auto caption key: {key}")
        if auto_r.get("parse_status") != "OK":
            n_parse_failed_skipped += 1
            continue
        covered_auto_keys.add(key)

        # Build canonical auto claim map keyed strictly by stable claim_id
        auto_claims_map: dict[str, dict[str, Any]] = {}
        for idx, c_auto in enumerate(auto_r.get("claims", [])):
            cid = str(c_auto.get("claim_id") or make_stable_claim_id(key[0], key[1], key[2], idx))
            auto_claims_map[cid] = c_auto

        # Extract human labels by explicit claim_id (rejecting duplicate claim_ids or raw unlabeled list mismatch!)
        direct_labels: dict[str, str] = {}
        if "claim_labels_by_id" in h and isinstance(h["claim_labels_by_id"], dict):
            for cid, lbl in h["claim_labels_by_id"].items():
                cid_s = str(cid)
                lbl_s = str(lbl)
                if lbl_s not in valid_truth:
                    raise ValueError(f"Illegal human truth label {lbl_s!r} for claim {cid_s}")
                direct_labels[cid_s] = lbl_s
        elif "claim_annotations" in h and isinstance(h["claim_annotations"], list):
            for entry in h["claim_annotations"]:
                cid_s = str(entry["claim_id"])
                lbl_s = str(entry["label"])
                if cid_s in direct_labels:
                    raise ValueError(f"Duplicate human claim_id {cid_s!r} in caption {key}")
                if lbl_s not in valid_truth:
                    raise ValueError(f"Illegal human truth label {lbl_s!r} for claim {cid_s}")
                direct_labels[cid_s] = lbl_s
        elif "claim_labels" in h:
            raise ValueError(
                f"Legacy positional 'claim_labels' list without stable claim_id is forbidden (risk of silent truncation) on {key}; "
                "use `claim_labels_by_id` or `claim_annotations` with explicit `claim_id`."
            )

        # Process split / merge revisions
        revised_source_labels: dict[str, str] = {}
        for rev in h.get("claim_revisions", []):
            rtype = str(rev.get("revision_type", "")).lower()
            src_ids = [str(s) for s in rev.get("source_claim_ids", [])]
            res_claims = list(rev.get("resulting_claims", []))
            if rtype not in ("split", "merge") or not src_ids or not res_claims:
                raise ValueError(f"Malformed claim_revision on {key}: {rev}")
            if rtype == "split":
                n_splits += 1
            else:
                n_merges += 1
            res_labels = []
            for rc in res_claims:
                rl = str(rc.get("label", ""))
                if rl not in valid_truth:
                    raise ValueError(f"Illegal human label {rl!r} inside {rtype} revision on {key}")
                res_labels.append(rl)
            # Conservative video-error precedence: if any split sub-claim is Video-Contradicted, source carries Video-Contradicted
            if VideoTruthLabel.VIDEO_CONTRADICTED.value in res_labels:
                agg_lbl = VideoTruthLabel.VIDEO_CONTRADICTED.value
            elif VideoTruthLabel.VIDEO_UNVERIFIABLE.value in res_labels:
                agg_lbl = VideoTruthLabel.VIDEO_UNVERIFIABLE.value
            elif VideoTruthLabel.VIDEO_SUPPORTED.value in res_labels:
                agg_lbl = VideoTruthLabel.VIDEO_SUPPORTED.value
            else:
                agg_lbl = VideoTruthLabel.NON_FACTUAL.value
            for scid in src_ids:
                if scid not in auto_claims_map:
                    raise ValueError(f"claim_revision references unknown source_claim_id {scid!r} on {key}")
                if scid in revised_source_labels:
                    raise ValueError(f"Duplicate claim_revision targeting source_claim_id {scid!r} on {key}")
                revised_source_labels[scid] = agg_lbl

        # Verify ZERO unknown claim IDs and ZERO missing claim IDs!
        unknown = sorted(set(direct_labels.keys()) - set(auto_claims_map.keys()))
        if unknown:
            raise ValueError(f"Human annotation has unknown claim_id(s) on {key}: {unknown}")

        covered_cids = set(direct_labels.keys()) | set(revised_source_labels.keys())
        missing = sorted(set(auto_claims_map.keys()) - covered_cids)
        if missing:
            raise ValueError(f"Incomplete human annotation on {key}: missing claim_id(s) {missing}")

        for cid, c_auto in auto_claims_map.items():
            h_lbl = revised_source_labels.get(cid, direct_labels.get(cid))
            assert h_lbl is not None
            if cid not in revised_source_labels:
                n_direct_matched += 1
            aligned.append(
                AlignedAtomRecord(
                    video_id=key[0],
                    arm=key[1],
                    model_id=key[2],
                    ref_label=str(c_auto["label"]),
                    human_label=h_lbl,
                )
            )

        # Audit human-added omitted atomic claims (extraction omission audit!)
        for ac in h.get("added_claims", []):
            ac_lbl = str(ac.get("label", ""))
            if ac_lbl not in valid_truth:
                raise ValueError(f"Illegal label {ac_lbl!r} in added_claims on {key}")
            n_added_claims += 1
            if ac_lbl == VideoTruthLabel.VIDEO_CONTRADICTED.value:
                n_added_video_errors += 1

    ok_auto_keys = {k for k, r in auto_by_key.items() if r.get("parse_status") == "OK"}
    missing_captions = sorted(ok_auto_keys - covered_auto_keys)
    if missing_captions:
        raise ValueError(
            f"Incomplete A2 human annotations: {len(missing_captions)} auto-evaluated caption(s) missing (e.g. {missing_captions[:3]})"
        )

    total_true_video_errors = sum(
        1 for a in aligned if a.human_label == VideoTruthLabel.VIDEO_CONTRADICTED.value
    ) + n_added_video_errors

    return {
        "aligned_records": aligned,
        "reconciliation_audit": {
            "package_sha256_verified": bool(expected_package_sha256),
            "package_sha256": pkg_sha or expected_package_sha256 or "",
            "n_captions_reconciled": len(covered_auto_keys),
            "n_auto_parse_failed_skipped": n_parse_failed_skipped,
            "n_aligned_auto_claims": len(aligned),
            "n_direct_matched_claims": n_direct_matched,
            "n_split_revisions": n_splits,
            "n_merge_revisions": n_merges,
            "n_added_omitted_claims": n_added_claims,
            "n_omitted_video_errors": n_added_video_errors,
            "omitted_error_share": round(n_added_video_errors / total_true_video_errors, 4)
            if total_true_video_errors > 0
            else None,
        },
    }


def _binary_agreement_metrics(auto_vals: list[int], human_vals: list[int]) -> dict[str, Any]:
    n = len(auto_vals)
    if n == 0:
        return {
            "n": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "cohens_kappa": None,
            "gwets_ac1": None,
            "confusion": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        }
    tp = sum(1 for a, h in zip(auto_vals, human_vals) if a == 1 and h == 1)
    fp = sum(1 for a, h in zip(auto_vals, human_vals) if a == 1 and h == 0)
    fn = sum(1 for a, h in zip(auto_vals, human_vals) if a == 0 and h == 1)
    tn = sum(1 for a, h in zip(auto_vals, human_vals) if a == 0 and h == 0)
    acc = (tp + tn) / n
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec) > 0) else None
    sa = [str(x) for x in auto_vals]
    sh = [str(x) for x in human_vals]
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4) if prec is not None else None,
        "recall": round(rec, 4) if rec is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "cohens_kappa": cohens_kappa(sa, sh),
        "gwets_ac1": gwets_ac1(sa, sh, categories=["0", "1"]),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _compute_con_vs_neu_bias(
    paired_rows: list[tuple[dict[str, Any], dict[str, Any]]],
    field_name: str,
) -> dict[str, Any]:
    """Compute Con vs Neu differential evaluation bias: (auto_Con - auto_Neu) - (human_Con - human_Neu)."""
    con_a = [int(a[field_name]) for a, _ in paired_rows if a["arm"] == ExpArm.E3_CON.value]
    neu_a = [int(a[field_name]) for a, _ in paired_rows if a["arm"] == ExpArm.E3_NEU.value]
    con_h = [int(h[field_name]) for a, h in paired_rows if a["arm"] == ExpArm.E3_CON.value]
    neu_h = [int(h[field_name]) for a, h in paired_rows if a["arm"] == ExpArm.E3_NEU.value]
    if not con_a or not neu_a:
        return {
            "n_con": len(con_a),
            "n_neu": len(neu_a),
            "auto_rate_con": None,
            "auto_rate_neu": None,
            "human_rate_con": None,
            "human_rate_neu": None,
            "delta_auto_con_minus_neu": None,
            "delta_human_con_minus_neu": None,
            "differential_bias": None,
        }
    ra_con = sum(con_a) / len(con_a)
    ra_neu = sum(neu_a) / len(neu_a)
    rh_con = sum(con_h) / len(con_h)
    rh_neu = sum(neu_h) / len(neu_h)
    da = ra_con - ra_neu
    dh = rh_con - rh_neu
    return {
        "n_con": len(con_a),
        "n_neu": len(neu_a),
        "auto_rate_con": round(ra_con, 4),
        "auto_rate_neu": round(ra_neu, 4),
        "human_rate_con": round(rh_con, 4),
        "human_rate_neu": round(rh_neu, 4),
        "delta_auto_con_minus_neu": round(da, 4),
        "delta_human_con_minus_neu": round(dh, 4),
        "differential_bias": round(da - dh, 4),
    }


def audit_e3_evaluator_agreement(
    auto_e3_rows: list[dict[str, Any]],
    human_e3_payload: dict[str, Any] | list[dict[str, Any]],
    expected_package_sha256: str | None = None,
    min_accuracy: float = 0.80,
    max_abs_con_neu_diff_bias: float = 0.15,
    min_paired_items: int = 4,
) -> dict[str, Any]:
    """Independent Y_true and Y_conflict human-vs-auto agreement + Con vs Neu differential bias audit for Pilot A3."""
    if isinstance(human_e3_payload, dict):
        pkg_sha = str(human_e3_payload.get("package_sha256", "") or "")
        if expected_package_sha256 and pkg_sha and pkg_sha != expected_package_sha256:
            raise ValueError(
                f"E3 bundle version mismatch: human package_sha256={pkg_sha!r} != expected={expected_package_sha256!r}"
            )
        human_rows = list(human_e3_payload.get("records", []))
    else:
        pkg_sha = ""
        human_rows = list(human_e3_payload)

    auto_by_key = {(str(r["query_id"]), str(r["arm"]), str(r["model_id"])): r for r in auto_e3_rows}
    human_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for h in human_rows:
        k = (str(h["query_id"]), str(h["arm"]), str(h["model_id"]))
        if k in human_by_key:
            raise ValueError(f"Duplicate E3 human annotation for key {k}")
        human_by_key[k] = h

    missing_keys = sorted(set(auto_by_key) - set(human_by_key))
    if missing_keys:
        raise ValueError(f"Incomplete E3 human annotations: {len(missing_keys)} item(s) missing (e.g. {missing_keys[:3]})")

    paired: list[tuple[dict[str, Any], dict[str, Any]]] = []
    n_auto_parse_failed = 0
    for k, a in sorted(auto_by_key.items()):
        if a.get("parse_status", "OK") != "OK" or a.get("y_true") is None or a.get("y_conflict") is None:
            n_auto_parse_failed += 1
            continue
        paired.append((a, human_by_key[k]))

    yt_metrics = _binary_agreement_metrics([int(a["y_true"]) for a, _ in paired], [int(h["y_true"]) for _, h in paired])
    yc_metrics = _binary_agreement_metrics(
        [int(a["y_conflict"]) for a, _ in paired], [int(h["y_conflict"]) for _, h in paired]
    )

    con_neu_overall = {
        "y_true": _compute_con_vs_neu_bias(paired, "y_true"),
        "y_conflict": _compute_con_vs_neu_bias(paired, "y_conflict"),
    }

    by_model: dict[str, Any] = {}
    models = sorted({a["model_id"] for a, _ in paired})
    for mid in models:
        sub = [(a, h) for a, h in paired if a["model_id"] == mid]
        by_model[mid] = {
            "y_true": _binary_agreement_metrics([int(a["y_true"]) for a, _ in sub], [int(h["y_true"]) for _, h in sub]),
            "y_conflict": _binary_agreement_metrics(
                [int(a["y_conflict"]) for a, _ in sub], [int(h["y_conflict"]) for _, h in sub]
            ),
            "con_vs_neu_bias": {
                "y_true": _compute_con_vs_neu_bias(sub, "y_true"),
                "y_conflict": _compute_con_vs_neu_bias(sub, "y_conflict"),
            },
        }

    fail_reasons: list[str] = []
    inconclusive_reasons: list[str] = []

    if len(paired) < min_paired_items:
        inconclusive_reasons.append(f"Only {len(paired)} valid paired E3 items (< {min_paired_items}).")
    if yt_metrics["accuracy"] is not None and yt_metrics["accuracy"] < min_accuracy:
        fail_reasons.append(f"Y_true accuracy={yt_metrics['accuracy']:.3f} < {min_accuracy}")
    if yc_metrics["accuracy"] is not None and yc_metrics["accuracy"] < min_accuracy:
        fail_reasons.append(f"Y_conflict accuracy={yc_metrics['accuracy']:.3f} < {min_accuracy}")

    for ep in ("y_true", "y_conflict"):
        db = con_neu_overall[ep]["differential_bias"]
        if db is None:
            inconclusive_reasons.append(f"Con vs Neu differential bias undefined for {ep} (missing Con or Neu arms).")
        elif abs(db) > max_abs_con_neu_diff_bias:
            fail_reasons.append(
                f"Con vs Neu differential bias on {ep}={db:+.3f} exceeds +/-{max_abs_con_neu_diff_bias}"
            )

    if fail_reasons:
        verdict = VerdictStatus.FAIL.value
    elif inconclusive_reasons:
        verdict = VerdictStatus.INCONCLUSIVE.value
    else:
        verdict = VerdictStatus.PASS.value

    return {
        "verdict": verdict,
        "fail_reasons": fail_reasons,
        "inconclusive_reasons": inconclusive_reasons,
        "package_sha256": pkg_sha or expected_package_sha256 or "",
        "n_paired_valid": len(paired),
        "n_auto_parse_failed": n_auto_parse_failed,
        "y_true_agreement": yt_metrics,
        "y_conflict_agreement": yc_metrics,
        "con_vs_neu_differential_bias": con_neu_overall,
        "by_model": by_model,
    }


def audit_cross_model_dream1k_bias(
    videollama_records: list[AlignedAtomRecord],
    modern_vlm_records: list[AlignedAtomRecord],
    max_recall_gap: float = 0.20,
    max_precision_gap: float = 0.20,
    max_paired_bias_gap: float = 0.35,
    n_boot: int = 300,
) -> dict[str, Any]:
    """Verify DREAM-1K proxy evaluator does not show severe differential bias between Video-LLaMA and Modern VLM."""
    vl_cal = audit_evaluator_calibration(videollama_records, n_boot=n_boot)
    vlm_cal = audit_evaluator_calibration(modern_vlm_records, n_boot=n_boot)

    r1, r2 = vl_cal["overall_3x4"]["rec_err_to_contra"], vlm_cal["overall_3x4"]["rec_err_to_contra"]
    p1, p2 = vl_cal["overall_3x4"]["prec_contra_to_err"], vlm_cal["overall_3x4"]["prec_contra_to_err"]
    b1 = vl_cal["paired_delta_a7_minus_a0"]["mean_differential_bias"]
    b2 = vlm_cal["paired_delta_a7_minus_a0"]["mean_differential_bias"]

    rec_gap = round(abs(r1 - r2), 4) if (r1 is not None and r2 is not None) else None
    prec_gap = round(abs(p1 - p2), 4) if (p1 is not None and p2 is not None) else None
    bias_gap = round(abs(b1 - b2), 4) if (b1 is not None and b2 is not None) else None

    fail_reasons: list[str] = []
    inconclusive_reasons: list[str] = []
    if rec_gap is not None and rec_gap > max_recall_gap:
        fail_reasons.append(f"Cross-model DREAM-1K error recall gap={rec_gap:.3f} > {max_recall_gap}")
    if prec_gap is not None and prec_gap > max_precision_gap:
        fail_reasons.append(f"Cross-model DREAM-1K precision gap={prec_gap:.3f} > {max_precision_gap}")
    if bias_gap is not None and bias_gap > max_paired_bias_gap:
        fail_reasons.append(f"Cross-model DREAM-1K paired-delta bias gap={bias_gap:.3f} > {max_paired_bias_gap}")
    if rec_gap is None and prec_gap is None and bias_gap is None:
        inconclusive_reasons.append("Insufficient cross-model DREAM-1K errors/paired deltas to compute bias gaps.")

    verdict = (
        VerdictStatus.FAIL.value
        if fail_reasons
        else (VerdictStatus.INCONCLUSIVE.value if inconclusive_reasons else VerdictStatus.PASS.value)
    )
    return {
        "verdict": verdict,
        "fail_reasons": fail_reasons,
        "inconclusive_reasons": inconclusive_reasons,
        "recall_gap": rec_gap,
        "precision_gap": prec_gap,
        "paired_delta_bias_gap": bias_gap,
        "videollama_audit": vl_cal,
        "modern_vlm_audit": vlm_cal,
    }
