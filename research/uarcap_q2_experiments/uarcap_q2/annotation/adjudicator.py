"""Dual-annotator agreement (Cohen's kappa, Gwet's AC1), disagreement extraction, and 2-annotator time ledger."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..common.io_utils import atomic_write_json, read_json, read_jsonl
from .package_builder import unblind_annotations


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


def _load_annotator_export(path: Path | str) -> list[dict[str, Any]]:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        return read_jsonl(p)
    obj = read_json(p)
    if isinstance(obj, dict) and "records" in obj:
        return list(obj["records"])
    if isinstance(obj, dict):
        # Raw progress.json dict keyed by task_key
        return list(obj.values())
    return list(obj)


def adjudicate_e2_annotations(
    ann1_export_path: Path | str,
    ann2_export_path: Path | str,
    blind_key_path: Path | str,
    out_dir: Path | str,
    resolutions_path: Path | str | None = None,
    adjudication_joint_sec: float = 0.0,
) -> dict[str, Any]:
    """End-to-end E2 dual-annotator unblinding, disagreement extraction, and canonical adjudicated JSON export."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key_obj = read_json(blind_key_path)
    pkg_sha = str(key_obj.get("package_sha256", ""))

    u1 = unblind_annotations(_load_annotator_export(ann1_export_path), blind_key_path)
    u2 = unblind_annotations(_load_annotator_export(ann2_export_path), blind_key_path)

    by_cap_1 = {(r["query_id"], r["arm"], r["model_id"]): r for r in u1}
    by_cap_2 = {(r["query_id"], r["arm"], r["model_id"]): r for r in u2}

    map1: dict[str, str] = {}
    map2: dict[str, str] = {}
    for k in sorted(set(by_cap_1) | set(by_cap_2)):
        if k in by_cap_1:
            for cid, lbl in by_cap_1[k].get("claim_labels_by_id", {}).items():
                map1[cid] = str(lbl)
        if k in by_cap_2:
            for cid, lbl in by_cap_2[k].get("claim_labels_by_id", {}).items():
                map2[cid] = str(lbl)

    sec1 = sum(float(r.get("elapsed_sec", 0.0)) for r in u1)
    sec2 = sum(float(r.get("elapsed_sec", 0.0)) for r in u2)
    summary = extract_disagreements_and_time_summary(map1, map2, sec1, sec2, adjudication_joint_sec=adjudication_joint_sec)

    resolutions: dict[str, str] = {}
    if resolutions_path and Path(resolutions_path).is_file():
        raw_res = read_json(resolutions_path)
        if isinstance(raw_res, dict) and "resolutions" in raw_res:
            resolutions = {str(k): str(v) for k, v in raw_res["resolutions"].items()}
        elif isinstance(raw_res, dict) and isinstance(raw_res.get("disagreements"), list):
            for it in raw_res["disagreements"]:
                if it.get("resolved_label"):
                    resolutions[str(it["item_key"])] = str(it["resolved_label"])
        elif isinstance(raw_res, list):
            for it in raw_res:
                if it.get("resolved_label"):
                    resolutions[str(it["item_key"])] = str(it["resolved_label"])
        elif isinstance(raw_res, dict):
            resolutions = {str(k): str(v) for k, v in raw_res.items()}

    unresolved = [d for d in summary["disagreements"] if d["item_key"] not in resolutions]
    if unresolved:
        dis_file = atomic_write_json(
            out / "e2_disagreements_to_resolve.json",
            {
                "package_sha256": pkg_sha,
                "n_unresolved": len(unresolved),
                "instructions": "Fill `resolved_label` for each item and re-run adjudicate_annotations.py --resolutions <this_file>",
                "disagreements": [{**d, "resolved_label": ""} for d in unresolved],
            },
        )
        return {
            "status": "DISAGREEMENTS_PENDING_RESOLUTION",
            "package_sha256": pkg_sha,
            "agreement_summary": summary,
            "disagreements_file": str(dis_file),
            "adjudicated_path": None,
        }

    records: list[dict[str, Any]] = []
    for cap_key in sorted(set(by_cap_1) | set(by_cap_2)):
        r1 = by_cap_1.get(cap_key, {})
        r2 = by_cap_2.get(cap_key, {})
        all_cids = sorted(set(r1.get("claim_labels_by_id", {})) | set(r2.get("claim_labels_by_id", {})))
        resolved_labels: dict[str, str] = {}
        for cid in all_cids:
            if cid in resolutions:
                resolved_labels[cid] = resolutions[cid]
            elif cid in map1 and cid in map2:
                resolved_labels[cid] = map1[cid]
            else:
                resolved_labels[cid] = map1.get(cid, map2[cid])
        # Merge omitted added_claims & split/merge revisions from both annotators (dedup by text/source)
        seen_added: set[tuple[str, str]] = set()
        added_merged: list[dict[str, Any]] = []
        for ac in list(r1.get("added_claims", [])) + list(r2.get("added_claims", [])):
            sig = (str(ac.get("text", "")).strip().lower(), str(ac.get("label", "")))
            if sig not in seen_added:
                seen_added.add(sig)
                added_merged.append(ac)
        revs_merged = list(r1.get("claim_revisions", [])) or list(r2.get("claim_revisions", []))
        records.append(
            {
                "query_id": cap_key[0],
                "arm": cap_key[1],
                "model_id": cap_key[2],
                "claim_labels_by_id": resolved_labels,
                "added_claims": added_merged,
                "claim_revisions": revs_merged,
                "core_events": list(r1.get("core_events", [])) or list(r2.get("core_events", [])),
            }
        )

    adj_payload = {
        "schema_version": "uarcap-q2-adjudicated-e2-v2",
        "package_sha256": pkg_sha,
        "agreement_summary": summary,
        "records": records,
    }
    adj_path = atomic_write_json(out / "adjudicated_e2.json", adj_payload)
    return {
        "status": "ADJUDICATED_COMPLETE",
        "package_sha256": pkg_sha,
        "agreement_summary": summary,
        "adjudicated_path": str(adj_path),
        "n_captions": len(records),
    }


def adjudicate_e3_annotations(
    ann1_export_path: Path | str,
    ann2_export_path: Path | str,
    blind_key_path: Path | str,
    out_dir: Path | str,
    resolutions_path: Path | str | None = None,
    adjudication_joint_sec: float = 0.0,
) -> dict[str, Any]:
    """End-to-end E3 dual-annotator unblinding, independent Y_true/Y_conflict agreement, & adjudicated JSON export."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key_obj = read_json(blind_key_path)
    pkg_sha = str(key_obj.get("package_sha256", ""))

    u1 = unblind_annotations(_load_annotator_export(ann1_export_path), blind_key_path)
    u2 = unblind_annotations(_load_annotator_export(ann2_export_path), blind_key_path)

    by_k1 = {(r["query_id"], r["arm"], r["model_id"]): r for r in u1}
    by_k2 = {(r["query_id"], r["arm"], r["model_id"]): r for r in u2}

    map1_t = {f"{k[0]}::{k[1]}::{k[2]}::y_true": str(int(r["y_true"])) for k, r in by_k1.items()}
    map2_t = {f"{k[0]}::{k[1]}::{k[2]}::y_true": str(int(r["y_true"])) for k, r in by_k2.items()}
    map1_c = {f"{k[0]}::{k[1]}::{k[2]}::y_conflict": str(int(r["y_conflict"])) for k, r in by_k1.items()}
    map2_c = {f"{k[0]}::{k[1]}::{k[2]}::y_conflict": str(int(r["y_conflict"])) for k, r in by_k2.items()}

    sec1 = sum(float(r.get("elapsed_sec", 0.0)) for r in u1)
    sec2 = sum(float(r.get("elapsed_sec", 0.0)) for r in u2)
    sum_t = extract_disagreements_and_time_summary(map1_t, map2_t, sec1, sec2, adjudication_joint_sec=adjudication_joint_sec)
    sum_c = extract_disagreements_and_time_summary(map1_c, map2_c, sec1, sec2, adjudication_joint_sec=adjudication_joint_sec)
    all_dis = sum_t["disagreements"] + sum_c["disagreements"]

    resolutions: dict[str, int] = {}
    if resolutions_path and Path(resolutions_path).is_file():
        raw_res = read_json(resolutions_path)
        items = raw_res.get("disagreements", []) if isinstance(raw_res, dict) else raw_res
        if isinstance(raw_res, dict) and "resolutions" in raw_res:
            resolutions = {str(k): int(v) for k, v in raw_res["resolutions"].items()}
        elif isinstance(items, list):
            for it in items:
                if it.get("resolved_value") not in (None, ""):
                    resolutions[str(it["item_key"])] = int(it["resolved_value"])

    unresolved = [d for d in all_dis if d["item_key"] not in resolutions]
    if unresolved:
        dis_file = atomic_write_json(
            out / "e3_disagreements_to_resolve.json",
            {
                "package_sha256": pkg_sha,
                "n_unresolved": len(unresolved),
                "instructions": "Fill `resolved_value` (0 or 1) for each item and re-run adjudicate_annotations.py --task e3 --resolutions <this_file>",
                "disagreements": [{**d, "resolved_value": ""} for d in unresolved],
            },
        )
        return {
            "status": "DISAGREEMENTS_PENDING_RESOLUTION",
            "package_sha256": pkg_sha,
            "y_true_agreement": sum_t,
            "y_conflict_agreement": sum_c,
            "disagreements_file": str(dis_file),
            "adjudicated_path": None,
        }

    records: list[dict[str, Any]] = []
    for k in sorted(set(by_k1) | set(by_k2)):
        kt = f"{k[0]}::{k[1]}::{k[2]}::y_true"
        kc = f"{k[0]}::{k[1]}::{k[2]}::y_conflict"
        yt = resolutions[kt] if kt in resolutions else int(map1_t.get(kt, map2_t[kt]))
        yc = resolutions[kc] if kc in resolutions else int(map1_c.get(kc, map2_c[kc]))
        records.append(
            {
                "query_id": k[0],
                "arm": k[1],
                "model_id": k[2],
                "y_true": yt,
                "y_conflict": yc,
            }
        )

    adj_payload = {
        "schema_version": "uarcap-q2-adjudicated-e3-v2",
        "package_sha256": pkg_sha,
        "y_true_agreement": sum_t,
        "y_conflict_agreement": sum_c,
        "records": records,
    }
    adj_path = atomic_write_json(out / "adjudicated_e3.json", adj_payload)
    return {
        "status": "ADJUDICATED_COMPLETE",
        "package_sha256": pkg_sha,
        "y_true_agreement": sum_t,
        "y_conflict_agreement": sum_c,
        "adjudicated_path": str(adj_path),
        "n_items": len(records),
    }
