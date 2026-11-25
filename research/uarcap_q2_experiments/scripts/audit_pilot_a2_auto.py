#!/usr/bin/env python3
"""Independent integrity audit of the Pilot A2 automatic stage (no human claims yet).

Checks input reuse, dedup semantics, per-arm claim ID mapping, exact-evidence reuse,
blindness of annotator bundles, frame coverage, raw-response completeness, and that the
core-event lock predates the judge run.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uarcap_q2.annotation.package_builder import assert_blind_payload_clean  # noqa: E402
from uarcap_q2.common.io_utils import sha256_file  # noqa: E402


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _sha_obj(obj) -> str:
    return sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2-dir", type=Path, required=True)
    ap.add_argument("--a1-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    out = args.a2_dir
    checks: dict[str, dict] = {}

    def check(name: str, ok: bool, **details) -> None:
        checks[name] = {"ok": bool(ok), **details}

    rows = _read_jsonl(out / "a2_ref_fact_eval.jsonl")
    raw = _read_jsonl(out / "a2_judge_raw_calls.jsonl")
    rec = json.loads((out / "a2_judge_run_receipt.json").read_text(encoding="utf-8"))
    lock = json.loads((out / "a2_core_event_lock.json").read_text(encoding="utf-8"))
    ver = json.loads((out / "a2_input_verification.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "a2_bundle_manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads((out / "calibration_receipt.json").read_text(encoding="utf-8"))

    qids = [e["query_id"] for e in lock["entries"]]
    arms = ["ARM_A0", "ARM_A1", "ARM_A7"]
    keys = [(r["query_id"], r["arm"]) for r in rows]
    expected = {(q, a) for q in qids for a in arms}
    check(
        "eval_rows_complete_90",
        len(rows) == 90 and set(keys) == expected and len(set(keys)) == 90,
        n_rows=len(rows), missing=sorted(f"{q}/{a}" for q, a in (expected - set(keys)))[:10],
    )
    check(
        "all_parse_status_ok",
        all(r["parse_status"] == "OK" for r in rows) and rec["n_parse_failures"] == 0,
        parse_failures=rec["n_parse_failures"], statuses=sorted({r["parse_status"] for r in rows}),
    )

    # Exact-input duplicate reuse: identical (qid, caption) => identical claim texts/labels/evidence.
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["query_id"], r["caption"])].append(r)
    reuse_viol = []
    id_viol = []
    for key, group in groups.items():
        sig = [_sha_obj([(c["claim"], c["label"], c.get("evidence", "")) for c in g.get("claims", [])]) for g in group]
        if len(set(sig)) != 1:
            reuse_viol.append({"key": key[0], "caption_sha": sha256(key[1].encode()).hexdigest()[:12]})
        ids = {c["claim_id"] for g in group for c in g.get("claims", [])}
        if len(group) > 1 and len(ids) != sum(len(g.get("claims", [])) for g in group):
            id_viol.append({"key": key[0], "arms": [g["arm"] for g in group]})
    check(
        "identical_captions_have_identical_verdicts",
        not reuse_viol,
        n_unique_caption_groups=len(groups), violations=reuse_viol[:5],
        n_duplicate_reuse_sum=sum(len(g) - 1 for g in groups.values()),
    )
    check("per_arm_claim_ids_unique", not id_viol, violations=id_viol[:5])

    # Reference fields used + core-event verdict alignment
    lock_core = {e["query_id"]: (1 if e.get("core_event_candidate") else 0) for e in lock["entries"]}
    ref_viol = []
    core_viol = []
    for q in qids:
        qrows = [r for r in rows if r["query_id"] == q]
        shas = {r.get("reference_description_sha256") for r in qrows}
        if len(shas) != 1 or any(not r.get("reference_fields_used") for r in qrows):
            ref_viol.append(q)
        for r in qrows:
            if r["parse_status"] == "OK":
                n_core = len(r.get("core_event_verdicts", []))
                if n_core != lock_core[q] or (lock_core[q] and r.get("r_core_ref") is None):
                    core_viol.append(f"{q}/{r['arm']}")
    check("reference_fields_consistent", not ref_viol, violations=ref_viol[:5],
          fields=sorted({tuple(r.get("reference_fields_used", [])) for r in rows})[:3])
    check("core_event_verdicts_aligned", not core_viol, violations=core_viol[:5])

    # Blindness + coverage of bundles
    pkg = out / "blind_annotation_package"
    bundle_report = {}
    blind_viol = []
    for ann in ("A1", "A2"):
        bp = pkg / f"blind_tasks_e2_{ann}.json"
        bundle = json.loads(bp.read_text(encoding="utf-8"))
        try:
            assert_blind_payload_clean(bundle)
        except Exception as exc:  # noqa: BLE001
            blind_viol.append(f"{ann}:{exc}")
        text = bp.read_text(encoding="utf-8")
        if "ARM_A0" in text or "ARM_A1" in text or "ARM_A7" in text:
            blind_viol.append(f"{ann}:arm_token")
        n_caps = sum(len(t.get("captions", [])) for t in bundle.get("tasks", []))
        n_frames_ok = sum(1 for t in bundle.get("tasks", []) if len(t.get("sampled_frame_paths", [])) == 16)
        bundle_report[ann] = {"n_tasks": len(bundle.get("tasks", [])), "n_captions": n_caps, "n_tasks_with_16_frames": n_frames_ok}
        if len(bundle.get("tasks", [])) != 30 or n_caps != 90 or n_frames_ok != 30:
            blind_viol.append(f"{ann}:coverage")
    check("blind_bundles_clean_and_complete", not blind_viol, bundles=bundle_report, violations=blind_viol)

    key = json.loads((pkg / "blind_key_e2_DO_NOT_OPEN.json").read_text(encoding="utf-8"))
    mapping = key.get("mapping", {})
    mapped_arms = {v["arm"] for v in mapping.values()}
    mapped_qids = {v["query_id"] for v in mapping.values()}
    n_auto_claims = sum(len(r.get("claims", [])) for r in rows)
    n_key_claims = sum(len(v.get("claim_id_map", {})) for v in mapping.values())
    per_annotator: dict[str, int] = defaultdict(int)
    for v in mapping.values():
        per_annotator[str(v.get("annotator_id"))] += 1
    check(
        "blind_key_maps_all_conditions",
        len(mapping) == 180
        and per_annotator == {"A1": 90, "A2": 90}
        and mapped_arms == set(arms)
        and len(mapped_qids) == 30
        and n_key_claims == 2 * n_auto_claims,
        n_blind_captions=len(mapping), per_annotator=dict(per_annotator), arms=sorted(mapped_arms),
        n_key_claims=n_key_claims, n_auto_claims=n_auto_claims,
    )

    frame_dirs = [d for d in (pkg / "frames").iterdir() if d.is_dir()] if (pkg / "frames").is_dir() else []
    bad_frames = [d.name for d in frame_dirs if len(list(d.glob("*.jpg"))) != 16 or any(f.stat().st_size == 0 for f in d.glob("*.jpg"))]
    check(
        "frame_previews_complete",
        len(frame_dirs) == 30 and not bad_frames,
        n_frame_dirs=len(frame_dirs), bad=bad_frames[:5],
    )

    # Raw judge calls
    empty_raw = [i for i, r in enumerate(raw) if not str(r.get("raw_response", "")).strip()]
    check(
        "raw_judge_outputs_saved",
        bool(raw) and not empty_raw and rec.get("n_raw_judge_calls") == len(raw),
        n_calls=len(raw), empty=len(empty_raw), retries=rec.get("n_retries"),
        stages=sorted({r["stage"] for r in raw}),
    )

    # Core-event lock predates the judge run and is built from human metadata only.
    lock_time = lock.get("timestamp_utc")
    rec_time = rec.get("timestamp_utc")
    check(
        "core_event_lock_precedes_judge",
        bool(lock_time and rec_time and lock_time < rec_time) and all(e.get("core_event_candidate") for e in lock["entries"]),
        lock_time=lock_time, judge_time=rec_time, n_entries=len(lock["entries"]),
    )
    check(
        "a1_inputs_sha_verified",
        ver.get("all_verified") is True,
        a1_status=ver.get("a1_run_status"), metadata_match=ver.get("metadata", {}).get("match"),
    )
    check(
        "judge_receipt_consistent",
        rec.get("n_items") == 90 and rec.get("n_unique_captions_evaluated") == len(groups)
        and rec.get("a1_predictions_sha256") == ver["a1_files"]["predictions.jsonl"]["actual_sha256"],
        judge=rec.get("judge_model_id"), family=rec.get("judge_family"), unique=rec.get("n_unique_captions_evaluated"),
    )
    check(
        "status_human_validation_pending",
        receipt.get("status") == "PILOT_A2_HUMAN_VALIDATION_PENDING" and receipt.get("human_calibration_completed") is False,
        status=receipt.get("status"),
    )

    artifacts = [
        "a2_ref_fact_eval.jsonl", "a2_judge_raw_calls.jsonl", "a2_judge_run_receipt.json",
        "a2_input_verification.json", "a2_core_event_lock.json", "a2_bundle_manifest.json",
        "calibration_receipt.json", "a2_auto_diagnostics.json", "PILOT_A2_AUTO_DIAGNOSTICS.md",
        "blind_annotation_package/blind_tasks_e2_A1.json", "blind_annotation_package/blind_tasks_e2_A2.json",
    ]
    artifact_info = {a: {"exists": (out / a).is_file(), "bytes": (out / a).stat().st_size if (out / a).is_file() else None,
                         "sha256": sha256_file(out / a) if (out / a).is_file() else None} for a in artifacts}

    all_ok = all(c["ok"] for c in checks.values())
    payload = {
        "audit_schema": "uarcap_q2.pilot_a2_auto_audit.v1",
        "all_checks_passed": all_ok,
        "checks": checks,
        "artifacts": artifact_info,
        "package_sha256": manifest.get("package_sha256"),
        "source_verification_sha256": sha256_file(out / "a2_input_verification.json"),
        "core_event_lock_sha256": lock.get("lock_sha256"),
    }
    (out / "AUDIT_PILOT_A2_AUTO.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# PILOT A2 AUTOMATIC STAGE — INDEPENDENT AUDIT", "", f"- **All checks passed**: `{all_ok}`", "",
             "| Check | Result | Details |", "| --- | --- | --- |"]
    for name, c in checks.items():
        details = "; ".join(f"{k}={v}" for k, v in c.items() if k != "ok")
        lines.append(f"| `{name}` | {'PASS' if c['ok'] else 'FAIL'} | {details} |")
    lines += ["", "## Artifacts", "", "| File | Bytes | SHA-256 |", "| --- | --- | --- |"]
    for a, info in artifact_info.items():
        lines.append(f"| `{a}` | {info['bytes']} | `{info['sha256']}` |")
    (out / "AUDIT_PILOT_A2_AUTO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"all_checks_passed": all_ok, "checks": {k: v["ok"] for k, v in checks.items()}}, indent=2))
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
