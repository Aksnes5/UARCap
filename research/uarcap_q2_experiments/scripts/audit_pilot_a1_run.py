#!/usr/bin/env python3
"""Independent post-run integrity audit for Pilot A1 (does not trust runner summaries).

Verifies:
  1. 30 frozen queries x 3 arms = 90 unique predictions, all non-empty.
  2. Prediction <-> actual-input-trace hash linkage (1:1).
  3. ARM_A0 strict zero external evidence.
  4. ARM_A1 visual evidence is an exact prefix of ARM_A7; sizes 1 and 7.
  5. Per-query visual input hash identical across arms; 16 frame indices.
  6. Retrieval evidence present for every query, top-20 recorded, reranked >= 7,
     first 7 reranked representatives equal ARM_A7 visual evidence.
  7. Query set equals the frozen sampling manifest and query_manifest.json.
  8. Wall/GPU timing and peak VRAM recorded.

Writes AUDIT_PILOT_A1.json and AUDIT_PILOT_A1.md next to the run artifacts.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ARMS = ["ARM_A0", "ARM_A1", "ARM_A7"]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--frozen-manifest", type=Path, required=True)
    args = ap.parse_args(argv)
    out = args.run_dir

    preds = _read_jsonl(out / "predictions.jsonl")
    traces = _read_jsonl(out / "input_traces.jsonl")
    retrieval = _read_jsonl(out / "retrieval_candidates.jsonl")
    receipt_path = out / "run_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {}
    qman_path = out / "query_manifest.json"
    qman = json.loads(qman_path.read_text(encoding="utf-8")) if qman_path.is_file() else {}
    frozen = json.loads(args.frozen_manifest.read_text(encoding="utf-8"))
    frozen_ids = [e["query_id"] for e in frozen["selected"]]

    checks: dict[str, dict] = {}

    def check(name: str, ok: bool, **details) -> None:
        checks[name] = {"ok": bool(ok), **details}

    pred_keys = [(p.get("query_id"), p.get("arm"), p.get("model_id")) for p in preds]
    expected_keys = {(q, a, preds[0].get("model_id") if preds else None) for q in frozen_ids for a in ARMS}
    check(
        "predictions_complete_90_unique",
        len(preds) == 90 and len(set(pred_keys)) == 90 and set(pred_keys) == expected_keys,
        n_predictions=len(preds),
        n_unique=len(set(pred_keys)),
        n_expected=90,
        missing=sorted(f"{q}/{a}" for q, a, _ in (expected_keys - set(pred_keys))),
        extra=sorted(f"{q}/{a}" for q, a, _ in (set(pred_keys) - expected_keys)),
    )

    trace_map = {(t.get("query_id"), t.get("arm"), t.get("model_id")): t for t in traces}
    mismatch = []
    for p in preds:
        key = (p.get("query_id"), p.get("arm"), p.get("model_id"))
        t = trace_map.get(key)
        if t is None or p.get("trace_sha256") != t.get("trace_sha256") or p.get("input_trace") != {
            k: v for k, v in t.items() if k not in ("run_config_fingerprint", "trace_sha256")
        }:
            mismatch.append(key)
    check(
        "prediction_trace_hash_linkage",
        len(traces) == 90 and len(trace_map) == 90 and not mismatch,
        n_traces=len(traces),
        mismatches=mismatch[:10],
    )

    bad_captions = [f"{p['query_id']}/{p['arm']}" for p in preds if not str(p.get("caption", "")).strip() or not p.get("ok", True)]
    check("all_generations_nonempty_ok", not bad_captions, bad=bad_captions[:10], n_bad=len(bad_captions))

    a0_viol = []
    for p in preds:
        if p.get("arm") != "ARM_A0":
            continue
        t = p.get("input_trace", {})
        if t.get("text_evidence") or t.get("visual_evidence_ids") or t.get("visual_evidence_scores"):
            a0_viol.append(p["query_id"])
    check("a0_strict_zero_evidence", len([p for p in preds if p.get("arm") == "ARM_A0"]) == 30 and not a0_viol, violations=a0_viol)

    by_q_arm = {(p.get("query_id"), p.get("arm")): p for p in preds}
    prefix_viol = []
    for q in frozen_ids:
        a1 = by_q_arm.get((q, "ARM_A1"))
        a7 = by_q_arm.get((q, "ARM_A7"))
        if not a1 or not a7:
            prefix_viol.append(f"{q}:missing")
            continue
        t1, t7 = a1.get("input_trace", {}), a7.get("input_trace", {})
        ids1, ids7 = t1.get("visual_evidence_ids", []), t7.get("visual_evidence_ids", [])
        s1, s7 = t1.get("visual_evidence_scores", []), t7.get("visual_evidence_scores", [])
        if (
            len(ids1) != 1
            or len(ids7) != 7
            or ids7[:1] != ids1
            or [round(x, 6) for x in s7[:1]] != [round(x, 6) for x in s1]
            or t1.get("text_evidence") != t7.get("text_evidence", [None])[:1]
        ):
            prefix_viol.append(q)
    check("a1_prefix_of_a7", not prefix_viol, violations=prefix_viol)

    frame_viol = []
    for q in frozen_ids:
        hashes = set()
        for arm in ARMS:
            p = by_q_arm.get((q, arm))
            if not p:
                frame_viol.append(f"{q}/{arm}:missing")
                continue
            t = p.get("input_trace", {})
            hashes.add(t.get("visual_input_sha256"))
            fidx = t.get("frame_indices") or []
            if len(fidx) != 16:
                frame_viol.append(f"{q}/{arm}:frames={len(fidx)}")
        if len(hashes) != 1:
            frame_viol.append(f"{q}:hash_drift")
    check("visual_input_consistency_16_frames", not frame_viol, violations=frame_viol[:10], n_violations=len(frame_viol))

    ret_by_q = {r.get("query_id"): r for r in retrieval}
    ret_viol = []
    for q in frozen_ids:
        r = ret_by_q.get(q)
        if not r:
            ret_viol.append(f"{q}:missing")
            continue
        top20 = r.get("original_top20") or []
        rer = r.get("reranked_d") or []
        if len(top20) < 7 or len(rer) < 7:
            ret_viol.append(f"{q}:top20={len(top20)},reranked={len(rer)}")
            continue
        a7 = by_q_arm.get((q, "ARM_A7"))
        if a7:
            reps = [c.get("clip_ids", [None])[0] for c in rer[:7]]
            if reps != a7.get("input_trace", {}).get("visual_evidence_ids"):
                ret_viol.append(f"{q}:a7_vs_reranked")
            # D consensus reranker orders by rerank_score; `score` keeps the original X-CLIP similarity.
            q_scores = [round(float(c.get("rerank_score", 0)), 6) for c in rer]
            if q_scores != sorted(q_scores, reverse=True):
                ret_viol.append(f"{q}:rerank_order")
            if [c.get("rerank_rank") for c in rer] != list(range(len(rer))):
                ret_viol.append(f"{q}:rerank_rank")
    check("retrieval_evidence_complete", len(retrieval) == 30 and not ret_viol, n_rows=len(retrieval), violations=ret_viol[:10])

    qset = set(qman.get("query_ids", []))
    check(
        "frozen_manifest_matches_run",
        qset == set(frozen_ids) and qman.get("count") == 30,
        n_manifest=len(frozen_ids),
        n_run=len(qset),
        missing=sorted(set(frozen_ids) - qset),
        extra=sorted(qset - set(frozen_ids)),
    )

    wall = receipt.get("wall_seconds")
    peak = receipt.get("peak_vram_bytes")
    gpu_secs = sum(float(p.get("wall_seconds", 0) or 0) for p in preds)
    check(
        "gpu_timing_recorded",
        wall is not None and peak is not None and gpu_secs > 0,
        run_wall_seconds=wall,
        sum_generation_seconds=round(gpu_secs, 3),
        peak_vram_bytes=peak,
        peak_vram_gib=round((peak or 0) / (1024**3), 2),
    )

    artifacts = [
        "predictions.jsonl",
        "input_traces.jsonl",
        "retrieval_candidates.jsonl",
        "query_manifest.json",
        "run_receipt.json",
        "PILOT_A1_EXECUTION_REPORT.md",
        "PILOT_A1_INTEGRITY_REPORT.md",
        "PILOT_A1_GO_NO_GO.md",
    ]
    artifact_info = {
        name: {
            "exists": (out / name).is_file(),
            "bytes": (out / name).stat().st_size if (out / name).is_file() else None,
            "sha256": _sha_file(out / name),
        }
        for name in artifacts
    }

    all_ok = all(c["ok"] for c in checks.values())
    payload = {
        "audit_schema": "uarcap_q2.pilot_a1_independent_audit.v1",
        "run_dir": str(out),
        "frozen_manifest_sha256": frozen.get("manifest_sha256"),
        "all_checks_passed": all_ok,
        "checks": checks,
        "run_receipt_status": receipt.get("status"),
        "run_receipt_go_no_go": receipt.get("go_no_go_verdict"),
        "artifacts": artifact_info,
    }
    (out / "AUDIT_PILOT_A1.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# PILOT A1 INDEPENDENT AUDIT",
        "",
        f"- **All checks passed**: `{all_ok}`",
        f"- **Run receipt status**: `{payload['run_receipt_status']}`",
        f"- **Go/No-Go**: `{payload['run_receipt_go_no_go']}`",
        f"- **Frozen sampling manifest SHA-256**: `{payload['frozen_manifest_sha256']}`",
        "",
        "| Check | Result | Details |",
        "| --- | --- | --- |",
    ]
    for name, c in checks.items():
        details = "; ".join(f"{k}={v}" for k, v in c.items() if k != "ok")
        lines.append(f"| `{name}` | {'PASS' if c['ok'] else 'FAIL'} | {details} |")
    lines += ["", "## Artifacts", "", "| File | Bytes | SHA-256 |", "| --- | --- | --- |"]
    for name, info in artifact_info.items():
        lines.append(f"| `{name}` | {info['bytes']} | `{info['sha256']}` |")
    (out / "AUDIT_PILOT_A1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"all_checks_passed": all_ok, "checks": {k: v["ok"] for k, v in checks.items()}}, indent=2))
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
