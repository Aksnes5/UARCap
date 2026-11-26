#!/usr/bin/env python3
"""Pilot A2 machine-assisted cross-judge over the 119 unique atomic claims.

Reuses the existing Gemma labels, asks one independent local text model to re-label
each unique claim against the full human reference (caption + spatial + temporal),
and reports machine-machine agreement. This is MACHINE-ASSISTED evidence, never human Gold.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import time

BASE = Path(os.environ.get("OVERNIGHT_BASE", "/home/bn/uarcap_q2_runtime/overnight/20261126"))
TASK_ID = os.environ.get("OVERNIGHT_TASK_ID", "a2_cross_judge")
PROGRESS = BASE / "progress" / f"{TASK_ID}.json"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def atomic_write(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def append_jsonl(p: Path, row: dict) -> None:
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def progress(done: int, total: int, phase: str, model: str, **extra) -> None:
    atomic_write(PROGRESS, {"task": TASK_ID, "phase": phase, "model": model,
                            "completed": done, "total": total, "updated_utc": utcnow(), **extra})


def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    toks = [t for t in s.split() if t not in {"a", "an", "the"}]
    return " ".join(toks)


def strip_fence(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def canonical_label(raw: str) -> str | None:
    m = {
        "entailed": "entailed", "entail": "entailed", "entailment": "entailed", "supported": "entailed",
        "contradicted": "contradicted", "contradiction": "contradicted", "contradict": "contradicted",
        "conflict": "contradicted", "conflicting": "contradicted",
        "unmentioned": "unmentioned", "neutral": "unmentioned", "unknown": "unmentioned",
        "not_mentioned": "unmentioned", "not mentioned": "unmentioned",
    }
    return m.get((raw or "").strip().lower().replace("-", "_"))


def parse_single(raw: str) -> dict:
    if not raw or not raw.strip():
        raise ValueError("empty response")
    obj = json.loads(strip_fence(raw))
    label = canonical_label(str(obj.get("label", "")))
    if label is None:
        raise ValueError(f"illegal label {obj.get('label')!r}")
    ev = obj.get("evidence")
    evidence = "" if ev is None or str(ev).strip().lower() in ("", "none", "null") else str(ev).strip()
    if label in ("entailed", "contradicted") and not evidence:
        raise ValueError("definite label without evidence")
    return {
        "label": label,
        "evidence": evidence,
        "ambiguous": bool(obj.get("ambiguous", False)),
        "needs_video": bool(obj.get("needs_video", False)),
        "atomic_split_error": bool(obj.get("atomic_split_error", False)),
    }


SYS = (
    "You are a strict fact-checking judge. Given a REFERENCE (human caption, spatial description, temporal description) "
    "and one CLAIM, classify the claim: entailed (the reference supports it), contradicted (the reference clearly conflicts), "
    "unmentioned (the reference is insufficient). Use ONLY the reference, no outside knowledge. For entailed/contradicted, "
    "evidence must be a non-empty reference span or reason. Also report whether the claim is semantically ambiguous, whether "
    "checking the actual video is needed, and whether the atomic split itself looks wrong. Output strict JSON only: "
    '{"label":"entailed|contradicted|unmentioned","evidence":"...","ambiguous":false,"needs_video":false,"atomic_split_error":false}'
)


def build_reference(lock_entry: dict) -> str:
    parts = []
    if lock_entry.get("reference_caption"):
        parts.append("Caption: " + lock_entry["reference_caption"])
    if lock_entry.get("spatial_caption"):
        parts.append("Spatial description: " + lock_entry["spatial_caption"])
    if lock_entry.get("temporal_caption"):
        parts.append("Temporal description: " + lock_entry["temporal_caption"])
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2-dir", type=Path, default=Path("/home/bn/uarcap_q2_runtime/outputs/pilot_a2/pilot_a2_carebench_eval_v1"))
    ap.add_argument("--out-dir", type=Path, default=Path("/home/bn/uarcap_q2_runtime/outputs/pilot_a2_machine"))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--model-family", default="llama")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--chunk", type=int, default=12)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="debug: evaluate only the first N unique claims")
    ap.add_argument("--seed-labels",type=Path)
    ap.add_argument("--live-labels",type=Path)
    ap.add_argument("--seed-sha256")
    args = ap.parse_args()

    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    from uarcap_q2.common.gpu_lock import GPULockConfig, acquire_cooperative_gpu_lock
    from uarcap_q2.evaluators.backends import JudgeBackendConfig, LocalVLLMJudgeBackend

    rows = read_jsonl(args.a2_dir / "a2_ref_fact_eval.jsonl")
    lock = json.loads((args.a2_dir / "a2_core_event_lock.json").read_text(encoding="utf-8"))
    ref_by_qid = {e["query_id"]: build_reference(e) for e in lock["entries"]}

    uniq: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    row_map: list[dict] = []
    for r in rows:
        if r.get("parse_status") != "OK":
            continue
        ref = ref_by_qid[r["query_id"]]
        for c in r.get("claims", []):
            key = (r["query_id"], norm(c["claim"]))
            if key not in uniq:
                uniq[key] = {"query_id": r["query_id"], "claim": c["claim"], "reference": ref,
                             "gemma_label": c["label"], "gemma_evidence": c.get("evidence", ""),
                             "arms": [], "claim_ids": []}
                order.append(key)
            uniq[key]["arms"].append(r["arm"])
            uniq[key]["claim_ids"].append(c.get("claim_id"))
        row_map.append(r)

    if args.limit and args.limit > 0:
        order = order[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.out_dir / "A2_CROSS_JUDGE_LABELS.jsonl"
    raw_path = args.out_dir / "A2_CROSS_JUDGE_RAW_CALLS.jsonl"
    done = {}
    if labels_path.is_file():
        for x in read_jsonl(labels_path):
            done[(x["query_id"], x["norm_key"])] = x
    from uarcap_q2.common.io_utils import sha256_file, primitive_hash, read_json
    PROGRESS.parent.mkdir(parents=True,exist_ok=True)
    if not order:
        raise ValueError("Missing cross-judge facts")
    binding={"eval_sha256":sha256_file(args.a2_dir/"a2_ref_fact_eval.jsonl"),
             "core_sha256":sha256_file(args.a2_dir/"a2_core_event_lock.json"),
             "judge":args.model_id,"judge_path":args.model_path,
             "max_model_len":args.max_model_len,"max_new_tokens":args.max_new_tokens}
    bound=args.out_dir/"CROSS_JUDGE_INPUT_BINDING.json"
    if bound.exists() and read_json(bound)!=binding:
        raise ValueError("Cross-judge configuration drift")
    if done and not bound.exists():
        raise ValueError("No binding for existing cross-judge rows")
    atomic_write(bound,binding)
    if args.seed_labels:
        if sha256_file(args.seed_labels)!=args.seed_sha256:
            raise ValueError("Legacy seed cross-judge hash mismatch")
        for row in read_jsonl(args.seed_labels):
            key=(row["query_id"],row["norm_key"])
            if key not in uniq or norm(row["claim"])!=key[1] or row.get("judge_model_id")!=args.model_id:
                raise ValueError("Legacy seed does not match scored claim and judge")
            if key not in done and row.get("llama_label") in ("entailed","contradicted","unmentioned"):
                append_jsonl(labels_path,row)
                done[key]=row
    if args.live_labels and args.live_labels.is_file():
        for row in read_jsonl(args.live_labels):
            key=(row["query_id"],row["norm_key"])
            if key in uniq and key not in done and norm(row["claim"])==key[1] and (
                row.get("judge_model_id")==args.model_id and row.get("llama_label") in ("entailed","contradicted","unmentioned")
                and row.get("gemma_label")==uniq[key]["gemma_label"]):
                append_jsonl(labels_path,{**row,"reused_live_label_sha256":sha256_file(args.live_labels)})
                done[key]=row
    progress(len(done), len(order), "prepare", args.model_id)

    backend = LocalVLLMJudgeBackend(
        JudgeBackendConfig(
            backend_type="vllm_offline", model_id=args.model_id, model_family=args.model_family,
            local_model_path=str(args.model_path), temperature=0.0,
            max_new_tokens=args.max_new_tokens, max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem, language_model_only=False,
        )
    )
    lock_cfg = GPULockConfig(
        lock_path=Path("/home/bn/videoufo_research/gpu.lock"), run_id=TASK_ID,
        gpu_uuid="GPU-66d83fcc-2695-33a7-33aa-af30b7e3d26d", require_nvidia_smi_idle=True,
        sidecar_dir=args.out_dir,
    )

    def prompts_for(keys: list[tuple[str, str]], attempt: int) -> list[tuple[str, str]]:
        out = []
        for key in keys:
            u = uniq[key]
            up = f"[TASK:CLAIM_CHECK]\nREFERENCE:\n{u['reference']}\nCLAIM: {u['claim']}"
            if attempt:
                up += "\nReturn ONLY the JSON object; labels must be exactly entailed/contradicted/unmentioned; give non-empty evidence for entailed/contradicted."
            out.append((SYS, up))
        return out

    pending = [k for k in order if k not in done]
    with (acquire_cooperative_gpu_lock(lock_cfg) if pending else nullcontext()):
        for attempt in range(args.max_retries + 1):
            if not pending:
                break
            for start in range(0, len(pending), args.chunk):
                batch = pending[start:start + args.chunk]
                t0 = time.time()
                responses = backend.complete_batch(prompts_for(batch, attempt))
                for key, raw in zip(batch, responses):
                    ok = False
                    err = ""
                    try:
                        parsed = parse_single(raw)
                        ok = True
                    except Exception as exc:
                        parsed = None
                        err = str(exc)
                    append_jsonl(raw_path, {"task": TASK_ID, "query_id": key[0], "norm_key": key[1],
                                            "attempt": attempt, "raw_response": raw, "parsed_ok": ok,
                                            "parse_error": err, "elapsed_sec": round(time.time() - t0, 3)})
                    if ok:
                        rec = {"query_id": key[0], "norm_key": key[1], "claim": uniq[key]["claim"],
                               "gemma_label": uniq[key]["gemma_label"], "llama_label": parsed["label"],
                               "llama_evidence": parsed["evidence"], "ambiguous": parsed["ambiguous"],
                               "needs_video": parsed["needs_video"],
                               "atomic_split_error": parsed["atomic_split_error"],
                               "judge_model_id": args.model_id, "arms": sorted(set(uniq[key]["arms"]))}
                        append_jsonl(labels_path, rec)
                        done[key] = rec
                progress(len(done), len(order), f"judge_attempt_{attempt}", args.model_id)
            pending = [k for k in order if k not in done]

    # ---- machine-machine agreement ----
    keys = [k for k in order if k in done]
    gemma = [uniq[k]["gemma_label"] for k in keys]
    llama = [done[k]["llama_label"] for k in keys]
    n = len(keys)
    agree = sum(1 for a, b in zip(gemma, llama) if a == b)
    po = agree / n if n else 0.0
    ca, cb = Counter(gemma), Counter(llama)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in ("entailed", "contradicted", "unmentioned")) if n else 0.0
    kappa = (po - pe) / (1 - pe) if n and pe < 1 else None

    per_arm = defaultdict(lambda: {"n": 0, "agree": 0, "gemma": Counter(), "llama": Counter()})
    for r in row_map:
        arm = r["arm"]
        for c in r.get("claims", []):
            k = (r["query_id"], norm(c["claim"]))
            if k not in done:
                continue
            per_arm[arm]["n"] += 1
            per_arm[arm]["gemma"][c["label"]] += 1
            per_arm[arm]["llama"][done[k]["llama_label"]] += 1
            if c["label"] == done[k]["llama_label"]:
                per_arm[arm]["agree"] += 1
    per_arm_out = {a: {"n": v["n"], "agreement": round(v["agree"] / v["n"], 4) if v["n"] else None,
                       "gemma_counts": dict(v["gemma"]), "llama_counts": dict(v["llama"])}
                   for a, v in per_arm.items()}

    disagreements = [
        {"query_id": k[0], "claim": uniq[k]["claim"], "gemma": uniq[k]["gemma_label"],
         "llama": done[k]["llama_label"], "llama_evidence": done[k]["llama_evidence"],
         "needs_video": done[k]["needs_video"], "ambiguous": done[k]["ambiguous"],
         "atomic_split_error": done[k]["atomic_split_error"]}
        for k in keys if uniq[k]["gemma_label"] != done[k]["llama_label"]
    ]
    report = {
        "schema": "uarcap_q2.a2_cross_judge.v1",
        "status": "MACHINE_ASSISTED_UNCALIBRATED",
        "generated_utc": utcnow(),
        "judge_model_id": args.model_id,
        "judge_model_path": str(args.model_path),
        "n_unique_claims": len(order),
        "n_evaluated": n,
        "n_parse_failed": len(order) - n,
        "observed_agreement_unique": round(po, 4),
        "cohen_kappa_unique": round(kappa, 4) if kappa is not None else None,
        "label_counts_gemma": dict(ca),
        "label_counts_second_judge": dict(cb),
        "per_arm_claims": per_arm_out,
        "n_disagreements": len(disagreements),
        "n_need_video": sum(1 for k in keys if done[k]["needs_video"]),
        "n_ambiguous": sum(1 for k in keys if done[k]["ambiguous"]),
        "n_atomic_split_suspect": sum(1 for k in keys if done[k]["atomic_split_error"]),
        "disagreements": disagreements,
        "warning": "Machine-assisted cross-judge only; NOT human Gold and NOT a human agreement statistic.",
    }
    atomic_write(args.out_dir / "A2_CROSS_JUDGE_AGREEMENT.json", report)
    progress(n, len(order), "completed", args.model_id)
    print(json.dumps({k: report[k] for k in ("n_unique_claims", "n_evaluated", "observed_agreement_unique",
                                             "cohen_kappa_unique", "n_disagreements", "n_need_video")}, indent=2))
    return 0 if n == len(order) else 2


if __name__ == "__main__":
    raise SystemExit(main())
