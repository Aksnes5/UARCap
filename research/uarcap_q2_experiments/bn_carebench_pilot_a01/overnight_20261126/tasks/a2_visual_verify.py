#!/usr/bin/env python3
"""Machine-assisted visual verification of flagged A2 claims using real 16-frame input.

Flags: Gemma/Llama disagreement, Gemma UNMENTIONED, or visually sensitive claims
(object/attribute/count/position/color). For each, Qwen3.8 sees the exact 16 frames used
by the original generator and answers two separate questions:
  - video_truth: does the actual video support the claim?
  - visible_in_16_frames: is the evidence observable in these 16 frames?
This is NOT human Gold and does NOT replace the pending human validation.
"""

from __future__ import annotations

import argparse
import base64
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
TASK_ID = os.environ.get("OVERNIGHT_TASK_ID", "a2_visual_verify")
PROGRESS = BASE / "progress" / f"{TASK_ID}.json"
REPO = "/home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments"
sys.path.insert(0, REPO)

VISUAL_CUES = re.compile(r"\b(red|blue|green|yellow|black|white|brown|gray|grey|pink|purple|orange|wearing|shirt|jacket|dress|hat|glasses|hair|shoes|pants|coat|two|three|four|five|one|left|right|behind|front|under|above|next to|near|holding|with|table|wall|floor|room|kitchen|street|yard)\b", re.I)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def append_jsonl(p: Path, row: dict) -> None:
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_write(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def progress(done: int, total: int, phase: str, **extra) -> None:
    atomic_write(PROGRESS, {"task": TASK_ID, "phase": phase, "model": "Qwen3.8-27B-FP8",
                            "completed": done, "total": total, "updated_utc": utcnow(), **extra})


def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    return " ".join(t for t in s.split() if t not in {"a", "an", "the"})


def strip_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def parse(raw: str) -> dict:
    obj = json.loads(strip_fence(raw))
    vt = str(obj.get("video_truth", "")).strip().lower()
    vt = {"supported": "entailed", "supported_entailed": "entailed", "contradicted": "contradicted",
          "unverifiable": "unverifiable", "unknown": "unverifiable", "uncertain": "unverifiable"}.get(vt, vt)
    if vt not in ("entailed", "contradicted", "unverifiable"):
        raise ValueError(f"bad video_truth {obj.get('video_truth')!r}")
    vis = str(obj.get("visible_in_16_frames", "")).strip().lower()
    if vis not in ("yes", "no", "partial"):
        raise ValueError(f"bad visible flag {obj.get('visible_in_16_frames')!r}")
    return {"video_truth": vt, "visible_in_16_frames": vis,
            "evidence": str(obj.get("evidence", "")).strip(),
            "ambiguity": str(obj.get("ambiguity", "")).strip()}


SYS = (
    "You are a careful visual fact-checker. You are shown 16 uniformly sampled frames from a real video "
    "(the same 16-frame input used by another model). A REFERENCE (human caption/spatial/temporal) is provided "
    "as auxiliary context but may be incomplete. Decide independently: (1) video_truth: does the actual video "
    "support the CLAIM (entailed), clearly conflict (contradicted), or is it unverifiable? (2) visible_in_16_frames: "
    "is the evidence for your judgement observable in these 16 frames (yes/no/partial)? Give concrete visual evidence. "
    'Output JSON only: {"video_truth":"entailed|contradicted|unverifiable","visible_in_16_frames":"yes|no|partial","evidence":"...","ambiguity":"..."}'
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2-dir", type=Path, default=Path("/home/bn/uarcap_q2_runtime/outputs/pilot_a2/pilot_a2_carebench_eval_v1"))
    ap.add_argument("--cross-judge-dir", type=Path, default=Path("/home/bn/uarcap_q2_runtime/outputs/pilot_a2_machine"))
    ap.add_argument("--frames-dir", type=Path, default=Path("/home/bn/uarcap_q2_runtime/outputs/pilot_a2/pilot_a2_carebench_eval_v1/blind_annotation_package/frames"))
    ap.add_argument("--out-dir", type=Path, default=Path("/home/bn/uarcap_q2_runtime/outputs/pilot_a2_machine_visual"))
    ap.add_argument("--model-path", default="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/models/Qwen3.8-27B-FP8")
    ap.add_argument("--model-id", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-claims", type=int, default=120)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from uarcap_q2.common.gpu_lock import GPULockConfig, acquire_cooperative_gpu_lock

    rows = read_jsonl(args.a2_dir / "a2_ref_fact_eval.jsonl")
    lock = json.loads((args.a2_dir / "a2_core_event_lock.json").read_text(encoding="utf-8"))
    ref_by_qid = {e["query_id"]: "\n".join(x for x in [
        ("Caption: " + e["reference_caption"]) if e.get("reference_caption") else "",
        ("Spatial description: " + e["spatial_caption"]) if e.get("spatial_caption") else "",
        ("Temporal description: " + e["temporal_caption"]) if e.get("temporal_caption") else "",
    ] if x) for e in lock["entries"]}
    cross = {}
    labels_path = args.cross_judge_dir / "A2_CROSS_JUDGE_LABELS.jsonl"
    if labels_path.is_file():
        for r in read_jsonl(labels_path):
            cross[(r["query_id"], r["norm_key"])] = r

    # unique claims + flags
    uniq: dict[tuple[str, str], dict] = {}
    for r in rows:
        if r.get("parse_status") != "OK":
            continue
        for c in r.get("claims", []):
            key = (r["query_id"], norm(c["claim"]))
            u = uniq.setdefault(key, {"query_id": r["query_id"], "claim": c["claim"], "reference": ref_by_qid[r["query_id"]],
                                      "gemma_label": c["label"], "arms": [], "reason": set()})
            u["arms"].append(r["arm"])
            other = cross.get(key)
            if other and other["llama_label"] != c["label"]:
                u["reason"].add("judge_disagreement")
            if c["label"] == "unmentioned":
                u["reason"].add("gemma_unmentioned")
            if VISUAL_CUES.search(c["claim"]):
                u["reason"].add("visual_attribute_object_position")
            if other and other.get("needs_video"):
                u["reason"].add("second_judge_needs_video")
    flagged = [u for u in uniq.values() if u["reason"]]
    flagged.sort(key=lambda u: (0 if "judge_disagreement" in u["reason"] else 1, u["query_id"], u["claim"]))
    flagged = flagged[: args.max_claims]
    progress(0, len(flagged), "prepare", n_flagged=len(flagged), n_unique=len(uniq))

    out_labels = args.out_dir / "A2_VISUAL_VERIFICATION_RESULTS.jsonl"
    raw_path = args.out_dir / "A2_VISUAL_RAW_CALLS.jsonl"
    done = set()
    if out_labels.is_file():
        for r in read_jsonl(out_labels):
            done.add((r["query_id"], r["norm_key"]))
    pending = [u for u in flagged if (u["query_id"], norm(u["claim"])) not in done]

    lock_cfg = GPULockConfig(lock_path=Path("/home/bn/videoufo_research/gpu.lock"), run_id=TASK_ID,
                             gpu_uuid="GPU-66d83fcc-2695-33a7-33aa-af30b7e3d26d",
                             require_nvidia_smi_idle=True, sidecar_dir=args.out_dir)
    with acquire_cooperative_gpu_lock(lock_cfg):
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.model_path, gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
                  max_num_seqs=64, limit_mm_per_prompt={"image": 16}, quantization="fp8", seed=42)
        sp = SamplingParams(temperature=0.0, max_tokens=256)
        for start in range(0, len(pending), args.batch):
            batch = pending[start:start + args.batch]
            convs, meta = [], []
            for u in batch:
                fdir = args.frames_dir / u["query_id"]
                fpaths = sorted(fdir.glob("frame_*.jpg"))
                if len(fpaths) != 16:
                    append_jsonl(raw_path, {"query_id": u["query_id"], "error": f"frames={len(fpaths)}"})
                    continue
                urls = []
                h = sha256()
                for p in fpaths:
                    b = p.read_bytes()
                    h.update(b)
                    urls.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")}})
                user = [
                    {"type": "text", "text": f"REFERENCE (auxiliary, may be incomplete):\n{u['reference']}\n\nCLAIM: {u['claim']}"},
                    *urls,
                    {"type": "text", "text": "Return the JSON verdict now."},
                ]
                convs.append([{"role": "system", "content": SYS}, {"role": "user", "content": user}])
                meta.append((u, h.hexdigest(), len(fpaths)))
            if not convs:
                continue
            t0 = time.time()
            outs = llm.chat(convs, sampling_params=sp, chat_template_kwargs={"enable_thinking": False})
            for (u, fsha, nframes), o in zip(meta, outs):
                raw = str(o.outputs[0].text)
                ok, parsed, err = False, None, ""
                try:
                    parsed = parse(raw); ok = True
                except Exception as exc:
                    err = str(exc)
                append_jsonl(raw_path, {"query_id": u["query_id"], "norm_key": norm(u["claim"]), "claim": u["claim"],
                                        "raw_response": raw, "parsed_ok": ok, "parse_error": err,
                                        "frames_sha256": fsha, "n_frames": nframes, "elapsed_sec": round(time.time() - t0, 3)})
                if ok:
                    rec = {"query_id": u["query_id"], "norm_key": norm(u["claim"]), "claim": u["claim"],
                           "gemma_label": u["gemma_label"], "llama_label": (cross.get((u["query_id"], norm(u["claim"]))) or {}).get("llama_label"),
                           "reason": sorted(u["reason"]), **parsed,
                           "frames_sha256": fsha, "n_frames": nframes, "judge_model_id": args.model_id,
                           "machine_assisted_only": True, "qwen_self_evaluation_caveat": True}
                    append_jsonl(out_labels, rec)
                    done.add((u["query_id"], norm(u["claim"])))
            progress(len(done), len(flagged), "visual_verify", peak_completed=len(done))

    results = read_jsonl(out_labels) if out_labels.is_file() else []
    by_reason = defaultdict(lambda: Counter())
    for r in results:
        for reason in r.get("reason", []):
            by_reason[reason][r["video_truth"]] += 1
    summary = {
        "schema": "uarcap_q2.a2_visual_verification.v1", "status": "MACHINE_ASSISTED_UNCALIBRATED",
        "generated_utc": utcnow(), "model_id": args.model_id, "model_path": args.model_path,
        "n_flagged": len(flagged), "n_verified": len(results), "n_parse_failed": len(flagged) - len(results),
        "video_truth_counts": dict(Counter(r["video_truth"] for r in results)),
        "visible_16f_counts": dict(Counter(r["visible_in_16_frames"] for r in results)),
        "by_reason": {k: dict(v) for k, v in by_reason.items()},
        "n_still_uncertain": sum(1 for r in results if r["video_truth"] == "unverifiable" or r["visible_in_16_frames"] != "yes"),
        "warning": "Qwen is machine assistance only; its outputs are never human Gold, and it is not used as an independent confirmation when Qwen also generates text in A3.",
    }
    atomic_write(args.out_dir / "A2_VISUAL_VERIFICATION_SUMMARY.json", summary)
    progress(len(results), len(flagged), "completed")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
