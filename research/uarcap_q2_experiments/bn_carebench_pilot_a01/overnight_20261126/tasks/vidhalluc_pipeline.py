#!/usr/bin/env python3
"""VidHalluc-ACH exploratory E3 pipeline (machine-assisted; never human Gold).

Stages: extract -> prepare -> screen_text -> screen_visual -> approve ->
generate_videollama / generate_qwen -> evaluate -> report.
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
import random
import re
import subprocess
import sys
import time

BASE = Path(os.environ.get("OVERNIGHT_BASE", "/home/bn/uarcap_q2_runtime/overnight/20261126"))
TASK_ID = os.environ.get("OVERNIGHT_TASK_ID", "vidhalluc")
PROGRESS = BASE / "progress" / f"{TASK_ID}.json"
REPO = "/home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments"
sys.path.insert(0, REPO)
Q2 = Path("/home/bn/uarcap_q2_runtime")
VH = Q2 / "data" / "vidhalluc"
ZIP = VH / "data" / "ACH_videos.zip"
META = VH / "ach_mcq.json"
EXTRACT = VH / "videos"
WORK = Q2 / "outputs" / "pilot_e3_exploratory"
ZIP_SHA = "8f163f29a72f5bc8d15720847b1c0b125c8a593d3bdaedfabbeb4f83e3c25af5"
ZIP_BYTES = 16754119398
VLLM = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/.venv_rag_vllm/bin/python"
GEMMA = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/models_campaign/M04_gemma3_27b"
QWEN = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/models/Qwen3.8-27B-FP8"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []


def append_jsonl(p: Path, row: dict) -> None:
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_write(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def progress(phase: str, done: int, total: int, **extra) -> None:
    atomic_write(PROGRESS, {"task": TASK_ID, "phase": phase, "completed": done, "total": total,
                            "updated_utc": utcnow(), **extra})


def sha256_file(p: Path) -> str:
    h = sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def safe_extract(zip_path: Path, dest: Path) -> int:
    import zipfile
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"unsafe zip member {name}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"symlink in zip {name}")
            target = (dest / name).resolve()
            if dest.resolve() not in target.parents and target != dest.resolve():
                raise ValueError(f"zip-slip {name}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                while True:
                    b = src.read(8 * 1024 * 1024)
                    if not b: break
                    dst.write(b)
            n += 1
    return n


def find_video_root(base: Path, clip_key_probe: str | None = None) -> Path:
    if clip_key_probe:
        for p in base.rglob(f"{clip_key_probe}.mp4"):
            return p.parent
    best = base
    best_n = 0
    for d in base.rglob("*"):
        if d.is_dir():
            n = sum(1 for _ in d.glob("*.mp4"))
            if n > best_n:
                best, best_n = d, n
    return best


def stage_extract() -> int:
    if not ZIP.is_file():
        print("HARD_BLOCK: video zip missing", flush=True); return 3
    if ZIP.stat().st_size != ZIP_BYTES:
        print("HARD_BLOCK: zip size mismatch", flush=True); return 3
    h = sha256_file(ZIP)
    if h != ZIP_SHA:
        print("HARD_BLOCK: zip sha mismatch", flush=True); return 3
    EXTRACT.mkdir(parents=True, exist_ok=True)
    n = safe_extract(ZIP, EXTRACT)
    receipt = {"status": "EXTRACTED", "zip": str(ZIP), "sha256": h, "files": n, "utc": utcnow()}
    atomic_write(WORK / "E3_EXTRACT_RECEIPT.json", receipt)
    print(json.dumps(receipt), flush=True)
    return 0


def stage_prepare() -> int:
    from uarcap_q2.datasets.vidhalluc import (load_vidhalluc_ach_mcq, deduplicate_and_filter_vidhalluc,
                                              sample_vidhalluc_pilot_candidates, write_vidhalluc_pilot_exclusion_manifest)
    import yaml
    cfg = yaml.safe_load(Path(f"{REPO}/configs/bn_paths.local.yaml").read_text(encoding="utf-8"))
    clips = Path(cfg["historical_frozen_assets"]["clips_full_tsv_path"])
    vufo = {ln.split("\t")[0].strip().rsplit(".", 1)[0] for ln in clips.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("clip_id")}
    # probe one clip key for the actual root
    raw = json.loads(META.read_text(encoding="utf-8"))
    probe = next(iter(next(iter(raw.values())).keys()))
    root = find_video_root(EXTRACT, probe)
    recs = load_vidhalluc_ach_mcq(META, root)
    kept, audit = deduplicate_and_filter_vidhalluc(recs, vufo)
    pilot, formal = sample_vidhalluc_pilot_candidates(kept, candidate_count=30, seed=20261125)
    WORK.mkdir(parents=True, exist_ok=True)
    write_vidhalluc_pilot_exclusion_manifest(pilot, WORK / "E3_PILOT_EXCLUSION.json", seed=20261125)
    rows = []
    for r in pilot:
        md = r.metadata or {}
        rows.append({"query_id": r.query_id, "clip_key": md.get("clip_key"), "source_id": r.source_id,
                     "video_path": r.video_path, "question": md.get("question"),
                     "true_action": r.target_true_fact, "conflict_action": r.candidate_conflict_fact,
                     "all_distractors": md.get("all_distractors", [])})
    atomic_write(WORK / "E3_CANDIDATES.json", {"schema": "uarcap_q2.e3_candidates.v1", "status": "MACHINE_ASSISTED_UNVALIDATED",
                                               "utc": utcnow(), "n": len(rows), "source_audit": audit, "candidates": rows})
    lock = {"schema": "uarcap_q2.e3_core_lock.v1", "created_before_any_model_call": True, "utc": utcnow(),
            "entries": [{"query_id": x["query_id"], "core_event_candidate": x["true_action"], "reference_caption": x["true_action"],
                         "clip_key": x["clip_key"], "source_id": x["source_id"]} for x in rows]}
    atomic_write(WORK / "E3_CORE_LOCK.json", lock)
    progress("prepare", len(rows), len(rows))
    print(json.dumps({"n_records": len(recs), "n_unique_sources": len(kept), "n_candidates": len(rows), "video_root": str(root), "audit": audit}, indent=2)[:2000])
    return 0


def _gemma_backend(gpu_mem=0.85):
    from uarcap_q2.evaluators.backends import JudgeBackendConfig, LocalVLLMJudgeBackend
    return LocalVLLMJudgeBackend(JudgeBackendConfig(backend_type="vllm_offline", model_id="RedHatAI/gemma-3-27b-it-FP8-dynamic",
        model_family="gemma", local_model_path=GEMMA, temperature=0.0, max_new_tokens=256, max_model_len=2048,
        gpu_memory_utilization=gpu_mem, language_model_only=True))


def _llama_backend(gpu_mem=0.95):
    from uarcap_q2.evaluators.backends import JudgeBackendConfig, LocalVLLMJudgeBackend
    return LocalVLLMJudgeBackend(JudgeBackendConfig(backend_type="vllm_offline", model_id="meta-llama/Llama-3.3-70B-Instruct-W4A16",
        model_family="llama", local_model_path="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Jnh_rag/models_campaign/M08_llama33_70b",
        temperature=0.0, max_new_tokens=256, max_model_len=2048, gpu_memory_utilization=gpu_mem))


def _parse_json(raw: str) -> dict:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s); s = re.sub(r"\s*```$", "", s)
    return json.loads(s)


def stage_screen_text() -> int:
    cands = json.loads((WORK / "E3_CANDIDATES.json").read_text())["candidates"]
    out = WORK / "E3_TEXT_SCREEN.jsonl"
    done = {r["query_id"] for r in read_jsonl(out)}
    pending = [c for c in cands if c["query_id"] not in done]
    sys_p = ("Check whether a correct and a conflict action pair is suitable for a controlled intervention. "
             "They must be mutually exclusive, both natural English, similar length, and the conflict must be a clear "
             'different action (not a synonym/substep). Output JSON {"mutually_exclusive":true/false,"natural":true/false,'
             '"length_ok":true/false,"conflict_plausibly_absent":true/false,"ambiguity":"..."}')
    backend = _gemma_backend()
    for i in range(0, len(pending), 12):
        batch = pending[i:i+12]
        prompts = [(sys_p, f"TRUE ACTION: {c['true_action']}\nCONFLICT ACTION: {c['conflict_action']}\nQUESTION: {c['question']}") for c in batch]
        resps = backend.complete_batch(prompts)
        for c, raw in zip(batch, resps):
            try:
                p = _parse_json(raw); ok = True
            except Exception as e:
                p, ok = {"parse_error": str(e)}, False
            append_jsonl(out, {"query_id": c["query_id"], "true_action": c["true_action"], "conflict_action": c["conflict_action"],
                               "screen": p, "parse_ok": ok, "raw": raw[:4000], "model": "gemma-3-27b-fp8"})
    progress(len(done)+len(pending), len(cands), "screen_text")
    return 0


def _decode_frames(video: Path, out_dir: Path) -> list[str]:
    from uarcap_q2.datasets.video_decoder import decode_and_sample_video
    frames, rep = decode_and_sample_video(video, num_frames=16)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, im in enumerate(frames):
        p = out_dir / f"frame_{i:02d}.jpg"; im.save(p, "JPEG", quality=92); paths.append(str(p))
    return paths


def stage_screen_visual() -> int:
    cands = json.loads((WORK / "E3_CANDIDATES.json").read_text())["candidates"]
    out = WORK / "E3_VISUAL_SCREEN.jsonl"
    done = {r["query_id"] for r in read_jsonl(out)}
    pending = [c for c in cands if c["query_id"] not in done]
    from vllm import LLM, SamplingParams
    llm = LLM(model=QWEN, gpu_memory_utilization=0.88, max_model_len=16384, max_num_seqs=64,
              limit_mm_per_prompt={"image": 16}, quantization="fp8", seed=42)
    sp = SamplingParams(temperature=0.0, max_tokens=256)
    sys_p = ("You see 16 real frames from a video. Two candidate actions are given: TRUE (from the dataset answer) and "
             "CONFLICT (a distractor). Decide independently: is TRUE visible? is CONFLICT absent? Are the frames sufficient? "
             'Output JSON {"true_present":"yes|no|partial","conflict_present":"yes|no|partial","frames_sufficient":"yes|no|partial","evidence":"..."}')
    for i in range(0, len(pending), 2):
        batch = pending[i:i+2]; convs = []; meta = []
        for c in batch:
            fpaths = sorted((WORK / "frames" / c["query_id"]).glob("frame_*.jpg"))
            if len(fpaths) != 16:
                try:
                    fpaths = [Path(p) for p in _decode_frames(Path(c["video_path"]), WORK / "frames" / c["query_id"])]
                except Exception as e:
                    append_jsonl(out, {"query_id": c["query_id"], "error": f"decode:{e}"}); continue
            content = [{"type": "text", "text": f"TRUE ACTION: {c['true_action']}\nCONFLICT ACTION: {c['conflict_action']}"}]
            for p in fpaths:
                content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()}})
            content.append({"type": "text", "text": "Return the JSON verdict."})
            convs.append([{"role": "system", "content": sys_p}, {"role": "user", "content": content}])
            meta.append(c)
        if not convs: continue
        outs = llm.chat(convs, sampling_params=sp, chat_template_kwargs={"enable_thinking": False})
        for c, o in zip(meta, outs):
            raw = str(o.outputs[0].text)
            try: p = _parse_json(raw); ok = True
            except Exception as e: p, ok = {"parse_error": str(e)}, False
            append_jsonl(out, {"query_id": c["query_id"], "screen": p, "parse_ok": ok, "raw": raw[:4000], "model": "qwen3.8-27b-fp8"})
    progress(len(done)+len(pending), len(cands), "screen_visual")
    return 0


def stage_approve() -> int:
    cands = {c["query_id"]: c for c in json.loads((WORK/"E3_CANDIDATES.json").read_text())["candidates"]}
    text = {r["query_id"]: r for r in read_jsonl(WORK/"E3_TEXT_SCREEN.jsonl")}
    vis = {r["query_id"]: r for r in read_jsonl(WORK/"E3_VISUAL_SCREEN.jsonl")}
    approved, review = [], []
    for qid in sorted(cands):
        c = cands[qid]; t = (text.get(qid) or {}).get("screen") or {}; v = (vis.get(qid) or {}).get("screen") or {}
        yn = lambda x: str(x).strip().lower()
        ok = (t.get("mutually_exclusive") and t.get("natural") and t.get("length_ok") and t.get("conflict_plausibly_absent")
              and yn(v.get("true_present")) == "yes" and yn(v.get("conflict_present")) == "no"
              and yn(v.get("frames_sufficient")) == "yes")
        rec = {**c, "text_screen": t, "visual_screen": v, "machine_approved": bool(ok)}
        if ok and len(approved) < 20:
            neu = random.Random(int(sha256(qid.encode()).hexdigest()[:8], 16)).choice(
                ["reading a book in a library", "typing on a laptop at a desk", "folding laundry indoors", "washing a car outdoors"])
            rec["neutral_text"] = neu
            approved.append(rec)
        else:
            review.append(rec)
    atomic_write(WORK/"E3_APPROVED_CANDIDATES.json", {"schema": "uarcap_q2.e3_approved.v1", "status": "MACHINE_ASSISTED_UNVALIDATED",
                 "utc": utcnow(), "n_approved": len(approved), "n_review": len(review), "approved": approved})
    atomic_write(WORK/"E3_HUMAN_REVIEW_CANDIDATES.json", {"utc": utcnow(), "items": review})
    print(json.dumps({"n_approved": len(approved), "n_review": len(review)}), flush=True)
    return 0 if approved else 2


def _load_approved():
    return json.loads((WORK/"E3_APPROVED_CANDIDATES.json").read_text())["approved"]


def stage_generate_videollama() -> int:
    from PIL import Image
    from uarcap_q2.common.gpu_lock import GPULockConfig, acquire_cooperative_gpu_lock
    from uarcap_q2.common.schemas import PilotStage
    from uarcap_q2.datasets.carebench import load_carebench_records
    from uarcap_q2.generators.base import BatchItem
    from uarcap_q2.generators.modern_vlm_adapter import hash_pil_frames
    from uarcap_q2.generators.videollama_adapter import FrozenVideoLLaMAAdapter, FrozenVideoLLaMAConfig
    from uarcap_q2.retrieval.arm_builder import build_e3_intervention_arms, verify_e3_visual_lock
    ap = _load_approved(); out = WORK/"videollama"; out.mkdir(parents=True, exist_ok=True)
    pred = out/"predictions.jsonl"; trace = out/"input_traces.jsonl"; fail = out/"failures.jsonl"
    done = {(r["query_id"], r["arm"]) for r in read_jsonl(pred)}
    adapter = FrozenVideoLLaMAAdapter(FrozenVideoLLaMAConfig(
        python_executable="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python",
        evcap_work_root="/home/bn/uar_v06_runtime/evcap_work",
        checkpoint_path="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/lj/EVCapCode/outputs/train_evcap_msrvtt_quality_text/train_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review.pt",
        llm_model_dir="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/models/vicuna-7b-v1.3",
        video_feature_dir="/home/bn/videoufo_research/features/videollama_faithful_v2",
        torch_home="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/torch_cache",
        hf_hub_cache="/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/hf_cache",
        expected_checkpoint_sha256="a8b5af79d55a2c7e684771475d645be6373208bdaae5a4211bee556581eb8002"))
    lock = GPULockConfig(lock_path=Path("/home/bn/videoufo_research/gpu.lock"), run_id=TASK_ID,
                         gpu_uuid="GPU-66d83fcc-2695-33a7-33aa-af30b7e3d26d", require_nvidia_smi_idle=True, sidecar_dir=out)
    with acquire_cooperative_gpu_lock(lock):
        for arm in ("E3_BASE", "E3_SUP", "E3_CON", "E3_NEU"):
            items = []
            for c in ap:
                if (c["query_id"], arm) in done: continue
                frames = [Image.open(p).convert("RGB") for p in sorted((WORK/"frames"/c["query_id"]).glob("frame_*.jpg"))]
                if len(frames) != 16: continue
                from uarcap_q2.common.schemas import QueryRecord
                q = QueryRecord(query_id=c["query_id"], dataset_id="VidHalluc-ACH", source_id=c["source_id"], source_id_status="VERIFIED_EXACT",
                                video_path=c["video_path"], video_sha256=None, duration=None, domain="ACH",
                                reference_description=c["true_action"], reference_events=[c["true_action"]], core_events=[c["true_action"]],
                                target_true_fact=c["true_action"], candidate_conflict_fact=c["conflict_action"], split="exploratory_e3",
                                eligibility_status="CANDIDATE_PENDING_VERIFICATION", metadata={})
                packets = build_e3_intervention_arms(c["query_id"], c["true_action"], c["conflict_action"], c["neutral_text"], base_visual_mode="BASE_A0")
                verify_e3_visual_lock(packets)
                items.append(BatchItem(query=q, packet=packets[arm], visual_input_sha256=hash_pil_frames(frames), frame_indices=list(range(16)), frame_images=frames))
            if not items: continue
            try:
                outs = adapter.generate_batch(items, PilotStage.PILOT_A3, work_dir=out/"work", run_id="e3_exploratory")
                for o in outs:
                    row = o.to_dict(); row["execution_mode"] = "EXPLORATORY_E3_UNCALIBRATED"
                    append_jsonl(pred, row); append_jsonl(trace, row["input_trace"])
            except Exception as e:
                for it in items: append_jsonl(fail, {"query_id": it.query.query_id, "arm": arm, "error": str(e)[:600]})
    progress(len(done), len(ap)*4, "videollama")
    return 0


def stage_generate_qwen() -> int:
    import yaml
    from uarcap_q2.common.gpu_lock import GPULockConfig, acquire_cooperative_gpu_lock
    from uarcap_q2.common.schemas import PilotStage
    from uarcap_q2.generators.base import BatchItem
    from uarcap_q2.generators.modern_vlm_adapter import hash_pil_frames
    from uarcap_q2.generators.qwen3_vllm_adapter import Qwen3VLLMConfig, Qwen3VLLMMultimodalAdapter
    from uarcap_q2.retrieval.arm_builder import build_e3_intervention_arms
    from PIL import Image
    ap = _load_approved(); out = WORK/"qwen"; out.mkdir(parents=True, exist_ok=True)
    pred = out/"predictions.jsonl"; done = {(r["query_id"], r["arm"]) for r in read_jsonl(pred)}
    cfg = Qwen3VLLMConfig(model_id="Qwen/Qwen3.8-27B-FP8", local_model_path=QWEN, backend_mode="vllm_offline",
                         pinned_revision="LOCAL_BN_QWEN38_FP8_66SHARDS", num_frames=16, multimodal_input_key="image",
                         prompt_regime="short_aligned", max_new_tokens=48, temperature=0.0, quantization="fp8",
                         gpu_memory_utilization=0.88, max_model_len=8192, max_num_seqs=64)
    adapter = Qwen3VLLMMultimodalAdapter(cfg)
    lock = GPULockConfig(lock_path=Path("/home/bn/videoufo_research/gpu.lock"), run_id=TASK_ID,
                         gpu_uuid="GPU-66d83fcc-2695-33a7-33aa-af30b7e3d26d", require_nvidia_smi_idle=True, sidecar_dir=out)
    with acquire_cooperative_gpu_lock(lock):
        for arm in ("E3_BASE", "E3_SUP", "E3_CON", "E3_NEU"):
            items = []
            for c in ap:
                if (c["query_id"], arm) in done: continue
                frames = [Image.open(p).convert("RGB") for p in sorted((WORK/"frames"/c["query_id"]).glob("frame_*.jpg"))]
                if len(frames) != 16: continue
                from uarcap_q2.common.schemas import QueryRecord
                q = QueryRecord(query_id=c["query_id"], dataset_id="VidHalluc-ACH", source_id=c["source_id"], source_id_status="VERIFIED_EXACT",
                                video_path=c["video_path"], video_sha256=None, duration=None, domain="ACH",
                                reference_description=c["true_action"], reference_events=[c["true_action"]], core_events=[c["true_action"]],
                                target_true_fact=c["true_action"], candidate_conflict_fact=c["conflict_action"], split="exploratory_e3",
                                eligibility_status="CANDIDATE_PENDING_VERIFICATION", metadata={})
                packets = build_e3_intervention_arms(c["query_id"], c["true_action"], c["conflict_action"], c["neutral_text"], base_visual_mode="BASE_A0")
                items.append(BatchItem(query=q, packet=packets[arm], visual_input_sha256=hash_pil_frames(frames), frame_indices=list(range(16)), frame_images=frames))
            for i in range(0, len(items), 5):
                batch = items[i:i+5]
                try:
                    outs = adapter.generate_batch(batch, PilotStage.PILOT_A3, work_dir=out/"work", run_id="e3_exploratory_qwen")
                    for o in outs:
                        row = o.to_dict(); row["execution_mode"] = "EXPLORATORY_E3_UNCALIBRATED"
                        append_jsonl(pred, row)
                except Exception as e:
                    append_jsonl(out/"failures.jsonl", {"arm": arm, "error": str(e)[:600], "n": len(batch)})
    progress(len(done), len(ap)*4, "qwen")
    return 0


def _eval_prompt(caption: str, target: str) -> tuple[str, str]:
    sp = ("Decide whether a caption expresses a specific target fact. Do not count merely related objects or a generic "
          "action. Distinguish explicit mention, logical entailment, negation and hedging. Output JSON only: "
          '{"expressed":0|1,"negated":true/false,"hedged":true/false,"evidence":"..."}')
    return sp, f"CAPTION: {caption}\nTARGET_FACT: {target}"


def stage_evaluate() -> int:
    from uarcap_q2.common.gpu_lock import GPULockConfig, acquire_cooperative_gpu_lock
    ap = {c["query_id"]: c for c in _load_approved()}
    preds = []
    for model_dir in (WORK/"videollama", WORK/"qwen"):
        for r in read_jsonl(model_dir/"predictions.jsonl"):
            if r["query_id"] in ap: preds.append({**r, "generator": model_dir.name})
    out = WORK/"evaluation"; out.mkdir(parents=True, exist_ok=True)
    rows_path = out/"E3_EVAL.jsonl"; done = {(r["query_id"], r["arm"], r["generator"]) for r in read_jsonl(rows_path)}
    todos = [p for p in preds if (p["query_id"], p["arm"], p["generator"]) not in done]
    lock = GPULockConfig(lock_path=Path("/home/bn/videoufo_research/gpu.lock"), run_id=TASK_ID,
                         gpu_uuid="GPU-66d83fcc-2695-33a7-33aa-af30b7e3d26d", require_nvidia_smi_idle=True, sidecar_dir=out)
    def run_judge(backend, name):
        for i in range(0, len(todos), 16):
            batch = todos[i:i+16]; prompts = []
            for p in batch:
                c = ap[p["query_id"]]
                prompts.append(_eval_prompt(p["caption"], c["true_action"]))
                prompts.append(_eval_prompt(p["caption"], c["conflict_action"]))
            resps = backend.complete_batch(prompts)
            for j, p in enumerate(batch):
                try: yt = _parse_json(resps[2*j]); yc = _parse_json(resps[2*j+1]); ok=True
                except Exception as e: yt, yc, ok = {"parse_error": str(e)}, {"parse_error": str(e)}, False
                append_jsonl(rows_path, {"query_id": p["query_id"], "arm": p["arm"], "generator": p["generator"], "captions": p["caption"],
                                         "y_true": int(yt.get("expressed", 0) or 0), "y_conflict": int(yc.get("expressed", 0) or 0),
                                         "true_detail": yt, "conflict_detail": yc, "judge": name, "parse_ok": ok})
    from uarcap_q2.common.gpu_runtime import release_gpu_memory
    with acquire_cooperative_gpu_lock(lock):
        gb = _gemma_backend(); run_judge(gb, "gemma-3-27b-fp8")
        try: gb._engine = None
        except Exception: pass
        del gb; release_gpu_memory(); time.sleep(8)
        lb = _llama_backend(); run_judge(lb, "llama-3.3-70b-w4a16")
        try: lb._engine = None
        except Exception: pass
        del lb; release_gpu_memory()
    allrows = read_jsonl(rows_path)
    by = defaultdict(dict)
    for r in allrows: by[(r["query_id"], r["arm"], r["generator"])][r["judge"]] = r
    agree = []
    for k, v in by.items():
        if len(v) == 2:
            a, b = list(v.values())
            agree.append({"key": k, "true_agree": a["y_true"] == b["y_true"], "conflict_agree": a["y_conflict"] == b["y_conflict"]})
    summary = {"schema": "uarcap_q2.e3_eval.v1", "status": "MACHINE_ASSISTED_UNCALIBRATED", "utc": utcnow(),
               "n_rows": len(allrows), "n_pairs": len(agree),
               "true_agreement": round(sum(x["true_agree"] for x in agree)/max(1,len(agree)), 4),
               "conflict_agreement": round(sum(x["conflict_agree"] for x in agree)/max(1,len(agree)), 4),
               "y_true_counts": {f"{k[0]}|{k[1]}": v for k, v in Counter((r["judge"], r["y_true"]) for r in allrows).items()},
               "y_conflict_counts": {f"{k[0]}|{k[1]}": v for k, v in Counter((r["judge"], r["y_conflict"]) for r in allrows).items()}}
    atomic_write(out/"E3_CROSS_JUDGE.json", summary)
    progress(len(by), len(preds), "evaluate")
    print(json.dumps({k: summary[k] for k in ("n_rows","n_pairs","true_agreement","conflict_agreement")}, indent=2))
    return 0



def stage_report() -> int:
    ap = json.loads((WORK/"E3_APPROVED_CANDIDATES.json").read_text()) if (WORK/"E3_APPROVED_CANDIDATES.json").is_file() else {}
    preds = {m: read_jsonl(WORK/m/"predictions.jsonl") for m in ("videollama","qwen")}
    ev = rj = read_jsonl(WORK/"evaluation/E3_EVAL.jsonl") if (WORK/"evaluation/E3_EVAL.jsonl").is_file() else []
    summary = {"schema": "uarcap_q2.e3_exploratory_receipt.v1", "status": "EXPLORATORY_E3_UNCALIBRATED",
               "utc": utcnow(), "n_approved_candidates": ap.get("n_approved", 0),
               "n_generations_videollama": len(preds["videollama"]), "n_generations_qwen": len(preds["qwen"]),
               "n_eval_rows": len(ev), "n_human_candidate_review": (json.loads((WORK/"E3_HUMAN_REVIEW_CANDIDATES.json").read_text()).get("n", 0) if (WORK/"E3_HUMAN_REVIEW_CANDIDATES.json").is_file() else 0),
               "note": "Machine-assisted exploratory E3; no human validation; no confirmatory statistics."}
    atomic_write(WORK/"E3_FINAL_RECEIPT.json", summary)
    (WORK/"E3_EXPLORATORY_REPORT.md").write_text("# E3 Exploratory Report\n\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2)); return 0


def stage_frames() -> int:
    from PIL import Image
    from uarcap_q2.datasets.video_decoder import decode_and_sample_video
    cands = json.loads((WORK/"E3_CANDIDATES.json").read_text())["candidates"]
    done = 0
    for i, c in enumerate(cands):
        d = WORK/"frames"/c["query_id"]
        if d.is_dir() and len(list(d.glob("frame_*.jpg"))) == 16:
            done += 1; continue
        try:
            frames, rep = decode_and_sample_video(Path(c["video_path"]), num_frames=16)
            d.mkdir(parents=True, exist_ok=True)
            for j, im in enumerate(frames): im.save(d/f"frame_{j:02d}.jpg", "JPEG", quality=92)
            done += 1
        except Exception as e:
            append_jsonl(WORK/"E3_FRAME_FAILURES.jsonl", {"query_id": c["query_id"], "error": str(e)[:400]})
        if i % 5 == 0: progress("frames", done, len(cands))
    progress("frames_done", done, len(cands))
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["extract","prepare","frames","screen_text","screen_visual","approve","generate_videollama","generate_qwen","evaluate","report"])
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    return {"extract":stage_extract,"prepare":stage_prepare,"frames":stage_frames,"screen_text":stage_screen_text,"screen_visual":stage_screen_visual,
            "approve":stage_approve,"generate_videollama":stage_generate_videollama,"generate_qwen":stage_generate_qwen,
            "evaluate":stage_evaluate,"report":stage_report}[args.stage]()


if __name__ == "__main__":
    raise SystemExit(main())
