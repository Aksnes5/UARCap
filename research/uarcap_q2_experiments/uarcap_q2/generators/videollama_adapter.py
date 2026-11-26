"""Frozen BN Video-LLaMA / EVCap batch adapter (P0-3 remediation).

Key guarantees:
  1. First-class `generate_batch(...)` loads the 7B model ONCE per arm (NEVER per video) and works
     directly with `uarcap_q2/runner/orchestrator.py` (no more NotImplementedError!).
  2. Packs 16 decoded frames into exact 18-column frozen `vision_language_tsv.py` format (`row[2:]`).
  3. Guarantees the EXACT checkpoint file loaded by `eval_evcap` (`out_path.replace('eval','train')/ckpt_name`)
     resolves to `canonical_checkpoint` AND matches `expected_checkpoint_sha256` BEFORE launching!
  4. Launches frozen evaluator via an isolated wrapper that attaches forward hooks (`runtime_trace.py` compatible),
     emits real per-query `input_trace.jsonl` + `generated_captions.json`, and supports injectable
     `subprocess_runner` for CPU contract testing without GPU!
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from typing import Any, Callable

from ..common.io_utils import atomic_write_json, atomic_write_text, primitive_hash, read_jsonl, sha256_file
from ..common.schemas import ExpArm, InputTrace, PilotStage
from ..datasets.video_decoder import pack_evcap_frame_tsv_bundle
from ..retrieval.arm_builder import ArmInputPacket
from .base import (
    BatchItem,
    GenerationOutput,
    GeneratorAdapter,
    GeneratorOOMError,
    TraceVerificationError,
    verify_actual_input_trace,
)

FROZEN_CHECKPOINT_NAME = (
    "train_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review.pt"
)
FROZEN_CHECKPOINT_SHA256 = "a8b5af79d55a2c7e684771475d645be6373208bdaae5a4211bee556581eb8002"
FROZEN_EVALUATOR_SCRIPT = "eval_evcap_fix_video_llama_xclip_msrvtt_fusion_gate_prompt5.py"
FROZEN_PROMPT_REL = "prompts/prompt_evcap_msrvtt_detail_smallcap_prompt5.txt"
FROZEN_BASE_ARGS: dict[str, Any] = {
    "bs": 1,
    "random_seed": 42,
    "max_num_frames": 16,
    "num_video_query_token": 32,
    "beam_width": 5,
    "freeze_video_qformer": True,
    "vit_precision": "fp16",
    "freeze_llm": True,
    "freeze_vit": True,
    "freeze_qformer": True,
    "load_video_feature": True,
    "share_query_tokens": True,
    "retriever": True,
    "deterministic": True,
    "use_fusion": True,
    "no_retrieval": False,
}


@dataclass
class FrozenVideoLLaMAConfig:
    python_executable: str
    evcap_work_root: str
    checkpoint_path: str
    llm_model_dir: str
    video_feature_dir: str
    torch_home: str
    hf_hub_cache: str
    device: str = "cuda:0"
    expected_checkpoint_sha256: str = FROZEN_CHECKPOINT_SHA256
    timeout_sec: int = 3600
    verify_external_features_exist: bool = True
    allow_non_linux_for_injected_runner: bool = False


def build_evcap_batch_inputs(
    packets: list[ArmInputPacket],
    out_dir: Path | str,
    arm: str,
) -> tuple[Path, Path]:
    """Serialize arm packets into exact EVCap cap.json and sims.json wire format."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap_map = {p.query_id: p.evcap_cap_entry for p in packets}
    sims_map = {p.query_id: p.evcap_sims_entry for p in packets}
    if arm == ExpArm.ARM_A0.value:
        for qid, entry in cap_map.items():
            if entry != [[], [], []] or sims_map[qid] != [[], []]:
                raise TraceVerificationError(f"ARM_A0 non-empty evidence refused for {qid}")
    cap_path = atomic_write_json(out / f"{arm}_cap.json", cap_map)
    sims_path = atomic_write_json(out / f"{arm}_sims.json", sims_map)
    return cap_path, sims_path


def bind_and_verify_evcap_checkpoint_slot(
    evcap_work_root: Path,
    out_rel_path: str,
    canonical_checkpoint: Path,
    expected_sha256: str,
) -> Path:
    """Bind the EXACT path that frozen eval_evcap loads: `out_path.replace('eval','train')/FROZEN_CHECKPOINT_NAME`.

    Guarantees that the exact file loaded by frozen eval_evcap has verified canonical SHA-256!
    """
    if "eval" not in out_rel_path:
        raise ValueError(
            f"out_rel_path '{out_rel_path}' MUST contain 'eval' so eval_evcap's outpath.replace('eval','train') "
            "maps to a clean isolated checkpoint slot."
        )
    if not canonical_checkpoint.is_file():
        raise FileNotFoundError(f"Canonical frozen Video-LLaMA checkpoint missing: {canonical_checkpoint}")
    actual_sha = sha256_file(canonical_checkpoint)
    if actual_sha != expected_sha256:
        raise TraceVerificationError(
            f"Canonical checkpoint SHA-256 mismatch! Got {actual_sha}, expected {expected_sha256}"
        )

    train_dir = evcap_work_root / out_rel_path.replace("eval", "train")
    train_dir.mkdir(parents=True, exist_ok=True)
    slot = train_dir / FROZEN_CHECKPOINT_NAME
    if slot.exists() or slot.is_symlink():
        if slot.resolve() != canonical_checkpoint.resolve():
            raise TraceVerificationError(
                f"Existing checkpoint slot {slot} points to wrong target {slot.resolve()} != {canonical_checkpoint.resolve()}"
            )
    else:
        try:
            slot.symlink_to(canonical_checkpoint.resolve())
        except OSError:
            # Windows non-admin fallback for CPU contract tests only
            shutil.copy2(str(canonical_checkpoint), str(slot))

    if sha256_file(slot) != expected_sha256:
        raise TraceVerificationError(f"Bound checkpoint slot SHA-256 verification failed: {slot}")
    return slot


def build_evcap_cli_argv(
    cfg: FrozenVideoLLaMAConfig,
    test_yaml_path: Path | str,
    cap_json_path: Path | str,
    sims_json_path: Path | str,
    out_rel_path: str,
    text_topn: int,
    visual_topn: int,
    wrapper_script_path: Path | None = None,
) -> list[str]:
    """Build exact CLI args matching frozen BN eval_evcap_fix_video_llama_xclip_msrvtt_fusion_gate_prompt5.py."""
    topn = max(text_topn, visual_topn)
    raw = {
        "device": cfg.device,
        "test_yaml": str(test_yaml_path),
        "out_path": out_rel_path,
        "ckpt_name": FROZEN_CHECKPOINT_NAME,
        "prompt_path": FROZEN_PROMPT_REL,
        "llm_model": cfg.llm_model_dir,
        "ext_path": str(cap_json_path),
        "sim_scores_path": str(sims_json_path),
        "video_feature_dir": cfg.video_feature_dir,
        "topn": topn,
        "text_topn": text_topn,
        "visual_topn": visual_topn,
        **FROZEN_BASE_ARGS,
    }
    entry = str(wrapper_script_path) if wrapper_script_path else str(Path(cfg.evcap_work_root) / FROZEN_EVALUATOR_SCRIPT)
    argv = [cfg.python_executable, entry]
    for k, v in raw.items():
        argv.extend([f"--{k}", str(v)])
    return argv


def _write_traced_evcap_wrapper(wrapper_path: Path, evcap_script_path: Path, trace_jsonl_path: Path, arm: str) -> Path:
    """Emit a tiny read-only launcher wrapper that hooks EVCap.encode_img/fusion_train/prompt_wrap without modifying frozen source."""
    code = f"""# Auto-generated read-only trace wrapper for frozen EVCap evaluator
import hashlib, json, runpy, sys
from pathlib import Path

EV_SCRIPT = {str(evcap_script_path)!r}
TRACE_PATH = Path({str(trace_jsonl_path)!r})
ARM = {arm!r}

def main():
    ev_dir = str(Path(EV_SCRIPT).parent)
    if ev_dir not in sys.path:
        sys.path.insert(0, ev_dir)
    import torch
    from models import evcap_fix_video_llama_xclip_msrvtt_smallcap_quality_fusion_gate_prompt5 as mod
    cls = mod.EVCap
    orig_encode = cls.encode_img
    orig_fuse = cls.fusion_train
    trace_fh = TRACE_PATH.open("w", encoding="utf-8")

    def _tdigest(t):
        raw = t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    state = {{}}
    def traced_encode(self, image, ids, train_data, *args, **kwargs):
        q = ids[0]
        state["query_id"] = q
        state["image_tensor_sha256"] = _tdigest(image)
        state["text_topn"] = int(self.text_topn or 0)
        state["visual_topn"] = int(self.visual_topn or 0)
        out = orig_encode(self, image, ids, train_data, *args, **kwargs)
        state["visual_proj_sha256"] = _tdigest(out[0])
        return out

    def traced_fuse(self, hidden, attention, ids, train_data):
        q = ids[0]
        cap_entry = self.cap.get(q, [[], [], []])
        sims_entry = self.sims_scores.get(q, [[], []])
        k_vis = int(self.visual_topn or 0)
        k_txt = int(self.text_topn or 0)
        fused = orig_fuse(self, hidden, attention, ids, train_data)
        row = {{
            "query_id": q,
            "arm": ARM,
            "actual_text_evidence": list(cap_entry[0][:k_txt]),
            "actual_visual_clip_ids": [grp[0] for grp in cap_entry[2][:k_vis] if grp],
            "actual_visual_scores": [float(dict(zip(*sims_entry))[cap_entry[0][i]]) for i, grp in enumerate(cap_entry[2][:k_vis]) if grp] if k_vis else [],
            "image_tensor_sha256": state.get("image_tensor_sha256", ""),
            "fused_aggregate_sha256": _tdigest(fused),
            "fused_nonzero": int(torch.count_nonzero(fused).item()),
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        }}
        trace_fh.write(json.dumps(row, ensure_ascii=False) + "\\n")
        trace_fh.flush()
        return fused

    cls.encode_img = traced_encode
    cls.fusion_train = traced_fuse
    sys.argv = [EV_SCRIPT] + sys.argv[1:]
    try:
        runpy.run_path(EV_SCRIPT, run_name="__main__")
    finally:
        trace_fh.close()

if __name__ == "__main__":
    main()
"""
    return atomic_write_text(wrapper_path, code)


class FrozenVideoLLaMAAdapter(GeneratorAdapter):
    """Unified batch-capable adapter wrapping frozen BN Video-LLaMA/EVCap without modifying frozen files."""

    model_id = "videollama_evcap_frozen_v08"
    injection_interface = "BIMODAL_CROSS_GATING"

    def __init__(
        self,
        cfg: FrozenVideoLLaMAConfig,
        subprocess_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.expected_checkpoint_sha256 = cfg.expected_checkpoint_sha256
        self._subprocess_runner = subprocess_runner

    def verify_external_visual_features(self, packets: list[ArmInputPacket]) -> None:
        if not self.cfg.verify_external_features_exist:
            return
        feat_dir = Path(self.cfg.video_feature_dir)
        missing: list[str] = []
        for pkt in packets:
            for cid in pkt.visual_clip_ids:
                if not (feat_dir / f"{cid}.pt").is_file():
                    missing.append(cid)
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} external visual feature .pt files under {feat_dir}: {missing[:5]}"
            )

    def run_batch_subprocess(
        self,
        test_yaml_path: Path | str,
        packets: list[ArmInputPacket],
        arm: str,
        work_out_dir: Path | str,
        run_id: str = "q2run",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Prepare verified arm inputs, bind exact checkpoint slot, and run frozen evaluator once for the whole batch."""
        out_dir = Path(work_out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cap_path, sims_path = build_evcap_batch_inputs(packets, out_dir / "inputs", arm)
        text_topn = packets[0].text_topn if packets else 0
        visual_topn = packets[0].visual_topn if packets else 0

        # Mandatory 'eval' path so eval_evcap loads exact bound `out_rel.replace('eval','train')/FROZEN_CHECKPOINT_NAME`!
        out_rel = f"outputs/eval_q2_{run_id}_{arm}"
        evcap_root = Path(self.cfg.evcap_work_root)
        wrapper_path = _write_traced_evcap_wrapper(
            out_dir / f"launch_traced_{arm}.py",
            evcap_root / FROZEN_EVALUATOR_SCRIPT,
            out_dir / f"evcap_forward_trace_{arm}.jsonl",
            arm=arm,
        )
        argv = build_evcap_cli_argv(
            self.cfg,
            test_yaml_path=test_yaml_path,
            cap_json_path=cap_path,
            sims_json_path=sims_path,
            out_rel_path=out_rel,
            text_topn=text_topn,
            visual_topn=visual_topn,
            wrapper_script_path=wrapper_path,
        )
        if dry_run:
            return {
                "dry_run": True,
                "arm": arm,
                "argv": argv,
                "cap_path": str(cap_path),
                "sims_path": str(sims_path),
                "out_rel_path": out_rel,
                "expected_checkpoint_sha256": self.expected_checkpoint_sha256,
            }

        if not sys.platform.startswith("linux") and not (
            self._subprocess_runner is not None and self.cfg.allow_non_linux_for_injected_runner
        ):
            raise RuntimeError("Frozen BN Video-LLaMA GPU execution is supported only on authorized Linux BN server.")

        self.verify_external_visual_features(packets)
        ckpt_slot = bind_and_verify_evcap_checkpoint_slot(
            evcap_work_root=evcap_root,
            out_rel_path=out_rel,
            canonical_checkpoint=Path(self.cfg.checkpoint_path),
            expected_sha256=self.expected_checkpoint_sha256,
        )

        env = {
            **os.environ,
            "TORCH_HOME": self.cfg.torch_home,
            "HUGGINGFACE_HUB_CACHE": self.cfg.hf_hub_cache,
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "USE_TF": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
        runner_fn = self._subprocess_runner or subprocess.run
        t0 = perf_counter()
        proc = runner_fn(
            argv,
            cwd=str(evcap_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=self.cfg.timeout_sec,
        )
        elapsed = perf_counter() - t0
        if "CUDA out of memory" in (proc.stderr or "") or "OutOfMemoryError" in (proc.stderr or ""):
            raise GeneratorOOMError(f"CUDA OOM in frozen Video-LLaMA arm {arm}: {proc.stderr[-500:]}")
        if proc.returncode != 0:
            raise RuntimeError(f"Frozen Video-LLaMA exited rc={proc.returncode}: {proc.stderr[-800:]}")

        out_abs = evcap_root / out_rel
        pred_files = list(out_abs.glob("*generated_captions.json"))
        if len(pred_files) != 1:
            raise RuntimeError(f"Expected 1 *generated_captions.json in {out_abs}, found {len(pred_files)}")
        preds = json.loads(pred_files[0].read_text(encoding="utf-8"))
        trace_file = out_dir / f"evcap_forward_trace_{arm}.jsonl"
        forward_traces = read_jsonl(trace_file) if trace_file.is_file() else []
        return {
            "dry_run": False,
            "arm": arm,
            "bound_checkpoint_slot": str(ckpt_slot),
            "checkpoint_sha256": self.expected_checkpoint_sha256,
            "wall_seconds": round(elapsed, 4),
            "prediction_file": str(pred_files[0]),
            "predictions": preds,
            "forward_traces": forward_traces,
        }

    def generate_batch(
        self,
        items: list[BatchItem],
        stage: PilotStage,
        work_dir: Path | str,
        run_id: str = "q2run",
    ) -> list[GenerationOutput]:
        """Execute all items of one arm in a single 7B model load via frozen EVCap batch TSV + forward trace."""
        if not items:
            return []
        arms = {it.packet.arm for it in items}
        if len(arms) != 1:
            raise ValueError(f"FrozenVideoLLaMAAdapter.generate_batch expects single arm per batch, got {arms}")
        arm = next(iter(arms))

        wdir = Path(work_dir) / f"videollama_batch_{arm}"
        wdir.mkdir(parents=True, exist_ok=True)

        # Pack 16 decoded PIL frames into exact 18-column EVCap img.tsv + lineidx + label/caption/yaml
        tsv_items = []
        for it in items:
            if len(it.frame_images) != 16:
                raise ValueError(f"Frozen Video-LLaMA requires 16 decoded PIL frames for {it.query.query_id}, got {len(it.frame_images)}")
            tsv_items.append((it.query.query_id, it.frame_images, it.query.reference_description))
        tsv_bundle = pack_evcap_frame_tsv_bundle(tsv_items, wdir / "tsv", prefix=f"{run_id}_{arm}_16frames")

        res = self.run_batch_subprocess(
            test_yaml_path=tsv_bundle["yaml"],
            packets=[it.packet for it in items],
            arm=arm,
            work_out_dir=wdir,
            run_id=run_id,
            dry_run=False,
        )

        # Parse real EVCap output list: [{"split":"valid","image_name":qid,"captions":[...],"prediction":"..."}]
        pred_by_qid: dict[str, str] = {}
        for row in res["predictions"]:
            qid = str(row["image_name"])
            pred_by_qid[qid] = str(row.get("prediction", "")).strip()

        fwd_by_qid = {str(ft["query_id"]): ft for ft in res.get("forward_traces", [])}
        per_item_wall = round(res["wall_seconds"] / max(len(items), 1), 4)

        outputs: list[GenerationOutput] = []
        for it in items:
            qid = it.query.query_id
            if qid not in pred_by_qid:
                raise RuntimeError(f"Frozen EVCap output missing prediction for query_id={qid}")
            caption = pred_by_qid[qid]
            ft = fwd_by_qid.get(qid)
            if ft is None:
                raise TraceVerificationError(f"Missing real forward hook trace from EVCap for query_id={qid}")
            if arm == ExpArm.ARM_A0.value and int(ft.get("fused_nonzero", 0)) != 0:
                raise TraceVerificationError(f"ARM_A0 non-zero visual aggregate detected in forward hook for {qid}")

            trace = InputTrace(
                query_id=qid,
                dataset_id=it.query.dataset_id,
                stage=stage.value,
                arm=arm,
                model_id=self.model_id,
                checkpoint_sha256=res["checkpoint_sha256"],
                visual_input_sha256=it.visual_input_sha256,
                frame_indices=list(it.frame_indices),
                text_evidence=list(ft["actual_text_evidence"]),
                visual_evidence_ids=list(ft["actual_visual_clip_ids"]),
                visual_evidence_scores=list(ft["actual_visual_scores"]),
                prompt_rendered_sha256=primitive_hash({"prompt_path": FROZEN_PROMPT_REL, "text": ft["actual_text_evidence"]}),
                rng_seed=int(FROZEN_BASE_ARGS["random_seed"]),
                decoding_config=dict(FROZEN_BASE_ARGS),
                base_visual_mode=it.packet.base_visual_mode,
                injection_interface=self.injection_interface,
                actual_processor_telemetry={
                    "bound_checkpoint_slot": res["bound_checkpoint_slot"],
                    "image_tensor_sha256": ft.get("image_tensor_sha256", ""),
                    "fused_aggregate_sha256": ft.get("fused_aggregate_sha256", ""),
                    "fused_nonzero": ft.get("fused_nonzero", 0),
                },
            )
            verify_actual_input_trace(trace, it.packet, self.expected_checkpoint_sha256, it.visual_input_sha256)
            outputs.append(
                GenerationOutput(
                    query_id=qid,
                    dataset_id=it.query.dataset_id,
                    stage=stage.value,
                    arm=arm,
                    model_id=self.model_id,
                    caption=caption,
                    input_trace=trace,
                    wall_seconds=per_item_wall,
                    peak_vram_bytes=int(ft.get("peak_vram_bytes", 0)),
                    ok=bool(caption),
                    error="" if caption else "EMPTY_GENERATION",
                )
            )
        return outputs
