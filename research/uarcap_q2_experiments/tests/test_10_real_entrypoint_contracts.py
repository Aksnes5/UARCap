"""Tests 10: Real-Entrypoint Contract Tests (P0-1 through P0-7 full remediation verification on CPU).

NOTE: These are strict CPU real-entrypoint contract tests (using injectable fake X-CLIP/EVCap/Processor/VLM/Judge
through the REAL CLI & REAL stage dispatch paths). They prove engineering contract integrity — NOT RTX A6000 GPU validity.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
import yaml

EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from deployment.run_stage_cli import build_parser, dispatch_stage
from deployment.stage_assets import main as stage_assets_main
from uarcap_q2.common.gates import AuthorizationError, verify_stage_authorization
from uarcap_q2.common.io_utils import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl, sha256_file
from uarcap_q2.common.schemas import ExpArm, PilotStage, RetrievalCandidate, StageRunStatus, VideoTruthLabel
from uarcap_q2.datasets.dream1k import load_dream1k_records
from uarcap_q2.datasets.video_decoder import decode_and_sample_video
from uarcap_q2.datasets.vidhalluc import load_vidhalluc_ach_mcq
from uarcap_q2.evaluators.backends import MockJudgeBackend
from uarcap_q2.generators.base import TraceVerificationError
from uarcap_q2.generators.mock_generator import MockGeneratorAdapter
from uarcap_q2.generators.modern_vlm_adapter import ModernVLMConfig, Qwen25VLAdapter
from uarcap_q2.generators.videollama_adapter import (
    FROZEN_CHECKPOINT_NAME,
    FrozenVideoLLaMAAdapter,
    FrozenVideoLLaMAConfig,
)
from uarcap_q2.interventions.vidhalluc_e3 import (
    E3ApplicabilityChecklist,
    apply_human_applicability_review,
    build_vidhalluc_e3_candidate,
)
from uarcap_q2.retrieval.arm_builder import build_e3_intervention_arms
from uarcap_q2.retrieval.reranker import MockXCLIPRetriever, rerank_d_consensus
from uarcap_q2.retrieval.videoufo_retriever import (
    VideoUFOIndexConfig,
    VideoUFOXCLIPRetriever,
    XCLIPBackend,
)
from uarcap_q2.runner.orchestrator import ResumeIntegrityError, RunSpec, run_natural_retrieval_stage


# ---- Helpers: Injectable CPU Fakes exercising REAL adapter code paths ----

class FakeXCLIPBackend(XCLIPBackend):
    backend_id = "fake_xclip_contract_cpu"

    def encode_video(self, video_tensor_tchw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert video_tensor_tchw.shape == (16, 3, 224, 224)
        mean_rgb = video_tensor_tchw.mean(axis=(0, 2, 3))
        v = np.zeros(8, dtype=np.float32)
        v[:3] = mean_rgb
        v[3] = 1.0
        v /= max(float(np.linalg.norm(v)), 1e-6)
        im = np.ones((4, 8), dtype=np.float32) * 0.05
        return v, im

    def score_text_chunk(self, query_video_emb: np.ndarray, query_img_feat: np.ndarray, text_chunk_emb: np.ndarray) -> np.ndarray:
        # Query-conditioned prompt addition + cosine scoring (exact VideoSpecificPrompt contract)
        cond = text_chunk_emb + 0.01 * query_img_feat.mean(axis=0, keepdims=True)
        norms = np.maximum(np.linalg.norm(cond, axis=1, keepdims=True), 1e-6)
        return (cond / norms) @ query_video_emb


class FakeModernVLMProcessor:
    def __init__(self) -> None:
        self.received_videos: list[Any] = []
        self.received_texts: list[str] = []

    def apply_chat_template(self, messages: list[dict[str, Any]], tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        user_txt = [c["text"] for c in messages[1]["content"] if c["type"] == "text"][0]
        return f"<sys>{messages[0]['content'][0]['text']}</sys><usr>{user_txt}</usr><asst>"

    def __call__(self, text: list[str], videos: list[Any] | None = None, padding: bool = True, return_tensors: str = "pt") -> dict[str, Any]:
        assert videos is not None and len(videos) == 1 and len(videos[0]) == 16, "Processor MUST receive 16 actual PIL frames!"
        self.received_videos.append(videos[0])
        self.received_texts.append(text[0])
        # Build deterministic numpy arrays mimicking torch tensors
        frame_arr = np.stack([np.asarray(im, dtype=np.float32) for im in videos[0]], axis=0)
        toks = np.arange(1, 1 + max(4, len(text[0].split())), dtype=np.int64)[None, :]
        return {"input_ids": toks, "pixel_values_videos": frame_arr}

    def batch_decode(self, trimmed: np.ndarray, skip_special_tokens: bool = True) -> list[str]:
        return ["A person is throwing grass in the garbage outdoors."]


class FakeModernVLMModel:
    def __init__(self) -> None:
        self.config = {"model_type": "qwen2_5_vl", "hidden_size": 3584}
        self.calls: int = 0

    def generate(self, **kwargs: Any) -> np.ndarray:
        self.calls += 1
        assert "pixel_values_videos" in kwargs and kwargs["pixel_values_videos"].shape[0] == 16
        in_ids = kwargs["input_ids"]
        extra = np.array([[101, 102, 103]], dtype=np.int64)
        return np.concatenate([in_ids, extra], axis=1)


def _make_fake_evcap_subprocess_runner(evcap_work_root: Path):
    """Fake subprocess runner that validates real CLI argv + 18-col TSV + bound checkpoint slot & writes real EVCap outputs."""
    calls: list[list[str]] = []

    def _run(argv: list[str], cwd: str, env: dict[str, str], capture_output: bool, text: bool, timeout: int) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        kv = dict(zip(argv[2::2], argv[3::2]))
        out_rel = kv["--out_path"]
        assert "eval" in out_rel
        # Verify exact checkpoint slot derived by frozen eval_evcap exists!
        ckpt_slot = Path(cwd) / out_rel.replace("eval", "train") / kv["--ckpt_name"]
        assert ckpt_slot.is_file(), f"Missing bound checkpoint slot: {ckpt_slot}"

        # Read real 18-col img.tsv via test_yaml
        ycfg = yaml.safe_load(Path(kv["--test_yaml"]).read_text(encoding="utf-8"))
        img_rows = [ln.split("\t") for ln in Path(ycfg["img"]).read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert all(len(r) == 18 and r[1] == "16" for r in img_rows), "EVCap TSV must have exact 18 columns!"

        caps_map = json.loads(Path(kv["--ext_path"]).read_text(encoding="utf-8"))
        sims_map = json.loads(Path(kv["--sim_scores_path"]).read_text(encoding="utf-8"))
        t_k = int(kv["--text_topn"])
        v_k = int(kv["--visual_topn"])

        out_dir = Path(cwd) / out_rel
        out_dir.mkdir(parents=True, exist_ok=True)
        preds = []
        traces = []
        arm_name = out_rel.rsplit("_", 2)[-2] + "_" + out_rel.rsplit("_", 1)[-1]
        for r in img_rows:
            qid = r[0]
            c_ent = caps_map[qid]
            s_ent = sims_map[qid]
            if v_k == 0 and t_k == 0:
                assert c_ent == [[], [], []] and s_ent == [[], []]
            preds.append({"split": "valid", "image_name": qid, "captions": ["ref"], "prediction": f"A character walks in {qid} ({t_k})."})
            s_dict = dict(zip(*s_ent)) if s_ent[0] else {}
            traces.append(
                {
                    "query_id": qid,
                    "arm": arm_name,
                    "actual_text_evidence": list(c_ent[0][:t_k]),
                    "actual_visual_clip_ids": [grp[0] for grp in c_ent[2][:v_k]],
                    "actual_visual_scores": [float(s_dict[c]) for c in c_ent[0][:v_k]] if v_k else [],
                    "image_tensor_sha256": "fake_img_tensor_sha",
                    "fused_aggregate_sha256": "fake_agg_sha",
                    "fused_nonzero": 0 if v_k == 0 else 128,
                    "peak_vram_bytes": 17000000000,
                }
            )
        outfile = "eval" + kv["--ckpt_name"][5:-3] + "_generated_captions.json"
        (out_dir / outfile).write_text(json.dumps(preds, indent=2), encoding="utf-8")

        # Write forward trace where wrapper expects it (next to launch_traced_{arm}.py)
        wrapper_p = Path(argv[1])
        arm_suffix = wrapper_p.stem.replace("launch_traced_", "")
        trace_p = wrapper_p.parent / f"evcap_forward_trace_{arm_suffix}.jsonl"
        atomic_write_jsonl(trace_p, traces)
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    return _run, calls


def _build_synthetic_videoufo_index(tmp_path: Path) -> VideoUFOIndexConfig:
    idx_dir = tmp_path / "videoufo_idx"
    feat_dir = idx_dir / "features"
    idx_dir.mkdir(parents=True)
    feat_dir.mkdir(parents=True)

    csv_lines = ["ID,Brief Caption,Topic"]
    tsv_lines = ["clip_id\tsource_id"]
    rng = np.random.default_rng(7)
    n_clips = 28
    embs = rng.normal(size=(n_clips, 8)).astype(np.float16)

    for i in range(n_clips):
        cid = f"VUFO{i:07d}.{i % 2}"
        # Create 2 duplicate captions to test strip-only grouping + lexicographical clip sort
        cap = "A character walks forward in the room." if i in (0, 1) else f"Synthetic VideoUFO scene {i} with distinct action {i % 5}."
        csv_lines.append(f"{cid},\"{cap}\",topic_{i % 3}")
        tsv_lines.append(f"{cid}\tVUFO{i:07d}")
        (feat_dir / f"{cid}.pt").write_bytes(b"FAKE_PT_FEATURE_BYTES")

    csv_p = idx_dir / "VideoUFO.csv"
    tsv_p = idx_dir / "clips_full.tsv"
    emb_p = idx_dir / "videoufo_brief_text_fp16.npy"
    csv_p.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    tsv_p.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")
    np.save(str(emb_p), embs)

    return VideoUFOIndexConfig(
        videoufo_csv_path=str(csv_p),
        clips_full_tsv_path=str(tsv_p),
        brief_text_emb_npy_path=str(emb_p),
        video_feature_dir=str(feat_dir),
        chunk_size=10,
        topk=20,
    )


# ---- 1. P0-2 Golden Historical Regression Test ----

def test_p02_golden_historical_d_rerank_exact_match() -> None:
    golden = read_json(EXP_ROOT / "fixtures" / "golden_historical_d_rerank_sample.json")
    assert golden["source_commit"] == "52b5ed3a69e4b84a8e16ed747f141eefb5b1e2ec"
    for qkey, entry in golden["queries"].items():
        cands = [
            RetrievalCandidate(caption=x["caption"], score=float(x["score"]), clip_ids=list(x["clip_ids"]), original_rank=i)
            for i, x in enumerate(entry["top20"])
        ]
        rr = rerank_d_consensus(cands)[:7]
        exp_caps, exp_q, exp_clips = entry["expected_d7_cap"]
        exp_sim_map = dict(zip(*entry["expected_d7_sims"]))
        assert [c.caption for c in rr] == exp_caps, f"Order mismatch on {qkey}"
        assert [c.clip_ids for c in rr] == exp_clips, f"Clip groups mismatch on {qkey}"
        assert all(abs(c.rerank_score - eq) < 1e-12 for c, eq in zip(rr, exp_q)), f"ROUGE-L score mismatch on {qkey}"
        assert all(abs(c.score - exp_sim_map[c.caption]) < 1e-12 for c in rr), f"Similarity mismatch on {qkey}"


# ---- 2. P0-2 Real VideoUFO X-CLIP Retriever Contract Test ----

def test_p02_real_videoufo_xclip_retriever_contract(synthetic_dream1k_dir: Path, tmp_path: Path) -> None:
    vcfg = _build_synthetic_videoufo_index(tmp_path)
    retriever = VideoUFOXCLIPRetriever(vcfg, xclip_backend=FakeXCLIPBackend(), verify_feature_files_exist=True)
    recs = load_dream1k_records(synthetic_dream1k_dir / "json" / "metadata.json", synthetic_dream1k_dir / "video", require_video_exists=True)
    top20 = retriever.retrieve_top20(recs[0])
    assert len(top20) == 20
    # Grouped caption has lexicographically sorted clip_ids
    grp = [c for c in top20 if c.caption == "A character walks forward in the room."]
    if grp:
        assert grp[0].clip_ids == sorted(grp[0].clip_ids) and len(grp[0].clip_ids) == 2
    mpath = retriever.write_asset_manifest(tmp_path / "retrieval_manifest.json")
    manifest = read_json(mpath)
    assert manifest["n_unique_caption_groups"] == 27 and manifest["clips_full_tsv_sha256"]


# ---- 3. P0-1 & P0-3 Real CLI Pilot A1 -> Frozen Video-LLaMA Batch Contract Test ----

def test_p01_p03_real_pilot_a1_batch(synthetic_dream1k_dir: Path, tmp_path: Path) -> None:
    # Real CLI without real assets MUST fail (exit code != 0), NEVER print READY_FOR_BN_REAL_ADAPTER and return 0!
    paths_yaml = tmp_path / "bn_paths_missing.yaml"
    paths_yaml.write_text(
        yaml.safe_dump(
            {
                "host": {"device": "cuda:0", "shared_gpu_lock": ""},
                "historical_frozen_assets": {
                    "videoufo_csv_path": str(tmp_path / "missing.csv"),
                    "clips_full_tsv_path": str(tmp_path / "missing.tsv"),
                    "brief_text_emb_npy_path": str(tmp_path / "missing.npy"),
                    "xclip_repo_dir": str(tmp_path / "missing_xclip"),
                    "xclip_ckpt_path": str(tmp_path / "missing.pth"),
                    "video_feature_dir": str(tmp_path / "missing_feats"),
                    "frozen_videollama_python": sys.executable,
                    "v06_runtime_evcap_work": str(tmp_path / "evcap"),
                    "canonical_checkpoint": str(tmp_path / "missing.pt"),
                    "canonical_checkpoint_sha256": "deadbeef",
                    "vicuna_7b_dir": str(tmp_path / "vicuna"),
                    "torch_home": str(tmp_path),
                    "hf_hub_cache": str(tmp_path),
                },
                "q2_writable_workspace": {"dream1k_dir": str(tmp_path / "missing_dream"), "vidhalluc_dir": str(tmp_path / "missing_vh")},
            }
        ),
        encoding="utf-8",
    )
    auth_a1 = atomic_write_json(tmp_path / "auth_a1.json", {"authorized_stages": ["PILOT_A1"], "approved": True, "approval_id": "OK_A1"})
    p = subprocess.run(
        [
            sys.executable, str(EXP_ROOT / "deployment" / "run_stage_cli.py"),
            "--stage", "PILOT_A1", "--paths-config", str(paths_yaml),
            "--stage-config", str(EXP_ROOT / "configs" / "pilot_a01_config.yaml"),
            "--out-dir", str(tmp_path / "out_fail"), "--auth-receipt", str(auth_a1),
        ],
        capture_output=True, text=True,
    )
    assert p.returncode == 2 and "READY_FOR_BN_REAL_ADAPTER" not in p.stdout

    # Now exercise REAL PILOT_A1 dispatch with real VideoUFOXCLIPRetriever + real FrozenVideoLLaMAAdapter.generate_batch!
    vcfg = _build_synthetic_videoufo_index(tmp_path)
    real_ret = VideoUFOXCLIPRetriever(vcfg, xclip_backend=FakeXCLIPBackend(), verify_feature_files_exist=True)

    evcap_root = tmp_path / "ev"
    evcap_root.mkdir()
    fake_ckpt = tmp_path / "c" / FROZEN_CHECKPOINT_NAME
    fake_ckpt.parent.mkdir()
    fake_ckpt.write_bytes(b"SYNTHETIC_FROZEN_CHECKPOINT_BYTES_FOR_CONTRACT_TEST")
    ckpt_sha = sha256_file(fake_ckpt)

    fake_subproc, calls = _make_fake_evcap_subprocess_runner(evcap_root)
    real_vl = FrozenVideoLLaMAAdapter(
        FrozenVideoLLaMAConfig(
            python_executable=sys.executable,
            evcap_work_root=str(evcap_root),
            checkpoint_path=str(fake_ckpt),
            llm_model_dir=str(tmp_path),
            video_feature_dir=vcfg.video_feature_dir,
            torch_home=str(tmp_path),
            hf_hub_cache=str(tmp_path),
            expected_checkpoint_sha256=ckpt_sha,
            verify_external_features_exist=True,
            allow_non_linux_for_injected_runner=True,
        ),
        subprocess_runner=fake_subproc,
    )
    paths_ok = tmp_path / "bn_paths_ok.yaml"
    paths_ok.write_text(
        yaml.safe_dump({"host": {"device": "cuda:0"}, "historical_frozen_assets": {}, "q2_writable_workspace": {"dream1k_dir": str(synthetic_dream1k_dir), "vidhalluc_dir": str(tmp_path)}}),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--stage", "PILOT_A1", "--paths-config", str(paths_ok),
            "--stage-config", str(EXP_ROOT / "configs" / "pilot_a01_config.yaml"),
            "--out-dir", str(tmp_path / "a1_real_out"), "--run-id", "contract_a1",
            "--auth-receipt", str(auth_a1),
        ]
    )
    rc, summary = dispatch_stage(args, injected_retriever=real_ret, injected_videollama=real_vl)
    assert rc == 0, f"PILOT_A1 failed: {summary}"
    assert summary["status"] == StageRunStatus.REAL_RUNTIME_PASS.value
    assert summary["completed_generations"] == 30  # 10 videos (2 per domain x 5 domains) x 3 arms
    # 7B Video-LLaMA loaded ONCE per arm (3 subprocess calls total for 15 generations, NOT 15 calls!)
    assert len(calls) == 3
    assert (tmp_path / "a1_real_out" / "PILOT_A1_GO_NO_GO.md").is_file()


# ---- 4. P0-4 Modern VLM Real Frame Handoff, Processor Telemetry & E3 Visual Parity ----

def test_p04_modern_vlm_real_frames_processor_telemetry_and_e3_parity(pathological_videos: dict[str, Path], tmp_path: Path) -> None:
    model_dir = tmp_path / "qwen_local"
    model_dir.mkdir()
    atomic_write_json(model_dir / "model_revision.json", {"revision": "rev_abc_2026"})

    fake_proc = FakeModernVLMProcessor()
    fake_model = FakeModernVLMModel()
    vlm = Qwen25VLAdapter(
        ModernVLMConfig(local_model_path=str(model_dir), pinned_revision="rev_abc_2026", device="cpu"),
        processor=fake_proc,
        model=fake_model,
    )

    imgs, rep = decode_and_sample_video(pathological_videos["dynamic_24f"], num_frames=16)
    e3_pkts = build_e3_intervention_arms("vidhalluc_1", "throwing grass", "raking leaves", "standing outside", base_visual_mode="BASE_A0")
    from uarcap_q2.common.schemas import QueryRecord, SourceIdStatus

    q = QueryRecord("VidHalluc-ACH", "vidhalluc_1", "vcCwvRYqU2I", SourceIdStatus.VERIFIED_EXACT.value, str(pathological_videos["dynamic_24f"]), rep.video_sha256, 2.0, "ACH", "throwing grass")

    # Missing frames MUST raise TraceVerificationError!
    with pytest.raises(TraceVerificationError, match="NO actual video frames"):
        vlm.generate_one(q, e3_pkts["E3_BASE"], PilotStage.PILOT_A3, rep.combined_visual_sha256, rep.sampled_indices, frame_images=None)

    # Run all 4 E3 arms through REAL Qwen25VLAdapter.generate_one
    pv_hashes = set()
    for arm_name in ("E3_BASE", "E3_SUP", "E3_CON", "E3_NEU"):
        out = vlm.generate_one(q, e3_pkts[arm_name], PilotStage.PILOT_A3, rep.combined_visual_sha256, rep.sampled_indices, frame_images=imgs)
        assert out.ok and out.input_trace.injection_interface == "PROMPT_TEXT_ONLY"
        assert out.input_trace.visual_evidence_ids == [] and out.input_trace.visual_evidence_scores == []
        tel = out.input_trace.actual_processor_telemetry
        assert tel["actual_frames_received"] == 16
        assert tel["actual_frames_sha256"] == rep.combined_visual_sha256
        assert tel["pixel_values_shape"][0] == 16 and tel["input_ids_len"] > 0
        pv_hashes.add(tel["pixel_values_sha256"])

    # Exact identical visual tensor across all 4 E3 arms!
    assert len(pv_hashes) == 1 and fake_model.calls == 4


# ---- 5. P0-5 & P0-7 Full Stage State Machine: Mock Isolation + A2 Human Pause + A3 + Formal B + Asset Unlock ----

def test_p05_p07_mock_cannot_unlock_real_stages_and_a2_a3_formal_b_chain(
    synthetic_dream1k_dir: Path, synthetic_vidhalluc_json: Path, pathological_videos: dict[str, Path], tmp_path: Path
) -> None:
    # A. MOCK_PASS receipt MUST NEVER unlock real PILOT_A2 or asset downloads!
    mock_a1 = atomic_write_json(tmp_path / "mock_a1.json", {"status": StageRunStatus.MOCK_PASS.value, "execution_mode": "cpu_mock"})
    auth_a2 = atomic_write_json(tmp_path / "auth_a2.json", {"authorized_stages": ["PILOT_A2"], "approved": True, "approval_id": "A2_OK"})
    with pytest.raises(AuthorizationError, match="MOCK_PASS"):
        verify_stage_authorization(PilotStage.PILOT_A2, auth_a2, prior_receipts={"PILOT_A1": mock_a1}, allow_cpu_mock=False)

    # Asset download CLI actively enforces stage_unlock (blocks VidHalluc 15.60 GiB without A3 auth + real A2!)
    rc_dl = stage_assets_main(["download", "--data-root", str(tmp_path), "--asset", "vidhalluc_ach_videos", "--confirm-download"])
    assert rc_dl == 3

    # B. Pilot A2 reads A1 outputs -> pauses at exit code 10 (HUMAN_VALIDATION_PENDING), NEVER fake PASS!
    real_a1_rec = atomic_write_json(tmp_path / "real_a1_rec.json", {"status": StageRunStatus.REAL_RUNTIME_PASS.value, "execution_mode": "real_authorized"})
    a1_preds = tmp_path / "a1_preds.jsonl"
    recs = load_dream1k_records(synthetic_dream1k_dir / "json" / "metadata.json", synthetic_dream1k_dir / "video")
    atomic_write_jsonl(
        a1_preds,
        [
            {"query_id": r.query_id, "arm": arm, "model_id": "videollama", "caption": "A person is raking leaves." if arm == "ARM_A7" else r.reference_description}
            for r in recs[:5]
            for arm in ("ARM_A0", "ARM_A7")
        ],
    )
    paths_cfg = tmp_path / "paths.yaml"
    paths_cfg.write_text(yaml.safe_dump({"host": {}, "historical_frozen_assets": {"clips_full_tsv_path": str(tmp_path / "empty.tsv")}, "q2_writable_workspace": {"dream1k_dir": str(synthetic_dream1k_dir), "vidhalluc_dir": str(synthetic_vidhalluc_json.parent)}}), encoding="utf-8")
    (tmp_path / "empty.tsv").write_text("clip_id\tsource_id\n", encoding="utf-8")

    args_a2_wait = build_parser().parse_args(
        [
            "--stage", "PILOT_A2", "--paths-config", str(paths_cfg),
            "--stage-config", str(EXP_ROOT / "configs" / "pilot_a23_config.yaml"),
            "--out-dir", str(tmp_path / "a2_out"), "--auth-receipt", str(auth_a2),
            "--prior-a1", str(real_a1_rec), "--a1-predictions", str(a1_preds),
        ]
    )
    rc_a2, rec_a2 = dispatch_stage(args_a2_wait, injected_ref_judge=MockJudgeBackend(model_family="mistral"))
    assert rc_a2 == 10 and rec_a2["status"] == StageRunStatus.HUMAN_VALIDATION_PENDING.value
    assert (tmp_path / "a2_out" / "blind_annotation_package" / "blind_tasks_e2_A1.json").is_file()

    # Pilot A3 MUST reject A2 while still HUMAN_VALIDATION_PENDING!
    auth_a3 = atomic_write_json(tmp_path / "auth_a3.json", {"authorized_stages": ["PILOT_A3"], "approved": True, "approval_id": "A3_OK"})
    with pytest.raises(AuthorizationError, match="HUMAN_VALIDATION_PENDING"):
        verify_stage_authorization(PilotStage.PILOT_A3, auth_a3, prior_receipts={"PILOT_A2": tmp_path / "a2_out" / "calibration_receipt.json"})

    # C. Pilot A3 with real A2 PASS + human-APPROVED E3 candidate runs 4 arms on BOTH models + Blind E3 evaluator!
    real_a2_pass = atomic_write_json(tmp_path / "real_a2_pass.json", {"status": StageRunStatus.REAL_RUNTIME_PASS.value, "human_calibration_completed": True})
    vh_dir = tmp_path / "vh_stage"
    (vh_dir / "data").mkdir(parents=True)
    (vh_dir / "ach_mcq.json").write_text(synthetic_vidhalluc_json.read_text(encoding="utf-8"), encoding="utf-8")
    vh_recs = load_vidhalluc_ach_mcq(vh_dir / "ach_mcq.json", vh_dir / "data")
    # Copy dynamic MP4 for the approved clip
    c0 = build_vidhalluc_e3_candidate(vh_recs[0], neu_text="In this video clip, a person is outside in daylight.")
    (vh_dir / "data" / f"{vh_recs[0].metadata['clip_key']}.mp4").write_bytes(pathological_videos["dynamic_24f"].read_bytes())
    c0_ok = apply_human_applicability_review(
        c0, E3ApplicabilityChecklist(True, True, True, True, True, False, "A1")
    )
    approved_p = atomic_write_json(tmp_path / "approved_e3.json", [c0_ok.to_dict()])

    paths_a3 = tmp_path / "paths_a3.yaml"
    paths_a3.write_text(yaml.safe_dump({"host": {}, "historical_frozen_assets": {"clips_full_tsv_path": str(tmp_path / "empty.tsv")}, "q2_writable_workspace": {"dream1k_dir": str(synthetic_dream1k_dir), "vidhalluc_dir": str(vh_dir)}}), encoding="utf-8")
    args_a3 = build_parser().parse_args(
        [
            "--stage", "PILOT_A3", "--paths-config", str(paths_a3),
            "--stage-config", str(EXP_ROOT / "configs" / "pilot_a23_config.yaml"),
            "--out-dir", str(tmp_path / "a3_out"), "--auth-receipt", str(auth_a3),
            "--prior-a2", str(real_a2_pass), "--approved-e3-candidates", str(approved_p), "--limit", "2",
        ]
    )
    rc_a3, rec_a3 = dispatch_stage(
        args_a3,
        injected_retriever=MockXCLIPRetriever(),
        injected_videollama=MockGeneratorAdapter(model_id="real_vl_contract"),
        injected_modern_vlm=MockGeneratorAdapter(model_id="real_vlm_contract"),
        injected_e3_judge=MockJudgeBackend(model_family="mistral"),
    )
    assert rc_a3 == 0 and rec_a3["status"] == StageRunStatus.REAL_RUNTIME_PASS.value
    assert rec_a3["n_e3_evaluated_captions"] == 8  # 1 clip x 4 arms x 2 models

    # D. Formal B executes ONLY when A1/A2/A3 REAL_RUNTIME_PASS + all frozen bindings match!
    formal_cfg = EXP_ROOT / "configs" / "formal_b_candidate.yaml"
    asset_man = EXP_ROOT / "DATA_AND_MODEL_ASSET_MANIFEST.json"
    pilot_excl = atomic_write_json(tmp_path / "pilot_excl.json", {"excluded_query_ids": [recs[0].query_id]})
    formal_auth = atomic_write_json(
        tmp_path / "formal_b_auth.json",
        {
            "stage": "FORMAL_B",
            "approved": True,
            "approval_id": "USER_FORMAL_B_TEST",
            "frozen_pilot_params_confirmed": True,
            "frozen_formal_config_sha256": sha256_file(formal_cfg),
            "evaluator_version": "mistral-24b-frozen-v1",
            "asset_manifest_sha256": sha256_file(asset_man),
            "pilot_exclusion_manifest_sha256": sha256_file(pilot_excl),
        },
    )
    args_fb = build_parser().parse_args(
        [
            "--stage", "FORMAL_B", "--paths-config", str(paths_a3),
            "--stage-config", str(formal_cfg), "--out-dir", str(tmp_path / "formal_out"),
            "--auth-receipt", str(formal_auth), "--prior-a1", str(real_a1_rec),
            "--prior-a2", str(real_a2_pass), "--prior-a3", str(tmp_path / "a3_out" / "run_receipt.json"),
            "--asset-manifest", str(asset_man), "--pilot-exclusion", str(pilot_excl),
            "--approved-e3-candidates", str(approved_p), "--limit", "2",
        ]
    )
    rc_fb, rec_fb = dispatch_stage(
        args_fb,
        injected_retriever=MockXCLIPRetriever(),
        injected_videollama=MockGeneratorAdapter(model_id="real_vl_contract"),
        injected_modern_vlm=MockGeneratorAdapter(model_id="real_vlm_contract"),
        injected_ref_judge=MockJudgeBackend(model_family="mistral"),
        injected_e3_judge=MockJudgeBackend(model_family="mistral"),
    )
    assert rc_fb == 0 and rec_fb["status"] == StageRunStatus.REAL_RUNTIME_PASS.value
    assert (tmp_path / "formal_out" / "formal_b_statistics.json").is_file()


# ---- 6. P0-6 Trace-Before-Commit, Truncation, Config Drift, and Full-Run Resume Audit ----

def test_p06_trace_failure_never_commits_and_resume_rejects_truncation_and_config_drift(
    synthetic_dream1k_dir: Path, tmp_path: Path
) -> None:
    recs = load_dream1k_records(synthetic_dream1k_dir / "json" / "metadata.json", synthetic_dream1k_dir / "video")
    out_dir = tmp_path / "p06_trace_fail"
    # MockGeneratorAdapter returns a non-null GenerationOutput with wrong checkpoint -> orchestrator verifies AFTER generate_one!
    s = run_natural_retrieval_stage(
        RunSpec("p06_1", PilotStage.PILOT_A1, "DREAM-1K", [ExpArm.ARM_A0], out_dir, allow_cpu_mock=True),
        [recs[0]], MockXCLIPRetriever(), MockGeneratorAdapter(fault_mode="wrong_checkpoint"), skip_video_decode=True,
    )
    assert s["status"] == StageRunStatus.FAILED.value
    # CRITICAL P0-6 PROOF: zero rows written to predictions.jsonl or input_traces.jsonl!
    assert read_jsonl(out_dir / "predictions.jsonl") == []
    assert read_jsonl(out_dir / "input_traces.jsonl") == []
    assert len(read_jsonl(out_dir / "failures.jsonl")) == 1

    # Clean partial run (query 0 only), then resume with DIFFERENT checkpoint -> MUST raise ResumeIntegrityError!
    out_res = tmp_path / "p06_resume"
    spec1 = RunSpec("p06_r", PilotStage.PILOT_A1, "DREAM-1K", [ExpArm.ARM_A0], out_res, allow_cpu_mock=True)
    run_natural_retrieval_stage(spec1, [recs[0]], MockXCLIPRetriever(), MockGeneratorAdapter(expected_checkpoint_sha256="ckpt_v1"), skip_video_decode=True)
    with pytest.raises(ResumeIntegrityError, match="Config/model version drift"):
        run_natural_retrieval_stage(spec1, recs[:2], MockXCLIPRetriever(), MockGeneratorAdapter(expected_checkpoint_sha256="ckpt_v2"), skip_video_decode=True)

    # Truncated JSONL line on resume -> MUST raise ResumeIntegrityError!
    with (out_res / "predictions.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"query_id": "corrupt_truncated')
    with pytest.raises(ResumeIntegrityError, match="Truncated or malformed"):
        run_natural_retrieval_stage(spec1, recs[:2], MockXCLIPRetriever(), MockGeneratorAdapter(expected_checkpoint_sha256="ckpt_v1"), skip_video_decode=True)


# ---- 7. P0-7 Standalone E1 CLI Entrypoint Contract Test ----

def test_p07_e1_cli_entrypoint(tmp_path: Path) -> None:
    ra = atomic_write_json(tmp_path / "ra.json", {"q1": {"ARM_A1": 0.5, "ARM_A7": 0.7}, "q2": {"ARM_A1": 0.6, "ARM_A7": 0.4}})
    rb = atomic_write_json(tmp_path / "rb.json", {"q1": {"ARM_A1": 0.52, "ARM_A7": 0.68}, "q2": {"ARM_A1": 0.58, "ARM_A7": 0.42}})
    sm = atomic_write_json(tmp_path / "sm.json", {"q1": "S1", "q2": "S2"})
    args = build_parser().parse_args(
        [
            "--stage", "E1", "--paths-config", str(EXP_ROOT / "configs" / "bn_paths.example.yaml"),
            "--stage-config", str(EXP_ROOT / "configs" / "pilot_a01_config.yaml"),
            "--out-dir", str(tmp_path / "e1_out"), "--e1-ref-a", str(ra), "--e1-ref-b", str(rb), "--e1-source-map", str(sm),
        ]
    )
    rc, res = dispatch_stage(args)
    assert rc == 0 and res["n_queries"] == 2 and (tmp_path / "e1_out" / "e1_report.json").is_file()


def test_p06_full_run_resume_metrics_cover_earlier_empty_output(synthetic_dream1k_dir: Path, tmp_path: Path) -> None:
    recs = load_dream1k_records(synthetic_dream1k_dir / "json" / "metadata.json", synthetic_dream1k_dir / "video")
    out_dir = tmp_path / "p06_full_metrics"
    spec = RunSpec("p06_m", PilotStage.PILOT_A1, "DREAM-1K", [ExpArm.ARM_A0], out_dir, allow_cpu_mock=True)
    # Run 1: query 0 produces empty caption (still trace-valid, committed with ok=False)
    s1 = run_natural_retrieval_stage(spec, [recs[0]], MockXCLIPRetriever(), MockGeneratorAdapter(fault_mode="empty_output"), skip_video_decode=True)
    assert s1["empty_output_rate"] == 1.0 and s1["status"] == StageRunStatus.FAILED.value
    # Run 2: resume with query 0 + query 1 using clean generator -> query 0 skipped, query 1 clean;
    # FULL-RUN empty_output_rate MUST be 0.50 (1/2), NOT 0.0 (0/1 newly completed)!
    s2 = run_natural_retrieval_stage(spec, recs[:2], MockXCLIPRetriever(), MockGeneratorAdapter(), skip_video_decode=True)
    assert s2["resumed_skipped"] == 1 and s2["newly_completed"] == 1
    assert s2["empty_output_rate"] == pytest.approx(0.5, abs=1e-6)
    assert s2["status"] == StageRunStatus.FAILED.value


def test_p05_pilot_a2_human_annotation_return_and_a3_unverified_e3_pause(
    synthetic_dream1k_dir: Path, synthetic_vidhalluc_json: Path, tmp_path: Path
) -> None:
    recs = load_dream1k_records(synthetic_dream1k_dir / "json" / "metadata.json", synthetic_dream1k_dir / "video")
    real_a1 = atomic_write_json(tmp_path / "a1.json", {"status": StageRunStatus.REAL_RUNTIME_PASS.value, "execution_mode": "real_authorized"})
    auth_a2 = atomic_write_json(tmp_path / "auth_a2.json", {"authorized_stages": ["PILOT_A2"], "approved": True, "approval_id": "A2"})
    a1_preds = tmp_path / "a1_preds.jsonl"
    atomic_write_jsonl(
        a1_preds,
        [
            {"query_id": r.query_id, "arm": arm, "model_id": "vl", "caption": r.reference_description}
            for r in recs[:3]
            for arm in ("ARM_A0", "ARM_A7")
        ],
    )
    paths_cfg = tmp_path / "paths.yaml"
    (tmp_path / "empty.tsv").write_text("clip_id\tsource_id\n", encoding="utf-8")
    paths_cfg.write_text(
        yaml.safe_dump({"host": {}, "historical_frozen_assets": {"clips_full_tsv_path": str(tmp_path / "empty.tsv")}, "q2_writable_workspace": {"dream1k_dir": str(synthetic_dream1k_dir), "vidhalluc_dir": str(synthetic_vidhalluc_json.parent)}}),
        encoding="utf-8",
    )
    # 1. Supply adjudicated human annotations to Pilot A2 -> completes calibration (INCONCLUSIVE when 0 errors, human_calibration_completed=True)
    human_ann = atomic_write_json(
        tmp_path / "human_adj.json",
        [
            {"query_id": r.query_id, "arm": arm, "model_id": "vl", "claim_labels": [VideoTruthLabel.VIDEO_SUPPORTED.value] * 5}
            for r in recs[:3]
            for arm in ("ARM_A0", "ARM_A7")
        ],
    )
    args_a2 = build_parser().parse_args(
        [
            "--stage", "PILOT_A2", "--paths-config", str(paths_cfg),
            "--stage-config", str(EXP_ROOT / "configs" / "pilot_a23_config.yaml"),
            "--out-dir", str(tmp_path / "a2_cal"), "--auth-receipt", str(auth_a2),
            "--prior-a1", str(real_a1), "--a1-predictions", str(a1_preds), "--human-annotations", str(human_ann),
        ]
    )
    rc_a2, rec_a2 = dispatch_stage(args_a2, injected_ref_judge=MockJudgeBackend(model_family="mistral"))
    assert rec_a2["human_calibration_completed"] is True
    assert rec_a2["status"] == StageRunStatus.INCONCLUSIVE.value and rc_a2 == 2

    # 2. Pilot A3 WITHOUT --approved-e3-candidates pauses cleanly at exit code 10 (HUMAN_VALIDATION_PENDING)!
    real_a2_ok = atomic_write_json(tmp_path / "a2_ok.json", {"status": StageRunStatus.REAL_RUNTIME_PASS.value, "human_calibration_completed": True})
    auth_a3 = atomic_write_json(tmp_path / "auth_a3.json", {"authorized_stages": ["PILOT_A3"], "approved": True, "approval_id": "A3"})
    args_a3_wait = build_parser().parse_args(
        [
            "--stage", "PILOT_A3", "--paths-config", str(paths_cfg),
            "--stage-config", str(EXP_ROOT / "configs" / "pilot_a23_config.yaml"),
            "--out-dir", str(tmp_path / "a3_wait"), "--auth-receipt", str(auth_a3),
            "--prior-a2", str(real_a2_ok), "--limit", "2",
        ]
    )
    rc_a3w, rec_a3w = dispatch_stage(
        args_a3_wait,
        injected_retriever=MockXCLIPRetriever(),
        injected_videollama=MockGeneratorAdapter(model_id="vl"),
        injected_modern_vlm=MockGeneratorAdapter(model_id="vlm"),
        injected_e3_judge=MockJudgeBackend(model_family="mistral"),
    )
    assert rc_a3w == 10 and rec_a3w["status"] == StageRunStatus.HUMAN_VALIDATION_PENDING.value
    assert (tmp_path / "a3_wait" / "vidhalluc_candidates_pending_human_review.json").is_file()
