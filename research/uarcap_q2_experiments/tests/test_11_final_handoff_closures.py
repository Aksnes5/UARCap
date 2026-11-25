"""Tests 11: Final Offline Handoff Closures (4.1-4.6 + Section V + Qwen3.8-27B-FP8 vLLM Adapter).

Covers:
  1. Qwen3.8-27B-FP8 / Qwen3.5-27B vLLM multimodal adapter (offline engine, local HTTP, subprocess, thinking control,
     local shard check, 16-frame enforcement, E3 visual parity, PROMPT_TEXT_ONLY trace).
  2. A2 strict claim_id reconciliation (omitted claims, split/merge revisions, missing/duplicate/bundle-mismatch rejection,
     and bundle/auto-eval reuse on re-entry without re-invoking the LLM judge).
  3. A3 full human evaluation closed-loop (E3 blind export -> pause exit 10 -> Y_true & Y_conflict independent agreement,
     Con vs Neu differential bias detection, and cross-model DREAM-1K bias check).
  4. Formal B stratified probability sampling + near-dup/Pilot/same-source isolation + full stats (A1-A0, A7-A1, A7-A0,
     Core/Unmentioned, Two-Phase IPW, E3 secondary comparisons, real E3-Bridge execution, MSR-VTT bridge, NO `or 0.0` coercion!).
  5. Isolated X-CLIP & Text Judge subprocess interfaces + shared GPU lock + VRAM release.
  6. One-click `stage_assets.py prepare` (Route-2 chunk import -> safe extract -> normalize -> validate + zip-slip block).
  7. End-to-end Web UI export -> `adjudicate_annotations.py` (both E2 & E3) -> pipeline ingestion.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import zipfile
from typing import Any

import pytest
import yaml

EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from deployment.run_stage_cli import build_parser, dispatch_stage
from deployment.stage_assets import main as stage_assets_main
from scripts.adjudicate_annotations import main as adjudicate_cli_main
from uarcap_q2.annotation.web_app import AnnotationStore
from uarcap_q2.common.io_utils import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl, sha256_file
from uarcap_q2.common.schemas import (
    ExpArm,
    PilotStage,
    QueryRecord,
    RefRelation,
    SourceIdStatus,
    StageRunStatus,
    VerdictStatus,
    VideoTruthLabel,
)
from uarcap_q2.datasets.asset_transfer import split_file_into_chunks
from uarcap_q2.datasets.data_preparer import UnsafeArchiveError, safe_extract_zip
from uarcap_q2.datasets.dream1k import load_dream1k_records
from uarcap_q2.datasets.formal_sampling import (
    FormalSamplingConfig,
    build_formal_dream1k_manifest,
    build_formal_vidhalluc_manifest,
)
from uarcap_q2.datasets.video_decoder import decode_and_sample_video
from uarcap_q2.datasets.vidhalluc import load_vidhalluc_ach_mcq
from uarcap_q2.evaluators.backends import (
    JudgeBackendConfig,
    LocalVLLMJudgeBackend,
    MockJudgeBackend,
    SubprocessTransformersJudgeBackend,
)
from uarcap_q2.evaluators.calibration import (
    AlignedAtomRecord,
    audit_e3_evaluator_agreement,
    reconcile_a2_human_and_auto,
)
from uarcap_q2.generators.base import BatchItem, TraceVerificationError
from uarcap_q2.generators.mock_generator import MockGeneratorAdapter
from uarcap_q2.generators.qwen3_vllm_adapter import (
    Qwen3VLLMConfig,
    Qwen3VLLMMultimodalAdapter,
    strip_thinking_block,
)
from uarcap_q2.interventions.e3_bridge import NaturalConflictObservation
from uarcap_q2.interventions.vidhalluc_e3 import (
    E3ApplicabilityChecklist,
    apply_human_applicability_review,
    build_vidhalluc_e3_candidate,
)
from uarcap_q2.retrieval.arm_builder import build_e3_intervention_arms
from uarcap_q2.retrieval.reranker import MockXCLIPRetriever
from uarcap_q2.retrieval.videoufo_retriever import SubprocessVideoUFOXCLIPRetriever, VideoUFOIndexConfig
from uarcap_q2.runner.stage_pipelines import run_formal_b_pipeline, run_pilot_a2_pipeline, run_pilot_a3_pipeline


# ---- 1. Qwen3.8-27B-FP8 / Qwen3.5-27B vLLM Multimodal Video Adapter ----

class FakeVLLMEngine:
    def __init__(self, emit_think: bool = False) -> None:
        self.emit_think = emit_think
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages_batch: list[Any], sampling_params: Any = None, chat_template_kwargs: Any = None) -> list[dict[str, Any]]:
        self.calls.append({"n": len(messages_batch), "chat_template_kwargs": chat_template_kwargs})
        outs = []
        for msgs in messages_batch:
            # Verify video payload arrived cleanly
            user_c = msgs[1]["content"]
            assert any(part.get("type") == "video" and part["video"].shape[0] == 16 for part in user_c)
            txt = "<think>internal reasoning</think>A person is throwing grass in the garbage." if self.emit_think else "A person is throwing grass in the garbage."
            outs.append({"text": txt, "prompt_tokens": 120, "completion_tokens": 10})
        return outs


def test_qwen3_8_27b_fp8_vllm_multimodal_adapter_thinking_control_and_trace(
    pathological_videos: dict[str, Path], tmp_path: Path
) -> None:
    # Build synthetic local model dir with config.json + 2/2 shards verified
    mdir = tmp_path / "Qwen3.8-27B-FP8"
    mdir.mkdir()
    atomic_write_json(mdir / "config.json", {"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"], "vision_config": {"patch_size": 14}})
    atomic_write_json(mdir / "model.safetensors.index.json", {"weight_map": {"w1": "s1.safetensors", "w2": "s2.safetensors"}})
    (mdir / "s1.safetensors").write_bytes(b"s1")
    (mdir / "s2.safetensors").write_bytes(b"s2")

    imgs, rep = decode_and_sample_video(pathological_videos["dynamic_24f"], num_frames=16)
    q = QueryRecord("VidHalluc-ACH", "vh_q1", "vcCwvRYqU2I", SourceIdStatus.VERIFIED_EXACT.value, str(pathological_videos["dynamic_24f"]), rep.video_sha256, 2.0, "ACH", "throwing grass")
    e3_pkts = build_e3_intervention_arms("vh_q1", "throwing grass", "raking leaves", "standing outside", base_visual_mode="BASE_A0")

    # 1a. Offline vLLM engine mode + thinking control + <think> stripping + E3 visual parity
    fake_eng = FakeVLLMEngine(emit_think=True)
    adapter = Qwen3VLLMMultimodalAdapter(
        Qwen3VLLMConfig(
            model_id="Qwen/Qwen3.8-27B-FP8",
            local_model_path=str(mdir),
            backend_mode="vllm_offline",
            enable_thinking=False,
            expected_num_shards=2,
        ),
        vllm_engine=fake_eng,
    )
    # Missing frames MUST fail!
    with pytest.raises(TraceVerificationError, match="NO/invalid decoded video frames"):
        adapter.generate_one(q, e3_pkts["E3_BASE"], PilotStage.PILOT_A3, rep.combined_visual_sha256, rep.sampled_indices, frame_images=None)

    batch = [BatchItem(q, e3_pkts[a], rep.combined_visual_sha256, rep.sampled_indices, imgs) for a in ("E3_BASE", "E3_SUP", "E3_CON", "E3_NEU")]
    outs = adapter.generate_batch(batch, PilotStage.PILOT_A3)
    assert len(outs) == 4 and len(fake_eng.calls) == 1  # Single batch call!
    assert fake_eng.calls[0]["chat_template_kwargs"] == {"enable_thinking": False}
    vshas = set()
    for o in outs:
        assert o.ok and o.caption == "A person is throwing grass in the garbage."
        assert o.input_trace.injection_interface == "PROMPT_TEXT_ONLY"
        assert o.input_trace.visual_evidence_ids == []
        tel = o.input_trace.actual_processor_telemetry
        assert tel["enable_thinking"] is False and tel["thinking_tags_stripped"] is True
        assert tel["actual_frames_received"] == 16 and tel["actual_frames_sha256"] == rep.combined_visual_sha256
        vshas.add(tel["visual_payload_sha256"])
    assert len(vshas) == 1  # Exact identical visual payload across all 4 E3 arms!

    # 1b. Local OpenAI-compatible vLLM HTTP mode (16 JPEG data URLs + enable_thinking=False)
    seen_payloads: list[dict[str, Any]] = []

    def _fake_http(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        seen_payloads.append(payload)
        return {"choices": [{"message": {"content": "A person throws grass into a bin."}}], "usage": {"prompt_tokens": 90, "completion_tokens": 8}}

    http_adapter = Qwen3VLLMMultimodalAdapter(
        Qwen3VLLMConfig(
            model_id="Qwen/Qwen3.5-27B",
            local_model_path=str(mdir),
            backend_mode="vllm_openai_http",
            api_base_url="http://127.0.0.1:8000/v1",
            enable_thinking=False,
        ),
        http_requester=_fake_http,
    )
    o_http = http_adapter.generate_one(q, e3_pkts["E3_CON"], PilotStage.PILOT_A3, rep.combined_visual_sha256, rep.sampled_indices, frame_images=imgs)
    assert o_http.ok and seen_payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}
    img_parts = [p for p in seen_payloads[0]["messages"][1]["content"] if p.get("type") == "image_url"]
    assert len(img_parts) == 16

    # 1c. Missing shard detection
    (mdir / "s2.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete local model shards"):
        Qwen3VLLMMultimodalAdapter(
            Qwen3VLLMConfig(local_model_path=str(mdir), backend_mode="vllm_offline"), vllm_engine=fake_eng
        ).generate_one(q, e3_pkts["E3_BASE"], PilotStage.PILOT_A3, rep.combined_visual_sha256, rep.sampled_indices, frame_images=imgs)

    # 1d. Unclosed <think> stripping safety
    cleaned, had_t, _ = strip_thinking_block("<think>truncated thought without end")
    assert cleaned == "" and had_t is True


# ---- 2. A2 Strict Claim-ID Reconciliation, Revisions & Bundle Reuse ----

def test_a2_strict_claim_id_reconciliation_omissions_split_merge_and_bundle_reuse(
    synthetic_dream1k_dir: Path, tmp_path: Path
) -> None:
    recs = load_dream1k_records(synthetic_dream1k_dir / "json" / "metadata.json", synthetic_dream1k_dir / "video")
    a1_preds = tmp_path / "a1_preds.jsonl"
    atomic_write_jsonl(
        a1_preds,
        [
            {"query_id": recs[0].query_id, "arm": "ARM_A0", "model_id": "vl", "caption": "A person walks and waves near a car."},
            {"query_id": recs[0].query_id, "arm": "ARM_A7", "model_id": "vl", "caption": "A person is raking leaves and standing near a bicycle."},
        ],
    )

    class CountingJudge(MockJudgeBackend):
        def __init__(self) -> None:
            super().__init__(model_family="mistral")
            self.call_count = 0

        def complete(self, system_prompt: str, user_prompt: str) -> str:
            self.call_count += 1
            return super().complete(system_prompt, user_prompt)

    judge = CountingJudge()
    out_dir = tmp_path / "a2_reuse"
    r1 = run_pilot_a2_pipeline(
        a1_predictions_path=a1_preds,
        dream1k_annotation_path=synthetic_dream1k_dir / "json" / "metadata.json",
        dream1k_video_root=synthetic_dream1k_dir / "video",
        judge_backend=judge,
        out_dir=out_dir,
        allow_cpu_mock=True,
    )
    assert r1["status"] == StageRunStatus.HUMAN_VALIDATION_PENDING.value
    calls_after_first = judge.call_count
    assert calls_after_first > 0

    eval_rows = read_jsonl(out_dir / "a2_ref_fact_eval.jsonl")
    pkg_sha = r1["package_sha256"]

    # Reject wrong bundle SHA-256
    with pytest.raises(ValueError, match="bundle version mismatch"):
        reconcile_a2_human_and_auto(eval_rows, {"package_sha256": "wrong_sha", "records": []}, expected_package_sha256=pkg_sha)

    # Reject missing claim_id
    ev0 = eval_rows[0]
    with pytest.raises(ValueError, match="missing claim_id"):
        reconcile_a2_human_and_auto(
            [ev0],
            {"package_sha256": pkg_sha, "records": [{"query_id": ev0["query_id"], "arm": ev0["arm"], "model_id": ev0["model_id"], "claim_labels_by_id": {}}]},
            expected_package_sha256=pkg_sha,
        )

    # Reject duplicate claim_id
    cid0 = ev0["claims"][0]["claim_id"]
    with pytest.raises(ValueError, match="Duplicate human claim_id"):
        reconcile_a2_human_and_auto(
            [ev0],
            {
                "package_sha256": pkg_sha,
                "records": [
                    {
                        "query_id": ev0["query_id"],
                        "arm": ev0["arm"],
                        "model_id": ev0["model_id"],
                        "claim_annotations": [
                            {"claim_id": cid0, "label": VideoTruthLabel.VIDEO_SUPPORTED.value},
                            {"claim_id": cid0, "label": VideoTruthLabel.VIDEO_CONTRADICTED.value},
                        ],
                    }
                ],
            },
            expected_package_sha256=pkg_sha,
        )

    # Valid reconciliation WITH split revision + added omitted error claim -> re-entering A2 REUSES eval (0 new judge calls!)
    ev1 = eval_rows[1]
    c1_ids = [c["claim_id"] for c in ev1["claims"]]
    human_payload = {
        "package_sha256": pkg_sha,
        "records": [
            {
                "query_id": ev0["query_id"],
                "arm": ev0["arm"],
                "model_id": ev0["model_id"],
                "claim_labels_by_id": {c["claim_id"]: VideoTruthLabel.VIDEO_SUPPORTED.value for c in ev0["claims"]},
                "added_claims": [{"text": "A hallucinated red truck appears", "label": VideoTruthLabel.VIDEO_CONTRADICTED.value}],
            },
            {
                "query_id": ev1["query_id"],
                "arm": ev1["arm"],
                "model_id": ev1["model_id"],
                "claim_labels_by_id": {cid: VideoTruthLabel.VIDEO_SUPPORTED.value for cid in c1_ids[1:]},
                "claim_revisions": [
                    {
                        "revision_type": "split",
                        "source_claim_ids": [c1_ids[0]],
                        "resulting_claims": [
                            {"claim_id": c1_ids[0] + "_S1", "text": "a person", "label": VideoTruthLabel.VIDEO_SUPPORTED.value},
                            {"claim_id": c1_ids[0] + "_S2", "text": "raking leaves", "label": VideoTruthLabel.VIDEO_CONTRADICTED.value},
                        ],
                    }
                ],
            },
        ],
    }
    h_path = atomic_write_json(tmp_path / "a2_human_reconciled.json", human_payload)
    r2 = run_pilot_a2_pipeline(
        a1_predictions_path=a1_preds,
        dream1k_annotation_path=synthetic_dream1k_dir / "json" / "metadata.json",
        dream1k_video_root=synthetic_dream1k_dir / "video",
        judge_backend=judge,
        out_dir=out_dir,
        human_annotations_path=h_path,
        allow_cpu_mock=True,
    )
    assert judge.call_count == calls_after_first  # ZERO re-generation of claims/bundle!
    assert r2["reused_existing_auto_eval_and_bundle"] is True
    assert r2["reconciliation_audit"]["n_split_revisions"] == 1
    assert r2["reconciliation_audit"]["n_added_omitted_claims"] == 1
    assert r2["reconciliation_audit"]["n_omitted_video_errors"] == 1


# ---- 3. A3 E3 Agreement + Con vs Neu Differential Bias + Cross-Model DREAM-1K Bias ----

def test_a3_e3_differential_bias_detection_and_cross_model_dream1k_audit(
    synthetic_dream1k_dir: Path, synthetic_vidhalluc_json: Path, pathological_videos: dict[str, Path], tmp_path: Path
) -> None:
    # Detect severe Con vs Neu differential bias cleanly -> FAIL!
    auto_rows = [
        {"query_id": "q1", "arm": "E3_CON", "model_id": "m1", "y_true": 0, "y_conflict": 1, "parse_status": "OK"},
        {"query_id": "q2", "arm": "E3_CON", "model_id": "m1", "y_true": 0, "y_conflict": 1, "parse_status": "OK"},
        {"query_id": "q1", "arm": "E3_NEU", "model_id": "m1", "y_true": 1, "y_conflict": 0, "parse_status": "OK"},
        {"query_id": "q2", "arm": "E3_NEU", "model_id": "m1", "y_true": 1, "y_conflict": 0, "parse_status": "OK"},
    ]
    # Human says NO conflict on Con (delta_human = 0.0 while delta_auto = +1.0 -> bias = +1.0 -> FAIL!)
    human_rows = [
        {"query_id": "q1", "arm": "E3_CON", "model_id": "m1", "y_true": 1, "y_conflict": 0},
        {"query_id": "q2", "arm": "E3_CON", "model_id": "m1", "y_true": 1, "y_conflict": 0},
        {"query_id": "q1", "arm": "E3_NEU", "model_id": "m1", "y_true": 1, "y_conflict": 0},
        {"query_id": "q2", "arm": "E3_NEU", "model_id": "m1", "y_true": 1, "y_conflict": 0},
    ]
    biased = audit_e3_evaluator_agreement(auto_rows, human_rows)
    assert biased["verdict"] == VerdictStatus.FAIL.value
    assert biased["con_vs_neu_differential_bias"]["y_conflict"]["differential_bias"] == 1.0

    # Full A3 pipeline WITH cross-model DREAM-1K bias check
    recs = load_dream1k_records(synthetic_dream1k_dir / "json" / "metadata.json", synthetic_dream1k_dir / "video")
    vh_dir = tmp_path / "vh_a3"
    (vh_dir / "data").mkdir(parents=True)
    (vh_dir / "ach_mcq.json").write_text(synthetic_vidhalluc_json.read_text(encoding="utf-8"), encoding="utf-8")
    vh_recs = load_vidhalluc_ach_mcq(vh_dir / "ach_mcq.json", vh_dir / "data")
    c0 = apply_human_applicability_review(
        build_vidhalluc_e3_candidate(vh_recs[0], neu_text="In this video clip, a person is outside in daylight."),
        E3ApplicabilityChecklist(True, True, True, True, True, False, "A1"),
    )
    (vh_dir / "data" / f"{vh_recs[0].metadata['clip_key']}.mp4").write_bytes(pathological_videos["dynamic_24f"].read_bytes())
    appr = atomic_write_json(tmp_path / "appr.json", [c0.to_dict()])

    a3_dir = tmp_path / "a3_full"
    r_wait = run_pilot_a3_pipeline(
        dream1k_pilot_queries=recs[:2],
        retriever=MockXCLIPRetriever(),
        modern_vlm=MockGeneratorAdapter(model_id="qwen3_8_mock"),
        videollama_adapter=MockGeneratorAdapter(model_id="vl_mock"),
        vidhalluc_ach_mcq_path=vh_dir / "ach_mcq.json",
        vidhalluc_video_root=vh_dir / "data",
        e3_judge_backend=MockJudgeBackend(model_family="mistral"),
        out_dir=a3_dir,
        videoufo_source_ids=set(),
        approved_e3_candidates_path=appr,
        require_cross_model_dream1k_human=True,
        allow_cpu_mock=True,
        skip_video_decode=True,
    )
    assert r_wait["status"] == StageRunStatus.HUMAN_VALIDATION_PENDING.value

    e3_evals = read_jsonl(a3_dir / "e3_dual_endpoint_eval.jsonl")
    vlm_evals = read_jsonl(a3_dir / "dream1k_vlm_ref_eval.jsonl")
    e3_h = atomic_write_json(
        tmp_path / "e3_h.json",
        {"package_sha256": r_wait["e3_package_sha256"], "records": [{"query_id": r["query_id"], "arm": r["arm"], "model_id": r["model_id"], "y_true": r["y_true"], "y_conflict": r["y_conflict"]} for r in e3_evals]},
    )
    vlm_h = atomic_write_json(
        tmp_path / "vlm_h.json",
        {
            "package_sha256": r_wait["dream1k_vlm_package_sha256"],
            "records": [
                {"query_id": ev["query_id"], "arm": ev["arm"], "model_id": ev["model_id"], "claim_labels_by_id": {c["claim_id"]: VideoTruthLabel.VIDEO_CONTRADICTED.value if c["label"] == "contradicted" else VideoTruthLabel.VIDEO_SUPPORTED.value for c in ev["claims"]}}
                for ev in vlm_evals
            ],
        },
    )
    vl_aligned_p = atomic_write_json(
        tmp_path / "vl_aligned.json",
        [{"video_id": "v1", "arm": "ARM_A0", "model_id": "vl_mock", "ref_label": RefRelation.CONTRADICTED.value, "human_label": VideoTruthLabel.VIDEO_CONTRADICTED.value}],
    )
    r_done = run_pilot_a3_pipeline(
        dream1k_pilot_queries=recs[:2],
        retriever=MockXCLIPRetriever(),
        modern_vlm=MockGeneratorAdapter(model_id="qwen3_8_mock"),
        videollama_adapter=MockGeneratorAdapter(model_id="vl_mock"),
        vidhalluc_ach_mcq_path=vh_dir / "ach_mcq.json",
        vidhalluc_video_root=vh_dir / "data",
        e3_judge_backend=MockJudgeBackend(model_family="mistral"),
        out_dir=a3_dir,
        videoufo_source_ids=set(),
        approved_e3_candidates_path=appr,
        e3_human_annotations_path=e3_h,
        dream1k_vlm_human_annotations_path=vlm_h,
        a2_aligned_records_path=vl_aligned_p,
        require_cross_model_dream1k_human=True,
        allow_cpu_mock=True,
        skip_video_decode=True,
    )
    assert r_done["e3_human_validation_completed"] is True
    assert r_done["cross_model_dream1k_bias_checked"] is True
    assert (a3_dir / "a3_evaluator_validation_report.json").is_file()


# ---- 4. Formal B Sampling/Isolation + Full Stats (NO `or 0.0`!) + Two-Phase + E3-Bridge + MSR-VTT Bridge ----

def test_formal_b_sampling_full_stats_no_zero_coercion_e3_bridge_and_msrvtt(
    synthetic_dream1k_dir: Path, synthetic_vidhalluc_json: Path, pathological_videos: dict[str, Path], tmp_path: Path
) -> None:
    recs = load_dream1k_records(synthetic_dream1k_dir / "json" / "metadata.json", synthetic_dream1k_dir / "video")
    # Inject a near-duplicate DREAM-1K record and verify deterministic exclusion
    from dataclasses import replace

    dup_rec = replace(recs[2], query_id="dream1k_dup_999")
    sel_d, d_man = build_formal_dream1k_manifest(
        recs + [dup_rec],
        pilot_excluded_ids={recs[0].query_id},
        cfg=FormalSamplingConfig(dream1k_target_n=6, e2_validation_target_n=3, seed=42),
        out_manifest_path=tmp_path / "d_man.json",
    )
    assert recs[0].query_id not in {r.query_id for r in sel_d}
    assert d_man["n_auto_near_dup_excluded"] >= 1
    assert d_man["e2_validation_set"]["disjoint_from_pilot_verified"] is True
    assert len(d_man["e2_validation_set"]["validation_query_ids"]) == 3

    # VidHalluc same-source + Pilot source exclusion
    vh_raw = load_vidhalluc_ach_mcq(synthetic_vidhalluc_json, synthetic_vidhalluc_json.parent / "data")
    pilot_src = vh_raw[0].source_id
    sel_vh, vh_man = build_formal_vidhalluc_manifest(
        vh_raw, videoufo_source_ids=set(), pilot_excluded_query_ids=set(), pilot_excluded_source_ids={pilot_src} if pilot_src else set(),
        cfg=FormalSamplingConfig(seed=42),
    )
    assert all(r.source_id != pilot_src for r in sel_vh)
    assert vh_man["same_source_isolated"] is True

    # Run full Formal B with:
    # - 1 query deliberately failing judge parse (proving NO `or 0.0` coercion!)
    # - E2 validation human annotations for Two-Phase IPW correction
    # - Enabled E3-Bridge (min_clean_pairs=1) actually executed on both models!
    # - MSR-VTT historical bridge JSONs!
    class OneParseFailJudge(MockJudgeBackend):
        def complete(self, system_prompt: str, user_prompt: str) -> str:
            if "[TASK:3WAY_REF_NLI]" in user_prompt and "FAIL_PARSE_TRIGGER" in user_prompt:
                return "{broken json!!!"
            return super().complete(system_prompt, user_prompt)

    class TriggerCaptionGen(MockGeneratorAdapter):
        def generate_one(self, query, packet, stage, visual_input_sha256, frame_indices, frame_images=None):
            out = super().generate_one(query, packet, stage, visual_input_sha256, frame_indices, frame_images)
            if query.query_id == recs[1].query_id and packet.arm == ExpArm.ARM_A7.value:
                out.caption = "FAIL_PARSE_TRIGGER causes judge error."
            return out

    formal_cfg = EXP_ROOT / "configs" / "formal_b_candidate.yaml"
    asset_man = EXP_ROOT / "DATA_AND_MODEL_ASSET_MANIFEST.json"
    pilot_excl = atomic_write_json(tmp_path / "p_excl.json", {"excluded_query_ids": [recs[0].query_id]})
    a1_r = atomic_write_json(tmp_path / "a1.json", {"status": StageRunStatus.REAL_RUNTIME_PASS.value})
    a2_r = atomic_write_json(tmp_path / "a2.json", {"status": StageRunStatus.REAL_RUNTIME_PASS.value, "human_calibration_completed": True})
    a3_r = atomic_write_json(tmp_path / "a3.json", {"status": StageRunStatus.REAL_RUNTIME_PASS.value, "e3_human_validation_completed": True})
    fb_auth = atomic_write_json(
        tmp_path / "fb_auth.json",
        {
            "stage": "FORMAL_B",
            "approved": True,
            "approval_id": "FB_TEST",
            "frozen_pilot_params_confirmed": True,
            "frozen_formal_config_sha256": sha256_file(formal_cfg),
            "evaluator_version": "mistral-3.2-24b-frozen",
            "asset_manifest_sha256": sha256_file(asset_man),
            "pilot_exclusion_manifest_sha256": sha256_file(pilot_excl),
        },
    )

    vh_dir = tmp_path / "vh_fb"
    (vh_dir / "data").mkdir(parents=True)
    (vh_dir / "ach_mcq.json").write_text(synthetic_vidhalluc_json.read_text(encoding="utf-8"), encoding="utf-8")
    vh_recs = load_vidhalluc_ach_mcq(vh_dir / "ach_mcq.json", vh_dir / "data")
    c0 = apply_human_applicability_review(
        build_vidhalluc_e3_candidate(vh_recs[0], neu_text="In this video clip, a person is outside in daylight."),
        E3ApplicabilityChecklist(True, True, True, True, True, False, "A1"),
    )

    # E3-Bridge natural conflict observation on a selected formal DREAM-1K clip
    bridge_obs = [
        NaturalConflictObservation(
            query_id=recs[2].query_id,
            domain=recs[2].domain,
            verified_true_fact="a person walks across the yard",
            natural_videoufo_caption="A person is swimming in a pool.",
            natural_conflict_fact="a person swims in a pool",
            single_target_clean=True,
            conflict_strictly_absent_in_video=True,
            proposed_neu_text="In the video, the outdoor scene stays bright.",
            reviewer_id="A1",
        )
    ]

    # Pre-build E2 validation human annotations for every eligible clip so Two-Phase IPW runs cleanly
    val_human = atomic_write_json(
        tmp_path / "e2_val_human.json",
        {
            "records": [
                {"query_id": r.query_id, "arm": arm, "model_id": mid, "claim_labels_by_id": {"c1": VideoTruthLabel.VIDEO_SUPPORTED.value}}
                for r in recs[1:6]
                for arm in ("ARM_A0", "ARM_A7")
                for mid in ("vl_fb", "vlm_fb")
            ]
        },
    )
    ra = atomic_write_json(tmp_path / "ra.json", {"q1": {"ARM_A1": 0.5, "ARM_A7": 0.7}})
    rb = atomic_write_json(tmp_path / "rb.json", {"q1": {"ARM_A1": 0.55, "ARM_A7": 0.65}})
    sm = atomic_write_json(tmp_path / "sm.json", {"q1": "S1"})

    out_fb = tmp_path / "formal_b_full"
    rec = run_formal_b_pipeline(
        formal_dream1k_queries=recs[:6],
        formal_vidhalluc_queries_by_id={r.query_id: r for r in vh_recs},
        approved_e3_candidates=[c0],
        retriever=MockXCLIPRetriever(),
        videollama_adapter=TriggerCaptionGen(model_id="vl_fb"),
        modern_vlm=TriggerCaptionGen(model_id="vlm_fb"),
        ref_judge_backend=OneParseFailJudge(model_family="mistral"),
        e3_judge_backend=MockJudgeBackend(model_family="mistral"),
        out_dir=out_fb,
        auth_receipt_path=fb_auth,
        prior_receipts={"PILOT_A1": a1_r, "PILOT_A2": a2_r, "PILOT_A3": a3_r},
        formal_config_path=formal_cfg,
        asset_manifest_path=asset_man,
        pilot_exclusion_path=pilot_excl,
        pilot_excluded_query_ids={recs[0].query_id},
        e3_bridge_observations=bridge_obs,
        sampling_cfg=FormalSamplingConfig(dream1k_target_n=5, e2_validation_target_n=5, seed=7),
        e2_validation_human_annotations_path=val_human,
        msrvtt_historical_bridge={"e1_ref_a": ra, "e1_ref_b": rb, "e1_source_map": sm},
        e3_bridge_min_clean_pairs=1,
        skip_video_decode=True,
    )
    assert rec["status"] == StageRunStatus.REAL_RUNTIME_PASS.value
    assert rec["e3_bridge"]["executed"] is True
    stats = read_json(out_fb / "formal_b_statistics.json")

    # Verify parse failure was NOT coerced to 0.0!
    vl_e2 = stats["e2_by_model"]["videollama"]
    assert vl_e2["parse_failures_total"] == 1
    a7_a1_contra = vl_e2["paired_contrasts"]["n_contra_ref"]["a7_minus_a1"]
    assert a7_a1_contra["missingness"]["dropped_missing_or_parse_failed_n"] == 1
    assert recs[1].query_id in a7_a1_contra["missingness"]["arm_a"]["parse_failed_ids"]

    # Verify A1-A0, A7-A1, Core, Unmentioned, Two-Phase IPW, E3 secondary, E3-Bridge, & MSR-VTT bridge are ALL present!
    assert "a1_minus_a0" in vl_e2["paired_contrasts"]["n_entail_ref"]
    assert "r_core_ref" in vl_e2["paired_contrasts"] and "p_unment_ref" in vl_e2["paired_contrasts"]
    assert stats["e2_two_phase_error_correction"]["videollama"]["status"] == "OK"
    assert "sup_vs_neu" in stats["e3_by_model"]["videollama"]["secondary_comparisons"]
    assert stats["e3_bridge"]["executed"] is True and "videollama" in stats["e3_bridge"]["stats"]
    assert stats["msrvtt_historical_bridge"]["status"] == "COMPUTED_FROM_FROZEN_PER_QUERY_JSONS"


# ---- 5. Isolated X-CLIP & Judge Subprocess Interfaces + Shared GPU Lock ----

def test_isolated_xclip_and_judge_subprocess_interfaces_and_gpu_lock(tmp_path: Path) -> None:
    lock_p = tmp_path / "gpu.lock"
    xclip_calls: list[list[str]] = []

    def _fake_xclip_runner(argv, cwd, env, capture_output, text, timeout):
        xclip_calls.append(argv)
        req_p = Path(argv[argv.index("--request-json") + 1])
        resp_p = Path(argv[argv.index("--response-json") + 1])
        req = read_json(req_p)
        atomic_write_json(
            resp_p,
            {
                "status": "OK",
                "manifest": {"retriever_id": "subproc_xclip"},
                "results": {
                    q["query_id"]: [{"caption": "A clip caption.", "score": 0.85, "clip_ids": ["C001.0"], "source_ids": ["C001"], "original_rank": 0, "rerank_score": 0.0}]
                    for q in req["queries"]
                },
            },
        )
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    ret = SubprocessVideoUFOXCLIPRetriever(
        cfg=VideoUFOIndexConfig("a.csv", "b.tsv", "c.npy"),
        python_executable=sys.executable,
        gpu_lock_path=lock_p,
        require_nvidia_smi_idle=False,
        subprocess_runner=_fake_xclip_runner,
    )
    q1 = QueryRecord("DREAM-1K", "q1", None, SourceIdStatus.UNKNOWN_OPAQUE.value, "v1.mp4", "", 2.0, "d", "ref")
    q2 = QueryRecord("DREAM-1K", "q2", None, SourceIdStatus.UNKNOWN_OPAQUE.value, "v2.mp4", "", 2.0, "d", "ref")
    ret.prefetch_batch([q1, q2])
    assert len(xclip_calls) == 1  # Single batch subprocess call for both queries!
    assert ret.retrieve_top20(q1)[0].caption == "A clip caption."

    # Text judge subprocess + LocalVLLMJudgeBackend
    def _fake_judge_runner(argv, cwd, env, capture_output, text, timeout):
        req_p = Path(argv[argv.index("--requests-jsonl") + 1])
        resp_p = Path(argv[argv.index("--responses-jsonl") + 1])
        reqs = read_jsonl(req_p)
        atomic_write_jsonl(resp_p, [{"request_id": r["request_id"], "response": '{"claims": ["ok"]}'} for r in reqs])
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    jb = SubprocessTransformersJudgeBackend(
        JudgeBackendConfig(backend_type="vllm_offline", model_id="mistralai/Mistral-Small-3.2-24B-Instruct-2506", model_family="mistral", local_model_path=str(tmp_path)),
        python_executable=sys.executable,
        gpu_lock_path=lock_p,
        require_nvidia_smi_idle=False,
        subprocess_runner=_fake_judge_runner,
    )
    assert jb.complete("sys", "usr") == '{"claims": ["ok"]}'

    class FakeTextVLLM:
        def chat(self, convs, sampling_params=None, chat_template_kwargs=None):
            return [{"text": "vllm judge ok"} for _ in convs]

    lj = LocalVLLMJudgeBackend(JudgeBackendConfig("vllm_offline", "mistral-3.2-24b", "mistral"), vllm_engine=FakeTextVLLM())
    assert lj.complete("sys", "usr") == "vllm judge ok"


# ---- 6. One-Click Automated Data Preparation & Zip-Slip Protection ----

def test_one_click_data_prepare_safe_extract_normalize_and_zipslip_block(
    synthetic_dream1k_dir: Path, tmp_path: Path
) -> None:
    # 6a. Zip-slip traversal MUST be blocked!
    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("../escape.txt", "pwned")
    with pytest.raises(UnsafeArchiveError, match="Blocked unsafe zip member"):
        safe_extract_zip(evil_zip, tmp_path / "extract_evil")

    # 6b. Full Route-2 chunk import -> safe extract -> nested dir normalize -> schema/correspondence check -> READY!
    src_zip = tmp_path / "video.zip"
    with zipfile.ZipFile(src_zip, "w") as zf:
        for mp4 in sorted((synthetic_dream1k_dir / "video").glob("*.mp4")):
            # Intentionally nest inside `nested_video/` to test normalization!
            zf.write(mp4, arcname=f"nested_video/{mp4.name}")

    chunks_dir = tmp_path / "chunks"
    manifest_p = split_file_into_chunks(src_zip, chunks_dir, chunk_bytes=4096)

    data_root = tmp_path / "bn_data"
    (data_root / "dream1k" / "json").mkdir(parents=True)
    (data_root / "dream1k" / "json" / "metadata.json").write_text(
        (synthetic_dream1k_dir / "json" / "metadata.json").read_text(encoding="utf-8"), encoding="utf-8"
    )

    rc = stage_assets_main(
        [
            "prepare", "--data-root", str(data_root), "--dataset", "dream1k",
            "--transfer-manifest", str(manifest_p), "--skip-archive-size-check",
        ]
    )
    assert rc == 0
    readiness = read_json(data_root / "dream1k_readiness.json")
    assert readiness["ready"] is True and readiness["n_missing_or_empty_videos"] == 0
    assert stage_assets_main(["check-ready", "--data-root", str(data_root), "--dataset", "dream1k"]) == 0


# ---- 7. End-to-End Web UI Export -> Adjudication CLI -> A2 & A3 Ingestion ----

def test_end_to_end_web_ui_export_adjudication_cli_for_both_e2_and_e3(
    synthetic_dream1k_dir: Path, pathological_videos: dict[str, Path], tmp_path: Path
) -> None:
    recs = load_dream1k_records(synthetic_dream1k_dir / "json" / "metadata.json", synthetic_dream1k_dir / "video")
    a1_preds = tmp_path / "a1.jsonl"
    atomic_write_jsonl(a1_preds, [{"query_id": recs[0].query_id, "arm": "ARM_A0", "model_id": "vl", "caption": recs[0].reference_description}])
    a2_out = tmp_path / "a2_ui_loop"
    r_wait = run_pilot_a2_pipeline(
        a1_preds, synthetic_dream1k_dir / "json" / "metadata.json", synthetic_dream1k_dir / "video",
        MockJudgeBackend(model_family="mistral"), a2_out, allow_cpu_mock=True,
    )
    b_a1 = Path(r_wait["blind_annotation_files"]["tasks_A1"])
    b_a2 = Path(r_wait["blind_annotation_files"]["tasks_A2"])
    b_key = Path(r_wait["blind_annotation_files"]["blind_key"])

    # Simulate both annotators clicking in web UI AnnotationStore (with 1 intentional disagreement on first claim!)
    exports = {}
    for ann, first_lbl in (("A1", VideoTruthLabel.VIDEO_SUPPORTED.value), ("A2", VideoTruthLabel.VIDEO_CONTRADICTED.value)):
        bundle_p = b_a1 if ann == "A1" else b_a2
        store = AnnotationStore(bundle_p, tmp_path / "ui_saves")
        for t in store.bundle["tasks"]:
            cap_payload = {}
            for c in t["captions"]:
                cl_map = {cl["claim_id"]: (first_lbl if idx == 0 else VideoTruthLabel.VIDEO_SUPPORTED.value) for idx, cl in enumerate(c["claims"])}
                cap_payload[c["blind_caption_id"]] = {"claims": cl_map, "added_claims": [], "claim_revisions": []}
            store.save_record(t["blind_video_id"], {"task_key": t["blind_video_id"], "task_type": "E2_FACT_AUDIT", "captions": cap_payload, "elapsed_sec": 30, "completed": True})
        exp_p = tmp_path / f"export_{ann}.json"
        exp_p.write_bytes(store.export_json_bytes())
        exports[ann] = exp_p

    # Step 1 of adjudication detects disagreement (exit code 10)
    adj_dir = tmp_path / "adj_e2"
    rc_dis = adjudicate_cli_main(["--task", "e2", "--ann1", str(exports["A1"]), "--ann2", str(exports["A2"]), "--blind-key", str(b_key), "--out-dir", str(adj_dir)])
    assert rc_dis == 10
    dis_obj = read_json(adj_dir / "e2_disagreements_to_resolve.json")
    assert dis_obj["n_unresolved"] == 1
    dis_obj["disagreements"][0]["resolved_label"] = VideoTruthLabel.VIDEO_SUPPORTED.value
    res_p = atomic_write_json(adj_dir / "resolved.json", dis_obj)

    # Step 2 resolves disagreement -> outputs canonical adjudicated_e2.json accepted directly by A2!
    rc_ok = adjudicate_cli_main(["--task", "e2", "--ann1", str(exports["A1"]), "--ann2", str(exports["A2"]), "--blind-key", str(b_key), "--out-dir", str(adj_dir), "--resolutions", str(res_p)])
    assert rc_ok == 0
    r_cal = run_pilot_a2_pipeline(
        a1_preds, synthetic_dream1k_dir / "json" / "metadata.json", synthetic_dream1k_dir / "video",
        MockJudgeBackend(model_family="mistral"), a2_out, human_annotations_path=adj_dir / "adjudicated_e2.json", allow_cpu_mock=True,
    )
    assert r_cal["human_calibration_completed"] is True and r_cal["reused_existing_auto_eval_and_bundle"] is True

    # Now verify E3 web UI export -> adjudicate_annotations.py --task e3 -> canonical adjudicated_e3.json!
    from uarcap_q2.annotation.package_builder import build_blind_e3_annotation_package

    e3_pkg = build_blind_e3_annotation_package(
        [
            {
                "query_id": "vh_1",
                "arm": "E3_CON",
                "model_id": "qwen3_8",
                "video_path": str(pathological_videos["dynamic_24f"]),
                "caption": "A person rakes leaves.",
                "target_true_fact": "throwing grass",
                "target_conflict_fact": "raking leaves",
            }
        ],
        ["A1", "A2"],
        tmp_path / "e3_ui_pkg",
    )
    e3_exports = {}
    for ann, agree in (("A1", True), ("A2", False)):
        st = AnnotationStore(e3_pkg[f"tasks_{ann}"], tmp_path / "e3_ui_saves")
        for t in st.bundle["tasks"]:
            a_is_con = t["target_a"] == "raking leaves"
            st.save_record(
                t["blind_task_id"],
                {
                    "task_key": t["blind_task_id"],
                    "task_type": "E3_DUAL_TARGET",
                    "e3": {
                        "target_a_expressed": a_is_con if agree else (not a_is_con),
                        "target_b_expressed": (not a_is_con),
                    },
                    "elapsed_sec": 20,
                    "completed": True,
                },
            )
        ep = tmp_path / f"e3_exp_{ann}.json"
        ep.write_bytes(st.export_json_bytes())
        e3_exports[ann] = ep

    adj_e3_dir = tmp_path / "adj_e3"
    rc_e3_dis = adjudicate_cli_main(
        ["--task", "e3", "--ann1", str(e3_exports["A1"]), "--ann2", str(e3_exports["A2"]), "--blind-key", str(e3_pkg["blind_key"]), "--out-dir", str(adj_e3_dir)]
    )
    assert rc_e3_dis == 10
    e3_dis = read_json(adj_e3_dir / "e3_disagreements_to_resolve.json")
    for d in e3_dis["disagreements"]:
        d["resolved_value"] = 1 if d["item_key"].endswith("y_conflict") else 0
    e3_res_p = atomic_write_json(adj_e3_dir / "e3_resolved.json", e3_dis)
    rc_e3_ok = adjudicate_cli_main(
        ["--task", "e3", "--ann1", str(e3_exports["A1"]), "--ann2", str(e3_exports["A2"]), "--blind-key", str(e3_pkg["blind_key"]), "--out-dir", str(adj_e3_dir), "--resolutions", str(e3_res_p)]
    )
    assert rc_e3_ok == 0
    adj_e3 = read_json(adj_e3_dir / "adjudicated_e3.json")
    assert adj_e3["records"][0]["y_conflict"] == 1 and adj_e3["records"][0]["y_true"] == 0
