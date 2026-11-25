"""CaReBench Final Offline Handoff Test Suite.

Covers all 9 required verification domains:
  1. Real CaReBench metadata schema parsing (original ID, parent ID, category/subcategory, missing duration -> None,
     human captions vs non-Gold LLM-extracted events/objects, opening overview candidate).
  2. FineAction parent-video deduplication & honest residual overlap statement.
  3. Reproducible proportional category-stratified sampling with subcategory spread and zero parent leakage.
  4. End-to-end `--e2-dataset {carebench,dream1k}` switching across A0, A1, A2, A3, Formal B, E3-Bridge, E5.
  5. Strict Pilot / debug / near-dup / independent validation / formal main isolation (including parent-ID isolation).
  6. 3-way evaluation inputs (ENTAILED/CONTRADICTED/UNMENTIONED), reference support vs conflict rates, and
     explicit rule-based auxiliary static/dynamic/mixed/unclassified classification.
  7. A1/A2/A3/Formal B dataset-ID consistency (CaReBench outputs NEVER mislabeled as DREAM-1K!).
  8. CaReBench data preparation pipeline (prepare, zip-slip block, check-ready, download helper safety guard).
  9. Git ignore rule regression ensuring all 6 `uarcap_q2/datasets/*.py` modules (5 recovered + carebench.py)
     are never excluded by `.gitignore`, while raw video archives remain ignored.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import yaml

EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_ROOT.parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from deployment.run_stage_cli import build_parser, dispatch_stage
from deployment.stage_assets import KNOWN_ASSETS, main as stage_assets_main
from fixtures.build_synthetic_fixtures import make_dynamic_mp4
from scripts.download_carebench_stage import main as download_carebench_main
from uarcap_q2.common.io_utils import atomic_write_json, read_json, read_jsonl, sha256_file
from uarcap_q2.common.schemas import EligibilityStatus, PilotStage, SourceIdStatus, StageRunStatus, VideoTruthLabel
from uarcap_q2.datasets.carebench import (
    CAREBENCH_CATEGORIES,
    deduplicate_carebench_by_parent_video,
    extract_opening_overview_candidate,
    get_carebench_parent_id,
    load_carebench_records,
    parse_fineaction_parent_id,
    sample_carebench_stratified_pilot,
    write_carebench_pilot_exclusion_manifest,
)
from uarcap_q2.datasets.data_preparer import UnsafeArchiveError, safe_extract_zip
from uarcap_q2.datasets.formal_sampling import FormalSamplingConfig, build_formal_e2_manifest
from uarcap_q2.datasets.vidhalluc import load_vidhalluc_ach_mcq
from uarcap_q2.evaluators.backends import MockJudgeBackend
from uarcap_q2.evaluators.reference_fact_eval import (
    ThreeWayReferenceFactEvaluator,
    classify_claim_fact_dimension,
)
from uarcap_q2.generators.mock_generator import MockGeneratorAdapter
from uarcap_q2.interventions.e3_bridge import NaturalConflictObservation
from uarcap_q2.interventions.vidhalluc_e3 import E3ApplicabilityChecklist, build_vidhalluc_e3_candidate
from uarcap_q2.retrieval.dedup import screen_e2_near_duplicates
from uarcap_q2.retrieval.reranker import MockXCLIPRetriever
from uarcap_q2.runner.stage_pipelines import run_formal_b_pipeline
from uarcap_q2.statistics.power_planner import plan_formal_e2_sample_size


def _build_synthetic_carebench_dir(base_dir: Path, n_unique_parents: int = 40, n_multi_clip_parents: int = 6) -> Path:
    """Create a realistic CaReBench directory matching real MCG-NJU/CaReBench json/metadata.json schema."""
    cb_dir = base_dir / "carebench"
    (cb_dir / "json").mkdir(parents=True, exist_ok=True)
    (cb_dir / "videos").mkdir(parents=True, exist_ok=True)

    subcats = {
        "Socializing, Relaxing": ["playing_guitar", "watching_tv", "chatting_outdoors"],
        "Sports, Excercise": ["running_track", "doing_yoga", "lifting_weights"],
        "Personal Care": ["apply_eyebrows", "brushing_hair", "washing_face"],
        "Household Activities": ["folding_laundry", "washing_dishes", "sweeping_floor"],
    }
    # Interleaved approximate real CaReBench category shares (30% / 30% / 20% / 20% over 10-slot cycle)
    cat_cycle = [
        "Socializing, Relaxing",
        "Sports, Excercise",
        "Personal Care",
        "Household Activities",
        "Socializing, Relaxing",
        "Sports, Excercise",
        "Personal Care",
        "Household Activities",
        "Socializing, Relaxing",
        "Sports, Excercise",
    ]

    rows = []
    # Build one valid MP4 once and copy bytes to keep test execution super fast
    seed_mp4 = cb_dir / "videos" / "_seed_tmp.mp4"
    make_dynamic_mp4(seed_mp4, num_frames=16, width=96, height=96, fps=8, seed_offset=10)
    seed_bytes = seed_mp4.read_bytes()
    seed_mp4.unlink()

    colors = ["red", "blue", "green", "yellow", "purple", "orange", "white", "black", "brown", "silver"]
    rooms = ["kitchen", "studio", "garden", "hallway", "balcony", "garage", "office", "porch", "terrace", "loft"]
    items = ["notebook", "kettle", "basket", "towel", "mirror", "lamp", "pillow", "pitcher", "tray", "blanket"]

    for idx in range(n_unique_parents):
        parent_id = f"v_{idx + 1:08d}"
        cat = cat_cycle[idx % len(cat_cycle)]
        sc_list = subcats[cat]
        subcat = sc_list[idx % len(sc_list)]
        color = colors[idx % len(colors)]
        room = rooms[(idx // 2) % len(rooms)]
        item = items[(idx // 3) % len(items)]
        n_clips = 2 if idx < n_multi_clip_parents else 1
        for clip_num in range(1, n_clips + 1):
            vname = f"{parent_id}_{clip_num}.mp4"
            (cb_dir / "videos" / vname).write_bytes(seed_bytes)
            # 94%+ open with overview template, last 2 use non-template opening to test edge case
            if idx < n_unique_parents - 2:
                opening = f"This video showcases participant_{idx} performing {subcat.replace('_', ' ')} inside a {room}."
            else:
                opening = f"Participant_{idx} wearing a {color} jacket demonstrates {subcat.replace('_', ' ')} in a {room}."
            spatial = f"{opening} Subject_{idx} holds a {color} {item} near fixture_{idx} with {idx + 2} props."
            temporal = f"{opening} First subject_{idx} lifts the {item} and completes step_{idx} smoothly."
            full_cap = f"{opening} Subject_{idx} holds a {color} {item} near fixture_{idx} with {idx + 2} props. First subject_{idx} lifts the {item} and completes step_{idx} smoothly."
            rows.append(
                {
                    "video": vname,
                    "category": cat,
                    "subcategory": subcat,
                    "caption": full_cap,
                    "spatial_caption": spatial,
                    "temporal_caption": temporal,
                    "events": [f"Person performs {subcat.replace('_', ' ')}"],
                    "objects": ["There is a person wearing a red shirt.", "There is a wooden table."],
                }
            )
    (cb_dir / "json" / "metadata.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return cb_dir


def test_carebench_schema_provenance_missing_fields_and_parent_dedup(tmp_path: Path) -> None:
    cb_dir = _build_synthetic_carebench_dir(tmp_path, n_unique_parents=40, n_multi_clip_parents=6)
    recs = load_carebench_records(cb_dir / "json" / "metadata.json", cb_dir / "videos", require_video_exists=True)
    assert len(recs) == 46  # 40 unique parents + 6 sibling clips

    r0 = recs[0]
    assert r0.dataset_id == "CaReBench"
    assert r0.query_id.startswith("carebench_v_")
    # FineAction anonymous parent ID is NOT a verified original YouTube ID!
    assert r0.source_id is None
    assert r0.source_id_status == SourceIdStatus.UNKNOWN_OPAQUE.value
    # Missing duration preserved explicitly as None (never fabricated!)
    assert r0.duration is None
    assert "duration" in r0.metadata["missing_fields"]
    # Explicit provenance separation: human captions vs GPT-4 extracted events/objects
    assert r0.metadata["llm_extracted_facts_provenance"] == "GPT4_EXTRACTED_NOT_INDEPENDENT_HUMAN_GOLD"
    assert set(r0.metadata["human_reference_fields"]) == {"caption", "spatial_caption", "temporal_caption"}
    assert r0.metadata["core_event_candidate_status"] == "UNVERIFIED_CANDIDATE_REQUIRES_PILOT_A2_AUDIT"
    assert r0.metadata["parent_video_id"].startswith("v_")
    assert r0.metadata["category"] in CAREBENCH_CATEGORIES
    assert r0.metadata["subcategory"] != "unknown"
    assert len(r0.core_events) == 1

    # Opening overview template detector handles both template and non-template openings
    s_tpl, is_tpl = extract_opening_overview_candidate("This video showcases a woman applying makeup. She smiles.")
    assert s_tpl == "This video showcases a woman applying makeup." and is_tpl is True
    s_nt, is_nt = extract_opening_overview_candidate("A man rides a bike on a street. Trees line the road.")
    assert s_nt == "A man rides a bike on a street." and is_nt is False
    assert parse_fineaction_parent_id("v_00000030_5.mp4") == ("v_00000030", 5)

    # Parent-video deduplication collapses 46 clips -> 40 unique FineAction parents deterministically
    deduped, audit = deduplicate_carebench_by_parent_video(recs)
    assert len(deduped) == 40
    assert audit["n_raw_clips"] == 46
    assert audit["n_unique_fineaction_parents"] == 40
    assert audit["n_multi_clip_parents"] == 6
    assert audit["n_dropped_same_parent_siblings"] == 6
    assert audit["parent_video_dedup_does_not_prove_independent_youtube_sources"] is True
    assert len({get_carebench_parent_id(r) for r in deduped}) == 40


def test_carebench_reproducible_stratified_pilot_and_formal_isolation(tmp_path: Path) -> None:
    cb_dir = _build_synthetic_carebench_dir(tmp_path, n_unique_parents=40, n_multi_clip_parents=6)
    recs = load_carebench_records(cb_dir / "json" / "metadata.json", cb_dir / "videos", require_video_exists=True)

    p1, rem1, audit1 = sample_carebench_stratified_pilot(recs, total_queries=12, seed=20261125)
    p2, rem2, _ = sample_carebench_stratified_pilot(recs, total_queries=12, seed=20261125)
    assert [r.query_id for r in p1] == [r.query_id for r in p2]
    assert [r.query_id for r in rem1] == [r.query_id for r in rem2]
    assert len(p1) == 12
    assert len(rem1) == 28  # 40 unique parents - 12 pilot parents
    # All 4 categories represented proportionally
    assert set(audit1["category_pilot_allocation"].keys()) == set(CAREBENCH_CATEGORIES)
    assert all(v >= 1 for v in audit1["category_pilot_allocation"].values())
    # Zero parent overlap between Pilot and remaining pool
    pilot_parents = {get_carebench_parent_id(r) for r in p1}
    rem_parents = {get_carebench_parent_id(r) for r in rem1}
    assert pilot_parents.isdisjoint(rem_parents)

    # Persist exclusion manifest including extra debug clip
    debug_rec = rem1[0]
    excl_path = write_carebench_pilot_exclusion_manifest(
        p1,
        tmp_path / "pilot_a_excluded_carebench_vids.json",
        seed=20261125,
        extra_debug_query_ids={debug_rec.query_id},
        extra_debug_parent_ids={get_carebench_parent_id(debug_rec)},
    )
    excl_doc = read_json(excl_path)
    assert excl_doc["dataset_id"] == "CaReBench"
    assert len(excl_doc["excluded_parent_video_ids"]) == 13

    # Build Formal B manifest from ALL raw 46 clips -> verifies parent dedup + pilot/debug/near-dup/val isolation!
    near_dup_qid = rem1[1].query_id
    sel_formal, f_man = build_formal_e2_manifest(
        all_records=recs,
        pilot_excluded_ids=set(excl_doc["excluded_query_ids"]),
        pilot_excluded_parent_ids=set(excl_doc["excluded_parent_video_ids"]),
        extra_debug_query_ids={debug_rec.query_id},
        extra_debug_parent_ids={get_carebench_parent_id(debug_rec)},
        explicit_near_dup_excluded_ids={near_dup_qid},
        cfg=FormalSamplingConfig(e2_target_n=20, e2_validation_target_n=6, seed=42),
        out_manifest_path=tmp_path / "formal_carebench_analysis_manifest.json",
    )
    assert f_man["dataset_id"] == "CaReBench"
    assert f_man["n_after_parent_dedup"] == 40
    assert f_man["n_selected_formal"] == 20
    assert f_man["parent_isolation_verified"] is True
    assert f_man["e2_validation_set"]["n_validation_selected"] == 6
    assert f_man["e2_validation_set"]["disjoint_from_pilot_verified"] is True
    assert f_man["e2_validation_set"]["disjoint_from_formal_main_verified"] is True
    assert f_man["n_formal_main_excluding_validation"] == 14
    sel_parents = {get_carebench_parent_id(r) for r in sel_formal}
    assert sel_parents.isdisjoint(pilot_parents)
    assert get_carebench_parent_id(debug_rec) not in sel_parents
    assert near_dup_qid not in {r.query_id for r in sel_formal}

    # Verify conservative near-duplicate screening and power planner defaults for CaReBench
    nd_rep = screen_e2_near_duplicates(
        sel_formal[:2],
        top1_visual_matches={sel_formal[0].query_id: {"clip_id": "ufo_1", "caption": "unrelated", "cosine": 0.95}},
        dataset_id="CaReBench",
    )
    assert nd_rep["dataset_id"] == "CaReBench"
    assert nd_rep["exact_source_zero_overlap_claimed"] is False
    assert "FineAction" in nd_rep["residual_overlap_uncertainty_statement"]

    pwr = plan_formal_e2_sample_size(pilot_delta_sd=0.55)
    assert pwr["dataset_id"] == "CaReBench"
    assert pwr["planned_range"] == [600, 750]
    assert 600 <= pwr["recommended_formal_n"] <= 750


def test_three_way_evaluator_reference_rates_and_auxiliary_static_dynamic_classifier() -> None:
    # Explicit rule-based classification (static / dynamic / mixed / unclassified)
    assert classify_claim_fact_dimension("There is a red wooden table in the kitchen.") == "static_entity_attribute"
    assert classify_claim_fact_dimension("The person walks and jumps forward.") == "dynamic_action_temporal"
    assert classify_claim_fact_dimension("A woman wearing a blue shirt chops vegetables on a wooden table.") == "mixed"
    assert classify_claim_fact_dimension("An object appears.") == "unclassified"

    ev = ThreeWayReferenceFactEvaluator(MockJudgeBackend(model_family="mistral"), generator_model_family="qwen")
    res = ev.evaluate_caption(
        query_id="carebench_v_00000001_1",
        arm="ARM_A7",
        model_id="Qwen/Qwen3.8-27B-FP8",
        caption="A woman in a red shirt washes dishes near a wooden table.",
        reference_description="This video showcases a woman washing dishes. She wears a blue shirt near a wooden table.",
        core_events=["This video showcases a woman washing dishes."],
    )
    assert res.parse_status == "OK"
    assert res.n_claims >= 1
    assert res.p_entail_ref is not None and 0.0 <= res.p_entail_ref <= 1.0
    assert res.p_contra_ref is not None and 0.0 <= res.p_contra_ref <= 1.0
    assert res.p_unment_ref is not None and 0.0 <= res.p_unment_ref <= 1.0
    assert set(res.auxiliary_dimension_breakdown.keys()) == {
        "static_entity_attribute",
        "dynamic_action_temporal",
        "mixed",
        "unclassified",
    }


def test_carebench_end_to_end_cli_a0_a1_a2_a3_formal_b_and_dream1k_backup_switch(
    synthetic_dream1k_dir: Path,
    synthetic_vidhalluc_json: Path,
    tmp_path: Path,
) -> None:
    cb_dir = _build_synthetic_carebench_dir(tmp_path, n_unique_parents=36, n_multi_clip_parents=4)
    paths_yaml = tmp_path / "paths_both.yaml"
    paths_yaml.write_text(
        yaml.safe_dump(
            {
                "host": {"device": "cuda:0", "min_free_disk_gb": 1},
                "historical_frozen_assets": {"clips_full_tsv_path": str(tmp_path / "empty.tsv")},
                "q2_writable_workspace": {
                    "q2_root": str(tmp_path),
                    "data_root": str(tmp_path),
                    "carebench_dir": str(cb_dir),
                    "dream1k_dir": str(synthetic_dream1k_dir),
                    "vidhalluc_dir": str(synthetic_vidhalluc_json.parent),
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "empty.tsv").write_text("clip_id\n", encoding="utf-8")

    # 1. Verify default dry-run selects CaReBench AND explicit --e2-dataset dream1k switches to DREAM-1K backup!
    args_cb_dry = build_parser().parse_args(
        [
            "--stage", "PILOT_A1",
            "--paths-config", str(paths_yaml),
            "--stage-config", str(EXP_ROOT / "configs" / "pilot_a01_config.yaml"),
            "--out-dir", str(tmp_path / "dry_cb"),
            "--dry-run",
        ]
    )
    rc_dry, plan_cb = dispatch_stage(args_cb_dry)
    assert rc_dry == 0 and plan_cb["e2_dataset"] == "carebench" and plan_cb["e2_dataset_id"] == "CaReBench"

    args_dr_dry = build_parser().parse_args(
        [
            "--stage", "PILOT_A1",
            "--e2-dataset", "dream1k",
            "--paths-config", str(paths_yaml),
            "--stage-config", str(EXP_ROOT / "configs" / "pilot_a01_config.yaml"),
            "--out-dir", str(tmp_path / "dry_dr"),
            "--dry-run",
        ]
    )
    rc_dr, plan_dr = dispatch_stage(args_dr_dry)
    assert rc_dr == 0 and plan_dr["e2_dataset"] == "dream1k" and plan_dr["e2_dataset_id"] == "DREAM-1K"

    # 2. Pilot A0 on CaReBench
    a0_dir = tmp_path / "a0_out"
    rc_a0, rep_a0 = dispatch_stage(
        build_parser().parse_args(
            [
                "--stage", "PILOT_A0",
                "--e2-dataset", "carebench",
                "--paths-config", str(paths_yaml),
                "--stage-config", str(EXP_ROOT / "configs" / "pilot_a01_config.yaml"),
                "--out-dir", str(a0_dir),
                "--cpu-mock",
            ]
        )
    )
    assert rc_a0 == 0 and rep_a0["e2_dataset_selected"] == "CaReBench"

    # 3. Pilot A1 on CaReBench (limit=4 -> 12 gens) — verify outputs labeled CaReBench, NEVER DREAM-1K!
    a1_dir = tmp_path / "a1_cb"
    rc_a1, sum_a1 = dispatch_stage(
        build_parser().parse_args(
            [
                "--stage", "PILOT_A1",
                "--e2-dataset", "carebench",
                "--paths-config", str(paths_yaml),
                "--stage-config", str(EXP_ROOT / "configs" / "pilot_a01_config.yaml"),
                "--out-dir", str(a1_dir),
                "--run-id", "pilot_a1_carebench_test",
                "--limit", "4",
                "--cpu-mock",
            ]
        ),
        injected_retriever=MockXCLIPRetriever(),
        injected_videollama=MockGeneratorAdapter(model_id="mock_vl_cb"),
    )
    assert rc_a1 == 0
    assert sum_a1["dataset_id"] == "CaReBench"
    assert (a1_dir / "pilot_a_excluded_carebench_vids.json").is_file()
    assert not (a1_dir / "pilot_a_excluded_dream1k_vids.json").exists()
    assert (a1_dir / "pilot_a1_carebench_sampling_audit.json").is_file()
    a1_preds = read_jsonl(a1_dir / "predictions.jsonl")
    assert len(a1_preds) == 12
    assert all(p["query_id"].startswith("carebench_") for p in a1_preds)

    # 4. Pilot A2 on CaReBench (first pass pauses at exit 10, stamps dataset_id == CaReBench)
    a2_dir = tmp_path / "a2_cb"
    rc_a2, rec_a2 = dispatch_stage(
        build_parser().parse_args(
            [
                "--stage", "PILOT_A2",
                "--e2-dataset", "carebench",
                "--paths-config", str(paths_yaml),
                "--stage-config", str(EXP_ROOT / "configs" / "pilot_a23_config.yaml"),
                "--out-dir", str(a2_dir),
                "--a1-predictions", str(a1_dir / "predictions.jsonl"),
                "--cpu-mock",
            ]
        ),
        injected_ref_judge=MockJudgeBackend(model_family="mistral"),
    )
    assert rc_a2 == 10
    assert rec_a2["dataset_id"] == "CaReBench"
    assert rec_a2["status"] == StageRunStatus.HUMAN_VALIDATION_PENDING.value

    # 5. Pilot A3 on CaReBench (first pass runs Modern VLM on CaReBench -> pauses for VidHalluc E3 review)
    a3_dir = tmp_path / "a3_cb"
    rc_a3, rec_a3 = dispatch_stage(
        build_parser().parse_args(
            [
                "--stage", "PILOT_A3",
                "--e2-dataset", "carebench",
                "--paths-config", str(paths_yaml),
                "--stage-config", str(EXP_ROOT / "configs" / "pilot_a23_config.yaml"),
                "--out-dir", str(a3_dir),
                "--limit", "3",
                "--cpu-mock",
            ]
        ),
        injected_retriever=MockXCLIPRetriever(),
        injected_videollama=MockGeneratorAdapter(model_id="mock_vl_cb"),
        injected_modern_vlm=MockGeneratorAdapter(model_id="mock_qwen38_cb"),
        injected_e3_judge=MockJudgeBackend(model_family="mistral"),
    )
    assert rc_a3 == 10
    assert rec_a3["e2_dataset_id"] == "CaReBench"
    assert "dream1k_modern_vlm_summary" not in rec_a3
    assert (a3_dir / "carebench_modern_vlm" / "predictions.jsonl").is_file()
    assert not (a3_dir / "dream1k_modern_vlm").exists()

    # 6. Formal B on CaReBench + VidHalluc + E3-Bridge (verifying zero DREAM-1K mislabeling!)
    all_cb = load_carebench_records(cb_dir / "json" / "metadata.json", cb_dir / "videos", require_video_exists=True)
    vh_recs = load_vidhalluc_ach_mcq(synthetic_vidhalluc_json, synthetic_vidhalluc_json.parent / "data")
    vh_by_qid = {r.query_id: r for r in vh_recs}
    e3_cands = []
    for r in vh_recs[:2]:
        c = build_vidhalluc_e3_candidate(r, neu_text="In the video, the person continues moving naturally.")
        c.checklist = E3ApplicabilityChecklist(True, True, True, True, True, reviewer_id="A1").to_dict()
        c.eligibility_status = EligibilityStatus.APPROVED.value
        e3_cands.append(c)

    # Create real-like frozen receipts to unlock Formal B in unit test
    p_a1 = atomic_write_json(tmp_path / "p_a1.json", {"stage": "PILOT_A1", "status": "REAL_RUNTIME_PASS"})
    p_a2 = atomic_write_json(tmp_path / "p_a2.json", {"stage": "PILOT_A2", "status": "REAL_RUNTIME_PASS", "human_calibration_completed": True})
    p_a3 = atomic_write_json(tmp_path / "p_a3.json", {"stage": "PILOT_A3", "status": "REAL_RUNTIME_PASS", "e3_human_validation_completed": True})
    formal_cfg = tmp_path / "formal_frozen_cb.yaml"
    formal_cfg.write_text(yaml.safe_dump({"stage": "FORMAL_B", "e2_dataset": "carebench", "e2_formal_n": 4}), encoding="utf-8")
    asset_man = tmp_path / "assets_cb.json"
    asset_man.write_text(json.dumps({"dataset": "CaReBench"}), encoding="utf-8")
    pilot_excl_file = a1_dir / "pilot_a_excluded_carebench_vids.json"
    pilot_excl_doc = read_json(pilot_excl_file)

    fb_auth = atomic_write_json(
        tmp_path / "fb_auth.json",
        {
            "stage": "FORMAL_B",
            "approved": True,
            "approval_id": "TEST_CB_FB",
            "frozen_pilot_params_confirmed": True,
            "evaluator_version": "mistral_v1",
            "frozen_formal_config_sha256": sha256_file(formal_cfg),
            "asset_manifest_sha256": sha256_file(asset_man),
            "pilot_exclusion_manifest_sha256": sha256_file(pilot_excl_file),
        },
    )

    # Create 1 CaReBench E3-Bridge observation from a non-pilot CaReBench clip
    non_pilot_cb = [r for r in all_cb if get_carebench_parent_id(r) not in set(pilot_excl_doc["excluded_parent_video_ids"])]
    bridge_obs = [
        NaturalConflictObservation(
            query_id=non_pilot_cb[0].query_id,
            domain=non_pilot_cb[0].domain,
            verified_true_fact="a person wears a red shirt",
            natural_videoufo_caption="A person wears a green coat outdoors.",
            natural_conflict_fact="a person wears a green coat",
            single_target_clean=True,
            conflict_strictly_absent_in_video=True,
            proposed_neu_text="In the video, the person remains visible in the scene.",
            reviewer_id="A1",
        )
    ]

    fb_dir = tmp_path / "fb_cb"
    fb_rec = run_formal_b_pipeline(
        formal_e2_queries=all_cb,
        formal_vidhalluc_queries_by_id=vh_by_qid,
        approved_e3_candidates=e3_cands,
        retriever=MockXCLIPRetriever(),
        videollama_adapter=MockGeneratorAdapter(model_id="mock_vl_cb"),
        modern_vlm=MockGeneratorAdapter(model_id="mock_qwen38_cb"),
        ref_judge_backend=MockJudgeBackend(model_family="mistral"),
        e3_judge_backend=MockJudgeBackend(model_family="mistral"),
        out_dir=fb_dir,
        auth_receipt_path=fb_auth,
        prior_receipts={"PILOT_A1": p_a1, "PILOT_A2": p_a2, "PILOT_A3": p_a3},
        formal_config_path=formal_cfg,
        asset_manifest_path=asset_man,
        pilot_exclusion_path=pilot_excl_file,
        pilot_excluded_query_ids=set(pilot_excl_doc["excluded_query_ids"]),
        pilot_excluded_parent_ids=set(pilot_excl_doc["excluded_parent_video_ids"]),
        e3_bridge_observations=bridge_obs,
        e3_bridge_min_clean_pairs=1,
        sampling_cfg=FormalSamplingConfig(e2_target_n=6, e2_validation_target_n=2, seed=7),
        skip_video_decode=True,
    )
    assert fb_rec["status"] == StageRunStatus.REAL_RUNTIME_PASS.value
    assert fb_rec["e2_dataset_id"] == "CaReBench"
    assert fb_rec["e3_dataset_id"] == "VidHalluc-ACH"
    assert "dream1k_manifest_sha256" not in fb_rec
    assert (fb_dir / "formal_carebench_analysis_manifest.json").is_file()
    assert not (fb_dir / "formal_dream1k_analysis_manifest.json").exists()
    fb_stats = read_json(fb_dir / "formal_b_statistics.json")
    assert fb_stats["e2_dataset_id"] == "CaReBench"
    assert "e2_auxiliary_static_dynamic_diagnostics" in fb_stats
    assert "action facts" in fb_stats["e3_causal_scope_notice"]
    assert fb_stats["e3_bridge"]["source_dataset_id"] == "CaReBench"


def test_carebench_data_preparer_and_download_helper_guards(tmp_path: Path) -> None:
    assert "carebench_metadata" in KNOWN_ASSETS and KNOWN_ASSETS["carebench_metadata"].gated is False
    assert "carebench_videos" in KNOWN_ASSETS and KNOWN_ASSETS["carebench_videos"].gated is False
    assert KNOWN_ASSETS["carebench_videos"].expected_bytes == 3918994506

    # Download helper dry-run succeeds cleanly & refuses full download without --confirm-download
    assert download_carebench_main(["--stage-dir", str(tmp_path / "stage"), "--dry-run"]) == 0
    assert download_carebench_main(["--stage-dir", str(tmp_path / "stage")]) == 2

    # Build a synthetic CaReBench zip with nested videos/videos/*.mp4 layout and test one-click prepare + check-ready
    cb_src = _build_synthetic_carebench_dir(tmp_path / "src", n_unique_parents=4, n_multi_clip_parents=1)
    data_root = tmp_path / "bn_data"
    (data_root / "carebench" / "json").mkdir(parents=True, exist_ok=True)
    (data_root / "carebench" / "videos").mkdir(parents=True, exist_ok=True)
    (data_root / "carebench" / "json" / "metadata.json").write_text(
        (cb_src / "json" / "metadata.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    zip_dst = data_root / "carebench" / "videos" / "videos.zip"
    with zipfile.ZipFile(zip_dst, "w") as zf:
        for mp4 in sorted((cb_src / "videos").glob("*.mp4")):
            zf.write(mp4, arcname=f"nested_videos/{mp4.name}")

    rc_prep = stage_assets_main(
        ["prepare", "--data-root", str(data_root), "--dataset", "carebench", "--skip-archive-size-check"]
    )
    assert rc_prep == 0
    rdy = read_json(data_root / "carebench_readiness.json")
    assert rdy["dataset"] == "carebench"
    assert rdy["ready"] is True
    assert rdy["n_metadata_records"] == 5
    assert rdy["n_unique_fineaction_parents"] == 4
    assert rdy["n_missing_or_empty_videos"] == 0
    assert stage_assets_main(["check-ready", "--data-root", str(data_root), "--dataset", "carebench"]) == 0

    # Verify zip-slip protection still blocks malicious archives
    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../escape.mp4", b"evil")
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(bad_zip, tmp_path / "extract_out")


def test_gitignore_regression_allows_all_dataset_python_modules_and_blocks_raw_videos() -> None:
    """Verify root & local .gitignore rules NEVER ignore Python modules under uarcap_q2/datasets/."""
    git_dir = REPO_ROOT / ".git"
    if not git_dir.exists():
        pytest.skip("Not running inside a Git working tree (e.g. git archive snapshot)")

    required_py = [
        "research/uarcap_q2_experiments/uarcap_q2/datasets/__init__.py",
        "research/uarcap_q2_experiments/uarcap_q2/datasets/asset_transfer.py",
        "research/uarcap_q2_experiments/uarcap_q2/datasets/carebench.py",
        "research/uarcap_q2_experiments/uarcap_q2/datasets/data_preparer.py",
        "research/uarcap_q2_experiments/uarcap_q2/datasets/dream1k.py",
        "research/uarcap_q2_experiments/uarcap_q2/datasets/formal_sampling.py",
        "research/uarcap_q2_experiments/uarcap_q2/datasets/video_decoder.py",
        "research/uarcap_q2_experiments/uarcap_q2/datasets/vidhalluc.py",
    ]
    for rel in required_py:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "check-ignore", rel],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode != 0, f"CRITICAL: .gitignore is excluding Python source file {rel}: {proc.stdout}"

    # Verify large raw video archives inside q2_experiments remain ignored
    proc_mp4 = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", "research/uarcap_q2_experiments/carebench_videos.zip"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_mp4.returncode == 0
