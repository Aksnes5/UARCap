"""Tests 05: Blind annotation packages, zero-leak verification, local HTTP web UI persistence/resume, and adjudication."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import urllib.request

import pytest

from uarcap_q2.annotation.adjudicator import extract_disagreements_and_time_summary
from uarcap_q2.annotation.package_builder import (
    assert_blind_payload_clean,
    build_blind_e2_annotation_package,
    build_blind_e3_annotation_package,
    normalize_claim_key,
    unblind_annotations,
)
from uarcap_q2.annotation.web_app import create_annotation_server


def test_blind_packages_have_zero_leaks_and_safe_unblinding(pathological_videos: dict[str, Path], tmp_path: Path) -> None:
    videos = [
        {
            "query_id": "dream1k_0001",
            "video_path": str(pathological_videos["dynamic_24f"]),
            "captions": [
                {"arm": "ARM_A0", "model_id": "vl", "caption": "A bar moves right.", "claims": ["A bar moves right"]},
                {"arm": "ARM_A7", "model_id": "vl", "caption": "A bar moves right near a tree.", "claims": ["A bar moves right", "near a tree"]},
            ],
        }
    ]
    out = build_blind_e2_annotation_package(videos, ["A1", "A2"], tmp_path / "e2_pkg", seed=99)
    raw_txt = out["tasks_A1"].read_text(encoding="utf-8")
    for secret_str in ("ARM_A0", "ARM_A7", "dream1k_0001"):
        assert secret_str not in raw_txt
    bundle = json.loads(raw_txt)
    assert_blind_payload_clean(bundle)

    # Shared normalized claim matches safely without exposing arm/model metadata
    c0, c1 = bundle["tasks"][0]["captions"]
    norms = {cl["norm_key"] for c in (c0, c1) for cl in c["claims"]}
    assert normalize_claim_key("A bar moves right.") in norms

    # Verify blindness leak detector catches intentional injection
    with pytest.raises(ValueError, match="BLINDNESS LEAK"):
        assert_blind_payload_clean({"tasks": [{"arm": "ARM_A7"}]})

    # E3 blind package randomizes target_a/target_b and unblinds accurately to y_true/y_conflict
    e3_items = [
        {
            "query_id": "vidhalluc_1",
            "arm": "E3_CON",
            "model_id": "vl",
            "video_path": str(pathological_videos["dynamic_24f"]),
            "caption": "A person rakes leaves.",
            "target_true_fact": "throwing grass",
            "target_conflict_fact": "raking leaves",
        }
    ]
    e3_out = build_blind_e3_annotation_package(e3_items, ["A1"], tmp_path / "e3_pkg", seed=123)
    e3_bundle = json.loads(e3_out["tasks_A1"].read_text(encoding="utf-8"))
    t = e3_bundle["tasks"][0]
    rec = {
        "blind_task_id": t["blind_task_id"],
        "target_a_expressed": t["target_a"] == "raking leaves",
        "target_b_expressed": t["target_b"] == "raking leaves",
    }
    unblinded = unblind_annotations([rec], e3_out["blind_key"])
    assert unblinded[0]["y_true"] == 0 and unblinded[0]["y_conflict"] == 1


def test_local_web_annotation_server_save_resume_export_and_media_guard(
    pathological_videos: dict[str, Path], tmp_path: Path
) -> None:
    videos = [
        {
            "query_id": "dream1k_0001",
            "video_path": str(pathological_videos["dynamic_24f"]),
            "captions": [{"arm": "ARM_A0", "model_id": "vl", "caption": "A bar moves.", "claims": ["A bar moves"]}],
        }
    ]
    pkg = build_blind_e2_annotation_package(videos, ["A1"], tmp_path / "pkg")
    save_dir = tmp_path / "saves"
    server, _store = create_annotation_server(pkg["tasks_A1"], save_dir, host="127.0.0.1", port=0)
    port = server.server_port
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{port}"
        html = urllib.request.urlopen(f"{base}/").read().decode("utf-8")
        assert "UARCap Q2 Blind Factuality Annotator" in html

        state = json.loads(urllib.request.urlopen(f"{base}/api/state").read().decode("utf-8"))
        tk = state["bundle"]["tasks"][0]["blind_video_id"]
        payload = json.dumps({"task_key": tk, "record": {"task_key": tk, "completed": True, "elapsed_sec": 42}}).encode("utf-8")
        req = urllib.request.Request(f"{base}/api/save", data=payload, headers={"Content-Type": "application/json"}, method="POST")
        assert json.loads(urllib.request.urlopen(req).read().decode("utf-8"))["ok"] is True

        csv_bytes = urllib.request.urlopen(f"{base}/api/export?fmt=csv").read().decode("utf-8")
        assert tk in csv_bytes and "42" in csv_bytes

        # Media path traversal / unauthorized file access blocked (403)
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(f"{base}/media?path=/etc/passwd")
        assert ei.value.code == 403
    finally:
        server.shutdown()
        server.server_close()

    # Verify resume from disk in a fresh server instance
    server2, store2 = create_annotation_server(pkg["tasks_A1"], save_dir, host="127.0.0.1", port=0)
    server2.server_close()
    assert store2.saved[tk]["completed"] is True and store2.saved[tk]["elapsed_sec"] == 42


def test_adjudicator_kappa_ac1_and_two_annotator_total_person_hours() -> None:
    a1 = {f"c{i}": "Video-Supported" if i < 8 else "Video-Contradicted" for i in range(10)}
    a2 = {f"c{i}": "Video-Supported" if i < 7 else "Video-Contradicted" for i in range(10)}
    res = extract_disagreements_and_time_summary(a1, a2, ann1_elapsed_sec=3600, ann2_elapsed_sec=3600, adjudication_joint_sec=900)
    assert res["n_disagreements"] == 1
    assert res["cohens_kappa"] is not None and res["cohens_kappa"] > 0.70
    assert res["gwets_ac1"] is not None and res["gwets_ac1"] > 0.75
    # 1h + 1h + 2 * 0.25h joint = 2.5 total person-hours!
    assert res["time_ledger"]["total_two_annotator_person_hours"] == 2.5
