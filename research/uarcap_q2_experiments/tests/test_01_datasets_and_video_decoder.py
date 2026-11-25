"""Tests 01: Unified data schema, DREAM-1K & VidHalluc loading, video decoding pathologies, ZIP safety, and asset transfer."""

from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

from uarcap_q2.common.io_utils import safe_extract_zip
from uarcap_q2.common.schemas import EligibilityStatus, QueryRecord, SourceIdStatus
from uarcap_q2.datasets.asset_transfer import reassemble_and_verify_chunks, split_file_into_chunks
from uarcap_q2.datasets.dream1k import (
    load_dream1k_records,
    sample_dream1k_stratified_pilot,
    write_pilot_exclusion_manifest,
)
from uarcap_q2.datasets.video_decoder import decode_and_sample_video, pack_evcap_frame_tsv_bundle
from uarcap_q2.datasets.vidhalluc import (
    deduplicate_and_filter_vidhalluc,
    extract_youtube_id,
    load_vidhalluc_ach_mcq,
    sample_vidhalluc_pilot_candidates,
)


def test_query_record_never_allows_fabricated_source_id() -> None:
    with pytest.raises(ValueError, match="never fabricate"):
        QueryRecord(
            dataset_id="DREAM-1K",
            query_id="dream1k_0001",
            source_id="fake_id",
            source_id_status=SourceIdStatus.UNKNOWN_OPAQUE.value,
            video_path="1.mp4",
            video_sha256=None,
            duration=5.0,
            domain="movie_animation",
            reference_description="test",
        ).validate()


def test_dream1k_load_stratified_pilot_and_exclusion(synthetic_dream1k_dir: Path, tmp_path: Path) -> None:
    recs = load_dream1k_records(
        synthetic_dream1k_dir / "json" / "metadata.json",
        synthetic_dream1k_dir / "video",
        require_video_exists=True,
    )
    assert len(recs) == 10
    assert all(r.source_id is None and r.source_id_status == SourceIdStatus.UNKNOWN_OPAQUE.value for r in recs)
    # Camera meta event ('The scene fades out.') filtered out of core_events
    assert all("fades out" not in " ".join(r.core_events).lower() for r in recs)
    assert all(1 <= len(r.core_events) <= 2 for r in recs)

    pilot1, rem1 = sample_dream1k_stratified_pilot(recs, per_domain=1, seed=42)
    pilot2, rem2 = sample_dream1k_stratified_pilot(recs, per_domain=1, seed=42)
    assert [r.query_id for r in pilot1] == [r.query_id for r in pilot2]
    assert len(pilot1) == 5 and len(rem1) == 5
    assert set(r.query_id for r in pilot1).isdisjoint(r.query_id for r in rem1)
    excl_path = write_pilot_exclusion_manifest(pilot1, tmp_path / "pilot_excl.json", seed=42)
    assert excl_path.is_file()


def test_vidhalluc_youtube_extraction_dedup_and_pending_status(synthetic_vidhalluc_json: Path, tmp_path: Path) -> None:
    assert extract_youtube_id("vcCwvRYqU2I_clip_4") == "vcCwvRYqU2I"
    recs = load_vidhalluc_ach_mcq(synthetic_vidhalluc_json, tmp_path)
    assert len(recs) == 6
    # CRITICAL: Every candidate starts as CANDIDATE_PENDING_VERIFICATION (distractors never auto-trusted!)
    assert all(r.eligibility_status == EligibilityStatus.CANDIDATE_PENDING_VERIFICATION.value for r in recs)

    kept, audit = deduplicate_and_filter_vidhalluc(
        recs,
        videoufo_source_ids={"OVERLAP0001"},
        historical_source_ids={"HISTMSRVTT1"},
    )
    assert audit["count_before"] == 6
    assert audit["overlap_videoufo_count"] == 1
    assert audit["overlap_historical_count"] == 1
    assert audit["same_source_duplicate_count"] == 1  # vcCwvRYqU2I appears twice -> max 1 kept
    assert audit["eligible_unique_source_count"] == 3
    assert len({r.source_id for r in kept}) == 3

    pilot, formal_pool = sample_vidhalluc_pilot_candidates(kept, candidate_count=2, seed=7)
    assert len(pilot) == 2 and len(formal_pool) == 1
    assert {r.source_id for r in pilot}.isdisjoint({r.source_id for r in formal_pool})


def test_video_decoder_all_pathologies_and_evcap_tsv_packing(
    pathological_videos: dict[str, Path], tmp_path: Path
) -> None:
    # 1. Normal dynamic 24-frame video
    imgs, rep = decode_and_sample_video(pathological_videos["dynamic_24f"], num_frames=16)
    assert rep.ok and rep.status == "OK"
    assert len(imgs) == 16 and rep.unique_frame_hashes >= 12 and rep.mean_consecutive_l1_diff > 1.0
    assert len(rep.frame_dhashes) == 16

    # 2. Pack into exact frozen EVCap 18-col TSV + lineidx + yaml (vision_language_tsv.py L311: qid + num_frames + 16 frames)
    bundle = pack_evcap_frame_tsv_bundle([("dream1k_0001", imgs, "a moving colored bar")], tmp_path / "tsv")
    assert bundle["yaml"].is_file() and bundle["img"].is_file() and bundle["img_lineidx"].is_file()
    cols = bundle["img"].read_text(encoding="utf-8").splitlines()[0].split("\t")
    assert len(cols) == 18 and cols[0] == "dream1k_0001" and cols[1] == "16"

    # 3. Short 4-frame video (< 16 frames -> deterministic resampling with warning)
    imgs_s, rep_s = decode_and_sample_video(pathological_videos["short_4f"], num_frames=16)
    assert rep_s.ok and rep_s.status == "SHORT_VIDEO_PADDED" and len(imgs_s) == 16

    # 4. Single-frame video -> rejected
    _, rep_1 = decode_and_sample_video(pathological_videos["single_1f"], num_frames=16)
    assert not rep_1.ok and rep_1.status == "SINGLE_FRAME"

    # 5. Variable frame rate (VFR) detected cleanly
    imgs_v, rep_v = decode_and_sample_video(pathological_videos["vfr_20f"], num_frames=16)
    assert rep_v.ok and rep_v.is_vfr_detected and len(imgs_v) == 16

    # 6. Historical repeated-first-frame bug -> REJECTED!
    _, rep_dup = decode_and_sample_video(pathological_videos["repeated_first_frame"], num_frames=16)
    assert not rep_dup.ok and rep_dup.status == "REPEATED_FIRST_FRAME"

    # 7. Corrupt video -> rejected
    _, rep_bad = decode_and_sample_video(pathological_videos["corrupt"], num_frames=16)
    assert not rep_bad.ok and rep_bad.status == "CORRUPT"

    # 8. Out-of-bounds temporal window -> rejected
    _, rep_oob = decode_and_sample_video(pathological_videos["dynamic_24f"], start_sec=50.0, end_sec=60.0)
    assert not rep_oob.ok and rep_oob.status == "OOB_WINDOW"


def test_zip_slip_blocked_and_chunked_offline_transfer(tmp_path: Path) -> None:
    bad_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../escape.txt", "pwned")
    with pytest.raises(ValueError, match="Zip-Slip"):
        safe_extract_zip(bad_zip, tmp_path / "extract_out")

    # Route 2 chunked split + verified reassembly
    src = tmp_path / "archive.bin"
    src.write_bytes(b"UARCAP_Q2_DATA_CHUNK_" * 500)
    manifest = split_file_into_chunks(src, tmp_path / "chunks", chunk_bytes=2048)
    res = reassemble_and_verify_chunks(manifest, tmp_path / "reassembled.bin")
    assert res["status"] == "IMPORTED_VERIFIED"
    assert (tmp_path / "reassembled.bin").read_bytes() == src.read_bytes()
