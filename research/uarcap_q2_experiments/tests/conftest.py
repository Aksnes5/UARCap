"""Pytest shared fixtures for UARCap Q2 offline CPU test suite."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from fixtures.build_synthetic_fixtures import (
    make_corrupt_mp4,
    make_dynamic_mp4,
    make_repeated_first_frame_mp4,
)
from uarcap_q2.datasets.dream1k import DREAM1K_DOMAINS


@pytest.fixture(scope="session")
def synthetic_dream1k_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a tiny isolated 10-clip synthetic DREAM-1K dataset (2 clips x 5 domains) with dynamic MP4s."""
    root = tmp_path_factory.mktemp("dream1k_synth")
    vdir = root / "video"
    jdir = root / "json"
    vdir.mkdir(parents=True)
    jdir.mkdir(parents=True)

    meta_rows = []
    base_vids = [1, 2, 201, 202, 401, 402, 601, 602, 801, 802]
    for idx, vid in enumerate(base_vids):
        dom = DREAM1K_DOMAINS[idx // 2]
        mp4 = make_dynamic_mp4(vdir / f"{vid}.mp4", num_frames=20, fps=10, seed_offset=idx)
        meta_rows.append(
            {
                "idx": str(vid),
                "vid": vid,
                "video_file": mp4.name,
                "source": dom,
                "duration": 2.0,
                "description": f"A character walks forward and waves near a table in scene {vid}.",
                "events": [
                    f"A character walks forward in scene {vid}.",
                    "The character waves near a table.",
                    "The scene fades out.",
                ],
                "n_subjects": 1,
                "n_shots": 1,
                "n_events": 3,
            }
        )
    (jdir / "metadata.json").write_text(json.dumps(meta_rows, indent=2), encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def pathological_videos(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("pathological_vids")
    return {
        "dynamic_24f": make_dynamic_mp4(root / "dynamic_24f.mp4", num_frames=24, fps=12),
        "short_4f": make_dynamic_mp4(root / "short_4f.mp4", num_frames=4, fps=12),
        "single_1f": make_dynamic_mp4(root / "single_1f.mp4", num_frames=1, fps=10),
        "vfr_20f": make_dynamic_mp4(root / "vfr_20f.mp4", num_frames=20, fps=10, vfr=True),
        "repeated_first_frame": make_repeated_first_frame_mp4(root / "repeated_first.mp4", num_frames=20, fps=10),
        "corrupt": make_corrupt_mp4(root / "corrupt.mp4"),
    }


@pytest.fixture(scope="session")
def synthetic_vidhalluc_json(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("vidhalluc_synth")
    payload = {
        "1": {
            "vcCwvRYqU2I_clip_4": {
                "Question": "What is the prominent action in the video?",
                "Choices": {
                    "A": "collecting trash from the yard",
                    "B": "throwing grass in the garbage",
                    "C": "watering plants",
                    "D": "raking leaves",
                },
                "Correct Answer": "B",
            },
            "vcCwvRYqU2I_clip_2": {
                "Question": "What is the predominant action captured in the video?",
                "Choices": {
                    "A": "throwing grass in the garbage",
                    "B": "raking leaves",
                    "C": "trimming hedges",
                    "D": "watering plants",
                },
                "Correct Answer": "B",
            },
        },
        "2": {
            "OBb4013eIc8_clip_1": {
                "Question": "What is the primary action?",
                "Choices": {
                    "A": "slicing vegetables",
                    "B": "boiling water",
                    "C": "washing dishes",
                    "D": "peeling potatoes",
                },
                "Correct Answer": "A",
            },
            "OVERLAP0001_clip_1": {
                "Question": "What is the primary action?",
                "Choices": {
                    "A": "riding a bicycle",
                    "B": "walking a dog",
                    "C": "running fast",
                    "D": "swimming",
                },
                "Correct Answer": "A",
            },
        },
        "3": {
            "HISTMSRVTT1_clip_2": {
                "Question": "What is happening?",
                "Choices": {"A": "playing guitar", "B": "playing piano"},
                "Correct Answer": "A",
            },
            "UNIQSRC0002_clip_3": {
                "Question": "What is happening?",
                "Choices": {"A": "pouring coffee", "B": "drinking tea"},
                "Correct Answer": "A",
            },
        },
    }
    p = root / "ach_mcq.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p
