"""Synthetic dynamic & pathological MP4 fixture generator for Windows CPU testing (PyAV).

CRITICAL:
  - Strictly isolated from real data; generated into pytest tmp_path (never committed as large binary assets).
  - Dynamic videos change color gradients and moving bars every frame so repeated-first-frame bugs are caught!
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np


def make_dynamic_mp4(
    out_path: Path | str,
    num_frames: int = 24,
    width: int = 128,
    height: int = 96,
    fps: int = 12,
    vfr: bool = False,
    seed_offset: int = 0,
) -> Path:
    """Create a real MP4 whose pixel content visibly moves and changes color on every frame."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(p), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        pts_tick = 0
        for i in range(num_frames):
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img[:, :, 0] = (30 + (i * 9 + seed_offset * 17) % 210)
            img[:, :, 1] = (40 + (i * 13 + seed_offset * 7) % 200)
            img[:, :, 2] = (50 + (i * 19) % 190)
            # Moving bright white block across x and y so consecutive frames always have strong L1 diff
            x0 = int((i / max(num_frames, 1)) * (width - 24))
            y0 = int(((i * 3) % max(height - 20, 1)))
            img[y0 : y0 + 20, x0 : x0 + 24, :] = 250

            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = pts_tick
            pts_tick += (1 if i % 2 == 0 else 3) if vfr else 1
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return p


def make_repeated_first_frame_mp4(
    out_path: Path | str,
    num_frames: int = 20,
    width: int = 128,
    height: int = 96,
    fps: int = 10,
) -> Path:
    """Create a pathological MP4 where all frames are pixel-identical (simulating historical val497 bug)."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    static = np.full((height, width, 3), 120, dtype=np.uint8)
    static[20:50, 20:60, :] = [200, 40, 40]
    with av.open(str(p), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for i in range(num_frames):
            frame = av.VideoFrame.from_ndarray(static, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return p


def make_corrupt_mp4(out_path: Path | str) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"NOT_A_VALID_MP4_HEADER_CORRUPT_BYTES_00001111")
    return p
