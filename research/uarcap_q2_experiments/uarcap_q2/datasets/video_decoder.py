"""Strict video decoding, uniform 16-frame sampling, repeated-first-frame guard, dHash, and EVCap TSV packing.

Directly prevents recurrence of the historical val497 raw-video frame-input bug by:
  1. Decoding real frames sequentially via PyAV (never repeating frame 0 on seek failure).
  2. Recording exact sampled frame indices, PTS timestamps, per-frame SHA-256 digests, and
     consecutive-frame pixel L1 differences.
  3. Explicitly flagging corrupted, zero-frame, single-frame, out-of-bounds temporal windows,
     insufficient-frame, and repeated-first-frame anomalies.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import av
import numpy as np
from PIL import Image

from ..common.io_utils import atomic_write_text, sha256_bytes, sha256_file


class VideoDecodeError(ValueError):
    """Raised when a video cannot be decoded or violates temporal/frame integrity."""


@dataclass
class FrameSamplingReport:
    video_path: str
    video_sha256: str | None
    ok: bool
    status: str  # OK | CORRUPT | ZERO_FRAMES | SINGLE_FRAME | SHORT_VIDEO_PADDED | OOB_WINDOW | REPEATED_FIRST_FRAME
    total_decoded_frames: int
    requested_num_frames: int
    sampled_indices: list[int] = field(default_factory=list)
    sampled_pts_sec: list[float] = field(default_factory=list)
    duration_sec: float | None = None
    is_vfr_detected: bool = False
    unique_frame_hashes: int = 0
    mean_consecutive_l1_diff: float = 0.0
    frame_sha256s: list[str] = field(default_factory=list)
    frame_dhashes: list[str] = field(default_factory=list)
    combined_visual_sha256: str = ""
    warnings: list[str] = field(default_factory=list)
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_image_dhash(img: Image.Image, hash_size: int = 8) -> str:
    """Compute a 64-bit perceptual difference hash (dHash) as 16-char hex."""
    gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(bool(b))
    return f"{bits:016x}"


def hamming_distance_hex(h1: str, h2: str) -> int:
    return (int(h1, 16) ^ int(h2, 16)).bit_count()


def uniform_frame_indices(total_frames: int, num_frames: int) -> list[int]:
    if total_frames <= 0 or num_frames <= 0:
        raise ValueError("total_frames and num_frames must be positive")
    if total_frames == 1:
        return [0] * num_frames
    lin = np.linspace(0, total_frames - 1, num=num_frames)
    return [int(round(x)) for x in lin]


def decode_and_sample_video(
    video_path: Path | str,
    num_frames: int = 16,
    target_size: tuple[int, int] = (224, 224),
    start_sec: float | None = None,
    end_sec: float | None = None,
    strict_reject_repeated_first_frame: bool = True,
    max_decode_frames: int = 20000,
) -> tuple[list[Image.Image], FrameSamplingReport]:
    """Decode video with PyAV, validate temporal bounds, and uniformly sample num_frames RGB images."""
    vpath = Path(video_path)
    if not vpath.is_file():
        rep = FrameSamplingReport(
            video_path=str(vpath),
            video_sha256=None,
            ok=False,
            status="CORRUPT",
            total_decoded_frames=0,
            requested_num_frames=num_frames,
            error_message=f"Video file does not exist: {vpath}",
        )
        return [], rep

    vsha = sha256_file(vpath)
    frames_rgb: list[np.ndarray] = []
    pts_list: list[float] = []
    warnings: list[str] = []
    duration_sec: float | None = None

    try:
        with av.open(str(vpath)) as container:
            vstreams = [s for s in container.streams if s.type == "video"]
            if not vstreams:
                raise VideoDecodeError("No video stream found")
            stream = vstreams[0]
            tb = float(stream.time_base) if stream.time_base is not None else None
            if container.duration is not None:
                duration_sec = float(container.duration) / float(av.time_base)
            elif stream.duration is not None and tb is not None:
                duration_sec = float(stream.duration) * tb

            for idx, frame in enumerate(container.decode(stream)):
                if idx >= max_decode_frames:
                    warnings.append(f"Reached max_decode_frames={max_decode_frames}")
                    break
                arr = frame.to_ndarray(format="rgb24")
                frames_rgb.append(arr)
                if frame.pts is not None and tb is not None:
                    pts_list.append(float(frame.pts) * tb)
                else:
                    pts_list.append(float(idx) / max(float(stream.average_rate or 25.0), 1.0))
    except Exception as exc:
        rep = FrameSamplingReport(
            video_path=str(vpath),
            video_sha256=vsha,
            ok=False,
            status="CORRUPT",
            total_decoded_frames=0,
            requested_num_frames=num_frames,
            error_message=f"PyAV decode failed: {exc}",
        )
        return [], rep

    total = len(frames_rgb)
    if total == 0:
        rep = FrameSamplingReport(
            video_path=str(vpath),
            video_sha256=vsha,
            ok=False,
            status="ZERO_FRAMES",
            total_decoded_frames=0,
            requested_num_frames=num_frames,
            duration_sec=duration_sec,
            error_message="Decoded 0 video frames",
        )
        return [], rep

    if duration_sec is None and pts_list:
        duration_sec = max(pts_list[-1] - pts_list[0], 0.0)

    # Detect Variable Frame Rate (VFR)
    is_vfr = False
    if len(pts_list) >= 4:
        deltas = np.diff(np.array(pts_list, dtype=np.float64))
        pos_deltas = deltas[deltas > 1e-6]
        if len(pos_deltas) >= 3 and (float(pos_deltas.max()) / float(pos_deltas.min())) > 1.25:
            is_vfr = True
            warnings.append("Variable frame rate (VFR) detected; sampling uniformly by decoded frame index.")

    # Optional temporal window validation
    valid_indices = list(range(total))
    if start_sec is not None or end_sec is not None:
        s0 = 0.0 if start_sec is None else float(start_sec)
        e0 = (duration_sec if duration_sec is not None else pts_list[-1]) if end_sec is None else float(end_sec)
        if s0 < 0 or e0 < s0 or (duration_sec is not None and s0 > duration_sec + 0.25):
            rep = FrameSamplingReport(
                video_path=str(vpath),
                video_sha256=vsha,
                ok=False,
                status="OOB_WINDOW",
                total_decoded_frames=total,
                requested_num_frames=num_frames,
                duration_sec=duration_sec,
                is_vfr_detected=is_vfr,
                error_message=f"Invalid or out-of-bounds temporal window [{s0}, {e0}] for duration={duration_sec}",
            )
            return [], rep
        valid_indices = [i for i, t in enumerate(pts_list) if s0 - 1e-3 <= t <= e0 + 1e-3]
        if not valid_indices:
            rep = FrameSamplingReport(
                video_path=str(vpath),
                video_sha256=vsha,
                ok=False,
                status="OOB_WINDOW",
                total_decoded_frames=total,
                requested_num_frames=num_frames,
                duration_sec=duration_sec,
                is_vfr_detected=is_vfr,
                error_message=f"No decoded frames fall within window [{s0}, {e0}]",
            )
            return [], rep

    sub_total = len(valid_indices)
    rel_idx = uniform_frame_indices(sub_total, num_frames)
    sampled_abs_idx = [valid_indices[i] for i in rel_idx]
    sampled_pts = [round(pts_list[i], 4) for i in sampled_abs_idx]

    pil_images: list[Image.Image] = []
    frame_hashes: list[str] = []
    dhashes: list[str] = []
    resized_arrays: list[np.ndarray] = []

    for abs_i in sampled_abs_idx:
        im = Image.fromarray(frames_rgb[abs_i]).resize(target_size, Image.Resampling.BILINEAR)
        pil_images.append(im)
        arr_r = np.asarray(im, dtype=np.uint8)
        resized_arrays.append(arr_r)
        frame_hashes.append(sha256_bytes(arr_r.tobytes()))
        dhashes.append(compute_image_dhash(im))

    unique_hashes = len(set(frame_hashes))
    diffs = [
        float(np.mean(np.abs(resized_arrays[k + 1].astype(np.float32) - resized_arrays[k].astype(np.float32))))
        for k in range(len(resized_arrays) - 1)
    ]
    mean_diff = float(np.mean(diffs)) if diffs else 0.0
    combined_sha = sha256_bytes("|".join(frame_hashes).encode("utf-8"))

    status = "OK"
    ok = True
    err = ""

    if sub_total == 1:
        status = "SINGLE_FRAME"
        ok = False
        err = "Video contains only 1 usable frame; cannot satisfy multi-frame video captioning protocol."
    elif sub_total < num_frames:
        status = "SHORT_VIDEO_PADDED"
        warnings.append(f"Video has only {sub_total} frames (< {num_frames}); deterministic nearest-frame resampling used.")
    elif (unique_hashes == 1) or (mean_diff < 0.5 and unique_hashes <= 4):
        status = "REPEATED_FIRST_FRAME"
        if strict_reject_repeated_first_frame:
            ok = False
            err = (
                f"Sampled frames are frozen/repeated (unique_hashes={unique_hashes}, mean_l1_diff={mean_diff:.4f}; "
                "repeated-first-frame bug guard triggered to prevent historical val497 failure mode)."
            )
        else:
            warnings.append("Sampled frames appear frozen/static (repeated-first-frame suspected).")

    rep = FrameSamplingReport(
        video_path=str(vpath),
        video_sha256=vsha,
        ok=ok,
        status=status,
        total_decoded_frames=total,
        requested_num_frames=num_frames,
        sampled_indices=sampled_abs_idx,
        sampled_pts_sec=sampled_pts,
        duration_sec=round(duration_sec, 4) if duration_sec is not None else None,
        is_vfr_detected=is_vfr,
        unique_frame_hashes=unique_hashes,
        mean_consecutive_l1_diff=round(mean_diff, 4),
        frame_sha256s=frame_hashes,
        frame_dhashes=dhashes,
        combined_visual_sha256=combined_sha,
        warnings=warnings,
        error_message=err,
    )
    return pil_images, rep


def pack_evcap_frame_tsv_bundle(
    sampled_items: list[tuple[str, list[Image.Image], str]],
    out_dir: Path | str,
    prefix: str = "dream1k_pilot30_16frames",
) -> dict[str, Path]:
    """Pack sampled 16-frame videos into the exact TSV + lineidx + YAML bundle expected by frozen EVCap.

    Each item is (query_id, list_of_16_PIL_images, reference_caption).
    Produces:
      - {prefix}.img.tsv + {prefix}.img.lineidx
      - {prefix}.label.tsv + {prefix}.label.lineidx
      - {prefix}.caption.tsv + {prefix}.caption.lineidx
      - {prefix}.yaml
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img_tsv = out / f"{prefix}.img.tsv"
    img_idx = out / f"{prefix}.img.lineidx"
    lbl_tsv = out / f"{prefix}.label.tsv"
    lbl_idx = out / f"{prefix}.label.lineidx"
    cap_tsv = out / f"{prefix}.caption.tsv"
    cap_idx = out / f"{prefix}.caption.lineidx"
    yaml_path = out / f"{prefix}.yaml"

    img_offsets: list[int] = []
    lbl_offsets: list[int] = []
    cap_offsets: list[int] = []

    with img_tsv.open("wb") as f_img, lbl_tsv.open("wb") as f_lbl, cap_tsv.open("wb") as f_cap:
        for qid, frames, ref_cap in sampled_items:
            if len(frames) != 16:
                raise ValueError(f"Expected 16 frames for frozen EVCap TSV, got {len(frames)} for {qid}")
            b64_frames: list[str] = []
            for im in frames:
                buf = BytesIO()
                im.save(buf, format="JPEG", quality=92)
                b64_frames.append(base64.b64encode(buf.getvalue()).decode("ascii"))
            img_offsets.append(f_img.tell())
            # Exact frozen vision_language_tsv.py L311 layout:
            # len(row) >= decoder_num_frames + 2 (=18), row[0]=key, row[1]=num_frames meta, row[2:]=16 base64 frames!
            img_line = f"{qid}\t{len(b64_frames)}\t" + "\t".join(b64_frames) + "\n"
            f_img.write(img_line.encode("utf-8"))

            cap_payload = json.dumps([{"caption": ref_cap}], ensure_ascii=False)
            lbl_offsets.append(f_lbl.tell())
            f_lbl.write(f"{qid}\t{cap_payload}\n".encode("utf-8"))
            cap_offsets.append(f_cap.tell())
            f_cap.write(f"{qid}\t{cap_payload}\n".encode("utf-8"))

    atomic_write_text(img_idx, "\n".join(str(x) for x in img_offsets) + ("\n" if img_offsets else ""))
    atomic_write_text(lbl_idx, "\n".join(str(x) for x in lbl_offsets) + ("\n" if lbl_offsets else ""))
    atomic_write_text(cap_idx, "\n".join(str(x) for x in cap_offsets) + ("\n" if cap_offsets else ""))
    yaml_body = (
        f"img: {img_tsv.resolve().as_posix()}\n"
        f"label: {lbl_tsv.resolve().as_posix()}\n"
        f"caption: {cap_tsv.resolve().as_posix()}\n"
    )
    atomic_write_text(yaml_path, yaml_body)
    return {
        "img": img_tsv,
        "img_lineidx": img_idx,
        "label": lbl_tsv,
        "label_lineidx": lbl_idx,
        "caption": cap_tsv,
        "caption_lineidx": cap_idx,
        "yaml": yaml_path,
    }
