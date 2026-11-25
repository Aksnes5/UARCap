"""Real VideoUFO X-CLIP query-conditioned retriever faithfully ported from frozen BN scripts.

Directly reproduces:
  - 52b5ed3a:scripts/extract_query_features_v2.py (16 center frames, short-side 256 NEAREST, 224 center crop,
    ImageNet mean/std normalization, X-CLIP k600_16_16.pth video + 196-patch spatial image features)
  - 52b5ed3a:scripts/make_external_retrieval_test.py (VideoUFO.csv + manifests/clips_full.tsv strip-only
    caption grouping, lexicographically sorted clip_ids, rep_rows=id2row[groups[k][0]], chunked
    query-conditioned VideoSpecificPrompt scoring `tf = tf + prompts_generator(tf, img)` + cosine top-20)

Supports both:
  1. Real BN GPU backend (`BNXCLIPBackend` loading frozen `models.xclip_all` + `k600_16_16.pth`)
  2. Injectable CPU fake X-CLIP backend for contract testing on Windows without GPU/1M-index!
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

import av
import numpy as np
from PIL import Image

from ..common.io_utils import atomic_write_json, sha256_file
from ..common.schemas import QueryRecord, RetrievalCandidate
from .reranker import RetrieverAdapter

XCLIP_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
XCLIP_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


@dataclass
class VideoUFOIndexConfig:
    videoufo_csv_path: str  # /home/bn/datasets/VideoUFO/metadata/VideoUFO.csv
    clips_full_tsv_path: str  # /home/bn/videoufo_research/manifests/clips_full.tsv
    brief_text_emb_npy_path: str  # /home/bn/videoufo_research/embeddings/videoufo_brief_text_fp16.npy
    xclip_repo_dir: str = "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/EvcapProject/X-CLIP"
    xclip_ckpt_path: str = (
        "/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/EvcapProject/X-CLIP/modelpth/k600_16_16.pth"
    )
    video_feature_dir: str = "/home/bn/videoufo_research/features/videollama_faithful_v2"
    num_frames: int = 16
    chunk_size: int = 100000
    topk: int = 20

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preprocess_video_for_xclip(video_path: Path | str, num_frames: int = 16) -> np.ndarray:
    """Exact replica of extract_query_features_v2.py::load_clip using PyAV + PIL NEAREST (no cv2 required).

    Returns float32 array of shape (T, 3, 224, 224) normalized with ImageNet MEAN/STD.
    """
    frames_rgb: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise ValueError(f"No video stream in {video_path}")
        for frame in container.decode(stream):
            frames_rgb.append(frame.to_ndarray(format="rgb24"))
    n = len(frames_rgb)
    if n == 0:
        raise ValueError(f"Decoded 0 frames from {video_path}")
    # Exact center-frame indexing from extract_query_features_v2.py L33:
    idx = np.clip(((np.arange(num_frames) + 0.5) * n / num_frames).astype(int), 0, n - 1)
    out = np.empty((num_frames, 224, 224, 3), dtype=np.float32)
    for i, fi in enumerate(idx):
        fr = frames_rgb[int(fi)]
        h, w = fr.shape[:2]
        if h <= w:
            nh, nw = 256, max(256, int(round(w * 256 / h)))
        else:
            nw, nh = 256, max(256, int(round(h * 256 / w)))
        pil_fr = Image.fromarray(fr).resize((nw, nh), Image.Resampling.NEAREST)
        arr = np.asarray(pil_fr, dtype=np.float32)
        y0, x0 = (nh - 224) // 2, (nw - 224) // 2
        crop = arr[y0 : y0 + 224, x0 : x0 + 224]
        out[i] = (crop - XCLIP_MEAN) / XCLIP_STD
    return np.ascontiguousarray(np.transpose(out, (0, 3, 1, 2)))


class XCLIPBackend(ABC):
    """Pluggable X-CLIP video encoder & VideoSpecificPrompt text-chunk scorer."""

    backend_id: str = "abstract_xclip"

    @abstractmethod
    def encode_video(self, video_tensor_tchw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (video_emb [D], spatial_img_feat [P, D])."""

    @abstractmethod
    def score_text_chunk(
        self,
        query_video_emb: np.ndarray,
        query_img_feat: np.ndarray,
        text_chunk_emb: np.ndarray,
    ) -> np.ndarray:
        """Apply VideoSpecificPrompt conditioning and return cosine similarity vector [chunk_len]."""


class BNXCLIPBackend(XCLIPBackend):
    """Real BN GPU X-CLIP backend loading frozen models.xclip_all + k600_16_16.pth."""

    backend_id = "bn_xclip_k600_16_16"

    def __init__(self, xclip_repo_dir: str, xclip_ckpt_path: str, num_frames: int = 16) -> None:
        self.xclip_repo_dir = xclip_repo_dir
        self.xclip_ckpt_path = xclip_ckpt_path
        self.num_frames = num_frames
        self._model = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        if not Path(self.xclip_repo_dir).is_dir():
            raise FileNotFoundError(f"Frozen X-CLIP repo directory missing: {self.xclip_repo_dir}")
        if not Path(self.xclip_ckpt_path).is_file():
            raise FileNotFoundError(f"Frozen X-CLIP checkpoint missing: {self.xclip_ckpt_path}")
        import os
        import torch

        class _SilentLog:
            def info(self, _m: str) -> None:
                pass

        if self.xclip_repo_dir not in sys.path:
            sys.path.insert(0, self.xclip_repo_dir)
        old_cwd = os.getcwd()
        try:
            os.chdir(self.xclip_repo_dir)
            from models import xclip_all as xclip  # type: ignore[import-not-found]

            sd = torch.load(self.xclip_ckpt_path, map_location="cpu")["model"]
            self._model = (
                xclip.build_model(
                    sd,
                    T=self.num_frames,
                    droppath=0.0,
                    use_checkpoint=False,
                    logger=_SilentLog(),
                    prompts_alpha=0.1,
                    prompts_layers=2,
                    use_cache=True,
                    mit_layers=1,
                    text_batch_size=0,
                )
                .cuda()
                .eval()
            )
        finally:
            os.chdir(old_cwd)

    def encode_video(self, video_tensor_tchw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._lazy_load()
        import torch

        x = torch.from_numpy(video_tensor_tchw).unsqueeze(0).cuda().half()
        with torch.no_grad(), torch.cuda.amp.autocast():
            v, im = self._model.encode_video(x)
            im = im.mean(dim=1)
        return v[0].float().cpu().numpy(), im[0].float().cpu().numpy()

    def score_text_chunk(
        self,
        query_video_emb: np.ndarray,
        query_img_feat: np.ndarray,
        text_chunk_emb: np.ndarray,
    ) -> np.ndarray:
        self._lazy_load()
        import torch
        import torch.nn.functional as F

        qv = F.normalize(torch.from_numpy(query_video_emb).float().cuda(), dim=0).half()
        img = torch.from_numpy(query_img_feat).float().cuda().half().unsqueeze(0)
        tf = torch.from_numpy(np.asarray(text_chunk_emb)).cuda().half().unsqueeze(0)
        with torch.no_grad(), torch.cuda.amp.autocast():
            tf = tf + self._model.prompts_generator(tf, img)
            sc = qv @ F.normalize(tf[0], dim=1).T
        return sc.float().cpu().numpy()


def load_videoufo_grouped_index(cfg: VideoUFOIndexConfig) -> dict[str, Any]:
    """Load VideoUFO.csv + manifests/clips_full.tsv and construct exact strip-only caption groups."""
    csv_p = Path(cfg.videoufo_csv_path)
    tsv_p = Path(cfg.clips_full_tsv_path)
    emb_p = Path(cfg.brief_text_emb_npy_path)
    for label, p in (("VideoUFO.csv", csv_p), ("clips_full.tsv", tsv_p), ("brief_text_emb.npy", emb_p)):
        if not p.is_file():
            raise FileNotFoundError(f"Required frozen VideoUFO retrieval asset missing ({label}): {p}")

    csv.field_size_limit(10**9)
    ids: list[str] = []
    brief: list[str] = []
    topic: list[str] = []
    with csv_p.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        i_id, i_b = header.index("ID"), header.index("Brief Caption")
        i_t = header.index("Topic") if "Topic" in header else None
        for row in reader:
            if row:
                ids.append(row[i_id])
                brief.append(row[i_b].strip())
                topic.append((row[i_t].strip().lower() or "unknown") if i_t is not None else "unknown")

    id2row = {c: i for i, c in enumerate(ids)}
    full_ids = [
        ln.split("\t")[0].strip()
        for ln in tsv_p.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("clip_id")
    ]

    groups: dict[str, list[str]] = {}
    order: list[str] = []
    for cid in full_ids:
        if cid not in id2row:
            continue
        key = brief[id2row[cid]]
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(cid)
    for k in groups:
        groups[k].sort()

    rep_rows = np.array([id2row[groups[k][0]] for k in order], dtype=np.int64)
    emb_mmap = np.load(str(emb_p), mmap_mode="r")
    if emb_mmap.shape[0] < int(rep_rows.max() + 1) if len(rep_rows) else False:
        raise ValueError(
            f"Embedding matrix rows ({emb_mmap.shape[0]}) smaller than max rep_row ({int(rep_rows.max())})"
        )

    return {
        "order": order,
        "groups": groups,
        "id2row": id2row,
        "topic": topic,
        "rep_rows": rep_rows,
        "emb_mmap": emb_mmap,
        "n_clips_manifest": len(full_ids),
        "n_unique_caption_groups": len(order),
        "embedding_shape": list(emb_mmap.shape),
    }


class VideoUFOXCLIPRetriever(RetrieverAdapter):
    """Real VideoUFO X-CLIP retriever reproducing frozen BN retrieval + representative clip verification."""

    retriever_id = "videoufo_xclip_faithful_v2"

    def __init__(
        self,
        cfg: VideoUFOIndexConfig,
        xclip_backend: XCLIPBackend | None = None,
        verify_feature_files_exist: bool = False,
    ) -> None:
        self.cfg = cfg
        self.verify_feature_files_exist = verify_feature_files_exist
        self.backend = xclip_backend or BNXCLIPBackend(
            cfg.xclip_repo_dir, cfg.xclip_ckpt_path, num_frames=cfg.num_frames
        )
        self._index: dict[str, Any] | None = None

    def _ensure_index(self) -> dict[str, Any]:
        if self._index is None:
            self._index = load_videoufo_grouped_index(self.cfg)
        return self._index

    def get_asset_manifest(self) -> dict[str, Any]:
        idx = self._ensure_index()
        return {
            "retriever_id": self.retriever_id,
            "xclip_backend": self.backend.backend_id,
            "videoufo_csv_path": self.cfg.videoufo_csv_path,
            "clips_full_tsv_path": self.cfg.clips_full_tsv_path,
            "clips_full_tsv_sha256": sha256_file(self.cfg.clips_full_tsv_path),
            "brief_text_emb_npy_path": self.cfg.brief_text_emb_npy_path,
            "n_clips_manifest": idx["n_clips_manifest"],
            "n_unique_caption_groups": idx["n_unique_caption_groups"],
            "embedding_shape": idx["embedding_shape"],
            "video_feature_dir": self.cfg.video_feature_dir,
            "topk": self.cfg.topk,
            "chunk_size": self.cfg.chunk_size,
        }

    def write_asset_manifest(self, out_path: Path | str) -> Path:
        return atomic_write_json(out_path, self.get_asset_manifest())

    def retrieve_top20(self, query: QueryRecord) -> list[RetrievalCandidate]:
        idx = self._ensure_index()
        order: list[str] = idx["order"]
        groups: dict[str, list[str]] = idx["groups"]
        rep_rows: np.ndarray = idx["rep_rows"]
        emb_mmap: np.ndarray = idx["emb_mmap"]

        vtensor = preprocess_video_for_xclip(query.video_path, num_frames=self.cfg.num_frames)
        qv, qim = self.backend.encode_video(vtensor)

        n_groups = len(order)
        k_eff = min(self.cfg.topk, n_groups)
        if k_eff == 0:
            raise ValueError("Loaded VideoUFO index has 0 caption groups")

        best_scores = np.full((0,), -1e9, dtype=np.float32)
        best_indices = np.zeros((0,), dtype=np.int64)

        for start in range(0, n_groups, self.cfg.chunk_size):
            end = min(start + self.cfg.chunk_size, n_groups)
            chunk_rows = rep_rows[start:end]
            chunk_emb = np.asarray(emb_mmap[chunk_rows], dtype=np.float32)
            scores = np.asarray(self.backend.score_text_chunk(qv, qim, chunk_emb), dtype=np.float32)
            local_k = min(k_eff, len(scores))
            top_local = np.argpartition(-scores, local_k - 1)[:local_k]
            cand_scores = np.concatenate([best_scores, scores[top_local]])
            cand_indices = np.concatenate([best_indices, top_local + start])
            # Exact deterministic top-k sort (score desc, index asc)
            sel = np.lexsort((cand_indices, -cand_scores))[:k_eff]
            best_scores = cand_scores[sel]
            best_indices = cand_indices[sel]

        feat_root = Path(self.cfg.video_feature_dir)
        out: list[RetrievalCandidate] = []
        for rank, (gi, sc) in enumerate(zip(best_indices.tolist(), best_scores.tolist())):
            cap = order[gi]
            cids = list(groups[cap])  # Already sorted lexicographically
            rep_clip = cids[0]
            if self.verify_feature_files_exist:
                pt_path = feat_root / f"{rep_clip}.pt"
                if not pt_path.is_file():
                    raise FileNotFoundError(f"Missing cached external visual feature for representative clip: {pt_path}")
            out.append(
                RetrievalCandidate(
                    caption=cap,
                    score=float(sc),
                    clip_ids=cids,
                    source_ids=sorted({c.rsplit(".", 1)[0] for c in cids}),
                    original_rank=rank,
                )
            )
        return out


class SubprocessVideoUFOXCLIPRetriever(RetrieverAdapter):
    """Cross-env subprocess retriever invoking `xclip_worker.py` in the frozen X-CLIP/Video-LLaMA env.

    Allows `uarcap_q2_cpu` to run real VideoUFO X-CLIP retrieval without installing CUDA in the CPU env,
    with shared GPU lock enforcement and guaranteed VRAM release when the worker exits.
    """

    retriever_id = "videoufo_xclip_faithful_v2_subprocess"

    def __init__(
        self,
        cfg: VideoUFOIndexConfig,
        python_executable: str | None = None,
        conda_env: str | None = None,
        work_dir: Path | str | None = None,
        gpu_lock_path: Path | None = None,
        gpu_uuid: str | None = None,
        require_nvidia_smi_idle: bool = False,
        verify_feature_files_exist: bool = True,
        subprocess_runner: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.python_executable = python_executable
        self.conda_env = conda_env
        self.work_dir = Path(work_dir) if work_dir else None
        self.gpu_lock_path = gpu_lock_path
        self.gpu_uuid = gpu_uuid
        self.require_nvidia_smi_idle = require_nvidia_smi_idle
        self.verify_feature_files_exist = verify_feature_files_exist
        self.subprocess_runner = subprocess_runner
        self._cache: dict[str, list[RetrievalCandidate]] = {}
        self._manifest: dict[str, Any] | None = None

    def prefetch_batch(self, queries: list[QueryRecord]) -> dict[str, list[RetrievalCandidate]]:
        import tempfile
        from ..common.gpu_lock import GPULockConfig
        from ..common.gpu_runtime import run_isolated_gpu_subprocess
        from ..common.io_utils import read_json

        missing = [q for q in queries if q.query_id not in self._cache]
        if not missing:
            return {q.query_id: self._cache[q.query_id] for q in queries}

        worker_script = Path(__file__).resolve().parent / "xclip_worker.py"
        lock_cfg = (
            GPULockConfig(
                lock_path=self.gpu_lock_path,
                run_id="xclip_retriever_batch",
                gpu_uuid=self.gpu_uuid,
                require_nvidia_smi_idle=self.require_nvidia_smi_idle,
            )
            if self.gpu_lock_path
            else None
        )

        with tempfile.TemporaryDirectory(prefix="uarcap_xclip_", dir=str(self.work_dir) if self.work_dir else None) as td:
            req_p = Path(td) / "xclip_request.json"
            resp_p = Path(td) / "xclip_response.json"
            atomic_write_json(
                req_p,
                {
                    "index_config": self.cfg.to_dict(),
                    "verify_feature_files_exist": self.verify_feature_files_exist,
                    "queries": [q.to_dict() for q in missing],
                },
            )
            proc = run_isolated_gpu_subprocess(
                script_or_module=str(worker_script),
                script_args=["--request-json", str(req_p), "--response-json", str(resp_p)],
                python_executable=self.python_executable,
                conda_env=self.conda_env,
                gpu_lock_cfg=lock_cfg,
                subprocess_runner=self.subprocess_runner,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"X-CLIP worker failed (rc={proc.returncode}): {proc.stderr[-1000:]}")
            if not resp_p.is_file():
                raise RuntimeError(f"X-CLIP worker succeeded without writing response JSON: {resp_p}")
            payload = read_json(resp_p)
            self._manifest = payload.get("manifest")
            for qid, cand_dicts in payload["results"].items():
                self._cache[qid] = [RetrievalCandidate(**cd) for cd in cand_dicts]
        return {q.query_id: self._cache[q.query_id] for q in queries}

    def retrieve_top20(self, query: QueryRecord) -> list[RetrievalCandidate]:
        if query.query_id not in self._cache:
            self.prefetch_batch([query])
        return self._cache[query.query_id]

    def write_asset_manifest(self, out_path: Path | str) -> Path:
        if self._manifest is None:
            raise RuntimeError("No X-CLIP retrieval executed yet; manifest unavailable.")
        return atomic_write_json(out_path, self._manifest)
