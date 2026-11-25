"""Faithful frozen D consensus reranker (full-caption mean pairwise ROUGE-L) and retriever adapters.

Preserves exact v0.8 reproducibility fidelity:
  - External caption deduplication groups by `.strip()` only (matching frozen UARCap implementation).
  - D quality score = mean pairwise full-caption ROUGE-L F1 across the top-M grouped candidates.
  - Provides clean RetrieverAdapter abstraction plus MockXCLIPRetriever for deterministic CPU E2E tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from hashlib import sha256
from typing import Any

from ..common.schemas import QueryRecord, RetrievalCandidate


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for tok_a in a:
        curr = [0] * (n + 1)
        for j, tok_b in enumerate(b, 1):
            if tok_a == tok_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = curr[j - 1] if curr[j - 1] >= prev[j] else prev[j]
        prev = curr
    return prev[n]


def rouge_l_f1(cand_a: str, cand_b: str) -> float:
    toks_a = cand_a.strip().split()
    toks_b = cand_b.strip().split()
    if not toks_a or not toks_b:
        return 0.0
    lcs = _lcs_length(toks_a, toks_b)
    if lcs == 0:
        return 0.0
    prec = lcs / len(toks_a)
    rec = lcs / len(toks_b)
    return 2.0 * prec * rec / (prec + rec)


def group_external_candidates_strip_only(raw_hits: list[dict[str, Any]], top_m: int = 20) -> list[RetrievalCandidate]:
    """Group external VideoUFO hits by caption.strip() only (exact v0.8 predecessor fidelity)."""
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for hit in raw_hits:
        cap = str(hit["caption"]).strip()
        if not cap:
            continue
        score = float(hit["score"])
        clip_id = str(hit["clip_id"])
        src_id = str(hit.get("source_id") or clip_id.rsplit(".", 1)[0])
        if cap not in grouped:
            if len(grouped) >= top_m:
                continue
            grouped[cap] = {"caption": cap, "score": score, "clip_ids": [clip_id], "source_ids": [src_id]}
        else:
            if clip_id not in grouped[cap]["clip_ids"]:
                grouped[cap]["clip_ids"].append(clip_id)
            if src_id not in grouped[cap]["source_ids"]:
                grouped[cap]["source_ids"].append(src_id)
            if score > grouped[cap]["score"]:
                grouped[cap]["score"] = score

    out: list[RetrievalCandidate] = []
    for idx, item in enumerate(grouped.values()):
        out.append(
            RetrievalCandidate(
                caption=item["caption"],
                score=item["score"],
                clip_ids=item["clip_ids"],
                source_ids=item["source_ids"],
                original_rank=idx,
            )
        )
    return out


def rerank_d_consensus(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    """Frozen D quality-aware consensus reranker: full-caption mean pairwise ROUGE-L F1."""
    n = len(candidates)
    if n == 0:
        return []
    scored: list[tuple[float, float, int, RetrievalCandidate]] = []
    for i, ci in enumerate(candidates):
        if n == 1:
            mean_rl = 1.0
        else:
            rl_sum = sum(rouge_l_f1(ci.caption, cj.caption) for j, cj in enumerate(candidates) if j != i)
            mean_rl = rl_sum / (n - 1)
        scored.append((mean_rl, ci.score, -(ci.original_rank if ci.original_rank is not None else i), ci))

    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    reranked: list[RetrievalCandidate] = []
    for r_idx, (mean_rl, _, _, cand) in enumerate(scored):
        reranked.append(
            RetrievalCandidate(
                caption=cand.caption,
                score=cand.score,
                clip_ids=list(cand.clip_ids),
                source_ids=list(cand.source_ids),
                rerank_score=round(mean_rl, 6),
                original_rank=cand.original_rank,
                rerank_rank=r_idx,
            )
        )
    return reranked


class RetrieverAdapter(ABC):
    @abstractmethod
    def retrieve_top20(self, query: QueryRecord) -> list[RetrievalCandidate]:
        """Return up to 20 grouped external VideoUFO candidates sorted by X-CLIP similarity."""


class MockXCLIPRetriever(RetrieverAdapter):
    """Deterministic CPU mock retriever returning 20 realistic VideoUFO-style candidates per query."""

    def __init__(self, corpus_pool: list[dict[str, Any]] | None = None) -> None:
        self.corpus_pool = corpus_pool or self._default_pool()

    @staticmethod
    def _default_pool() -> list[dict[str, Any]]:
        templates = [
            "a person is walking down a city street carrying a bag",
            "a man and a woman are talking indoors near a wooden table",
            "an animated character jumps across a colorful platform",
            "two dancers spin together on a stage with bright lights",
            "a red car drives quickly along an empty road",
            "four people gather around a laptop on a wooden table",
            "a cat plays with a small bowl on the kitchen floor",
            "someone throws cut grass into a green garbage bin",
            "a gardener rakes dry autumn leaves in a backyard",
            "a chef slices vegetables on a wooden cutting board",
            "children run through a grassy park under clear skies",
            "a musician plays an acoustic guitar on a small chair",
            "a dog catches a yellow ball in a sunny garden",
            "a swimmer dives into a blue indoor swimming pool",
            "a cyclist rides a bicycle along a narrow mountain trail",
            "a woman pours coffee into a white ceramic mug",
            "a student writes notes in a notebook at a library desk",
            "a group of friends laugh while eating dinner together",
            "a bird flies over calm ocean waves at sunset",
            "a mechanic repairs a bicycle wheel inside a small garage",
            "a person is talking and gesturing with both hands",
            "a character kneels down and signals to a small group",
        ]
        pool = []
        for idx, cap in enumerate(templates):
            yt = f"MOCKYT{idx:05d}"
            pool.append({"caption": cap, "clip_id": f"{yt}.{idx % 3}", "source_id": yt})
        return pool

    def retrieve_top20(self, query: QueryRecord) -> list[RetrievalCandidate]:
        ref_words = set(query.reference_description.lower().split())
        raw_hits: list[dict[str, Any]] = []
        for idx, entry in enumerate(self.corpus_pool):
            cap_words = set(entry["caption"].lower().split())
            overlap = len(ref_words & cap_words) / max(len(ref_words | cap_words), 1)
            tie_hash = int(sha256(f"{query.query_id}:{idx}".encode()).hexdigest()[:6], 16) / float(0xFFFFFF)
            score = round(0.20 + 0.55 * overlap + 0.08 * tie_hash, 6)
            raw_hits.append(
                {
                    "caption": entry["caption"],
                    "score": score,
                    "clip_id": entry["clip_id"],
                    "source_id": entry["source_id"],
                }
            )
        raw_hits.sort(key=lambda h: (-h["score"], h["clip_id"]))
        return group_external_candidates_strip_only(raw_hits, top_m=20)
