"""Hybrid retrieval: reciprocal rank fusion of the lexical (BM25) and dense (pgvector) rankings (ARCHITECTURE §51).

RRF uses ranks only, so BM25 scores and cosine similarities never need calibrating against each other:
score(d) = sum_r w_r / (k + rank_r(d)). Ties keep the order of first appearance, so fusion is deterministic.
"""

from __future__ import annotations

VERSION = "rrf-0.1.0"


def rrf(rankings: list[list[str]], k: int = 60, weights: list[float] | None = None, limit: int | None = None) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights or [1.0] * len(rankings)):
        for rank, doc in enumerate(ranking, 1):
            scores[doc] = scores.get(doc, 0.0) + w / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: -kv[1])  # sorted() is stable: ties keep first appearance
    return fused[:limit] if limit else fused
