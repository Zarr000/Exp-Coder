"""
Document Reranker.

Reranks retrieved documents by relevance.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class ScoredDocument:
    """Document with relevance score."""

    text: str
    score: float
    metadata: dict[str, Any]


class Reranker:
    """
    Reranks retrieved documents.

    Supports multiple reranking strategies.
    """

    def __init__(self, model: Optional[Any] = None) -> None:
        """Initialize reranker."""
        self.model = model

    def rerank_by_keyword_overlap(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5
    ) -> list[ScoredDocument]:
        """Rerank by keyword overlap."""
        query_terms = set(query.lower().split())
        results: list[ScoredDocument] = []

        for i, doc in enumerate(documents):
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            score = overlap / max(len(query_terms), 1)

            results.append(ScoredDocument(
                text=doc,
                score=score,
                metadata={"index": i}
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def rerank_by_embedding_similarity(
        self,
        query: str,
        documents: list[str],
        embeddings: Any,
        top_k: int = 5
    ) -> list[ScoredDocument]:
        """Rerank by embedding similarity."""
        if not hasattr(self, "_embeddings") or self._embeddings is None:
            return self.rerank_by_keyword_overlap(query, documents, top_k)

        query_emb = embeddings.encode([query])
        doc_embs = embeddings.encode(documents)

        scores = cosine_similarity(query_emb, doc_embs)[0]

        results = [
            ScoredDocument(
                text=doc,
                score=float(score),
                metadata={"index": i}
            )
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def rerank_by_length_penalty(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
        penalty_factor: float = 0.5
    ) -> list[ScoredDocument]:
        """Rerank with length penalty (favors concise answers)."""
        query_len = len(query.split())

        results: list[ScoredDocument] = []
        for i, doc in enumerate(documents):
            doc_len = len(doc.split())

            if doc_len < query_len:
                length_score = doc_len / query_len
            elif doc_len > query_len * 10:
                length_score = penalty_factor
            else:
                length_score = 1.0

            results.append(ScoredDocument(
                text=doc,
                score=length_score,
                metadata={"index": i, "doc_len": doc_len}
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def rerank_by_position(
        self,
        documents: list[str],
        top_k: int = 5,
        boost_recent: bool = True
    ) -> list[ScoredDocument]:
        """Rerank by position (recent items get boost)."""
        results = []

        for i, doc in enumerate(documents):
            if boost_recent:
                position_score = 1.0 / (i + 1)
            else:
                position_score = 1.0 - (i * 0.1)

            results.append(ScoredDocument(
                text=doc,
                score=position_score,
                metadata={"index": i}
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def rerank_combined(
        self,
        query: str,
        documents: list[str],
        strategies: list[str] = None,
        top_k: int = 5,
        weights: list[float] = None
    ) -> list[ScoredDocument]:
        """Combined reranking with multiple strategies."""
        strategies = strategies or ["keyword", "length"]
        weights = weights or [0.5, 0.5]

        all_scores = np.zeros(len(documents))

        if "keyword" in strategies:
            keyword_scores = np.array([
                self.rerank_by_keyword_overlap(query, documents, len(documents))[i].score
                for i in range(len(documents))
            ])
            all_scores += keyword_scores * weights[0]

        if "length" in strategies:
            length_scores = np.array([
                self.rerank_by_length_penalty(query, documents, len(documents))[i].score
                for i in range(len(documents))
            ])
            idx = 1 if len(weights) > 1 else 0
            all_scores += length_scores * weights[idx]

        results = [
            ScoredDocument(
                text=doc,
                score=float(all_scores[i]),
                metadata={"index": i}
            )
            for i, doc in enumerate(documents)
        ]

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between arrays."""
    dot_product = np.dot(a, b.T)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return dot_product / (norm_a[:, np.newaxis] * norm_b)