"""
Retriever for RAG.

Retrieves relevant context:
- Semantic search
- Hybrid search
- Re-ranking

Usage:
    retriever = Retriever(vector_store)
    results = await retriever.retrieve(query, top_k=5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved chunk."""

    text: str
    source: str
    score: float
    metadata: dict[str, Any]


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""

    top_k: int = 5
    min_score: float = 0.5
    hybrid_alpha: float = 0.5  # lexical vs semantic


class Retriever:
    """
    Retriever for RAG.

    Features:
    - Semantic retrieval
    - Keyword retrieval
    - Hybrid retrieval
    """

    def __init__(
        self,
        vector_store=None,
        config: Optional[RetrievalConfig] = None,
    ):
        """Initialize retriever."""
        self.vector_store = vector_store
        self.config = config or RetrievalConfig()
        self._embeddings = None

    def set_embeddings(self, embeddings) -> None:
        """Set embeddings model."""
        self._embeddings = embeddings

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks."""
        top_k = top_k or self.config.top_k

        if not self.vector_store:
            logger.warning("No vector store configured")
            return []

        # Get query embedding
        query_embedding = None
        if self._embeddings:
            query_embedding = await self._embeddings.embed(query)

        # Search vector store
        results = []

        if query_embedding:
            vector_results = await self.vector_store.search(
                query_embedding,
                top_k=top_k * 2,  # Get more for re-ranking
            )

            for result in vector_results:
                if result.score >= self.config.min_score:
                    results.append(RetrievedChunk(
                        text=result.text,
                        source=result.id,
                        score=result.score,
                        metadata=result.metadata,
                    ))

        # Trim to top_k
        return results[:top_k]

    async def retrieve_hybrid(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Hybrid retrieval (semantic + keyword)."""
        top_k = top_k or self.config.top_k
        alpha = self.config.hybrid_alpha

        # Get semantic results
        semantic_results = await self.retrieve(query, top_k=top_k)

        # Get keyword results
        keyword_results = await self._retrieve_keyword(query, top_k=top_k)

        # Combine and re-rank
        combined = {}
        seen = set()

        for r in semantic_results:
            key = r.source
            if key not in seen:
                combined[key] = alpha * r.score
                seen.add(key)

        for r in keyword_results:
            key = r.source
            if key not in seen:
                combined[key] = (1 - alpha) * r.score
                seen.add(key)
            else:
                combined[key] += (1 - alpha) * r.score

        # Sort by combined score
        sorted_keys = sorted(combined.keys(), key=lambda k: combined[k], reverse=True)

        # Build results
        results = []
        for key in sorted_keys[:top_k]:
            entry = await self.vector_store.get(key)
            if entry:
                results.append(RetrievedChunk(
                    text=entry.text,
                    source=key,
                    score=combined[key],
                    metadata=entry.metadata,
                ))

        return results

    async def _retrieve_keyword(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Keyword-based retrieval."""
        results = []

        if not self.vector_store:
            return results

        # Simple keyword matching
        query_words = set(query.lower().split())

        # This is a simplified version
        # Full implementation would use BM25 or similar
        return results

    async def retrieve_by_metadata(
        self,
        filters: dict[str, Any],
        limit: int = 10,
    ) -> list[RetrievedChunk]:
        """Retrieve by metadata filters."""
        results = []

        if not self.vector_store:
            return results

        # Apply filters
        def filter_func(entry):
            for key, value in filters.items():
                if entry.metadata.get(key) != value:
                    return False
            return True

        vector_results = await self.vector_store.search(
            [0] * self.vector_store.config.dimensions,
            top_k=limit,
            filter_func=filter_func,
        )

        for result in vector_results:
            results.append(RetrievedChunk(
                text=result.text,
                source=result.id,
                score=result.score,
                metadata=result.metadata,
            ))

        return results

    async def get_context(
        self,
        query: str,
        max_tokens: int = 2000,
    ) -> str:
        """Get combined context string."""
        results = await self.retrieve(query, top_k=10)

        context_parts = []
        total_chars = 0

        for result in results:
            # Rough token estimate: 4 chars per token
            estimated_tokens = len(result.text) // 4

            if total_chars + estimated_tokens > max_tokens * 4:
                break

            context_parts.append(result.text)
            total_chars += estimated_tokens

        return "\n\n".join(context_parts)


# Export
__all__ = [
    "Retriever",
    "RetrievalConfig",
    "RetrievedChunk",
]