"""
Vector Store for Expera AI.

Embeddings storage and retrieval:
- Store embeddings
- Similarity search
- CRUD operations

Usage:
    store = VectorStore(dimensions=384)
    await store.add("text", embedding)
    results = await store.search(query_embedding, top_k=5)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class VectorEntry:
    """A vector entry."""

    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Search result."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorStoreConfig:
    """Vector store configuration."""

    dimensions: int = 384
    storage_path: str = "data/vectors"
    max_entries: int = 100000
    metric: str = "cosine"  # cosine, euclidean, dot


class VectorStore:
    """
    Vector store.

    Features:
    - Add vectors
    - Search
    - Delete
    - Persistence
    """

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """Initialize vector store."""
        self.config = config or VectorStoreConfig()
        self._entries: dict[str, VectorEntry] = {}
        self._storage_path = Path(self.config.storage_path)

    async def add(
        self,
        id: str,
        text: str,
        embedding: list[float],
        metadata: Optional[dict] = None,
    ) -> bool:
        """Add a vector."""
        # Check dimensions
        if len(embedding) != self.config.dimensions:
            logger.error(f"Dimension mismatch: {len(embedding)} != {self.config.dimensions}")
            return False

        entry = VectorEntry(
            id=id,
            text=text,
            embedding=embedding,
            metadata=metadata or {},
        )

        self._entries[id] = entry
        return True

    async def get(self, id: str) -> Optional[VectorEntry]:
        """Get a vector."""
        return self._entries.get(id)

    async def delete(self, id: str) -> bool:
        """Delete a vector."""
        if id in self._entries:
            del self._entries[id]
            return True
        return False

    async def search(
        self,
        query: list[float],
        top_k: int = 5,
        filter_func: Optional[callable] = None,
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        if not self._entries:
            return []

        # Calculate similarities
        results = []

        for entry in self._entries.values():
            # Apply filter if provided
            if filter_func and not filter_func(entry):
                continue

            score = self._calculate_similarity(query, entry.embedding)

            results.append(SearchResult(
                id=entry.id,
                text=entry.text,
                score=score,
                metadata=entry.metadata,
            ))

        # Sort by score
        if self.config.metric == "cosine":
            results.sort(key=lambda r: r.score, reverse=True)
        else:
            results.sort(key=lambda r: r.score)

        return results[:top_k]

    def _calculate_similarity(
        self,
        a: list[float],
        b: list[float],
    ) -> float:
        """Calculate similarity."""
        if self.config.metric == "cosine":
            return self._cosine_similarity(a, b)
        elif self.config.metric == "euclidean":
            return -self._euclidean_distance(a, b)  # Negate for sorting
        elif self.config.metric == "dot":
            return self._dot_product(a, b)
        return 0.0

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return dot / (mag_a * mag_b)

    def _euclidean_distance(self, a: list[float], b: list[float]) -> float:
        """Calculate Euclidean distance."""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def _dot_product(self, a: list[float], b: list[float]) -> float:
        """Calculate dot product."""
        return sum(x * y for x, y in zip(a, b))

    async def save(self) -> bool:
        """Save to file."""
        try:
            self._storage_path.mkdir(parents=True, exist_ok=True)

            data = {}
            for id, entry in self._entries.items():
                data[id] = {
                    "text": entry.text,
                    "embedding": entry.embedding,
                    "metadata": entry.metadata,
                }

            with open(self._storage_path / "vectors.json", "w") as f:
                json.dump(data, f)

            logger.info(f"Saved {len(data)} vectors")
            return True

        except Exception as e:
            logger.error(f"Save failed: {e}")
            return False

    async def load(self) -> bool:
        """Load from file."""
        try:
            path = self._storage_path / "vectors.json"
            if not path.exists():
                return False

            with open(path) as f:
                data = json.load(f)

            for id, entry_data in data.items():
                self._entries[id] = VectorEntry(
                    id=id,
                    text=entry_data["text"],
                    embedding=entry_data["embedding"],
                    metadata=entry_data.get("metadata", {}),
                )

            logger.info(f"Loaded {len(self._entries)} vectors")
            return True

        except Exception as e:
            logger.error(f"Load failed: {e}")
            return False

    def __len__(self) -> int:
        """Get entry count."""
        return len(self._entries)


# Export
__all__ = [
    "VectorStore",
    "VectorStoreConfig",
    "VectorEntry",
    "SearchResult",
]