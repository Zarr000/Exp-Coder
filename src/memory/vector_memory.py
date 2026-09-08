"""
Vector Memory for Expera AI.

Stores embeddings for semantic search.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class VectorEntry:
    """Vector memory entry."""

    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class VectorMemory:
    """
    Vector memory storage.

    Uses simple cosine similarity for search.
    """

    def __init__(self, storage_path: Optional[str] = None, dim: int = 384) -> None:
        """Initialize vector memory."""
        self.storage_path = storage_path
        self.dim = dim
        self._entries: dict[str, VectorEntry] = {}
        self._load()

    def _load(self) -> None:
        """Load from storage."""
        if self.storage_path:
            path = self.storage_path / "vectors.json"
            if path.exists():
                data = json.loads(path.read_text())
                self._entries = {
                    k: VectorEntry(
                        id=e["id"],
                        text=e["text"],
                        embedding=e["embedding"],
                        metadata=e.get("metadata", {}),
                        created_at=datetime.fromisoformat(e["created_at"]),
                    )
                    for k, e in data.items()
                }

    def _save(self) -> None:
        """Save to storage."""
        if self.storage_path:
            path = self.storage_path / "vectors.json"
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                k: {
                    "id": e.id,
                    "text": e.text,
                    "embedding": e.embedding,
                    "metadata": e.metadata,
                    "created_at": e.created_at.isoformat(),
                }
                for k, e in self._entries.items()
            }
            path.write_text(json.dumps(data, indent=2))

    def _embed_simple(self, text: str) -> list[float]:
        """Simple embedding (hash-based)."""
        import hashlib
        hash_bytes = hashlib.sha256(text.encode()).digest()
        vec = [0.0] * self.dim

        for i, byte in enumerate(hash_bytes):
            idx = i % self.dim
            vec[idx] = float(byte) / 255.0

        return vec

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def add(self, text: str, metadata: Optional[dict[str, Any]] = None) -> VectorEntry:
        """Add entry."""
        entry = VectorEntry(
            id=str(len(self._entries)),
            text=text,
            embedding=self._embed_simple(text),
            metadata=metadata or {},
        )
        self._entries[entry.id] = entry
        self._save()
        return entry

    def search(self, query: str, top_k: int = 5) -> list[tuple[VectorEntry, float]]:
        """Search by similarity."""
        query_vec = self._embed_simple(query)
        results = []

        for entry in self._entries.values():
            sim = self._cosine_similarity(query_vec, entry.embedding)
            results.append((entry, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get(self, entry_id: str) -> Optional[VectorEntry]:
        """Get entry."""
        return self._entries.get(entry_id)

    def delete(self, entry_id: str) -> bool:
        """Delete entry."""
        if entry_id in self._entries:
            del self._entries[entry_id]
            self._save()
            return True
        return False

    def count(self) -> int:
        """Count entries."""
        return len(self._entries)