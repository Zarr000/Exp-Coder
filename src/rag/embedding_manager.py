"""
Embedding Manager for Expera AI.

Manages embeddings for RAG.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""

    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim: int = 384
    batch_size: int = 32
    normalize: bool = True


@dataclass
class EmbeddingEntry:
    """Embedding entry."""

    id: str
    text: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


class EmbeddingManager:
    """
    Embedding manager.

    Manages embeddings for retrieval.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        storage_path: Optional[str] = None,
    ) -> None:
        """Initialize embedding manager."""
        self.config = config or EmbeddingConfig()
        self.storage_path = Path(storage_path) if storage_path else None
        self._embeddings: dict[str, list[float]] = {}
        self._load()

    def _load(self) -> None:
        """Load embeddings."""
        if self.storage_path:
            path = self.storage_path / "embeddings.json"
            if path.exists():
                data = json.loads(path.read_text())
                self._embeddings = data.get("embeddings", {})

    def _save(self) -> None:
        """Save embeddings."""
        if self.storage_path:
            path = self.storage_path / "embeddings.json"
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {"embeddings": self._embeddings}
            path.write_text(json.dumps(data))

    def _embed_simple(self, text: str) -> list[float]:
        """Simple embedding (placeholder)."""
        import hashlib

        hash_bytes = hashlib.sha256(text.encode()).digest()
        vec = [0.0] * self.config.dim

        for i, byte in enumerate(hash_bytes):
            idx = i % self.config.dim
            vec[idx] = float(byte) / 255.0

        if self.config.normalize:
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]

        return vec

    def embed(self, text: str, entry_id: Optional[str] = None) -> list[float]:
        """Create embedding."""
        entry_id = entry_id or str(len(self._embeddings))
        vector = self._embed_simple(text)

        self._embeddings[entry_id] = vector
        self._save()

        return vector

    def embed_batch(self, texts: list[str]) -> dict[str, list[float]]:
        """Embed batch of texts."""
        results = {}

        for text in texts:
            entry_id = str(len(self._embeddings))
            results[entry_id] = self._embed_simple(text)
            self._embeddings[entry_id] = results[entry_id]

        self._save()
        return results

    def get(self, entry_id: str) -> Optional[list[float]]:
        """Get embedding."""
        return self._embeddings.get(entry_id)

    def delete(self, entry_id: str) -> bool:
        """Delete embedding."""
        if entry_id in self._embeddings:
            del self._embeddings[entry_id]
            self._save()
            return True
        return False

    def count(self) -> int:
        """Count embeddings."""
        return len(self._embeddings)