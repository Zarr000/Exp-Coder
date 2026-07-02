"""
Embeddings for RAG.

Text embeddings:
- Generate embeddings
- Multiple models
- Local fallback

Usage:
    embeddings = TextEmbeddings()
    vector = await embeddings.embed("Hello world")
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TextEmbeddings:
    """
    Text embeddings.

    Features:
    - Local embeddings
    - Remote API
    - Sentence transformers
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimensions: int = 384,
    ):
        """Initialize embeddings."""
        self.model_name = model
        self.dimensions = dimensions
        self._model = None

    async def embed(self, text: str) -> list[float]:
        """Get embedding for text."""
        # Try sentence-transformers first
        if self._model is None:
            await self._load_model()

        if self._model:
            return self._embed_transformers(text)

        # Fallback to simple hash-based embedding
        return self._embed_simple(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts."""
        embeddings = []

        for text in texts:
            emb = await self.embed(text)
            embeddings.append(emb)

        return embeddings

    async def _load_model(self) -> bool:
        """Load embedding model."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embeddings model: {self.model_name}")
            return True

        except ImportError:
            logger.warning("sentence-transformers not available")
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _embed_transformers(self, text: str) -> list[float]:
        """Get embedding from transformers."""
        if self._model is None:
            return self._embed_simple(text)

        try:
            embedding = self._model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return self._embed_simple(text)

    def _embed_simple(self, text: str) -> list[float]:
        """Simple hash-based embedding as fallback."""
        import hashlib

        # Create deterministic embedding from text
        hash_bytes = hashlib.sha256(text.encode()).digest()

        # Convert to float list
        embedding = []
        for i in range(self.dimensions):
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] - 128) / 128.0
            embedding.append(value)

        return embedding

    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.dimensions


# Export
__all__ = [
    "TextEmbeddings",
]