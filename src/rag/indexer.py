"""
Indexer for RAG.

Indexes documents:
- Chunking
- Embedding
- Storage

Usage:
    indexer = Indexer(vector_store)
    await indexer.index_directory("docs")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class IndexConfig:
    """Indexer configuration."""

    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_length: int = 50
    max_chunk_length: int = 1024


class Indexer:
    """
    Document indexer.

    Features:
    - Text chunking
    - Embedding generation
    - Vector storage
    """

    def __init__(
        self,
        vector_store=None,
        embeddings=None,
        config: Optional[IndexConfig] = None,
    ):
        """Initialize indexer."""
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.config = config or IndexConfig()

    def set_embeddings(self, embeddings) -> None:
        """Set embeddings model."""
        self.embeddings = embeddings

    def set_vector_store(self, vector_store) -> None:
        """Set vector store."""
        self.vector_store = vector_store

    async def index_text(
        self,
        text: str,
        source: str = "text",
        metadata: Optional[dict] = None,
    ) -> int:
        """Index a text document."""
        if not self.vector_store:
            logger.warning("No vector store configured")
            return 0

        # Chunk text
        chunks = self._chunk_text(text)

        # Embed each chunk
        count = 0
        for i, chunk in enumerate(chunks):
            chunk_id = f"{source}:{i}"

            # Get embedding
            embedding = None
            if self.embeddings:
                embedding = await self.embeddings.embed(chunk)
            else:
                # Use zero vector if no embeddings
                embedding = [0] * (self.vector_store.config.dimensions or 384)

            # Store
            await self.vector_store.add(
                id=chunk_id,
                text=chunk,
                embedding=embedding,
                metadata=metadata or {"source": source, "index": i},
            )

            count += 1

        logger.info(f"Indexed {count} chunks from {source}")
        return count

    async def index_file(
        self,
        file_path: str,
        metadata: Optional[dict] = None,
    ) -> int:
        """Index a file."""
        try:
            path = Path(file_path)
            text = path.read_text(encoding="utf-8")
            return await self.index_text(text, source=str(path), metadata=metadata)
        except Exception as e:
            logger.error(f"Failed to index file: {e}")
            return 0

    async def index_directory(
        self,
        dir_path: str,
        pattern: str = "**/*",
        metadata_func: Optional[callable] = None,
    ) -> int:
        """Index all files in a directory."""
        path = Path(dir_path)
        if not path.exists():
            logger.error(f"Directory not found: {dir_path}")
            return 0

        count = 0

        for file_path in path.glob(pattern):
            if not file_path.is_file():
                continue

            # Skip binary files
            if file_path.suffix in (".pyc", ".class", ".o", ".so"):
                continue

            metadata = {}
            if metadata_func:
                metadata = metadata_func(file_path)

            n = await self.index_file(str(file_path), metadata)
            count += n

        logger.info(f"Indexed {count} chunks from {dir_path}")
        return count

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks."""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.config.chunk_size

            if end >= text_len:
                chunks.append(text[start:])
                break

            # Try to break at sentence boundary
            chunk = text[start:end]
            sentences = re.split(r"(?<=[.!?])\s+", chunk)

            if len(sentences) > 1:
                # Find last sentence boundary
                last_end = start
                for sent in sentences[:-1]:
                    last_end += len(sent) + 1

                if last_end - start >= self.config.min_chunk_length:
                    end = last_end

            chunk = text[start:end].strip()
            if len(chunk) >= self.config.min_chunk_length:
                chunks.append(chunk)

            start = end - self.config.chunk_overlap

        return chunks

    async def index_from_texts(
        self,
        texts: list[str],
        source_prefix: str = "doc",
    ) -> int:
        """Index multiple texts."""
        count = 0

        for i, text in enumerate(texts):
            n = await self.index_text(text, source=f"{source_prefix}_{i}")
            count += n

        return count


# Export
__all__ = [
    "Indexer",
    "IndexConfig",
]