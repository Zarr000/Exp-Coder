"""
RAG Pipeline for Expera AI.

Unified RAG pipeline combining indexing, retrieval, and generation.

Usage:
    pipeline = RAGPipeline()
    await pipeline.index("path/to/repo")
    results = await pipeline.retrieve("query")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .indexer import Indexer, IndexConfig
from .retriever import RetrievedChunk, RetrievalConfig, Retriever
from .chunker import TextChunker, Chunk
from .embeddings import TextEmbeddings
from .reranker import Reranker

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """RAG pipeline configuration."""

    # Indexing
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_file_size: int = 1024 * 1024  # 1MB

    # Retrieval
    top_k: int = 5
    min_score: float = 0.5
    hybrid_alpha: float = 0.5

    # Reranking
    rerank: bool = True
    rerank_top_k: int = 3

    # Storage
    storage_path: Optional[Path] = None


@dataclass
class RAGResult:
    """RAG pipeline result."""

    query: str
    chunks: list[RetrievedChunk]
    context: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    """
    Unified RAG pipeline.

    Features:
    - Unified indexing
    - Hybrid retrieval
    - Re-ranking
    - Context formation
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize RAG pipeline."""
        self.config = config or RAGConfig()

        # Components
        self.indexer = Indexer(
            storage_path=self.config.storage_path,
            config=IndexConfig(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            ),
        )
        self.retriever = Retriever(
            vector_store=self.indexer.vector_store,
            config=RetrievalConfig(
                top_k=self.config.top_k,
                min_score=self.config.min_score,
                hybrid_alpha=self.config.hybrid_alpha,
            ),
        )
        self.reranker = Reranker() if self.config.rerank else None
        self.embedder = TextEmbeddings()

        self._indexed = False

    async def index(
        self,
        path: str,
        language: Optional[str] = None,
    ) -> int:
        """Index documents or repository."""
        from .repository_indexer import RepositoryIndexer

        repo_path = Path(path)

        if repo_path.is_file():
            # Index single file
            return await self.indexer.index_file(str(repo_path))
        elif repo_path.is_dir():
            # Index repository
            indexer = RepositoryIndexer(
                storage_path=self.config.storage_path,
                embedder=self.embedder,
            )
            return await indexer.index_repository(
                str(repo_path),
                language=language,
            )
        else:
            logger.error(f"Path not found: {path}")
            return 0

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks."""
        if not self._indexed:
            logger.warning("No documents indexed yet")

        top_k = top_k or self.config.top_k

        # Initial retrieval
        chunks = await self.retriever.retrieve(query, top_k=top_k * 2)

        # Re-rank if enabled
        if self.reranker and chunks:
            reranked = await self.reranker.rerank(query, chunks)
            chunks = reranked[:top_k]

        return chunks

    async def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> RAGResult:
        """Query the RAG pipeline."""
        chunks = await self.retrieve(question, top_k=top_k)

        # Form context
        context = self._form_context(chunks)

        return RAGResult(
            query=question,
            chunks=chunks,
            context=context,
            metadata={
                "num_chunks": len(chunks),
                "total_chars": sum(len(c.text) for c in chunks),
            },
        )

    def _form_context(self, chunks: list[RetrievedChunk]) -> str:
        """Form context from chunks."""
        parts = []

        for i, chunk in enumerate(chunks):
            parts.append(f"--- Source {i + 1} ({chunk.source}) ---\n{chunk.text}")

        return "\n\n".join(parts)

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        stats = {
            "indexed": self._indexed,
            "top_k": self.config.top_k,
            "rerank": self.config.rerank,
        }

        if self.indexer.vector_store:
            stats["vector_count"] = self.indexer.vector_store.count()

        return stats


# Export
__all__ = [
    "RAGPipeline",
    "RAGConfig",
    "RAGResult",
]