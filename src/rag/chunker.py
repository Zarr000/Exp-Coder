"""
Text Chunker.

Chunks text into overlapping segments for RAG.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Chunk:
    """Text chunk."""

    text: str
    start_idx: int
    end_idx: int
    metadata: dict[str, Any]


class TextChunker:
    """
    Chunks documents into overlapping segments.

    Supports multiple chunking strategies.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ) -> None:
        """Initialize chunker."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_by_tokens(self, text: str, tokenizer: Any) -> list[Chunk]:
        """Chunk text by token count."""
        tokens = tokenizer.encode(text)
        chunks: list[Chunk] = []

        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]

            if len(chunk_tokens) < self.min_chunk_size // 4:
                if chunks:
                    chunks[-1].text += tokenizer.decode(chunk_tokens)
                continue

            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=i,
                end_idx=i + len(chunk_tokens),
                metadata={"token_count": len(chunk_tokens)}
            ))

        return chunks

    def chunk_by_sentences(self, text: str, sentence_tokenizer: Any = None) -> list[Chunk]:
        """Chunk text by sentences."""
        if sentence_tokenizer:
            sentences = list(sentence_tokenizer(text))
        else:
            sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks: list[Chunk] = []
        current_chunk: list[str] = []
        current_size = 0
        start_idx = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(Chunk(
                    text=" ".join(current_chunk),
                    start_idx=start_idx,
                    end_idx=start_idx + current_size,
                    metadata={"num_sentences": len(current_chunk)}
                ))

                overlap_text = " ".join(current_chunk)[-self.chunk_overlap:]
                current_chunk = [overlap_text, sentence]
                current_size = len(overlap_text) + sentence_size
                start_idx += len(overlap_text)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        if current_chunk:
            chunks.append(Chunk(
                text=" ".join(current_chunk),
                start_idx=start_idx,
                end_idx=start_idx + current_size,
                metadata={"num_sentences": len(current_chunk)}
            ))

        return chunks

    def chunk_by_paragraphs(self, text: str) -> list[Chunk]:
        """Chunk text by paragraphs."""
        paragraphs = text.split("\n\n")
        chunks: list[Chunk] = []
        start_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            chunks.append(Chunk(
                text=para,
                start_idx=start_idx,
                end_idx=start_idx + len(para),
                metadata={"paragraph": True}
            ))
            start_idx += len(para) + 2

        return chunks

    def chunk_code(self, code: str) -> list[Chunk]:
        """Chunk code by functions/classes."""
        patterns = [
            r"^(?:def|class|async def|async class)\s+\w+",
            r"^@(?:property|staticmethod|classmethod)",
            r"^\s{0,4}def\s+\w+",
        ]

        lines = code.split("\n")
        chunks: list[Chunk] = []
        current_block: list[str] = []
        start_idx = 0

        for i, line in enumerate(lines):
            is_definition = any(re.match(p, line.strip()) for p in patterns)

            if is_definition and current_block:
                chunk_text = "\n".join(current_block)
                chunks.append(Chunk(
                    text=chunk_text,
                    start_idx=start_idx,
                    end_idx=start_idx + len(chunk_text),
                    metadata={"type": "code_block"}
                ))
                current_block = [line]
                start_idx = i
            else:
                current_block.append(line)

        if current_block:
            chunk_text = "\n".join(current_block)
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=start_idx + len(chunk_text),
                metadata={"type": "code_block"}
            ))

        return chunks

    def chunk(
        self,
        text: str,
        strategy: str = "tokens",
        tokenizer: Optional[Any] = None
    ) -> list[Chunk]:
        """Chunk text with specified strategy."""
        if strategy == "tokens":
            if tokenizer:
                return self.chunk_by_tokens(text, tokenizer)
            return self.chunk_by_sentences(text)
        elif strategy == "sentences":
            return self.chunk_by_sentences(text)
        elif strategy == "paragraphs":
            return self.chunk_by_paragraphs(text)
        elif strategy == "code":
            return self.chunk_code(text)
        else:
            return self.chunk_by_sentences(text)