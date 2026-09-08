"""
Documentation RAG for Expera AI.

Documentation RAG:
- Index markdown
- Extract headers
- Link sections

Usage:
    rag = DocumentationRAG()
    await rag.index_docs("docs")
    results = await rag.search("installation")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DocSection:
    """A documentation section."""

    title: str
    content: str
    level: int
    line_start: int
    path: str


class DocumentationRag:
    """
    Documentation RAG.

    Features:
    - Markdown parsing
    - Header extraction
    - Cross-reference linking
    """

    def __init__(self, indexer=None):
        """Initialize documentation RAG."""
        self.indexer = indexer

    async def index_docs(
        self,
        docs_path: str,
    ) -> int:
        """Index documentation."""
        from .indexer import Indexer

        if not self.indexer:
            self.indexer = Indexer()

        path = Path(docs_path)
        if not path.exists():
            logger.error(f"Docs not found: {docs_path}")
            return 0

        count = 0

        for file_path in path.glob("**/*.md"):
            if file_path.is_file():
                # Extract sections
                sections = self.extract_sections(file_path.read_text())

                # Index each section
                for section in sections:
                    await self.indexer.index_text(
                        section.content,
                        source=f"{file_path}:{section.title}",
                        metadata={"title": section.title, "level": section.level},
                    )
                    count += 1

        logger.info(f"Indexed {count} sections from {docs_path}")
        return count

    def extract_sections(self, content: str) -> list[DocSection]:
        """Extract documentation sections."""
        sections = []
        lines = content.split("\n")

        # Header pattern
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        current_title = "root"
        current_level = 0
        current_content = []
        start_line = 0

        for i, line in enumerate(lines):
            match = header_pattern.match(line)
            if match:
                # Save previous section
                if current_content:
                    sections.append(DocSection(
                        title=current_title,
                        content="\n".join(current_content),
                        level=current_level,
                        line_start=start_line,
                        path="",
                    ))

                # Start new section
                hashes = match.group(1)
                current_title = match.group(2).strip()
                current_level = len(hashes)
                current_content = [line]
                start_line = i

            else:
                current_content.append(line)

        # Add last section
        if current_content:
            sections.append(DocSection(
                title=current_title,
                content="\n".join(current_content),
                level=current_level,
                line_start=start_line,
                path="",
            ))

        return sections

    def extract_table_of_contents(
        self,
        content: str,
    ) -> list[tuple[str, int]]:
        """Extract table of contents."""
        toc = []

        lines = content.split("\n")
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        for i, line in enumerate(lines):
            match = header_pattern.match(line)
            if match:
                hashes = match.group(1)
                title = match.group(2).strip()
                level = len(hashes)
                toc.append((title, level))

        return toc


# Export
__all__ = [
    "DocumentationRag",
    "DocSection",
]