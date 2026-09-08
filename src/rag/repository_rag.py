"""
Repository RAG for Expera AI.

Code repository RAG:
- Index code files
- Extract functions/classes
- Metadata extraction

Usage:
    rag = RepositoryRAG()
    await rag.index_repository("path/to/repo")
    results = await rag.search("function_name")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """A code chunk."""

    type: str  # function, class, import, etc.
    name: str
    text: str
    line_start: int
    line_end: int
    language: str


class RepositoryRag:
    """
    Repository RAG for code.

    Features:
    - Code parsing
    - Function extraction
    - Class extraction
    """

    def __init__(self, indexer=None):
        """Initialize repository RAG."""
        self.indexer = indexer

    async def index_repository(
        self,
        repo_path: str,
        language: Optional[str] = None,
    ) -> int:
        """Index a repository."""
        from .indexer import Indexer

        if not self.indexer:
            self.indexer = Indexer()

        path = Path(repo_path)
        if not path.exists():
            logger.error(f"Repository not found: {repo_path}")
            return 0

        # Auto-detect language
        if language is None:
            language = self._detect_language(path)

        # Find and index code files
        patterns = self._get_file_patterns(language)
        count = 0

        for pattern in patterns:
            for file_path in path.glob(pattern):
                if file_path.is_file():
                    n = await self.indexer.index_file(str(file_path))
                    count += n

        logger.info(f"Indexed {count} chunks from {repo_path}")
        return count

    def _detect_language(self, path: Path) -> str:
        """Detect primary language."""
        extensions = {}

        for file_path in path.rglob("*"):
            if file_path.is_file() and file_path.suffix:
                ext = file_path.suffix
                extensions[ext] = extensions.get(ext, 0) + 1

        # Map to language
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
        }

        for ext, count in extensions.items():
            if ext in ext_map:
                return ext_map[ext]

        return "python"

    def _get_file_patterns(self, language: str) -> list[str]:
        """Get file patterns for language."""
        patterns = {
            "python": ["**/*.py"],
            "javascript": ["**/*.js", "**/*.jsx"],
            "typescript": ["**/*.ts", "**/*.tsx"],
            "rust": ["**/*.rs"],
            "go": ["**/*.go"],
            "java": ["**/*.java"],
        }

        return patterns.get(language, ["**/*.py"])

    def extract_functions(self, code: str, language: str) -> list[CodeChunk]:
        """Extract functions from code."""
        if language == "python":
            return self._extract_python_functions(code)
        elif language == "javascript":
            return self._extract_js_functions(code)
        elif language == "typescript":
            return self._extract_ts_functions(code)

        return []

    def _extract_python_functions(self, code: str) -> list[CodeChunk]:
        """Extract Python functions."""
        chunks = []
        lines = code.split("\n")

        # Pattern for function definition
        pattern = r"^(\s*)def\s+(\w+)\s*\("

        for i, line in enumerate(lines):
            match = re.match(pattern, line)
            if match:
                name = match.group(2)
                start = i

                # Find end (next definition or dedent)
                chunks.append(CodeChunk(
                    type="function",
                    name=name,
                    text=line,
                    line_start=start,
                    line_end=start,
                    language="python",
                ))

        return chunks

    def _extract_js_functions(self, code: str) -> list[CodeChunk]:
        """Extract JavaScript functions."""
        chunks = []

        # Function patterns
        patterns = [
            r"function\s+(\w+)\s*\(",
            r"const\s+(\w+)\s*=\s*\([^)]*)\s*=>",
            r"(\w+)\s*\([^)]*\)\s*\{",
        ]

        lines = code.split("\n")
        for i, line in enumerate(lines):
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    name = match.group(1)
                    chunks.append(CodeChunk(
                        type="function",
                        name=name,
                        text=line,
                        line_start=i,
                        line_end=i,
                        language="javascript",
                    ))

        return chunks

    def _extract_ts_functions(self, code: str) -> list[CodeChunk]:
        """Extract TypeScript functions."""
        return self._extract_js_functions(code)


# Export
__all__ = [
    "RepositoryRag",
    "CodeChunk",
]