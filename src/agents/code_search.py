"""
Code Search Engine for Expera AI Coding Agent.

Provides semantic and syntactic code search.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SearchResult:
    """Code search result."""

    file: str
    line: int
    content: str
    match_type: str
    score: float = 0.0


@dataclass
class SearchOptions:
    """Code search options."""

    extensions: list[str] = field(default_factory=lambda: [".py"])
    exclude_patterns: list[str] = field(default_factory=lambda: ["__pycache__", ".git", "venv", "env", "*.pyc"])
    case_sensitive: bool = False
    regex: bool = False
    whole_word: bool = False
    max_results: int = 100


class CodeSearch:
    """
    Code search engine.

    Features:
    - Text search
    - Regex search
    - AST-based search
    - Symbol search
    - Similarity search
    """

    def __init__(self, root_path: Optional[str] = None) -> None:
        """Initialize code search."""
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self._index: dict[str, dict] = {}

    def search(
        self,
        query: str,
        options: Optional[SearchOptions] = None,
    ) -> list[SearchResult]:
        """Search code."""
        options = options or SearchOptions()
        results: list[SearchResult] = []

        pattern = query
        if not options.regex:
            pattern = re.escape(query)

        if options.whole_word:
            pattern = r"\b" + pattern + r"\b"

        flags = 0 if options.case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)

        for file_path in self._iterate_files(options):
            try:
                content = file_path.read_text()
                lines = content.split("\n")

                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        results.append(SearchResult(
                            file=str(file_path),
                            line=i,
                            content=line.strip(),
                            match_type="text",
                            score=1.0,
                        ))

            except Exception:
                continue

            if len(results) >= options.max_results:
                break

        return results

    def _iterate_files(self, options: SearchOptions):
        """Iterate over files."""
        for ext in options.extensions:
            pattern = f"**/*{ext}"
            for path in self.root_path.glob(pattern):
                if any(p for p in options.exclude_patterns if p in str(path)):
                    continue
                yield path

    def search_symbol(
        self,
        symbol: str,
        kind: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search for symbol definitions."""
        results: list[SearchResult] = []

        for file_path in self._iterate_files(SearchOptions()):
            try:
                content = file_path.read_text()
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    is_match = False

                    if isinstance(node, ast.FunctionDef) and symbol in node.name:
                        is_match = True
                        match_type = "function"

                    elif isinstance(node, ast.ClassDef) and symbol in node.name:
                        is_match = True
                        match_type = "class"

                    elif isinstance(node, ast.Name) and symbol == node.id:
                        is_match = True
                        match_type = "name"

                    if is_match and (kind is None or kind == match_type):
                        results.append(SearchResult(
                            file=str(file_path),
                            line=node.lineno,
                            content=f"{match_type} {symbol}",
                            match_type=match_type,
                            score=1.0,
                        ))

            except Exception:
                continue

        return results

    def search_definitions(self, query: str) -> list[SearchResult]:
        """Search for definitions only."""
        results = self.search_symbol(query, "function")
        results += self.search_symbol(query, "class")

        return sorted(results, key=lambda r: r.line)

    def search_usages(self, symbol: str) -> list[SearchResult]:
        """Find all usages of symbol."""
        results: list[SearchResult] = []

        for file_path in self._iterate_files(SearchOptions()):
            try:
                content = file_path.read_text()
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and node.id == symbol:
                        results.append(SearchResult(
                            file=str(file_path),
                            line=node.lineno,
                            content=content.split("\n")[node.lineno - 1].strip(),
                            match_type="usage",
                            score=0.8,
                        ))

            except Exception:
                continue

        return results

    def find_duplicates(self, threshold: float = 0.9) -> list[tuple[str, str]]:
        """Find duplicate code blocks."""
        duplicates: list[tuple[str, str]] = []
        blocks: dict[str, str] = {}

        for file_path in self._iterate_files(SearchOptions()):
            try:
                content = file_path.read_text()
                lines = [
                    " ".join(line.split())
                    for line in content.split("\n")
                    if line.strip() and not line.strip().startswith("#")
                ]

                block = "\n".join(lines[:10])
                if block:
                    blocks[str(file_path)] = block

            except Exception:
                continue

        checked = set()
        for path1, block1 in blocks.items():
            for path2, block2 in blocks.items():
                if path1 == path2 or (path2, path1) in checked:
                    continue

                checked.add((path1, path2))

                if block1 == block2:
                    duplicates.append((path1, path2))

        return duplicates

    def index_files(self, extensions: Optional[list[str]] = None) -> dict[str, dict]:
        """Index files for faster search."""
        extensions = extensions or [".py"]
        self._index = {}

        for file_path in self._iterate_files(SearchOptions(extensions=extensions)):
            self._index[str(file_path)] = {
                "size": file_path.stat().st_size,
                "mtime": file_path.stat().st_mtime,
            }

        return self._index

    def get_indexed_files(self) -> list[str]:
        """Get list of indexed files."""
        return list(self._index.keys())