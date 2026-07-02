"""
Repository Indexer for Expera AI.

Indexes code repositories for semantic search.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class IndexedFile:
    """Indexed file."""

    path: str
    content: str
    symbols: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RepositoryIndex:
    """Repository index."""

    root_path: str
    files: list[IndexedFile] = field(default_factory=list)
    total_symbols: int = 0


class RepositoryIndexer:
    """
    Repository indexer.

    Indexes code files for search.
    """

    def __init__(self, root_path: Optional[str] = None) -> None:
        """Initialize repository indexer."""
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self._index: list[IndexedFile] = []

    def _extract_symbols(self, content: str, path: str) -> list[dict[str, Any]]:
        """Extract symbols from code."""
        symbols = []

        if path.endswith(".py"):
            try:
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        symbols.append({
                            "name": node.name,
                            "kind": "function",
                            "line": node.lineno,
                        })

                    elif isinstance(node, ast.ClassDef):
                        symbols.append({
                            "name": node.name,
                            "kind": "class",
                            "line": node.lineno,
                        })

                    elif isinstance(node, ast.AsyncFunctionDef):
                        symbols.append({
                            "name": node.name,
                            "kind": "async_function",
                            "line": node.lineno,
                        })

            except Exception:
                pass

        return symbols

    def index_file(self, file_path: str) -> Optional[IndexedFile]:
        """Index single file."""
        full_path = self.root_path / file_path

        if not full_path.exists():
            return None

        try:
            content = full_path.read_text(encoding="utf-8")
            symbols = self._extract_symbols(content, file_path)

            return IndexedFile(
                path=file_path,
                content=content,
                symbols=symbols,
                metadata={
                    "size": full_path.stat().st_size,
                },
            )

        except Exception:
            return None

    def index_all(self, extensions: Optional[list[str]] = None) -> RepositoryIndex:
        """Index all files in repository."""
        extensions = extensions or [".py", ".js", ".ts", ".md"]
        files: list[IndexedFile] = []

        for ext in extensions:
            for path in self.root_path.rglob(f"*{ext}"):
                if self._should_index(path):
                    file_path = str(path.relative_to(self.root_path))
                    if indexed := self.index_file(file_path):
                        files.append(indexed)

        total = sum(len(f.symbols) for f in files)

        return RepositoryIndex(
            root_path=str(self.root_path),
            files=files,
            total_symbols=total,
        )

    def _should_index(self, path: Path) -> bool:
        """Check if file should be indexed."""
        ignore = ["__pycache__", ".git", "node_modules", "venv", ".venv", "dist", "build"]
        return not any(ign in str(path) for ign in ignore)

    def update_file(self, file_path: str) -> bool:
        """Update index for file."""
        if indexed := self.index_file(file_path):
            for i, f in enumerate(self._index):
                if f.path == file_path:
                    self._index[i] = indexed
                    return True
            self._index.append(indexed)
            return True
        return False

    def remove_file(self, file_path: str) -> bool:
        """Remove file from index."""
        self._index = [f for f in self._index if f.path != file_path]
        return True

    def get_symbols(self, name: str) -> list[dict[str, Any]]:
        """Get symbols by name."""
        results = []

        for f in self._index:
            for sym in f.symbols:
                if sym["name"] == name:
                    results.append({
                        "symbol": sym,
                        "file": f.path,
                    })

        return results

    def to_json(self) -> str:
        """Export index to JSON."""
        return json.dumps({
            "root_path": str(self.root_path),
            "files": [
                {
                    "path": f.path,
                    "symbols": f.symbols,
                    "metadata": f.metadata,
                }
                for f in self._index
            ],
            "total_symbols": sum(len(f.symbols) for f in self._index),
        }, indent=2)