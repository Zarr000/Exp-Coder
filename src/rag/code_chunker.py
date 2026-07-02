"""
Code Chunker for Expera AI.

Specialized chunking for code.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CodeChunk:
    """Code chunk."""

    text: str
    language: str
    kind: str
    file: str
    line_start: int
    line_end: int
    imports: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class CodeChunker:
    """
    Specialized code chunker.

    Chunks by functions, classes, and modules.
    """

    def __init__(self, max_lines: int = 200) -> None:
        """Initialize code chunker."""
        self.max_lines = max_lines

    def chunk_python(self, content: str, file_path: str) -> list[CodeChunk]:
        """Chunk Python code."""
        chunks = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunk = self._extract_function(node, content, file_path)
                    if chunk:
                        chunks.append(chunk)

                elif isinstance(node, ast.ClassDef):
                    chunk = self._extract_class(node, content, file_path)
                    if chunk:
                        chunks.append(chunk)

        except Exception:
            pass

        if not chunks:
            chunks.append(CodeChunk(
                text=content,
                language="python",
                kind="module",
                file=file_path,
                line_start=1,
                line_end=len(content.split("\n")),
            )

        return chunks

    def _extract_function(
        self,
        node: ast.FunctionDef,
        content: str,
        file_path: str,
    ) -> Optional[CodeChunk]:
        """Extract function as chunk."""
        lines = content.split("\n")
        start = node.lineno - 1
        end = (node.end_lineno or node.lineno) - 1

        func_lines = lines[start:end]
        func_code = "\n".join(func_lines)

        imports = self._extract_imports(node)

        return CodeChunk(
            text=func_code,
            language="python",
            kind="function",
            file=file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            imports=imports,
            metadata={"name": node.name},
        )

    def _extract_class(
        self,
        node: ast.ClassDef,
        content: str,
        file_path: str,
    ) -> Optional[CodeChunk]:
        """Extract class as chunk."""
        lines = content.split("\n")
        start = node.lineno - 1
        end = (node.end_lineno or node.lineno) - 1

        class_lines = lines[start:end]
        class_code = "\n".join(class_lines)

        imports = self._extract_imports(node)

        methods = [
            n.name for n in node.body
            if isinstance(n, ast.FunctionDef)
        ]

        return CodeChunk(
            text=class_code,
            language="python",
            kind="class",
            file=file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            imports=imports,
            metadata={"name": node.name, "methods": methods},
        )

    def _extract_imports(self, node: ast.AST) -> list[str]:
        """Extract imports from node."""
        imports = []

        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                for alias in child.names:
                    imports.append(alias.name)

            elif isinstance(child, ast.ImportFrom):
                if child.module:
                    imports.append(child.module)

        return imports

    def chunk_file(self, file_path: str, content: str) -> list[CodeChunk]:
        """Chunk any file."""
        path = Path(file_path)
        ext = path.suffix

        if ext == ".py":
            return self.chunk_python(content, file_path)

        return [
            CodeChunk(
                text=content,
                language=ext[1:],
                kind="file",
                file=file_path,
                line_start=1,
                line_end=len(content.split("\n")),
            )
        ]

    def chunk_directory(
        self,
        directory: str,
    ) -> list[CodeChunk]:
        """Chunk all files in directory."""
        chunks = []
        dir_path = Path(directory)

        for path in dir_path.rglob("*.py"):
            try:
                content = path.read_text(encoding="utf-8")
                rel_path = str(path.relative_to(dir_path))
                chunks.extend(self.chunk_file(rel_path, content))

            except Exception:
                continue

        return chunks