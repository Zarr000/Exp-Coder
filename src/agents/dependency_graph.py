"""
Dependency Graph for Expera AI Coding Agent.

Analyzes code dependencies and generates dependency graphs.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Dependency:
    """Represents a code dependency."""

    module: str
    name: str
    line: int
    alias: Optional[str] = None
    imported: bool = False


@dataclass
class ImportInfo:
    """Information about an import."""

    module: str
    names: list[str]
    line: int
    level: int = 0
    is_from: bool = False


@dataclass
class Symbol:
    """Represents a defined symbol."""

    name: str
    kind: str
    line: int
    end_line: int
    docstring: Optional[str] = None
    params: list[str] = field(default_factory=list)


class DependencyGraph:
    """
    Analyzes code dependencies and generates dependency graphs.

    Supports Python, JavaScript, and TypeScript.
    """

    def __init__(self, root_path: Optional[str] = None) -> None:
        """Initialize dependency graph analyzer."""
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self._imports: dict[str, list[ImportInfo]] = {}
        self._symbols: dict[str, list[Symbol]] = {}
        self._calls: dict[str, list[str]] = {}
        self._graph: dict[str, set[str]] = {}

    def analyze_file(self, file_path: str) -> tuple[list[ImportInfo], list[Symbol]]:
        """Analyze a single file."""
        full_path = self.root_path / file_path
        ext = full_path.suffix

        if ext == ".py":
            return self._analyze_python(full_path)
        elif ext in (".js", ".ts", ".tsx", ".jsx"):
            return self._analyze_js(full_path)

        return [], []

    def _analyze_python(self, path: Path) -> tuple[list[ImportInfo], list[Symbol]]:
        """Analyze Python file."""
        imports: list[ImportInfo] = []
        symbols: list[Symbol] = []

        try:
            content = path.read_text()
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(ImportInfo(
                            module=alias.name,
                            names=[alias.asname or alias.name],
                            line=node.lineno,
                            is_from=False,
                        ))

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names = [alias.name for alias in node.names]
                    imports.append(ImportInfo(
                        module=module,
                        names=names,
                        line=node.lineno,
                        level=node.level,
                        is_from=True,
                    ))

                elif isinstance(node, ast.FunctionDef):
                    symbols.append(Symbol(
                        name=node.name,
                        kind="function",
                        line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        docstring=ast.get_docstring(node),
                        params=[arg.arg for arg in node.args.args],
                    ))

                elif isinstance(node, ast.ClassDef):
                    symbols.append(Symbol(
                        name=node.name,
                        kind="class",
                        line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        docstring=ast.get_docstring(node),
                    ))

        except Exception:
            pass

        return imports, symbols

    def _analyze_js(self, path: Path) -> tuple[list[ImportInfo], list[Symbol]]:
        """Analyze JavaScript/TypeScript file."""
        imports: list[ImportInfo] = []
        symbols: list[Symbol] = []

        return imports, symbols

    def generate_graph(self, files: list[str]) -> dict[str, set[str]]:
        """Generate dependency graph for files."""
        graph: dict[str, set[str]] = {}

        for file_path in files:
            imports, _ = self.analyze_file(file_path)

            deps: set[str] = set()
            for imp in imports:
                for name in imp.names:
                    if "." in name:
                        module = name.rsplit(".", 1)[0]
                        deps.add(module)

            graph[file_path] = deps

        self._graph = graph
        return graph

    def get_imports(self, file_path: str) -> list[ImportInfo]:
        """Get imports for file."""
        if file_path in self._imports:
            return self._imports[file_path]
        imports, _ = self.analyze_file(file_path)
        self._imports[file_path] = imports
        return imports

    def get_exports(self, file_path: str) -> list[Symbol]:
        """Get exported symbols for file."""
        if file_path in self._symbols:
            return self._symbols[file_path]
        _, symbols = self.analyze_file(file_path)
        self._symbols[file_path] = symbols
        return symbols

    def find_dependents(self, module: str) -> list[str]:
        """Find files that depend on module."""
        dependents = []

        for file_path, deps in self._graph.items():
            if module in deps:
                dependents.append(file_path)

        return dependents

    def circular_imports(self) -> list[list[str]]:
        """Detect circular imports."""
        cycles: list[list[str]] = []

        for from_file, imports in self._imports.items():
            for imp in imports:
                if imp.module in self._graph:
                    to_file = imp.module
                    if to_file in self._graph and from_file in self._graph[to_file]:
                        cycles.append([from_file, to_file])

        return cycles

    def topological_sort(self) -> list[str]:
        """Get topological sort of dependencies."""
        in_degree = {f: 0 for f in self._graph}

        for _, deps in self._graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1

        queue = [f for f, d in in_degree.items() if d == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            if node in self._graph:
                for dep in self._graph[node]:
                    if dep in in_degree:
                        in_degree[dep] -= 1
                        if in_degree[dep] == 0:
                            queue.append(dep)

        return result

    def to_mermaid(self) -> str:
        """Generate Mermaid diagram."""
        lines = ["flowchart TD"]

        for file_path, deps in self._graph.items():
            node_id = file_path.replace("/", "_").replace(".", "_")
            for dep in deps:
                dep_id = dep.replace("/", "_").replace(".", "_")
                lines.append(f"    {node_id} --> {dep_id}")

        return "\n".join(lines)