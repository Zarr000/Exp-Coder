"""
Repository Agent.

Understands and navigates code repositories.

Usage:
    python -m src.agents.repository_agent --path /path/to/repo --query "find main"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """File information."""

    path: str
    size: int
    modified: float
    language: str = ""


@dataclass
class Symbol:
    """Code symbol (function, class, etc.)."""

    name: str
    kind: str  # function, class, variable
    file: str
    line: int
    signature: str = ""


@dataclass
class RepoContext:
    """Repository context."""

    root: Path
    files: list[FileInfo] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    structure: dict = field(default_factory=dict)


LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".md": "markdown",
    ".rst": "rst",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sql": "sql",
}


class RepoAgent:
    """Understands and navigates repositories."""

    def __init__(self, root: Optional[Path] = None):
        self.root = root or Path.cwd()
        self.context: Optional[RepoContext] = None
        self._cache = {}

    async def index(self) -> RepoContext:
        """Index the repository."""
        if self.context:
            return self.context

        logger.info(f"Indexing repository: {self.root}")

        self.context = RepoContext(root=self.root)

        # Scan files
        for path in self.root.rglob("*"):
            if path.is_file():
                rel_path = path.relative_to(self.root)
                ext = path.suffix.lower()

                file_info = FileInfo(
                    path=str(rel_path),
                    size=path.stat().st_size,
                    modified=path.stat().st_mtime,
                    language=LANGUAGE_EXTENSIONS.get(ext, ""),
                )
                self.context.files.append(file_info)

                # Index symbols
                if file_info.language in ("python", "javascript", "typescript", "go", "rust"):
                    await self._index_symbols(path, file_info.language)

        logger.info(f"Indexed {len(self.context.files)} files")
        return self.context

    async def _index_symbols(self, path: Path, language: str) -> None:
        """Index symbols in a file."""
        if not self.context:
            return

        try:
            content = path.read_text(errors="ignore")
        except Exception:
            return

        if language == "python":
            # Find functions and classes
            for match in re.finditer(
                r"^(def|class)\s+(\w+)", content, re.MULTILINE
            ):
                kind = match.group(1)
                name = match.group(2)
                line = content[: match.start()].count("\n") + 1

                self.context.symbols.append(
                    Symbol(
                        name=name,
                        kind=kind,
                        file=str(path.relative_to(self.root)),
                        line=line,
                    )
                )

    async def find_file(
        self,
        pattern: str,
        language: Optional[str] = None,
    ) -> list[str]:
        """Find files by pattern."""
        await self.index()

        matches = []
        for file in self.context.files:
            if pattern in file.path:
                if language is None or file.language == language:
                    matches.append(file.path)

        return matches

    async def find_symbol(
        self,
        name: str,
        kind: Optional[str] = None,
    ) -> list[Symbol]:
        """Find symbols by name."""
        await self.index()

        matches = []
        for symbol in self.context.symbols:
            if name in symbol.name:
                if kind is None or symbol.kind == kind:
                    matches.append(symbol)

        return matches

    async def get_file_imports(self, file_path: str) -> list[str]:
        """Get imports for a file."""
        path = self.root / file_path

        if not path.exists():
            return []

        try:
            content = path.read_text(errors="ignore")
        except Exception:
            return []

        imports = []
        ext = path.suffix.lower()

        if ext == ".py":
            for match in re.finditer(
                r"^import\s+(\w+)|^from\s+(\S+)\s+import",
                content,
                re.MULTILINE,
            ):
                if match.group(1):
                    imports.append(match.group(1))
                elif match.group(2):
                    imports.append(match.group(2))

        return imports

    async def get_dependencies(self) -> dict:
        """Get dependency information."""
        deps = {}

        await self.index()

        for file in self.context.files:
            if file.path.endswith("package.json"):
                path = self.root / file.path
                try:
                    import json

                    data = json.loads(path.read_text())
                    deps[file.path] = data.get("dependencies", {})
                except Exception:
                    continue

        return deps

    async def search(
        self,
        query: str,
        language: Optional[str] = None,
    ) -> list[tuple[str, int, str]]:
        """Search for query in files."""
        await self.index()

        matches = []
        for file in self.context.files:
            if language and file.language != language:
                continue

            path = self.root / file.path

            try:
                content = path.read_text(errors="ignore")
                for i, line in enumerate(content.split("\n"), 1):
                    if query.lower() in line.lower():
                        matches.append((file.path, i, line.strip()))
            except Exception:
                continue

        return matches

    async def get_structure(self) -> dict:
        """Get directory structure."""
        await self.index()

        structure = {"root": str(self.root), "dirs": [], "files": []}

        dirs = set()
        for file in self.context.files:
            path = Path(file.path)
            if len(path.parts) > 1:
                dirs.add(str(path.parts[0]))

        structure["dirs"] = sorted(dirs)
        return structure


async def main():
    parser = argparse.ArgumentParser(description="Repository agent")
    parser.add_argument("--path", default=".", help="Repository path")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--find", help="Find file")
    args = parser.parse_args()

    agent = RepoAgent(Path(args.path))

    if args.find:
        files = await agent.find_file(args.find)
        print(f"Found {len(files)} files:")
        for f in files[:10]:
            print(f"  {f}")

    elif args.query:
        results = await agent.search(args.query)
        print(f"Found {len(results)} matches:")
        for path, line_num, line in results[:10]:
            print(f"  {path}:{line_num}: {line}")

    else:
        context = await agent.index()
        print(f"Indexed {len(context.files)} files, {len(context.symbols)} symbols")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())