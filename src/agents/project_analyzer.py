"""
Project Analyzer for Expera AI.

Analyzes code projects:
- File structure
- Dependencies
- Code metrics
- Language detection

Usage:
    analyzer = ProjectAnalyzer()
    report = await analyzer.analyze("path/to/project")
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """Information about a file."""

    path: str
    size: int
    extension: str
    language: str


@dataclass
class ProjectReport:
    """Project analysis report."""

    root_path: str
    files: list[FileInfo] = field(default_factory=list)
    languages: dict[str, int] = field(default_factory=dict)
    total_size: int = 0
    total_files: int = 0
    dependencies: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


# Extension to language mapping
EXTENSION_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
    ".lua": "lua",
    ".pl": "perl",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".txt": "text",
}


# Dependency patterns
DEPENDENCY_PATTERNS = {
    "python": [
        (r"import (\w+)", "import"),
        (r"from (\w+) import", "from"),
    ],
    "javascript": [
        (r"require\(['\"]([^'\"]+)['\"]\)", "require"),
        (r"import .* from ['\"]([^'\"]+)['\"]", "import"),
        (r"['\"]([^'\"]+)['\"]:?", "package"),
    ],
    "typescript": [
        (r"import .* from ['\"]([^'\"]+)['\"]", "import"),
        (r"import \(['\"]([^'\"]+)['\"]", "import"),
    ],
    "rust": [
        (r"use (\w+)", "use"),
        (r"extern crate (\w+)", "extern"),
    ],
}


class ProjectAnalyzer:
    """
    Analyzes code projects.

    Features:
    - File discovery
    - Language detection
    - Dependency analysis
    - Code metrics
    """

    def __init__(self):
        """Initialize project analyzer."""
        self.extension_map = EXTENSION_MAP.copy()
        self.ignore_patterns = {
            ".git",
            "__pycache__",
            "node_modules",
            "venv",
            ".venv",
            "dist",
            "build",
            ".pytest_cache",
            ".mypy_cache",
            ".tox",
        }

    async def analyze(
        self,
        project_path: str,
        recursive: bool = True,
    ) -> ProjectReport:
        """Analyze a project."""
        path = Path(project_path)
        if not path.exists():
            raise ValueError(f"Path does not exist: {project_path}")

        report = ProjectReport(root_path=str(path))

        # Find files
        files = self._find_files(path, recursive)
        report.files = files

        # Count by language
        for file in files:
            lang = file.language
            report.languages[lang] = report.languages.get(lang, 0) + 1

        report.total_files = len(files)
        report.total_size = sum(f.size for f in files)

        # Find dependencies
        report.dependencies = self._find_dependencies(files)

        # Calculate metrics
        report.metrics = self._calculate_metrics(files)

        return report

    def _find_files(self, path: Path, recursive: bool) -> list[FileInfo]:
        """Find all source files."""
        files = []

        if path.is_file():
            files.append(self._analyze_file(path))
            return files

        # Walk directory
        if recursive:
            for root, dirs, filenames in os.walk(path):
                # Filter ignored directories
                dirs[:] = [d for d in dirs if d not in self.ignore_patterns]

                for filename in filenames:
                    file_path = Path(root) / filename
                    if file_path.suffix not in self.ignore_patterns:
                        files.append(self._analyze_file(file_path))
        else:
            for file_path in path.iterdir():
                if file_path.is_file():
                    files.append(self._analyze_file(file_path))

        return files

    def _analyze_file(self, path: Path) -> FileInfo:
        """Analyze a single file."""
        ext = path.suffix.lower()
        language = self.extension_map.get(ext, "unknown")

        try:
            size = path.stat().st_size
        except Exception:
            size = 0

        return FileInfo(
            path=str(path),
            size=size,
            extension=ext,
            language=language,
        )

    def _find_dependencies(self, files: list[FileInfo]) -> list[str]:
        """Find dependencies."""
        dependencies = set()

        for file in files:
            lang = file.language
            patterns = DEPENDENCY_PATTERNS.get(lang, [])

            if not patterns:
                continue

            try:
                content = Path(file.path).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for pattern, _ in patterns:
                regex = re.compile(pattern)
                for match in regex.finditer(content):
                    dep = match.group(1)
                    if dep and not dep.startswith("."):
                        dependencies.add(dep)

        return sorted(dependencies)[:50]  # Limit to 50

    def _calculate_metrics(self, files: list[FileInfo]) -> dict[str, Any]:
        """Calculate project metrics."""
        metrics = {
            "file_count": len(files),
            "total_lines": 0,
            "avg_file_size": 0,
            "largest_file": None,
            "largest_size": 0,
        }

        total_lines = 0
        largest_file = None

        for file in files:
            if file.size > metrics["largest_size"]:
                metrics["largest_size"] = file.size
                largest_file = file.path

            # Count lines for text files
            if file.language in ("python", "javascript", "typescript", "rust", "go", "java", "c", "cpp"):
                try:
                    with open(file.path, encoding="utf-8", errors="ignore") as f:
                        lines = len(f.readlines())
                        total_lines += lines
                except Exception:
                    pass

        metrics["total_lines"] = total_lines
        metrics["avg_file_size"] = metrics["total_size"] // max(1, len(files))
        metrics["largest_file"] = largest_file

        return metrics

    def get_language_summary(self, report: ProjectReport) -> str:
        """Get language summary."""
        if not report.languages:
            return "No source files found"

        parts = []
        for lang, count in sorted(report.languages.items(), key=lambda x: -x[1]):
            parts.append(f"{lang}: {count}")

        return ", ".join(parts)

    def get_size_summary(self, report: ProjectReport) -> str:
        """Get size summary."""
        size_kb = report.total_size / 1024
        if size_kb < 1024:
            return f"{size_kb:.1f} KB"

        size_mb = size_kb / 1024
        return f"{size_mb:.1f} MB"


# Export
__all__ = [
    "ProjectAnalyzer",
    "ProjectReport",
    "FileInfo",
]