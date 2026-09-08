"""
File Tools for Expera AI.

File operations:
- Read/Write
- Copy/Move
- List/Search
- Metadata

Usage:
    tools = FileTools()
    content = tools.read("file.py")
    tools.write("new.py", content)
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """File information."""

    path: str
    size: int
    modified: float
    is_dir: bool
    is_file: bool


@dataclass
class FileOperationResult:
    """Result of file operation."""

    success: bool
    message: str
    data: Optional[any] = None


class FileTools:
    """
    File operations.

    Features:
    - Read/Write
    - Copy/Move
    - List/Search
    - Metadata
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

    def _resolve_path(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_dir / p

    def read(self, file_path: str) -> FileOperationResult:
        try:
            path = self._resolve_path(file_path)
            content = path.read_text(encoding="utf-8")
            return FileOperationResult(
                success=True,
                message=f"Read {len(content)} bytes",
                data=content,
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=str(e),
            )

    def read_lines(
        self,
        file_path: str,
        start: int = 0,
        end: Optional[int] = None,
    ) -> FileOperationResult:
        try:
            path = self._resolve_path(file_path)
            lines = path.read_text(encoding="utf-8").split("\n")
            end = end or len(lines)
            return FileOperationResult(
                success=True,
                message=f"Read lines {start}-{end}",
                data=lines[start:end],
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=str(e),
            )

    def write(
        self,
        file_path: str,
        content: str,
        append: bool = False,
    ) -> FileOperationResult:
        """Write file content."""
        try:
            path = self._resolve_path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if append else "w"
            with open(path, mode, encoding="utf-8") as f:
                f.write(content)

            return FileOperationResult(
                success=True,
                message=f"Wrote {len(content)} bytes",
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=str(e),
            )

    def copy(
        self,
        source: str,
        destination: str,
    ) -> FileOperationResult:
        """Copy file."""
        try:
            src = self._resolve_path(source)
            dst = self._resolve_path(destination)

            dst.parent.mkdir(parents=True, exist_ok=True)

            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

            return FileOperationResult(
                success=True,
                message=f"Copied {source} to {destination}",
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=str(e),
            )

    def move(
        self,
        source: str,
        destination: str,
    ) -> FileOperationResult:
        """Move file."""
        try:
            src = self._resolve_path(source)
            dst = self._resolve_path(destination)

            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

            return FileOperationResult(
                success=True,
                message=f"Moved {source} to {destination}",
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=str(e),
            )

    def delete(self, file_path: str) -> FileOperationResult:
        """Delete file."""
        try:
            path = self._resolve_path(file_path)

            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

            return FileOperationResult(
                success=True,
                message=f"Deleted {file_path}",
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=str(e),
            )

    def list(
        self,
        directory: str = ".",
        pattern: Optional[str] = None,
        recursive: bool = False,
    ) -> FileOperationResult:
        """List directory contents."""
        try:
            path = self._resolve_path(directory)
            results = []

            if recursive:
                for root, dirs, files in os.walk(path):
                    for f in files:
                        if pattern is None or pattern in f:
                            results.append(str(Path(root) / f))
            else:
                for p in path.iterdir():
                    if pattern is None or pattern in p.name:
                        results.append(str(p))

            return FileOperationResult(
                success=True,
                message=f"Found {len(results)} files",
                data=results,
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=str(e),
            )

    def search(
        self,
        directory: str,
        query: str,
        file_types: Optional[list[str]] = None,
    ) -> FileOperationResult:
        """Search for files containing query."""
        try:
            path = self._resolve_path(directory)
            results = []

            file_types = file_types or [".py", ".txt", ".md", ".json"]

            for ext in file_types:
                for p in path.rglob(f"*{ext}"):
                    try:
                        content = p.read_text(encoding="utf-8", errors="ignore")
                        if query.lower() in content.lower():
                            results.append(str(p))
                    except Exception:
                        pass

            return FileOperationResult(
                success=True,
                message=f"Found {len(results)} matches",
                data=results,
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=str(e),
            )

    def exists(self, file_path: str) -> bool:
        """Check if file exists."""
        return self._resolve_path(file_path).exists()

    def get_info(self, file_path: str) -> Optional[FileInfo]:
        """Get file information."""
        try:
            path = self._resolve_path(file_path)
            stat = path.stat()

            return FileInfo(
                path=str(path),
                size=stat.st_size,
                modified=stat.st_mtime,
                is_dir=path.is_dir(),
                is_file=path.is_file(),
            )
        except Exception:
            return None

    def create_directory(self, dir_path: str) -> FileOperationResult:
        """Create directory."""
        try:
            path = self._resolve_path(dir_path)
            path.mkdir(parents=True, exist_ok=True)

            return FileOperationResult(
                success=True,
                message=f"Created directory {dir_path}",
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=str(e),
            )


# Export
__all__ = [
    "FileTools",
    "FileInfo",
    "FileOperationResult",
]