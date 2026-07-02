"""
Filesystem Tool for Expera AI.

Provides file system operations: read, write, copy, move, delete, mkdir, list, search.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class FileResult:
    """Result of filesystem operation."""

    success: bool
    data: Optional[str] = None
    error: Optional[str] = None


class FilesystemTool:
    """
    Filesystem operations tool.

    Supports: read, write, copy, move, delete, mkdir, list, find, stat.
    """

    def __init__(self, base_path: Optional[str] = None) -> None:
        """Initialize filesystem tool."""
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self._lock = asyncio.Lock()

    async def _resolve(self, path: str) -> Path:
        """Resolve path relative to base."""
        p = Path(path)
        if p.isAbsolute():
            return p
        return self.base_path / p

    async def read(self, path: str, encoding: str = "utf-8") -> FileResult:
        """Read file contents."""
        async with self._lock:
            try:
                full_path = await self._resolve(path)
                content = full_path.read_text(encoding=encoding)
                return FileResult(success=True, data=content)
            except Exception as e:
                return FileResult(success=False, error=str(e))

    async def write(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        append: bool = False,
    ) -> FileResult:
        """Write file contents."""
        async with self._lock:
            try:
                full_path = await self._resolve(path)
                full_path.parent.mkdir(parents=True, exist_ok=True)

                mode = "a" if append else "w"
                with open(full_path, mode, encoding=encoding) as f:
                    f.write(content)

                return FileResult(success=True)
            except Exception as e:
                return FileResult(success=False, error=str(e))

    async def copy(self, src: str, dst: str, overwrite: bool = False) -> FileResult:
        """Copy file or directory."""
        async with self._lock:
            try:
                src_path = await self._resolve(src)
                dst_path = await self._resolve(dst)

                if dst_path.exists() and not overwrite:
                    return FileResult(success=False, error="Destination exists")

                if src_path.is_dir():
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=overwrite)
                else:
                    shutil.copy2(src_path, dst_path)

                return FileResult(success=True)
            except Exception as e:
                return FileResult(success=False, error=str(e))

    async def move(self, src: str, dst: str) -> FileResult:
        """Move file or directory."""
        async with self._lock:
            try:
                src_path = await self._resolve(src)
                dst_path = await self._resolve(dst)

                shutil.move(str(src_path), str(dst_path))
                return FileResult(success=True)
            except Exception as e:
                return FileResult(success=False, error=str(e))

    async def delete(self, path: str, recursive: bool = False) -> FileResult:
        """Delete file or directory."""
        async with self._lock:
            try:
                full_path = await self._resolve(path)

                if full_path.is_dir():
                    if recursive:
                        shutil.rmtree(full_path)
                    else:
                        full_path.rmdir()
                else:
                    full_path.unlink()

                return FileResult(success=True)
            except Exception as e:
                return FileResult(success=False, error=str(e))

    async def mkdir(self, path: str, parents: bool = True) -> FileResult:
        """Create directory."""
        async with self._lock:
            try:
                full_path = await self._resolve(path)
                full_path.mkdir(parents=parents, exist_ok=True)
                return FileResult(success=True)
            except Exception as e:
                return FileResult(success=False, error=str(e))

    async def list(
        self,
        path: str = ".",
        pattern: Optional[str] = None,
        recursive: bool = False,
    ) -> FileResult:
        """List directory contents."""
        async with self._lock:
            try:
                full_path = await self._resolve(path)

                if recursive:
                    items = list(full_path.rglob(pattern or "*"))
                else:
                    items = list(full_path.glob(pattern or "*"))

                result = "\n".join(str(p.relative_to(full_path.parent)) for p in items)
                return FileResult(success=True, data=result)
            except Exception as e:
                return FileResult(success=False, error=str(e))

    async def find(
        self,
        path: str = ".",
        name: Optional[str] = None,
        extension: Optional[str] = None,
    ) -> FileResult:
        """Find files matching criteria."""
        async with self._lock:
            try:
                full_path = await self._resolve(path)
                results: list[Path] = []

                for p in full_path.rglob("*"):
                    if name and name not in p.name:
                        continue
                    if extension and not p.suffix == extension:
                        continue
                    results.append(p)

                result = "\n".join(str(p) for p in results)
                return FileResult(success=True, data=result)
            except Exception as e:
                return FileResult(success=False, error=str(e))

    async def stat(self, path: str) -> FileResult:
        """Get file/directory stats."""
        async with self._lock:
            try:
                full_path = await self._resolve(path)
                st = full_path.stat()

                result = f"Size: {st.st_size}\nMode: {st.st_mode}\nModified: {st.st_mtime}"
                return FileResult(success=True, data=result)
            except Exception as e:
                return FileResult(success=False, error=str(e))

    async def exists(self, path: str) -> bool:
        """Check if path exists."""
        full_path = await self._resolve(path)
        return full_path.exists()

    async def is_file(self, path: str) -> bool:
        """Check if path is file."""
        full_path = await self._resolve(path)
        return full_path.is_file()

    async def is_dir(self, path: str) -> bool:
        """Check if path is directory."""
        full_path = await self._resolve(path)
        return full_path.is_dir()

    async def glob(self, path: str, pattern: str) -> FileResult:
        """Find files matching glob pattern."""
        async with self._lock:
            try:
                full_path = await self._resolve(path)
                results = list(full_path.glob(pattern))

                result = "\n".join(str(p) for p in results)
                return FileResult(success=True, data=result)
            except Exception as e:
                return FileResult(success=False, error=str(e))