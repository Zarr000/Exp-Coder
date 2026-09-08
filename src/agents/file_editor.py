"""
File Editor for Expera AI.

Edits files:
- Read/write files
- Apply patches
- Search and replace
- Line operations

Usage:
    editor = FileEditor()
    editor.read_file("main.py")
    editor.replace("old_code", "new_code")
    editor.write_file("main.py")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class EditOperation:
    """An edit operation."""

    type: str  # replace, insert, delete, append
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    old_text: Optional[str] = None
    new_text: Optional[str] = None
    pattern: Optional[str] = None
    position: str = "after"  # before, after


@dataclass
class EditResult:
    """Result of an edit operation."""

    success: bool
    message: str
    changes: int = 0


class FileEditor:
    """
    Edits source files.

    Features:
    - In-place editing
    - Search and replace
    - Line-based operations
    - Backup support
    """

    def __init__(self):
        """Initialize file editor."""
        self._content: str = ""
        self._file_path: Optional[Path] = None
        self._operations: list[EditOperation] = []

    def read_file(self, file_path: str) -> bool:
        """Read file content."""
        try:
            path = Path(file_path)
            self._content = path.read_text(encoding="utf-8")
            self._file_path = path
            logger.info(f"Read {len(self._content)} bytes from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            return False

    def write_file(
        self,
        file_path: Optional[str] = None,
        backup: bool = True,
    ) -> bool:
        """Write content to file."""
        path = Path(file_path) if file_path else self._file_path
        if not path:
            logger.error("No file path specified")
            return False

        try:
            # Backup if requested
            if backup and path.exists():
                backup_path = path.with_suffix(path.suffix + ".bak")
                backup_path.write_text(path.read_text(encoding="utf-8"))
                logger.info(f"Created backup: {backup_path}")

            path.write_text(self._content, encoding="utf-8")
            logger.info(f"Wrote {len(self._content)} bytes to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to write file: {e}")
            return False

    def get_content(self) -> str:
        """Get file content."""
        return self._content

    def get_lines(self) -> list[str]:
        """Get content as lines."""
        return self._content.split("\n")

    def replace(
        self,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace text."""
        if not old_text in self._content:
            return EditResult(False, "Text not found", 0)

        count = 0
        if replace_all:
            count = self._content.count(old_text)
            self._content = self._content.replace(old_text, new_text)
        else:
            count = 1
            self._content = self._content.replace(old_text, new_text, 1)

        logger.info(f"Replaced {count} occurrence(s)")
        return EditResult(True, f"Replaced {count} occurrence(s)", count)

    def replace_regex(
        self,
        pattern: str,
        new_text: str,
        replace_all: bool = True,
    ) -> EditResult:
        """Replace using regex."""
        try:
            regex = re.compile(pattern)
            matches = regex.findall(self._content)
            count = len(matches) if replace_all else min(1, len(matches))

            if replace_all:
                self._content = regex.sub(new_text, self._content)
            else:
                self._content = regex.sub(new_text, self._content, 1)

            logger.info(f"Replaced {count} match(es)")
            return EditResult(True, f"Replaced {count} match(es)", count)
        except re.error as e:
            logger.error(f"Invalid regex: {e}")
            return EditResult(False, f"Invalid regex: {e}", 0)

    def insert_at_line(
        self,
        line_number: int,
        text: str,
        position: str = "before",
    ) -> EditResult:
        """Insert text at line."""
        lines = self.get_lines()

        if line_number < 1 or line_number > len(lines) + 1:
            return EditResult(False, "Invalid line number", 0)

        idx = line_number - 1
        if position == "after":
            idx = min(line_number, len(lines))

        lines.insert(idx, text)
        self._content = "\n".join(lines)

        logger.info(f"Inserted at line {line_number} ({position})")
        return EditResult(True, f"Inserted at line {line_number}", 1)

    def delete_lines(
        self,
        start_line: int,
        end_line: Optional[int] = None,
    ) -> EditResult:
        """Delete lines."""
        lines = self.get_lines()

        if start_line < 1 or start_line > len(lines):
            return EditResult(False, "Invalid start line", 0)

        end = end_line if end_line else start_line
        if end < start_line or end > len(lines):
            return EditResult(False, "Invalid end line", 0)

        del lines[start_line - 1:end]
        self._content = "\n".join(lines)

        count = end - start_line + 1
        logger.info(f"Deleted {count} line(s)")
        return EditResult(True, f"Deleted {count} line(s)", count)

    def append(self, text: str) -> EditResult:
        """Append text to end."""
        if self._content:
            self._content += "\n"
        self._content += text

        logger.info(f"Appended {len(text)} characters")
        return EditResult(True, "Appended text", 1)

    def prepend(self, text: str) -> EditResult:
        """Prepend text to beginning."""
        text_to_add = text.rstrip()
        if self._content:
            text_to_add += "\n"
            self._content = text_to_add + self._content
        else:
            self._content = text_to_add

        logger.info(f"Prepended {len(text_to_add)} characters")
        return EditResult(True, "Prepended text", 1)

    def find_line(self, pattern: str, start: int = 0) -> Optional[int]:
        """Find line number matching pattern."""
        lines = self.get_lines()

        for i in range(start, len(lines)):
            if pattern in lines[i]:
                return i + 1

        return None

    def find_all(self, pattern: str) -> list[int]:
        """Find all lines matching pattern."""
        lines = self.get_lines()
        matches = []

        for i, line in enumerate(lines):
            if pattern in line:
                matches.append(i + 1)

        return matches

    def get_line(self, line_number: int) -> Optional[str]:
        """Get specific line."""
        lines = self.get_lines()

        if 1 <= line_number <= len(lines):
            return lines[line_number - 1]

        return None

    def line_count(self) -> int:
        """Get line count."""
        return len(self.get_lines())

    def is_modified(self) -> bool:
        """Check if content was modified."""
        return bool(self._operations)


# Export
__all__ = [
    "FileEditor",
    "EditOperation",
    "EditResult",
]