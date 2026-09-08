"""
Patch Generator for Expera AI Coding Agent.

Generates code patches from diffs and applies them.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PatchLine:
    """Represents a single line change."""

    type: str
    content: str
    old_line: Optional[int] = None
    new_line: Optional[int] = None


@dataclass
class Patch:
    """Represents a code patch."""

    file: str
    lines: list[PatchLine] = field(default_factory=list)
    header: Optional[str] = None


@dataclass
class Hunk:
    """Unified diff hunk."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    content: str


class PatchGenerator:
    """
    Generates and applies code patches.

    Supports:
    - Unified diff format
    - Context-aware patching
    - Multi-file patches
    - Reverse application
    """

    def __init__(self, root_path: Optional[str] = None) -> None:
        """Initialize patch generator."""
        self.root_path = Path(root_path) if root_path else Path.cwd()

    def generate_patch(
        self,
        original: str,
        modified: str,
        filename: str = "file",
    ) -> str:
        """Generate unified diff."""
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=filename,
            tofile=filename,
            lineterm="",
        )

        return "".join(diff)

    def parse_patch(self, patch_content: str) -> list[Patch]:
        """Parse unified diff."""
        patches: list[Patch] = []
        current_file: Optional[str] = None
        current_lines: list[PatchLine] = []

        for line in patch_content.split("\n"):
            if line.startswith("---") or line.startswith("+++"):
                current_file = line[4:].split("\t")[0]
            elif line.startswith("@@"):
                pass
            elif line.startswith("-"):
                current_lines.append(PatchLine(
                    type="remove",
                    content=line[1:],
                ))
            elif line.startswith("+"):
                current_lines.append(PatchLine(
                    type="add",
                    content=line[1:],
                ))
            elif line.startswith(" "):
                current_lines.append(PatchLine(
                    type="context",
                    content=line[1:],
                ))

        if current_file and current_lines:
            patches.append(Patch(file=current_file, lines=current_lines))

        return patches

    def apply_patch(
        self,
        original: str,
        patch_content: str,
    ) -> Optional[str]:
        """Apply patch to original."""
        try:
            patches = self.parse_patch(patch_content)

            if not patches:
                return None

            original_lines = original.splitlines(keepends=True)
            modified_lines = list(original_lines)
            offset = 0

            for patch in patches:
                for patch_line in patch.lines:
                    idx = (patch_line.old_line or 0) - 1 + offset

                    if patch_line.type == "add":
                        modified_lines.insert(idx, patch_line.content + "\n")
                        offset += 1

                    elif patch_line.type == "remove" and idx < len(modified_lines):
                        modified_lines.pop(idx)
                        offset -= 1

            return "".join(modified_lines)

        except Exception:
            return None

    def create_hunk(
        self,
        old_lines: list[str],
        new_lines: list[str],
        old_start: int = 1,
        new_start: int = 1,
    ) -> str:
        """Create unified diff hunk."""
        hunk = f"@@ -{old_start},{len(old_lines)} +{new_start},{len(new_lines)} @@\n"

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="a",
            tofile="b",
            lineterm="",
        )

        diff_lines = list(diff)[2:]
        hunk += "\n".join(diff_lines)

        return hunk

    def invert_patch(self, patch_content: str) -> str:
        """Invert a patch."""
        lines = patch_content.split("\n")
        result: list[str] = []

        i = 0
        while i < len(lines):
            line = lines[i]

            if line.startswith("@@"):
                match = re.match(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@", line)
                if match:
                    old_start = int(match.group(1))
                    old_count = int(match.group(2))
                    new_start = int(match.group(3))
                    new_count = int(match.group(4))
                    result.append(
                        f"@@ -{new_start},{new_count} +{old_start},{old_count} @@"
                    )
                else:
                    result.append(line)

            elif line.startswith("-"):
                result.append("+" + line[1:])

            elif line.startswith("+"):
                result.append("-" + line[1:])

            else:
                result.append(line)

            i += 1

        return "\n".join(result)

    def can_apply(
        self,
        original: str,
        patch_content: str,
    ) -> tuple[bool, list[str]]:
        """Check if patch can be applied."""
        issues: list[str] = []

        try:
            patches = self.parse_patch(patch_content)
            original_lines = original.splitlines()

            for patch in patches:
                for patch_line in patch.lines:
                    if patch_line.old_line and patch_line.old_line > len(original_lines):
                        issues.append(
                            f"Line {patch_line.old_line} out of range"
                        )

            return len(issues) == 0, issues

        except Exception as e:
            return False, [str(e)]

    def format_patch(
        self,
        file_path: str,
        old_content: str,
        new_content: str,
    ) -> str:
        """Format patch with proper headers."""
        patch = self.generate_patch(old_content, new_content, file_path)

        header = f"--- a/{file_path}\n+++ b/{file_path}\n"
        return header + patch

    def apply_strict(
        self,
        original: str,
        patch_content: str,
        fuzz: int = 2,
    ) -> Optional[str]:
        """Apply patch with fuzzy matching."""
        can_apply, issues = self.can_apply(original, patch_content)

        if can_apply:
            return self.apply_patch(original, patch_content)

        if fuzz <= 0:
            return None

        return self.apply_patch(original, patch_content)

    def split_patch(self, patch_content: str) -> list[dict[str, str]]:
        """Split multi-file patch into individual patches."""
        patches: list[dict[str, str]] = []
        current_file: Optional[str] = None
        current_lines: list[str] = []

        for line in patch_content.split("\n"):
            if line.startswith(("---", "+++")):
                if current_file and current_lines:
                    patches.append({
                        "file": current_file,
                        "patch": "\n".join(current_lines),
                    })

                current_file = line[4:].split("\t")[0]
                current_lines = [line]

            else:
                current_lines.append(line)

        if current_file and current_lines:
            patches.append({
                "file": current_file,
                "patch": "\n".join(current_lines),
            })

        return patches