"""
Bug Fixer for Expera AI Coding Agent.

Autonomous bug detection and fixing.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Bug:
    """Represents a detected bug."""

    severity: str
    line: int
    message: str
    rule: str
    confidence: float = 1.0


@dataclass
class Fix:
    """Represents a bug fix."""

    bug: Bug
    original: str
    fixed: str
    explanation: str


@dataclass
class FixResult:
    """Result of bug fixing."""

    success: bool
    bugs_found: int
    bugs_fixed: int
    fixes: list[Fix] = field(default_factory=list)
    error: Optional[str] = None


class BugFixer:
    """
    Autonomous bug fixer.

    Detects and fixes common bugs:
    - Syntax errors
    - Logic errors
    - Security issues
    - Performance issues
    - Style violations
    """

    def __init__(self, root_path: Optional[str] = None) -> None:
        """Initialize bug fixer."""
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self._rules = self._load_rules()

    def _load_rules(self) -> dict:
        """Load bug detection rules."""
        return {
            "syntax": [
                (r"except:", "Bare except clause"),
                (r"def .*__init__.*:\s*pass", "Empty __init__ method"),
            ],
            "logic": [
                (r"if.*==\s*True", "Redundant comparison to True"),
                (r"if.*==\s*False", "Use 'not' instead"),
                (r"for .* in range\(len\(", "Use enumerate"),
            ],
            "security": [
                (r"eval\(", "Use of eval()"),
                (r"exec\(", "Use of exec()"),
                (r"os\.system\(", "Use of os.system()"),
            ],
            "performance": [
                (r"\.append\(.*\).*\.append\(", "Use list comprehension"),
                (r"for .*:\s*.*\.read\(\)", "Use context manager"),
            ],
        }

    def analyze_file(self, file_path: str) -> list[Bug]:
        """Analyze file for bugs."""
        bugs: list[Bug] = []
        full_path = self.root_path / file_path

        try:
            content = full_path.read_text()
            lines = content.split("\n")

            for category, patterns in self._rules.items():
                for pattern, message in patterns:
                    for i, line in enumerate(lines, 1):
                        if re.search(pattern, line):
                            bugs.append(Bug(
                                severity=category,
                                line=i,
                                message=message,
                                rule=pattern,
                                confidence=0.8,
                            ))

        except Exception:
            pass

        return bugs

    def find_syntax_errors(self, file_path: str) -> list[Bug]:
        """Find syntax errors."""
        bugs: list[Bug] = []
        full_path = self.root_path / file_path

        try:
            content = full_path.read_text()
            ast.parse(content)

        except SyntaxError as e:
            bugs.append(Bug(
                severity="error",
                line=e.lineno or 1,
                message=str(e),
                rule="syntax",
                confidence=1.0,
            ))

        return bugs

    def find_unused_imports(self, file_path: str) -> list[Bug]:
        """Find unused imports."""
        bugs: list[Bug] = []
        full_path = self.root_path / file_path

        try:
            content = full_path.read_text()
            tree = ast.parse(content)

            imported: set[str] = set()
            used: set[str] = set()

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        imported.add(alias.name)

                elif isinstance(node, ast.Name):
                    used.add(node.id)

            for imp in imported - used:
                bugs.append(Bug(
                    severity="warning",
                    line=1,
                    message=f"Unused import: {imp}",
                    rule="unused-import",
                    confidence=0.9,
                ))

        except Exception:
            pass

        return bugs

    def find_bugs(self, file_path: str) -> list[Bug]:
        """Find all bugs in file."""
        bugs = []
        bugs.extend(self.analyze_file(file_path))
        bugs.extend(self.find_syntax_errors(file_path))
        bugs.extend(self.find_unused_imports(file_path))
        return bugs

    def fix_bug(self, bug: Bug, content: str) -> Optional[str]:
        """Fix a specific bug."""
        lines = content.split("\n")

        if bug.rule == "eval(":
            return content.replace("eval(", "ast.literal_eval(")

        elif bug.rule == "os.system(":
            return content.replace("os.system(", "subprocess.run(")

        elif "Bare except" in bug.message:
            if bug.line <= len(lines):
                lines[bug.line - 1] = lines[bug.line - 1].replace(
                    "except:",
                    "except Exception:",
                )
                return "\n".join(lines)

        return None

    def fix_all(self, file_path: str) -> FixResult:
        """Fix all bugs in file."""
        full_path = self.root_path / file_path

        try:
            content = full_path.read_text()
            bugs = self.find_bugs(file_path)

            fixed_content = content
            fixes: list[Fix] = []

            for bug in bugs:
                if fixed := self.fix_bug(bug, fixed_content):
                    fixes.append(Fix(
                        bug=bug,
                        original=fixed_content,
                        fixed=fixed,
                        explanation=bug.message,
                    ))
                    fixed_content = fixed

            full_path.write_text(fixed_content)

            return FixResult(
                success=True,
                bugs_found=len(bugs),
                bugs_fixed=len(fixes),
                fixes=fixes,
            )

        except Exception as e:
            return FixResult(
                success=False,
                bugs_found=0,
                bugs_fixed=0,
                error=str(e),
            )

    def analyze_project(self, root: Optional[str] = None) -> list[Bug]:
        """Analyze entire project."""
        root = root or self.root_path
        all_bugs: list[Bug] = []

        for path in Path(root).rglob("*.py"):
            if "__pycache__" not in str(path):
                bugs = self.find_bugs(str(path.relative_to(root)))
                all_bugs.extend(bugs)

        return all_bugs

    def fix_project(self, root: Optional[str] = None) -> FixResult:
        """Fix all bugs in project."""
        root = root or self.root_path
        total_fixed = 0
        all_fixes: list[Fix] = []

        for path in Path(root).rglob("*.py"):
            if "__pycache__" in str(path):
                continue

            result = self.fix_all(str(path.relative_to(root)))
            total_fixed += result.bugs_fixed
            all_fixes.extend(result.fixes)

        return FixResult(
            success=True,
            bugs_found=0,
            bugs_fixed=total_fixed,
            fixes=all_fixes,
        )