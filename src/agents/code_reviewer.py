"""
Code Reviewer for Expera AI.

Reviews code for issues:
- Style violations
- Bugs and errors
- Security issues
- Performance problems

Usage:
    reviewer = CodeReviewer()
    issues = await reviewer.review(code)
    for issue in issues:
        print(f"{issue.severity}: {issue.message}")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReviewIssue:
    """A code review issue."""

    severity: str  # error, warning, info
    category: str  # style, bug, security, performance
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    code: Optional[str] = None


@dataclass
class ReviewResult:
    """Result of code review."""

    issues: list[ReviewIssue] = field(default_factory=list)
    score: float = 0.0
    summary: str = ""


# Rule definitions
REVIEW_RULES = [
    # Python rules
    {
        "pattern": r"except:",
        "severity": "warning",
        "category": "bug",
        "message": "Bare except clause - caught all exceptions",
    },
    {
        "pattern": r"eval\(",
        "severity": "warning",
        "category": "security",
        "message": "eval() is dangerous - consider safer alternatives",
    },
    {
        "pattern": r"exec\(",
        "severity": "warning",
        "category": "security",
        "message": "exec() is dangerous - consider safer alternatives",
    },
    {
        "pattern": r"os\.system\(",
        "severity": "warning",
        "category": "security",
        "message": "os.system() can be a security risk",
    },
    {
        "pattern": r"subprocess\.call\([^,)]+\)",
        "severity": "info",
        "category": "security",
        "message": "subprocess.call with shell=True may be a security risk",
    },
    {
        "pattern": r"print\(.+\+",
        "severity": "info",
        "category": "style",
        "message": "Using + for string concatenation - consider f-strings",
    },
    {
        "pattern": r"from \S+ import \*",
        "severity": "warning",
        "category": "style",
        "message": "Wildcard imports (from x import *) are discouraged",
    },
    {
        "pattern": r"todo",
        "severity": "info",
        "category": "style",
        "message": "TODO comment found - should be addressed",
    },
    {
        "pattern": r"fixme",
        "severity": "info",
        "category": "style",
        "message": "FIXME comment found - should be addressed",
    },
    {
        "pattern": r"== None",
        "severity": "warning",
        "category": "style",
        "message": "Use 'is None' for None comparison",
    },
    {
        "pattern": r"!= None",
        "severity": "warning",
        "category": "style",
        "message": "Use 'is not None' for None comparison",
    },
    # Security rules
    {
        "pattern": r"password\s*=\s*[\"']",
        "severity": "warning",
        "category": "security",
        "message": "Hardcoded password detected",
    },
    {
        "pattern": r"api[_-]?key\s*=\s*[\"']",
        "severity": "warning",
        "category": "security",
        "message": "Hardcoded API key detected",
    },
    {
        "pattern": r"secret\s*=\s*[\"']",
        "severity": "warning",
        "category": "security",
        "message": "Hardcoded secret detected",
    },
    # Performance rules
    {
        "pattern": r"for .+ in .+:\s+for .+ in",
        "severity": "info",
        "category": "performance",
        "message": "Nested loops detected - consider optimization",
    },
    {
        "pattern": r"\.append\(.+\+",
        "severity": "info",
        "category": "performance",
        "message": "String concatenation in loop - use join()",
    },
]


class CodeReviewer:
    """
    Reviews code for issues.

    Features:
    - Multiple rule categories
    - Configurable rules
    - Severity levels
    - Line numbers
    """

    def __init__(self):
        """Initialize code reviewer."""
        self.rules = REVIEW_RULES.copy()
        self._custom_rules = []

    def add_rule(
        self,
        pattern: str,
        severity: str,
        category: str,
        message: str,
    ) -> None:
        """Add a custom rule."""
        self._custom_rules.append({
            "pattern": pattern,
            "severity": severity,
            "category": category,
            "message": message,
        })

    async def review(
        self,
        code: str,
        language: str = "python",
    ) -> ReviewResult:
        """Review code and return issues."""
        issues = []
        lines = code.split("\n")

        # Apply all rules
        rules = self.rules + self._custom_rules

        for rule in rules:
            pattern = rule["pattern"]
            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error:
                continue

            for line_num, line in enumerate(lines, 1):
                if regex.search(line):
                    issue = ReviewIssue(
                        severity=rule["severity"],
                        category=rule["category"],
                        message=rule["message"],
                        line=line_num,
                        code=line.strip(),
                    )
                    issues.append(issue)

        # Calculate score
        score = self._calculate_score(issues)

        # Generate summary
        summary = self._generate_summary(issues)

        return ReviewResult(
            issues=issues,
            score=score,
            summary=summary,
        )

    def _calculate_score(self, issues: list[ReviewIssue]) -> float:
        """Calculate code quality score."""
        # Start with perfect score
        score = 100.0

        # Deduct for issues
        for issue in issues:
            if issue.severity == "error":
                score -= 10
            elif issue.severity == "warning":
                score -= 5
            elif issue.severity == "info":
                score -= 1

        return max(0.0, score)

    def _generate_summary(self, issues: list[ReviewIssue]) -> str:
        """Generate review summary."""
        if not issues:
            return "No issues found. Code looks good!"

        # Count by severity
        errors = sum(1 for i in issues if i.severity == "error")
        warnings = sum(1 for i in issues if i.severity == "warning")
        infos = sum(1 for i in issues if i.severity == "info")

        parts = []
        if errors > 0:
            parts.append(f"{errors} error(s)")
        if warnings > 0:
            parts.append(f"{warnings} warning(s)")
        if infos > 0:
            parts.append(f"{infos} info(s)")

        return f"Found {', '.join(parts)}"

    async def review_file(
        self,
        file_path: str,
    ) -> Optional[ReviewResult]:
        """Review a file."""
        try:
            with open(file_path) as f:
                code = f.read()

            # Detect language from extension
            if file_path.endswith(".py"):
                language = "python"
            elif file_path.endswith(".js"):
                language = "javascript"
            elif file_path.endswith(".ts"):
                language = "typescript"
            elif file_path.endswith(".rs"):
                language = "rust"
            else:
                language = "python"

            return await self.review(code, language)

        except Exception as e:
            logger.error(f"Failed to review file: {e}")
            return None

    def get_rules(self) -> list[dict]:
        """Get all rules."""
        return self.rules + self._custom_rules


# Export
__all__ = [
    "CodeReviewer",
    "ReviewIssue",
    "ReviewResult",
]