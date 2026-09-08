"""
Debugger for Expera AI.

Helps debug code:
- Error analysis
- Stack trace parsing
- Suggested fixes
- Common patterns

Usage:
    debugger = Debugger()
    fixes = await debugger.analyze(error, code)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DebugFix:
    """Suggested fix for a bug."""

    description: str
    fix_code: Optional[str] = None
    confidence: float = 1.0
    pattern: str = ""


# Error patterns and fixes
ERROR_FIXES = [
    # Python errors
    {
        "pattern": r"SyntaxError:.*invalid syntax",
        "severity": "error",
        "category": "syntax",
        "description": "Syntax error - check for missing colons, parentheses, or indentation",
        "fixes": [
            "Check line ending with colon (:)",
            "Verify matching parentheses",
            "Ensure proper indentation",
        ],
    },
    {
        "pattern": r"IndentationError",
        "severity": "error",
        "category": "indentation",
        "description": "Indentation error - inconsistent tabs/spaces",
        "fixes": [
            "Use consistent indentation (spaces recommended)",
            "Configure editor to use 4 spaces",
            "Check for mixed tabs and spaces",
        ],
    },
    {
        "pattern": r"NameError.*'([^']+)' is not defined",
        "severity": "error",
        "category": "undefined",
        "description": "Variable not defined",
        "fixes": [
            "Define the variable before use",
            "Check for typos in variable name",
            "Import the required module",
        ],
    },
    {
        "pattern": r"TypeError:.*'([^']+)' object is not iterable",
        "severity": "error",
        "category": "type",
        "description": "Object is not iterable",
        "fixes": [
            "Convert to iterable (list, tuple, etc.)",
            "Check if object is None",
            "Use proper data structure",
        ],
    },
    {
        "pattern": r"TypeError:.*cannot concatenate 'str' and 'int'",
        "severity": "error",
        "category": "type",
        "description": "Cannot concatenate str and int",
        "fixes": [
            "Convert int to string with str()",
            "Use f-string for formatting",
            "Use .format() method",
        ],
    },
    {
        "pattern": r"IndexError: list index out of range",
        "severity": "error",
        "category": "index",
        "description": "List index out of range",
        "fixes": [
            "Check list length before indexing",
            "Use enumerate() for safe iteration",
            "Check if list is empty",
        ],
    },
    {
        "pattern": r"KeyError: '([^']+)'",
        "severity": "error",
        "category": "key",
        "description": "Key not found in dictionary",
        "fixes": [
            "Use .get() method with default",
            "Check if key exists with 'in'",
            "Add key before accessing",
        ],
    },
    {
        "pattern": r"AttributeError:.*'([^']+)' has no attribute '([^']+)'",
        "severity": "error",
        "category": "attribute",
        "description": "Object has no attribute",
        "fixes": [
            "Check correct attribute name",
            "Import required module",
            "Use hasattr() to check",
        ],
    },
    {
        "pattern": r"ImportError.*No module named '([^']+)'",
        "severity": "error",
        "category": "import",
        "description": "Module not found",
        "fixes": [
            "Install module with pip",
            "Check module name spelling",
            "Verify virtual environment",
        ],
    },
    {
        "pattern": r"ZeroDivisionError",
        "severity": "error",
        "category": "math",
        "description": "Division by zero",
        "fixes": [
            "Check divisor before division",
            "Handle zero case explicitly",
        ],
    },
    {
        "pattern": r"ValueError:.*",
        "severity": "error",
        "category": "value",
        "description": "Invalid value",
        "fixes": [
            "Validate input values",
            "Check expected format",
            "Use try/except for handling",
        ],
    },
    # Common warnings
    {
        "pattern": r"ResourceWarning: unclosed file",
        "severity": "warning",
        "category": "resource",
        "description": "File not closed",
        "fixes": [
            "Use 'with' statement",
            "Call file.close()",
            "Use context manager",
        ],
    },
    {
        "pattern": r"DeprecationWarning",
        "severity": "warning",
        "category": "deprecation",
        "description": "Using deprecated feature",
        "fixes": [
            "Update to new API",
            "Suppress with warnings.filterwarnings",
        ],
    },
]


class Debugger:
    """
    Helps debug code issues.

    Features:
    - Error pattern matching
    - Suggested fixes
    - Stack trace analysis
    """

    def __init__(self):
        """Initialize debugger."""
        self.fixes = ERROR_FIXES.copy()
        self._custom_fixes = []

    async def analyze(
        self,
        error: str,
        code: Optional[str] = None,
    ) -> list[DebugFix]:
        """Analyze error and suggest fixes."""
        fixes = []

        # Check each pattern
        for fix_def in self.fixes + self._custom_fixes:
            pattern = fix_def["pattern"]
            if re.search(pattern, error, re.IGNORECASE):
                fix = DebugFix(
                    description=fix_def["description"],
                    confidence=0.9 if fix_def["severity"] == "error" else 0.7,
                    pattern=pattern,
                )
                fixes.append(fix)

        # Add generic suggestions
        if not fixes:
            fixes.append(DebugFix(
                description="Check error message for details",
                confidence=0.5,
            ))

        return fixes

    async def analyze_stack_trace(
        self,
        traceback: str,
    ) -> dict:
        """Analyze stack trace."""
        lines = traceback.split("\n")
        result = {
            "error_type": "",
            "error_message": "",
            "file": "",
            "line": None,
            "frames": [],
        }

        # Parse error type and message
        for line in lines:
            if line.startswith("Traceback (most recent call last)"):
                continue

            match = re.match(r"  File \"(.+)\", line (\d+)", line)
            if match:
                result["frames"].append({
                    "file": match.group(1),
                    "line": int(match.group(2)),
                })
                continue

            match = re.match(r"(\w+Error): (.+)", line)
            if match:
                result["error_type"] = match.group(1)
                result["error_message"] = match.group(2)

        return result

    def add_fix(
        self,
        pattern: str,
        description: str,
        severity: str = "error",
    ) -> None:
        """Add custom fix."""
        self._custom_fixes.append({
            "pattern": pattern,
            "description": description,
            "severity": severity,
        })

    def get_common_errors(self) -> list[str]:
        """Get list of common error types."""
        errors = set()
        for fix in self.fixes:
            match = re.match(r"(\w+Error):", fix["pattern"])
            if match:
                errors.add(match.group(1))
        return sorted(errors)


# Export
__all__ = [
    "Debugger",
    "DebugFix",
]