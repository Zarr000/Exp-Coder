"""
Tool Registry for Expera AI.

Manages available tools:
- Registration
- Discovery
- Execution
- Documentation

Usage:
    registry = ToolRegistry()
    tool = registry.get("python_executor")
    result = await tool.execute(code="print('hello')")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Definition of a tool."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    function: Optional[Callable] = None
    category: str = "general"
    is_async: bool = False


@dataclass
class ToolResult:
    """Result of tool execution."""

    success: bool
    result: Any = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """
    Registry of available tools.

    Features:
    - Tool registration
    - Tool discovery
    - Execution
    - Validation
    """

    def __init__(self):
        """Initialize tool registry."""
        self._tools: dict[str, ToolDefinition] = {}
        self._categories: dict[str, set[str]] = {}

    def register(
        self,
        name: str,
        description: str,
        function: Optional[Callable] = None,
        parameters: Optional[dict] = None,
        category: str = "general",
    ) -> None:
        """Register a tool."""
        tool = ToolDefinition(
            name=name,
            description=description,
            function=function,
            parameters=parameters or {},
            category=category,
        )

        self._tools[name] = tool

        if category not in self._categories:
            self._categories[category] = set()
        self._categories[category].add(name)

        logger.info(f"Registered tool: {name}")

    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            tool = self._tools[name]
            category = tool.category

            del self._tools[name]

            if category in self._categories:
                self._categories[category].discard(name)

            logger.info(f"Unregistered tool: {name}")
            return True

        return False

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self, category: Optional[str] = None) -> list[ToolDefinition]:
        """List all tools."""
        if category:
            names = self._categories.get(category, set())
            return [self._tools[n] for n in names if n in self._tools]

        return list(self._tools.values())

    def get_categories(self) -> list[str]:
        """Get all categories."""
        return list(self._categories.keys())

    async def execute(
        self,
        name: str,
        **kwargs,
    ) -> ToolResult:
        """Execute a tool."""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool not found: {name}",
            )

        if not tool.function:
            return ToolResult(
                success=False,
                error=f"Tool has no function: {name}",
            )

        try:
            if tool.is_async:
                result = await tool.function(**kwargs)
            else:
                result = tool.function(**kwargs)

            return ToolResult(
                success=True,
                result=result,
            )

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return ToolResult(
                success=False,
                error=str(e),
            )

    def execute_sync(self, name: str, **kwargs) -> ToolResult:
        """Execute tool synchronously."""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool not found: {name}",
            )

        if not tool.function:
            return ToolResult(
                success=False,
                error=f"Tool has no function: {name}",
            )

        try:
            result = tool.function(**kwargs)

            return ToolResult(
                success=True,
                result=result,
            )

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return ToolResult(
                success=False,
                error=str(e),
            )

    def validate_parameters(self, name: str, params: dict) -> tuple[bool, str]:
        """Validate tool parameters."""
        tool = self._tools.get(name)
        if not tool:
            return False, f"Tool not found: {name}"

        required = tool.parameters.get("required", [])
        for param in required:
            if param not in params:
                return False, f"Missing required parameter: {param}"

        return True, ""

    def get_documentation(self, name: str) -> Optional[str]:
        """Get tool documentation."""
        tool = self._tools.get(name)
        if not tool:
            return None

        doc = f"# {tool.name}\n\n{tool.description}\n\n"

        if tool.parameters:
            doc += "## Parameters\n\n"
            for param, info in tool.parameters.items():
                doc += f"- {param}: {info.get('description', 'N/A')}\n"

        return doc


# Global registry instance
_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


# Export
__all__ = [
    "ToolRegistry",
    "ToolDefinition",
    "ToolResult",
    "get_registry",
]