"""
Tool Executor for Expera AI.

Unified execution of tools with streaming, retries, and error handling.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

from .registry import ToolDefinition, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Context for tool execution."""

    tool_name: str
    parameters: dict[str, Any]
    max_retries: int = 3
    timeout: float = 60.0
    stream: bool = False


@dataclass
class ExecutionResult:
    """Result of tool execution."""

    success: bool
    result: Any = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tokens: Optional[AsyncGenerator[str, None]] = None


class ToolExecutor:
    """
    Unified tool executor.

    Features:
    - Async execution
    - Automatic retries
    - Streaming support
    - Timeout handling
    - Error recovery
    """

    def __init__(self, registry: Optional[ToolRegistry] = None) -> None:
        """Initialize tool executor."""
        self.registry = registry or ToolRegistry()
        self._running: dict[str, asyncio.Task] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def execute(
        self,
        tool_name: str,
        parameters: Optional[dict[str, Any]] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
    ) -> ExecutionResult:
        """Execute tool with retries."""
        tool = self.registry.get(tool_name)

        if not tool:
            return ExecutionResult(
                success=False,
                error=f"Tool not found: {tool_name}"
            )

        last_error: Optional[str] = None

        for attempt in range(max_retries):
            try:
                if tool.is_async:
                    result = await asyncio.wait_for(
                        tool.execute(**(parameters or {})),
                        timeout=timeout,
                    )
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: tool.execute(**(parameters or {}))
                    )

                return ExecutionResult(
                    success=True,
                    result=result,
                    metadata={"attempts": attempt + 1},
                )

            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout}s"
                logger.warning(f"Tool {tool_name} timeout attempt {attempt + 1}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Tool {tool_name} error attempt {attempt + 1}: {e}")

            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        return ExecutionResult(
            success=False,
            error=last_error,
            metadata={"attempts": max_retries},
        )

    async def execute_stream(
        self,
        tool_name: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """Execute tool with streaming output."""
        tool = self.registry.get(tool_name)

        if not tool:
            yield json.dumps({"error": f"Tool not found: {tool_name}"})
            return

        try:
            result = await tool.execute(**(parameters or {}))

            if hasattr(result, "__iter__") and not isinstance(result, str):
                for item in result:
                    yield json.dumps({"chunk": item})
            else:
                yield json.dumps({"result": result})

        except Exception as e:
            yield json.dumps({"error": str(e)})

    async def execute_parallel(
        self,
        executions: list[dict[str, Any]],
    ) -> list[ExecutionResult]:
        """Execute multiple tools in parallel."""
        tasks = []
        for exec_data in executions:
            task = self.execute(
                exec_data["tool"],
                exec_data.get("parameters"),
                exec_data.get("max_retries", 3),
                exec_data.get("timeout", 60.0),
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [
            r if isinstance(r, ExecutionResult)
            else ExecutionResult(success=False, error=str(r))
            for r in results
        ]

    async def cancel(self, tool_name: str) -> bool:
        """Cancel running tool execution."""
        if tool_name in self._running:
            task = self._running[tool_name]
            task.cancel()
            del self._running[tool_name]
            return True
        return False

    def get_schema(self, tool_name: str) -> Optional[dict[str, Any]]:
        """Get tool JSON schema."""
        tool = self.registry.get(tool_name)
        if not tool:
            return None
        return tool.parameters

    def list_schemas(self) -> dict[str, dict[str, Any]]:
        """List all tool schemas."""
        schemas = {}
        for name, tool in self.registry._tools.items():
            schemas[name] = tool.parameters
        return schemas

    def validate_parameters(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """Validate tool parameters against schema."""
        tool = self.registry.get(tool_name)
        if not tool:
            return False, f"Tool not found: {tool_name}"

        schema = tool.parameters
        required = schema.get("required", [])

        for param in required:
            if param not in parameters:
                return False, f"Missing required parameter: {param}"

        return True, None