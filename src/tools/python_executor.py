"""
Python Executor for Expera AI.

Executes Python code:
- Sandboxed execution
- Timeout handling
- Output capture
- Error handling

Usage:
    executor = PythonExecutor()
    result = await executor.execute(code)
    print(result.output)
"""

from __future__ import annotations

import asyncio
import io
import logging
import sys
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution."""

    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0


class PythonExecutor:
    """
    Executes Python code.

    Features:
    - Output capture
    - Timeout
    - Error handling
    - Sandboxing options
    """

    def __init__(
        self,
        timeout: int = 30,
        max_output: int = 10000,
    ):
        """Initialize executor."""
        self.timeout = timeout
        self.max_output = max_output
        self._globals = {}
        self._locals = {}

    def set_context(self, **kwargs) -> None:
        """Set execution context."""
        self._globals.update(kwargs)

    def clear_context(self) -> None:
        """Clear execution context."""
        self._globals.clear()
        self._locals.clear()

    async def execute(
        self,
        code: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """Execute Python code."""
        import time

        timeout = timeout or self.timeout
        start_time = time.time()

        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            # Compile code
            compiled = compile(code, "<string>", "exec")

            # Run in async context
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    self._run_code,
                    compiled,
                ),
                timeout=timeout,
            )

            # Get output
            output = sys.stdout.getvalue()
            error = sys.stderr.getvalue()

            execution_time = time.time() - start_time

            # Trim output if needed
            if len(output) > self.max_output:
                output = output[:self.max_output] + "\n... (output truncated)"

            return ExecutionResult(
                success=True,
                output=output,
                execution_time=execution_time,
            )

        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timeout ({timeout}s)",
                execution_time=timeout,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_output = sys.stderr.getvalue() or str(e)

            return ExecutionResult(
                success=False,
                output=sys.stdout.getvalue(),
                error=error_output,
                execution_time=execution_time,
            )

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _run_code(self, compiled) -> None:
        """Run compiled code."""
        exec(compiled, self._globals, self._locals)

    async def execute_safe(
        self,
        code: str,
    ) -> ExecutionResult:
        """Execute in sandboxed mode."""
        # Restrict builtins
        safe_builtins = {
            "print": print,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "open": open,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "bool": bool,
            "type": type,
            "isinstance": isinstance,
        }

        # Create safe globals
        safe_globals = {
            "__builtins__": safe_builtins,
        }

        # Capture
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            compiled = compile(code, "<string>", "exec")
            exec(compiled, safe_globals, {})

            output = sys.stdout.getvalue()

            return ExecutionResult(
                success=True,
                output=output,
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                output=sys.stdout.getvalue(),
                error=str(e),
            )

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# Export
__all__ = [
    "PythonExecutor",
    "ExecutionResult",
]