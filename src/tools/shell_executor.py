"""
Shell Executor for Expera AI.

Executes shell commands:
- Command execution
- Output capture
- Timeout handling
- Security controls

Usage:
    executor = ShellExecutor()
    result = await executor.execute("ls -la")
    print(result.output)
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ShellResult:
    """Result of shell execution."""

    success: bool
    output: str
    error: Optional[str] = None
    return_code: int = 0
    execution_time: float = 0.0


class ShellExecutor:
    """
    Executes shell commands.

    Features:
    - Command execution
    - Output capture
    - Timeout
    - Security controls
    """

    def __init__(
        self,
        timeout: int = 60,
        max_output: int = 50000,
        allowed_commands: Optional[list[str]] = None,
        blocked_commands: Optional[list[str]] = None,
        working_dir: Optional[str] = None,
    ):
        """Initialize executor."""
        self.timeout = timeout
        self.max_output = max_output
        self.allowed_commands = allowed_commands
        self.blocked_commands = blocked_commands or [
            "rm -rf /",
            "dd if=",
            "mkfs",
            "fdisk",
            " parted",
            ":(){:|:&};:",
        ]
        self.working_dir = working_dir

    def _is_command_allowed(self, command: str) -> tuple[bool, str]:
        """Check if command is allowed."""
        # Check blocked
        for blocked in self.blocked_commands:
            if blocked in command:
                return False, f"Command blocked: {blocked}"

        # Check allowed if specified
        if self.allowed_commands:
            base_cmd = command.split()[0] if command.split() else ""
            if base_cmd not in self.allowed_commands:
                return False, f"Command not allowed: {base_cmd}"

        return True, ""

    async def execute(
        self,
        command: str,
        timeout: Optional[int] = None,
        shell: bool = True,
    ) -> ShellResult:
        """Execute shell command."""
        import time

        timeout = timeout or self.timeout

        # Security check
        allowed, reason = self._is_command_allowed(command)
        if not allowed:
            return ShellResult(
                success=False,
                output="",
                error=reason,
                return_code=1,
            )

        start_time = time.time()

        # Set working directory
        cwd = self.working_dir or os.getcwd()

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ShellResult(
                    success=False,
                    output="",
                    error=f"Command timeout ({timeout}s)",
                    return_code=124,
                )

            execution_time = time.time() - start_time

            # Decode output
            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")

            # Trim output
            if len(output) > self.max_output:
                output = output[:self.max_output] + "\n... (output truncated)"

            return ShellResult(
                success=process.returncode == 0,
                output=output,
                error=error_output or None,
                return_code=process.returncode,
                execution_time=execution_time,
            )

        except Exception as e:
            return ShellResult(
                success=False,
                output="",
                error=str(e),
                return_code=1,
            )

    async def execute_script(
        self,
        script: str,
        interpreter: str = "bash",
    ) -> ShellResult:
        """Execute script content."""
        import tempfile
        import stat

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f".{interpreter}",
            delete=False,
        ) as f:
            f.write(script)
            temp_path = f.name

        try:
            # Make executable
            os.chmod(temp_path, 0o755)

            # Execute
            return await self.execute(temp_path)

        finally:
            # Cleanup
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    def set_working_dir(self, path: str) -> None:
        """Set working directory."""
        path_obj = Path(path)
        if path_obj.exists() and path_obj.is_dir():
            self.working_dir = str(path_obj)
        else:
            logger.warning(f"Working directory not found: {path}")


# Export
__all__ = [
    "ShellExecutor",
    "ShellResult",
]