"""
Executor Agent.

Executes planned steps and handles tool invocations.

Usage:
    python -m src.agents.executor --plan plan.json
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from src.agents.planner import Plan, TaskStep

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result from executing a step."""

    step_id: int
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0


@dataclass
class ToolResult:
    """Result from a tool."""

    tool: str
    success: bool
    output: Any = None
    error: Optional[str] = None


class ToolExecutor:
    """Executes individual tools."""

    def __init__(self, llm=None, model_path: str = "checkpoints/expera-350m"):
        self.llm = llm
        self.model_path = model_path
        self._tools: dict[str, Callable] = {}

    def register_tool(self, name: str, func: Callable) -> None:
        """Register a tool."""
        self._tools[name] = func

    async def execute_tool(
        self,
        tool: str,
        parameters: dict,
    ) -> ToolResult:
        """Execute a tool."""
        if tool not in self._tools:
            return ToolResult(
                tool=tool,
                success=False,
                error=f"Unknown tool: {tool}",
            )

        try:
            result = await self._tools[tool](**parameters)
            return ToolResult(tool=tool, success=True, output=result)
        except Exception as e:
            return ToolResult(tool=tool, success=False, error=str(e))

    async def llm_tool(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Execute LLM generation."""
        if self.llm:
            result = await self.llm.generate(prompt, temperature=temperature)
            return result.text
        return "LLM not available"

    async def bash_tool(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: int = 60,
    ) -> str:
        """Execute bash command."""
        cwd = cwd or os.getcwd()

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            return result.stdout
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Command timed out: {command}")
        except Exception as e:
            raise RuntimeError(f"Command failed: {e}")

    async def file_tool(
        self,
        action: str,
        path: str,
        content: Optional[str] = None,
    ) -> dict:
        """File operations."""
        file_path = Path(path)

        if action == "read":
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            return {"content": file_path.read_text()}

        if action == "write":
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content or "")
            return {"path": str(file_path)}

        if action == "exists":
            return {"exists": file_path.exists()}

        if action == "list":
            if not file_path.is_dir():
                raise NotADirectoryError(f"Not a directory: {path}")
            return {"files": list(file_path.iterdir())}

        raise ValueError(f"Unknown action: {action}")

    async def search_tool(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: str = "*",
    ) -> list[str]:
        """Search files."""
        import re

        matches = []
        search_path = Path(path)

        for file in search_path.rglob(file_pattern):
            if file.is_file():
                try:
                    content = file.read_text(errors="ignore")
                    if re.search(pattern, content):
                        matches.append(str(file))
                except Exception:
                    continue

        return matches

    async def git_tool(
        self,
        command: str,
        cwd: Optional[str] = None,
    ) -> str:
        """Git operations."""
        return await self.bash_tool(f"git {command}", cwd=cwd)

    async def image_tool(
        self,
        action: str,
        prompt: str,
        **kwargs,
    ) -> dict:
        """Image generation tools."""
        # Import image clients
        try:
            from src.image import FluxClient, FluxConfig

            config = FluxConfig()
            async with FluxClient(config) as client:
                if action == "generate":
                    images = await client.generate(prompt, **kwargs)
                    return {"images": [len(images)]}
        except Exception as e:
            raise RuntimeError(f"Image generation failed: {e}")

        raise ValueError(f"Unknown action: {action}")

    def get_available_tools(self) -> list[str]:
        """Get list of available tools."""
        return list(self._tools.keys())


class Executor:
    """Executes task plans."""

    def __init__(self, llm=None):
        self.llm = llm
        self.tool_executor = ToolExecutor(llm)
        self._register_tools()

    def _register_tools(self) -> None:
        """Register default tools."""
        self.tool_executor.register_tool("llm", self.tool_executor.llm_tool)
        self.tool_executor.register_tool("bash", self.tool_executor.bash_tool)
        self.tool_executor.register_tool("file", self.tool_executor.file_tool)
        self.tool_executor.register_tool(
            "search", self.tool_executor.search_tool
        )
        self.tool_executor.register_tool("git", self.tool_executor.git_tool)
        self.tool_executor.register_tool("image", self.tool_executor.image_tool)

    async def execute_step(
        self,
        step: TaskStep,
        context: dict,
    ) -> ExecutionResult:
        """Execute a single step."""
        import time

        start_time = time.perf_counter()

        logger.info(f"Executing step {step.id}: {step.action} on {step.target}")

        # Get tool
        tool = step.tool or self._default_tool(step.action)
        parameters = step.parameters.copy()
        parameters.update(context)

        # Execute
        result = await self.tool_executor.execute_tool(tool, parameters)

        duration_ms = (time.perf_counter() - start_time) * 1000

        return ExecutionResult(
            step_id=step.id,
            success=result.success,
            output=result.output,
            error=result.error,
            duration_ms=duration_ms,
        )

    def _default_tool(self, action: str) -> str:
        """Get default tool for action."""
        tool_map = {
            "analyze": "llm",
            "generate": "llm",
            "edit": "file",
            "execute": "bash",
            "search": "search",
            "review": "llm",
            "image": "image",
        }
        return tool_map.get(action, "llm")

    async def execute_plan(
        self,
        plan: Plan,
        context: Optional[dict] = None,
    ) -> list[ExecutionResult]:
        """Execute a full plan."""
        context = context or {}
        results = []
        completed: set[int] = set()

        while len(completed) < len(plan.steps):
            # Get ready steps
            ready_steps = [
                step for step in plan.steps
                if step.is_ready(completed)
            ]

            if not ready_steps:
                logger.warning("No ready steps, likely circular dependency")
                break

            # Execute each ready step
            for step in ready_steps:
                result = await self.execute_step(step, context)
                results.append(result)

                if result.success:
                    completed.add(step.id)
                    context[f"step_{step.id}"] = result.output
                else:
                    logger.error(
                        f"Step {step.id} failed: {result.error}"
                    )
                    if plan.complexity.value == "simple":
                        raise RuntimeError(
                            f"Step {step.id} failed: {result.error}"
                        )

        return results

    async def execute_steps_parallel(
        self,
        steps: list[TaskStep],
        context: dict,
    ) -> list[ExecutionResult]:
        """Execute steps in parallel where possible."""
        results = []

        # Group by dependencies
        ready = [s for s in steps if not s.depends_on]

        while ready:
            # Execute ready steps in parallel
            tasks = [self.execute_step(step, context) for step in ready]
            batch_results = await asyncio.gather(*tasks)

            results.extend(batch_results)

            # Get next batch
            completed_ids = {r.step_id for r in batch_results if r.success}
            ready = [
                s for s in steps
                if s.id not in completed_ids
                and all(d in completed_ids for d in s.depends_on)
            ]

        return results


async def main():
    parser = argparse.ArgumentParser(description="Executor agent")
    parser.add_argument("--plan", type=Path, help="Plan JSON file")
    parser.add_argument("--step-id", type=int, help="Execute specific step")
    args = parser.parse_args()

    executor = Executor()

    if args.plan:
        import json

        with open(args.plan) as f:
            plan_data = json.load(f)

        # Reconstruct plan
        plan = Plan(
            task_type=plan_data.get("task_type", "UNKNOWN"),
            complexity=plan_data.get("complexity", "MODERATE"),
            goal=plan_data.get("goal", ""),
        )

        results = await executor.execute_plan(plan)
        print(f"Executed {len(results)} steps")

        for result in results:
            status = "✓" if result.success else "✗"
            print(f"{status} Step {result.step_id}: {result.duration_ms:.0f}ms")
            if result.error:
                print(f"  Error: {result.error}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())