"""
Agent Evaluation.

Evaluates agent performance on tasks.

Usage:
    python -m src.evaluation.eval_agent --model checkpoints/expera-350m --tasks data/eval/tasks.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentTask:
    """Agent task."""

    id: str
    description: str
    expected_action: str
    tools_available: list[str]
    success_criteria: str


@dataclass
class AgentResult:
    """Agent evaluation result."""

    task_id: str
    success: bool
    steps_taken: list[str]
    final_action: str = ""
    error: Optional[str] = None
    duration_ms: float = 0


@dataclass
class AgentBenchmarkResult:
    """Agent benchmark result."""

    name: str
    total: int
    passed: int
    pass_rate: float
    avg_steps: float
    avg_duration_ms: float
    results: list[AgentResult] = None


class AgentEvaluator:
    """Evaluates agent performance."""

    def __init__(self, model_path: str):
        self.model_path = model_path

    async def execute_tool(self, tool: str, parameters: dict) -> dict:
        """Execute a tool."""
        # This is a mock implementation
        if tool == "llm":
            return {"output": "Generated response"}
        elif tool == "bash":
            return {"output": "Command executed"}
        elif tool == "file":
            return {"output": "File operation done"}
        elif tool == "search":
            return {"output": ["file1.py", "file2.py"]}
        else:
            return {"error": f"Unknown tool: {tool}"}

    async def evaluate_task(
        self,
        task: AgentTask,
    ) -> AgentResult:
        """Evaluate a single task."""
        import time

        start_time = time.perf_counter()
        steps = []

        try:
            # Simple plan execution (mock)
            if "generate" in task.expected_action.lower():
                result = await self.execute_tool("llm", {"prompt": task.description})
                steps.append("llm")
            elif "search" in task.expected_action.lower():
                result = await self.execute_tool("search", {"pattern": task.description})
                steps.append("search")
            else:
                result = await self.execute_tool("bash", {"command": task.description})
                steps.append("bash")

            # Check success
            success = "error" not in result

            return AgentResult(
                task_id=task.id,
                success=success,
                steps_taken=steps,
                final_action=task.expected_action,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        except Exception as e:
            return AgentResult(
                task_id=task.id,
                success=False,
                steps_taken=steps,
                error=str(e),
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

    async def evaluate_file(
        self,
        file_path: Path,
    ) -> AgentBenchmarkResult:
        """Evaluate on task file."""
        tasks = []
        with open(file_path) as f:
            for line in f:
                item = json.loads(line)
                tasks.append(
                    AgentTask(
                        id=item.get("id", ""),
                        description=item.get("description", ""),
                        expected_action=item.get("expected_action", ""),
                        tools_available=item.get("tools", []),
                        success_criteria=item.get("success_criteria", ""),
                    )
                )

        results = []
        for task in tasks:
            result = await self.evaluate_task(task)
            results.append(result)

        passed = sum(1 for r in results if r.success)
        avg_steps = sum(len(r.steps_taken) for r in results) / len(results) if results else 0
        avg_duration = sum(r.duration_ms for r in results) / len(results) if results else 0

        return AgentBenchmarkResult(
            name=file_path.stem,
            total=len(results),
            passed=passed,
            pass_rate=passed / len(results) if results else 0.0,
            avg_steps=avg_steps,
            avg_duration_ms=avg_duration,
            results=results,
        )

    async def evaluate_directory(
        self,
        dir_path: Path,
    ) -> list[AgentBenchmarkResult]:
        """Evaluate on directory of task files."""
        results = []

        for file_path in dir_path.glob("*.jsonl"):
            result = await self.evaluate_file(file_path)
            results.append(result)

        return results


async def main():
    parser = argparse.ArgumentParser(description="Agent evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", type=Path)
    args = parser.parse_args()

    evaluator = AgentEvaluator(model_path=args.model)

    if args.tasks:
        if args.tasks.is_dir():
            results = await evaluator.evaluate_directory(args.tasks)
            for result in results:
                print(f"\n{result.name}:")
                print(f"  Pass Rate: {result.pass_rate * 100:.1f}%")
                print(f"  Avg Steps: {result.avg_steps:.1f}")
                print(f"  Avg Duration: {result.avg_duration_ms:.0f}ms")
        else:
            result = await evaluator.evaluate_file(args.tasks)
            print(f"\n{result.name}:")
            print(f"  Pass Rate: {result.pass_rate * 100:.1f}%")
            print(f"  Avg Steps: {result.avg_steps:.1f}")
            print(f"  Avg Duration: {result.avg_duration_ms:.0f}ms")

    else:
        print("Provide --tasks")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())