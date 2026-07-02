"""
Planner Agent.

Decomposes complex tasks into actionable steps.

Usage:
    python -m src.agents.planner --task "Create a web server" --context /path/to/project
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Types of tasks."""

    CODE_GENERATE = "code_generate"
    CODE_EDIT = "code_edit"
    CODE_REVIEW = "code_review"
    DEBUG = "debug"
    REFACTOR = "refactor"
    DOCUMENT = "document"
    SEARCH = "search"
    IMAGE_GENERATE = "image_generate"
    IMAGE_EDIT = "image_edit"
    RESEARCH = "research"
    UNKNOWN = "unknown"


class Complexity(str, Enum):
    """Task complexity levels."""

    SIMPLE = "simple"  # Single step
    MODERATE = "moderate"  # Few steps
    COMPLEX = "complex"  # Multiple steps with dependencies


@dataclass
class TaskStep:
    """A single step in a plan."""

    id: int
    action: str  # analyze, generate, edit, execute, review
    target: str  # file, function, etc.
    description: str
    depends_on: list[int] = field(default_factory=list)
    tool: Optional[str] = None  # llm, bash, git, file, search, image
    parameters: dict = field(default_factory=dict)

    def is_ready(self, completed: set[int]) -> bool:
        """Check if all dependencies are completed."""
        return all(dep in completed for dep in self.depends_on)


@dataclass
class Plan:
    """A task plan."""

    task_type: TaskType
    complexity: Complexity
    goal: str
    steps: list[TaskStep] = field(default_factory=list)
    context_files: list[str] = field(default_factory=list)
    estimated_steps: int = 0


class Planner:
    """Plans task execution."""

    def __init__(self, model=None):
        self.model = model

    async def plan(
        self,
        task: str,
        context: Optional[dict] = None,
    ) -> Plan:
        """
        Create a plan for the task.

        Args:
            task: Task description
            context: Additional context (project files, etc.)

        Returns:
            Plan with steps
        """
        context = context or {}

        # Analyze task type
        task_type = self._classify_task(task)

        # Analyze complexity
        complexity = self._classify_complexity(task, context)

        # Generate steps
        steps = await self._generate_steps(task, task_type, complexity, context)

        return Plan(
            task_type=task_type,
            complexity=complexity,
            goal=task,
            steps=steps,
            context_files=context.get("files", []),
            estimated_steps=len(steps),
        )

    def _classify_task(self, task: str) -> TaskType:
        """Classify the task type."""
        task_lower = task.lower()

        if any(kw in task_lower for kw in ["generate", "create", "write", "implement"]):
            if any(ext in task_lower for ext in [".py", ".js", ".ts", ".html", ".css"]):
                return TaskType.CODE_GENERATE
            return TaskType.CODE_GENERATE

        if any(kw in task_lower for kw in ["fix", "bug", "error", "crash"]):
            return TaskType.DEBUG

        if any(kw in task_lower for kw in ["edit", "modify", "update", "change"]):
            return TaskType.CODE_EDIT

        if any(kw in task_lower for kw in ["refactor", "improve", "clean"]):
            return TaskType.REFACTOR

        if any(kw in task_lower for kw in ["review", "check", "analyze"]):
            return TaskType.CODE_REVIEW

        if any(kw in task_lower for kw in ["image", "picture", "photo", "generate"]):
            return TaskType.IMAGE_GENERATE

        if any(kw in task_lower for kw in ["search", "find", "look"]):
            return TaskType.SEARCH

        return TaskType.UNKNOWN

    def _classify_complexity(self, task: str, context: dict) -> Complexity:
        """Classify task complexity."""
        # Check for multiple files
        files = context.get("files", [])
        if len(files) > 5:
            return Complexity.COMPLEX

        # Check for complex keywords
        complex_keywords = ["architecture", "system", "database", "api", "multiple"]
        if any(kw in task.lower() for kw in complex_keywords):
            return Complexity.COMPLEX

        # Check for simple keywords
        simple_keywords = ["simple", "single", "one"]
        if any(kw in task.lower() for kw in simple_keywords):
            return Complexity.SIMPLE

        return Complexity.MODERATE

    async def _generate_steps(
        self,
        task: str,
        task_type: TaskType,
        complexity: Complexity,
        context: dict,
    ) -> list[TaskStep]:
        """Generate steps for the task."""
        steps = []
        step_id = 1

        # Always start with analysis
        steps.append(
            TaskStep(
                id=step_id,
                action="analyze",
                target=context.get("root", "."),
                description=f"Analyze task: {task}",
                tool="repo",
                parameters={"task": task},
            )
        )
        step_id += 1

        if task_type == TaskType.CODE_GENERATE:
            if complexity == Complexity.COMPLEX:
                steps.append(
                    TaskStep(
                        id=step_id,
                        action="search",
                        target=".",
                        description="Find similar code for reference",
                        tool="search",
                    )
                )
                step_id += 1

            steps.append(
                TaskStep(
                    id=step_id,
                    action="generate",
                    target="new_file",
                    description="Generate code",
                    tool="llm",
                    parameters={
                        "prompt": task,
                        "type": "code",
                    },
                )
            )
            step_id += 1

            steps.append(
                TaskStep(
                    id=step_id,
                    action="execute",
                    target="test",
                    description="Verify code works",
                    tool="bash",
                    parameters={"command": "test"},
                )
            )

        elif task_type == TaskType.DEBUG:
            steps.append(
                TaskStep(
                    id=step_id,
                    action="search",
                    target="error_logs",
                    description="Find error context",
                    tool="bash",
                    parameters={"command": "logs"},
                )
            )
            step_id += 1

            steps.append(
                TaskStep(
                    id=step_id,
                    action="analyze",
                    target="error",
                    description="Analyze bug root cause",
                    tool="llm",
                )
            )
            step_id += 1

            steps.append(
                TaskStep(
                    id=step_id,
                    action="edit",
                    target="fix",
                    description="Apply fix",
                    tool="file",
                )
            )
            step_id += 1

            steps.append(
                TaskStep(
                    id=step_id,
                    action="execute",
                    target="test",
                    description="Verify fix",
                    tool="bash",
                )
            )

        elif task_type == TaskType.IMAGE_GENERATE:
            steps.append(
                TaskStep(
                    id=step_id,
                    action="generate",
                    target="prompt",
                    description="Enhance prompt",
                    tool="llm",
                    parameters={"prompt": task, "type": "image"},
                )
            )
            step_id += 1

            steps.append(
                TaskStep(
                    id=step_id,
                    action="generate",
                    target="image",
                    description="Generate image",
                    tool="image",
                )
            )

        else:
            # Generic task
            steps.append(
                TaskStep(
                    id=step_id,
                    action="execute",
                    target="task",
                    description="Execute task",
                    tool="llm",
                )
            )

        return steps

    def get_step(self, plan: Plan, step_id: int) -> Optional[TaskStep]:
        """Get a step by ID."""
        for step in plan.steps:
            if step.id == step_id:
                return step
        return None

    def get_ready_steps(self, plan: Plan, completed: set[int]) -> list[TaskStep]:
        """Get steps that are ready to execute."""
        return [
            step for step in plan.steps if step.is_ready(completed)
        ]


async def main():
    parser = argparse.ArgumentParser(description="Planner agent")
    parser.add_argument("--task", required=True)
    parser.add_argument("--context", default=".")
    args = parser.parse_args()

    planner = Planner()

    # Get context
    context = {"root": args.context}

    plan = await planner.plan(args.task, context)

    print(f"Task Type: {plan.task_type}")
    print(f"Complexity: {plan.complexity}")
    print(f"Estimated Steps: {plan.estimated_steps}")
    print("\nSteps:")
    for step in plan.steps:
        deps = f" (depends on {step.depends_on})" if step.depends_on else ""
        print(f"  {step.id}. {step.action}{deps}")
        print(f"     {step.description}")
        if step.tool:
            print(f"     Tool: {step.tool}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())