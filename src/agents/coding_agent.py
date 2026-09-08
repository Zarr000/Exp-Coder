"""
Coding Agent.

Specialized agent for code generation and editing.

Usage:
    python -m src.agents.coding_agent --task "create a web server"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.agents.executor import Executor, ExecutionResult
from src.agents.planner import Plan, Planner, TaskType
from src.agents.repository_agent import RepoAgent
from src.agents.tool_router import ToolRouter, ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class CodingContext:
    """Coding task context."""

    project_root: Path
    language: str
    framework: Optional[str] = None
    existing_files: list[str] = None


class CodingAgent:
    """Agent for code tasks."""

    def __init__(self, llm=None, project_root: Optional[Path] = None):
        self.llm = llm
        self.project_root = project_root or Path.cwd()
        self.planner = Planner(llm)
        self.executor = Executor(llm)
        self.repo = RepoAgent(self.project_root)
        self.router = ToolRouter()
        self._registry = ToolRegistry(self.router)

    async def generate_code(
        self,
        specification: str,
        language: str = "python",
        output_file: Optional[str] = None,
    ) -> ExecutionResult:
        """Generate code from specification."""
        logger.info(f"Generating {language} code: {specification}")

        # Create plan
        plan = await self.planner.plan(
            f"Generate {language} code: {specification}",
            {"files": [], "root": str(self.project_root)},
        )

        # Execute
        results = await self.executor.execute_plan(plan)

        return results[0] if results else None

    async def edit_code(
        self,
        file_path: str,
        edit_type: str,  # refactor, fix, enhance
        description: str,
    ) -> ExecutionResult:
        """Edit existing code."""
        logger.info(f"Editing {file_path}: {description}")

        # Get file content
        full_path = self.project_root / file_path
        if full_path.exists():
            content = full_path.read_text()
        else:
            content = ""

        # Create plan
        plan = await self.planner.plan(
            f"{edit_type} {file_path}: {description}",
            {"files": [file_path], "root": str(self.project_root)},
        )

        # Execute
        results = await self.executor.execute_plan(plan)

        return results[0] if results else None

    async def debug_code(
        self,
        error: str,
        file_path: Optional[str] = None,
    ) -> dict:
        """Debug code error."""
        logger.info(f"Debugging: {error}")

        # Search for similar code
        if file_path:
            matches = await self.repo.search(error, language="python")
            logger.info(f"Found {len(matches)} matches")

        # Create plan
        plan = await self.planner.plan(
            f"Fix bug: {error}",
            {"file": file_path, "root": str(self.project_root)},
        )

        # Execute
        results = await self.executor.execute_plan(plan)

        return {
            "plan": plan,
            "results": results,
        }

    async def review_code(
        self,
        file_path: str,
    ) -> dict:
        """Review code."""
        logger.info(f"Reviewing: {file_path}")

        # Get file content
        full_path = self.project_root / file_path
        if not full_path.exists():
            return {"error": "File not found"}

        content = full_path.read_text()

        # Analyze with LLM
        prompt = f"""Review this {self.repo.context.files[0].language if self.repo.context else 'code'}

{content}

Provide feedback on:
- Correctness
- Performance
- Security
- Style
"""
        # Create plan
        plan = await self.planner.plan(
            f"Review {file_path}",
            {"files": [file_path], "root": str(self.project_root)},
        )

        # Execute
        results = await self.executor.execute_plan(plan)

        return {
            "plan": plan,
            "results": results,
        }

    async def find_similar_code(
        self,
        query: str,
    ) -> list[dict]:
        """Find similar code examples."""
        matches = await self.repo.search(query)

        results = []
        for path, line_num, line in matches:
            results.append({
                "file": path,
                "line": line_num,
                "content": line,
            })

        return results

    async def create_from_template(
        self,
        template: str,
        output_file: str,
        variables: dict,
    ) -> ExecutionResult:
        """Create code from template."""
        logger.info(f"Creating from template: {template}")

        # Map templates
        templates = {
            "python/lambda": "import json\n\ndef handler(event, context):\n    return {\n        'statusCode': 200,\n        'body': json.dumps(event)\n    }\n",
            "python/fastapi": '''from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {{"message": "Hello World"}}
''',
            "python/flask": '''from flask import Flask

app = Flask(__name__)

@app.route("/")
def root():
    return "Hello World"
''',
            "javascript/express": '''const express = require("express");\nconst app = express();\n\napp.get("/", (req, res) => {{\n  res.json({{message: "Hello World"}});\n}});\n\napp.listen(3000);
''',
        }

        content = templates.get(template, "")
        if not content:
            return ExecutionResult(
                step_id=0, success=False, error=f"Unknown template: {template}"
            )

        # Apply variables
        for key, value in variables.items():
            content = content.replace(f"{{{{{key}}}}}", str(value))

        # Write file
        output_path = self.project_root / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)

        return ExecutionResult(
            step_id=1,
            success=True,
            output=str(output_path),
        )


async def main():
    parser = argparse.ArgumentParser(description="Coding agent")
    parser.add_argument("--task", required=True)
    parser.add_argument("--language", default="python")
    parser.add_argument("--output", help="Output file")
    args = parser.parse_args()

    agent = CodingAgent()

    if "generate" in args.task.lower() or "create" in args.task.lower():
        result = await agent.generate_code(
            args.task,
            language=args.language,
            output_file=args.output,
        )
        print(f"Generated: {result.output if result else 'failed'}")

    elif "fix" in args.task.lower() or "debug" in args.task.lower():
        result = await agent.debug_code(args.task)
        print(f"Results: {result}")

    else:
        print("Use --task with generate, fix, or review")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())