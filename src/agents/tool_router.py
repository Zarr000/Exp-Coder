"""
Tool Router.

Routes requests to appropriate tools based on:
- Request intent
- Available tools
- Tool capabilities

Usage:
    python -m src.agents.tool_router --intent "fix bug"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """Detected intents."""

    CODE_WRITE = "code_write"
    CODE_EDIT = "code_edit"
    CODE_READ = "code_read"
    CODE_DELETE = "code_delete"
    DEBUG_FIX = "debug_fix"
    SEARCH = "search"
    COMMAND = "command"
    IMAGE_GENERATE = "image_generate"
    IMAGE_EDIT = "image_edit"
    CHAT = "chat"
    QUESTION = "question"
    UNKNOWN = "unknown"


class ToolCapability(str, Enum):
    """Tool capabilities."""

    CODE_GENERATION = "code_generation"
    CODE_EDITING = "code_editing"
    CODE_READING = "code_reading"
    BASH_EXECUTION = "bash_execution"
    FILE_OPERATIONS = "file_operations"
    SEARCH = "search"
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDITING = "image_editing"
    REASONING = "reasoning"
    CHAT = "chat"


@dataclass
class Tool:
    """A registered tool."""

    name: str
    description: str
    capabilities: list[ToolCapability]
    parameters: dict = field(default_factory=dict)
    handler: Optional[Callable] = None


@dataclass
class ToolChoice:
    """Selected tool with parameters."""

    tool: str
    parameters: dict
    confidence: float = 1.0


class ToolRouter:
    """Routes requests to appropriate tools."""

    def __init__(self):
        self.tools: dict[str, Tool] = {}
        self._intent_keywords = {
            Intent.CODE_WRITE: ["write", "create", "implement", "generate"],
            Intent.CODE_EDIT: ["edit", "modify", "change", "update", "refactor"],
            Intent.CODE_READ: ["read", "show", "display", "view"],
            Intent.CODE_DELETE: ["delete", "remove"],
            Intent.DEBUG_FIX: ["fix", "bug", "error", "crash", "debug"],
            Intent.SEARCH: ["search", "find", "look", "grep"],
            Intent.COMMAND: ["run", "execute", "command", "bash"],
            Intent.IMAGE_GENERATE: ["generate", "create", "draw"],
            Intent.IMAGE_EDIT: ["edit", "modify", "inpaint"],
            Intent.CHAT: ["chat", "talk", "conversation"],
            Intent.QUESTION: ["what", "how", "why", "when", "explain"],
        }

    def register_tool(
        self,
        name: str,
        description: str,
        capabilities: list[ToolCapability],
        handler: Optional[Callable] = None,
    ) -> None:
        """Register a tool."""
        tool = Tool(
            name=name,
            description=description,
            capabilities=capabilities,
            handler=handler,
        )
        self.tools[name] = tool

    def detect_intent(self, request: str) -> Intent:
        """Detect intent from request."""
        request_lower = request.lower()

        # Check keywords
        for intent, keywords in self._intent_keywords.items():
            if any(kw in request_lower for kw in keywords):
                return intent

        return Intent.UNKNOWN

    def route(
        self,
        request: str,
        intent: Optional[Intent] = None,
    ) -> list[ToolChoice]:
        """Route request to tools."""
        intent = intent or self.detect_intent(request)

        # Map intent to tools
        if intent == Intent.CODE_WRITE:
            return [
                ToolChoice(
                    tool="llm",
                    parameters={"prompt": request, "type": "code"},
                )
            ]

        if intent == Intent.CODE_EDIT:
            return [
                ToolChoice(tool="search", parameters={"pattern": request}),
                ToolChoice(tool="llm", parameters={"prompt": request}),
            ]

        if intent == Intent.DEBUG_FIX:
            return [
                ToolChoice(tool="bash", parameters={"command": "logs"}),
                ToolChoice(tool="llm", parameters={"prompt": request}),
                ToolChoice(tool="file", parameters={"action": "edit"}),
            ]

        if intent == Intent.SEARCH:
            return [
                ToolChoice(
                    tool="search",
                    parameters={"pattern": request},
                )
            ]

        if intent == Intent.COMMAND:
            return [
                ToolChoice(
                    tool="bash",
                    parameters={"command": request},
                )
            ]

        if intent == Intent.IMAGE_GENERATE:
            return [
                ToolChoice(
                    tool="llm",
                    parameters={"prompt": request, "type": "image"},
                ),
                ToolChoice(
                    tool="image",
                    parameters={"action": "generate"},
                ),
            ]

        if intent == Intent.CHAT:
            return [
                ToolChoice(tool="llm", parameters={"prompt": request})
            ]

        # Default to LLM
        return [ToolChoice(tool="llm", parameters={"prompt": request})]

    def get_tools_for_capability(
        self,
        capability: ToolCapability,
    ) -> list[Tool]:
        """Get tools that provide a capability."""
        return [
            tool for tool in self.tools.values()
            if capability in tool.capabilities
        ]

    async def execute_choices(
        self,
        choices: list[ToolChoice],
    ) -> list[dict]:
        """Execute tool choices in order."""
        results = []

        for choice in choices:
            tool = self.tools.get(choice.tool)
            if not tool:
                results.append(
                    {"tool": choice.tool, "error": "Tool not found"}
                )
                continue

            if not tool.handler:
                results.append(
                    {"tool": choice.tool, "error": "No handler"}
                )
                continue

            try:
                output = await tool.handler(**choice.parameters)
                results.append({"tool": choice.tool, "output": output})
            except Exception as e:
                results.append({"tool": choice.tool, "error": str(e)})

        return results


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self, router: ToolRouter):
        self.router = router
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register default tools."""
        # LLM
        self.router.register_tool(
            name="llm",
            description="Language model for generation",
            capabilities=[
                ToolCapability.CODE_GENERATION,
                ToolCapability.REASONING,
                ToolCapability.CHAT,
            ],
        )

        # Bash
        self.router.register_tool(
            name="bash",
            description="Execute shell commands",
            capabilities=[ToolCapability.BASH_EXECUTION],
        )

        # File
        self.router.register_tool(
            name="file",
            description="File operations",
            capabilities=[
                ToolCapability.CODE_READING,
                ToolCapability.CODE_EDITING,
                ToolCapability.FILE_OPERATIONS,
            ],
        )

        # Search
        self.router.register_tool(
            name="search",
            description="Search files and code",
            capabilities=[ToolCapability.SEARCH],
        )

        # Git
        self.router.register_tool(
            name="git",
            description="Git version control",
            capabilities=[ToolCapability.BASH_EXECUTION],
        )

        # Image
        self.router.register_tool(
            name="image",
            description="Image generation and editing",
            capabilities=[
                ToolCapability.IMAGE_GENERATION,
                ToolCapability.IMAGE_EDITING,
            ],
        )

    def list_tools(self) -> list[dict]:
        """List all registered tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "capabilities": tool.capabilities,
            }
            for tool in self.router.tools.values()
        ]


async def main():
    parser = argparse.ArgumentParser(description="Tool router")
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    router = ToolRouter()
    registry = ToolRegistry(router)

    intent = router.detect_intent(args.request)
    print(f"Intent: {intent.value}")

    choices = router.route(args.request)
    print("\nRouting:")
    for choice in choices:
        print(f"  {choice.tool}: {choice.parameters}")

    print("\nAvailable tools:")
    for tool in registry.list_tools():
        print(f"  {tool['name']}: {tool['description']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())