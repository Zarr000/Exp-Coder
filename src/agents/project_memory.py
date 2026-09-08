"""
Project Memory for Expera AI Coding Agent.

Stores project-specific context and knowledge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class ProjectContext:
    """Project context."""

    name: str
    root_path: str
    language: str
    framework: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeEntity:
    """Code entity for memory."""

    name: str
    kind: str
    file: str
    line: int
    docstring: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ProjectMemory:
    """
    Project-specific memory.

    Stores:
    - Project structure
    - Code entities
    - Conventions
    - User preferences
    """

    def __init__(self, root_path: Optional[str] = None) -> None:
        """Initialize project memory."""
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self.memory_file = self.root_path / ".expera" / "memory.json"
        self._data: dict = self._load()

    def _load(self) -> dict:
        """Load memory from file."""
        if self.memory_file.exists():
            return json.loads(self.memory_file.read_text())
        return {"context": {}, "entities": [], "conventions": []}

    def _save(self) -> None:
        """Save memory to file."""
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.memory_file.write_text(json.dumps(self._data, indent=2))

    def set_context(self, context: ProjectContext) -> None:
        """Set project context."""
        self._data["context"] = {
            "name": context.name,
            "root_path": context.root_path,
            "language": context.language,
            "framework": context.framework,
            "dependencies": context.dependencies,
            "config": context.config,
        }
        self._save()

    def get_context(self) -> Optional[ProjectContext]:
        """Get project context."""
        ctx = self._data.get("context", {})
        if not ctx:
            return None

        return ProjectContext(
            name=ctx.get("name", ""),
            root_path=ctx.get("root_path", ""),
            language=ctx.get("language", ""),
            framework=ctx.get("framework"),
            dependencies=ctx.get("dependencies", []),
            config=ctx.get("config", {}),
        )

    def add_entity(self, entity: CodeEntity) -> None:
        """Add code entity."""
        entities = self._data.get("entities", [])
        entities.append({
            "name": entity.name,
            "kind": entity.kind,
            "file": entity.file,
            "line": entity.line,
            "docstring": entity.docstring,
            "metadata": entity.metadata,
        })
        self._data["entities"] = entities
        self._save()

    def get_entities(self, kind: Optional[str] = None) -> list[CodeEntity]:
        """Get code entities."""
        entities = self._data.get("entities", [])

        if kind:
            entities = [e for e in entities if e.get("kind") == kind]

        return [
            CodeEntity(
                name=e["name"],
                kind=e["kind"],
                file=e["file"],
                line=e["line"],
                docstring=e.get("docstring"),
                metadata=e.get("metadata", {}),
            )
            for e in entities
        ]

    def add_convention(self, name: str, pattern: str, description: str) -> None:
        """Add code convention."""
        conventions = self._data.get("conventions", [])
        conventions.append({
            "name": name,
            "pattern": pattern,
            "description": description,
        })
        self._data["conventions"] = conventions
        self._save()

    def get_conventions(self) -> list[dict]:
        """Get code conventions."""
        return self._data.get("conventions", [])

    def search_entities(self, query: str) -> list[CodeEntity]:
        """Search entities."""
        entities = self.get_entities()
        results = []

        for e in entities:
            if query in e.name or query in (e.docstring or ""):
                results.append(e)

        return results

    def forget_entity(self, name: str) -> None:
        """Remove entity."""
        entities = self._data.get("entities", [])
        entities = [e for e in entities if e.get("name") != name]
        self._data["entities"] = entities
        self._save()

    def clear(self) -> None:
        """Clear all memory."""
        self._data = {"context": {}, "entities": [], "conventions": []}
        self._save()

    def export(self) -> str:
        """Export memory to JSON."""
        return json.dumps(self._data, indent=2)

    def import_data(self, data: str) -> None:
        """Import memory from JSON."""
        self._data = json.loads(data)
        self._save()