"""
Semantic Memory for Expera AI.

Stores structured knowledge and concepts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class Concept:
    """Knowledge concept."""

    id: str
    name: str
    definition: str
    category: str
    attributes: dict[str, Any] = field(default_factory=dict)
    related: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Relationship:
    """Concept relationship."""

    from_concept: str
    to_concept: str
    relation_type: str
    strength: float = 1.0


class SemanticMemory:
    """
    Semantic memory storage.

    Stores structured knowledge and concepts.
    """

    def __init__(self, storage_path: Optional[str] = None) -> None:
        """Initialize semantic memory."""
        self.storage_path = storage_path
        self._concepts: dict[str, Concept] = {}
        self._relationships: list[Relationship] = []
        self._load()

    def _load(self) -> None:
        """Load from storage."""
        if self.storage_path:
            path = self.storage_path / "semantic.json"
            if path.exists():
                data = json.loads(path.read_text())
                self._concepts = {
                    k: Concept(
                        id=c["id"],
                        name=c["name"],
                        definition=c["definition"],
                        category=c["category"],
                        attributes=c.get("attributes", {}),
                        related=c.get("related", []),
                        created_at=datetime.fromisoformat(c["created_at"]),
                    )
                    for k, c in data.get("concepts", {}).items()
                }
                self._relationships = [
                    Relationship(
                        from_concept=r["from_concept"],
                        to_concept=r["to_concept"],
                        relation_type=r["relation_type"],
                        strength=r.get("strength", 1.0),
                    )
                    for r in data.get("relationships", [])
                ]

    def _save(self) -> None:
        """Save to storage."""
        if self.storage_path:
            path = self.storage_path / "semantic.json"
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "concepts": {
                    k: {
                        "id": c.id,
                        "name": c.name,
                        "definition": c.definition,
                        "category": c.category,
                        "attributes": c.attributes,
                        "related": c.related,
                        "created_at": c.created_at.isoformat(),
                    }
                    for k, c in self._concepts.items()
                },
                "relationships": [
                    {
                        "from_concept": r.from_concept,
                        "to_concept": r.to_concept,
                        "relation_type": r.relation_type,
                        "strength": r.strength,
                    }
                    for r in self._relationships
                ],
            }
            path.write_text(json.dumps(data, indent=2))

    def add_concept(
        self,
        name: str,
        definition: str,
        category: str,
        attributes: Optional[dict[str, Any]] = None,
    ) -> Concept:
        """Add concept."""
        concept = Concept(
            id=name.lower().replace(" ", "_"),
            name=name,
            definition=definition,
            category=category,
            attributes=attributes or {},
        )
        self._concepts[concept.id] = concept
        self._save()
        return concept

    def get_concept(self, name: str) -> Optional[Concept]:
        """Get concept."""
        return self._concepts.get(name.lower().replace(" ", "_"))

    def get_by_category(self, category: str) -> list[Concept]:
        """Get concepts by category."""
        return [c for c in self._concepts.values() if c.category == category]

    def add_relationship(
        self,
        from_concept: str,
        to_concept: str,
        relation_type: str,
        strength: float = 1.0,
    ) -> None:
        """Add relationship."""
        self._relationships.append(Relationship(
            from_concept=from_concept,
            to_concept=to_concept,
            relation_type=relation_type,
            strength=strength,
        ))
        self._save()

    def get_related(self, concept_name: str) -> list[Concept]:
        """Get related concepts."""
        related_ids = []

        for rel in self._relationships:
            if rel.from_concept == concept_name:
                related_ids.append(rel.to_concept)
            elif rel.to_concept == concept_name:
                related_ids.append(rel.from_concept)

        return [self._concepts[r] for r in related_ids if r in self._concepts]

    def search(self, query: str) -> list[Concept]:
        """Search concepts."""
        results = []
        query_lower = query.lower()

        for concept in self._concepts.values():
            if query_lower in concept.name.lower():
                results.append(concept)
            elif query_lower in concept.definition.lower():
                results.append(concept)

        return results

    def delete_concept(self, name: str) -> bool:
        """Delete concept."""
        key = name.lower().replace(" ", "_")
        if key in self._concepts:
            del self._concepts[key]
            self._relationships = [
                r for r in self._relationships
                if r.from_concept != key and r.to_concept != key
            ]
            self._save()
            return True
        return False

    def count(self) -> int:
        """Count concepts."""
        return len(self._concepts)