"""
Model Router for Expera AI.

Routes requests to appropriate models:
- By task type
- By model size
- By capability
- Load balancing

Usage:
    router = ModelRouter()
    model = router.route("code-gen", prompt_length=100)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a model."""

    name: str
    path: str
    size_mb: int
    parameters: int
    context_length: int
    capabilities: list[str] = field(default_factory=list)
    quantizations: list[str] = field(default_factory=list)
    is_available: bool = True
    is_loaded: bool = False


@dataclass
class RouteRule:
    """Rule for routing to a model."""

    task_type: str
    min_size_mb: int = 0
    max_size_mb: int = int(1e9)
    model_name: Optional[str] = None


class ModelRouter:
    """
    Routes requests to appropriate models.

    Features:
    - Task-based routing
    - Size-based selection
    - Load distribution
    - Fallback handling
    """

    def __init__(self):
        """Initialize model router."""
        self.models: dict[str, ModelInfo] = {}
        self.rules: list[RouteRule] = []
        self._load_default_models()

    def _load_default_models(self) -> None:
        """Load default model configurations."""
        # Expera model sizes
        default_models = [
            ModelInfo(
                name="expera-coder-120m",
                path="checkpoints/expera_coder_120m",
                size_mb=256,
                parameters=120_000_000,
                context_length=2048,
                capabilities=["code-gen", "chat", "infill"],
                quantizations=["fp16", "q4", "q5", "q8"],
            ),
            ModelInfo(
                name="expera-coder-350m",
                path="checkpoints/expera_coder_350m",
                size_mb=700,
                parameters=350_000_000,
                context_length=2048,
                capabilities=["code-gen", "chat", "infill", "reasoning"],
                quantizations=["fp16", "q4", "q5", "q8"],
            ),
            ModelInfo(
                name="expera-coder-1b",
                path="checkpoints/expera_coder_1b",
                size_mb=2_000,
                parameters=1_000_000_000,
                context_length=4096,
                capabilities=["code-gen", "chat", "infill", "reasoning", "completion"],
                quantizations=["fp16", "q4", "q5", "q8"],
            ),
            ModelInfo(
                name="expera-coder-3b",
                path="checkpoints/expera_coder_3b",
                size_mb=6_000,
                parameters=3_000_000_000,
                context_length=4096,
                capabilities=["code-gen", "chat", "infill", "reasoning", "completion", "long-context"],
                quantizations=["fp16", "q4", "q5", "q8"],
            ),
        ]

        for model in default_models:
            self.models[model.name] = model

        # Default routing rules
        self.rules = [
            RouteRule(task_type="code-gen", model_name="expera-coder-350m"),
            RouteRule(task_type="chat", model_name="expera-coder-120m"),
            RouteRule(task_type="infill", model_name="expera-coder-350m"),
            RouteRule(task_type="reasoning", min_size_mb=700, model_name="expera-coder-1b"),
            RouteRule(task_type="completion", min_size_mb=2000, model_name="expera-coder-1b"),
            RouteRule(task_type="long-context", min_size_mb=5000, model_name="expera-coder-3b"),
        ]

    def register_model(self, model: ModelInfo) -> None:
        """Register a model."""
        self.models[model.name] = model
        logger.info(f"Registered model: {model.name}")

    def unregister_model(self, name: str) -> None:
        """Unregister a model."""
        if name in self.models:
            del self.models[name]
            logger.info(f"Unregistered model: {name}")

    def add_rule(self, rule: RouteRule) -> None:
        """Add a routing rule."""
        self.rules.append(rule)
        logger.info(f"Added rule for task: {rule.task_type}")

    def route(
        self,
        task_type: str,
        prompt_length: int = 0,
        preferred_model: Optional[str] = None,
    ) -> Optional[ModelInfo]:
        """Route to appropriate model."""
        # Check preferred model first
        if preferred_model and preferred_model in self.models:
            model = self.models[preferred_model]
            if model.is_available:
                return model

        # Find matching rule
        for rule in self.rules:
            if rule.task_type == task_type:
                # Find model by name
                if rule.model_name and rule.model_name in self.models:
                    model = self.models[rule.model_name]
                    if model.is_available:
                        return model

                # Find by size constraints
                for name, model in self.models.items():
                    if (model.size_mb >= rule.min_size_mb
                        and model.size_mb <= rule.max_size_mb
                        and model.is_available):
                        return model

        # Fallback to smallest available model
        for name, model in sorted(self.models.items(), key=lambda x: x[1].size_mb):
            if model.is_available:
                return model

        return None

    def get_models_by_capability(self, capability: str) -> list[ModelInfo]:
        """Get models supporting a capability."""
        results = []
        for model in self.models.values():
            if capability in model.capabilities and model.is_available:
                results.append(model)
        return sorted(results, key=lambda x: x.size_mb)

    def get_models_by_quantization(self, quantization: str) -> list[ModelInfo]:
        """Get models with a quantization."""
        results = []
        for model in self.models.values():
            if quantization in model.quantizations and model.is_available:
                results.append(model)
        return sorted(results, key=lambda x: x.size_mb)

    def list_models(self) -> list[ModelInfo]:
        """List all registered models."""
        return list(self.models.values())

    def get_stats(self) -> dict:
        """Get router statistics."""
        return {
            "total_models": len(self.models),
            "available_models": sum(1 for m in self.models.values() if m.is_available),
            "loaded_models": sum(1 for m in self.models.values() if m.is_loaded),
        }


# Export
__all__ = [
    "ModelRouter",
    "ModelInfo",
    "RouteRule",
]