"""
LoRA Fine-tuning for Expera AI.

Implements:
- LoRA layers (lora_a, lora_b)
- Merge and save
- Load adapters
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """
    LoRA Configuration.

    Usage:
        config = LoRAConfig(
            r=16,  # rank
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
    """
    r: int = 16  # Rank
    lora_alpha: int = 32  # Alpha scaling
    lora_dropout: float = 0.05  # Dropout
    target_modules: Optional[List[str]] = None  # type: ignore  # Modules to apply
    bias: str = "none"  # none, all, lora_only
    modules_to_save: Optional[List[str]] = None  # Full finetune modules


class LoRALayer(nn.Module):
    """
    LoRA layer implementation.

    Adds low-rank adapters to linear layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        bias: Optional[Tensor] = None,
    ):
        """Initialize LoRA layer."""
        super().__init__()

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout)

        # Original layer (frozen)
        self.base_layer = nn.Linear(in_features, out_features, bias=bias is not None)
        self.base_layer.weight.requires_grad = False
        if bias is not None:
            self.base_layer.bias = bias
            self.base_layer.bias.requires_grad = False

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize LoRA weights."""
        nn.init.zeros_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        # Original output
        base_output = self.base_layer(x)

        # LoRA output
        lora_input = self.lora_dropout(x)
        lora_output = F.linear(
            F.linear(lora_input, self.lora_A) @ self.lora_B.T,
            self.base_layer.weight,
        )
        lora_output = lora_output * self.scaling

        return base_output + lora_output

    def merge(self):
        """Merge LoRA weights into base layer."""
        with torch.no_grad():
            # Compute merged weight
            merged_weight = self.base_layer.weight + self.lora_B @ self.lora_A * self.scaling
            self.base_layer.weight.copy_(merged_weight)

            # Reset LoRA
            nn.init.zeros_(self.lora_A)
            nn.init.zeros_(self.lora_B)

    def extra_repr(self) -> str:
        return f"r={self.r}, lora_alpha={self.lora_alpha}"


class LoRAWrapper(nn.Module):
    """
    Wraps module with LoRA adapters.

    Usage:
        wrapper = LoRAWrapper(model, config)
        # Train only LoRA parameters
        output = wrapper(input)
    """

    def __init__(
        self,
        module: nn.Module,
        config: LoRAConfig,
    ):
        """Initialize LoRA wrapper."""
        super().__init__()

        self.module = module
        self.config = config

        # Add LoRA to target modules
        self.lora_layers: Dict[str, LoRALayer] = {}
        self._apply_lora(module, config)

    def _apply_lora(self, module: nn.Module, config: LoRAConfig):
        """Apply LoRA to target modules."""
        target_modules = config.target_modules or ["q_proj", "v_proj"]

        for name, child in module.named_children():
            # Check if this is a target module
            if any(tm in name for tm in target_modules):
                if isinstance(child, nn.Linear):
                    # Wrap with LoRA
                    lora_layer = LoRALayer(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        r=config.r,
                        lora_alpha=config.lora_alpha,
                        lora_dropout=config.lora_dropout,
                        bias=child.bias,
                    )
                    self.lora_layers[name] = lora_layer

                    # Replace in parent
                    setattr(module, name, lora_layer)

            # Recurse
            self._apply_lora(child, config)

    def forward(self, *args, **kwargs):
        """Forward pass."""
        return self.module(*args, **kwargs)

    def merge(self):
        """Merge all LoRA layers."""
        for lora_layer in self.lora_layers.values():
            lora_layer.merge()


class LoRAManager:
    """
    Manages LoRA fine-tuning.

    Usage:
        manager = LoRAManager(model, config)
        manager.apply_lora()

        # Train
        loss.backward()

        # Merge and save
        manager.merge()
        manager.save_adapters("adapters.pt")
    """

    def __init__(
        self,
        model: nn.Module,
        config: LoRAConfig,
    ):
        """Initialize manager."""
        self.model = model
        self.config = config
        self.applied = False

    def apply_lora(self) -> nn.Module:
        """Apply LoRA to model."""
        if self.applied:
            logger.warning("LoRA already applied")
            return self.model

        self.model = LoRAWrapper(self.model, self.config)
        self.applied = True

        logger.info(f"Applied LoRA (r={self.config.r})")
        return self.model

    def merge(self):
        """Merge LoRA into base model."""
        for module in self.model.modules():
            if isinstance(module, LoRALayer):
                module.merge()

        logger.info("Merged LoRA weights")

    def save_adapters(self, path: str):
        """Save LoRA adapters."""
        state_dict = {}
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALayer):
                state_dict[f"{name}.lora_A"] = module.lora_A.data
                state_dict[f"{name}.lora_B"] = module.lora_B.data

        torch.save(state_dict, path)
        logger.info(f"Saved adapters to {path}")

    def load_adapters(self, path: str):
        """Load LoRA adapters."""
        state_dict = torch.load(path, map_location="cpu")

        for key, value in state_dict.items():
            # Parse name
            name = key.rsplit(".", 1)[0]
            param_name = key.rsplit(".", 1)[1]

            # Find module
            module = self.model.get_submodule(name)
            if hasattr(module, param_name):
                getattr(module, param_name).data = value

        logger.info(f"Loaded adapters from {path}")

    def get_trainable_params(self) -> Dict[str, Any]:
        """Get trainable parameters."""
        params = []
        for module in self.model.modules():
            if isinstance(module, LoRALayer):
                params.extend([module.lora_A, module.lora_B])

        return {"lora_params": params}


# Convenience functions


def apply_lora(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Apply LoRA to model.

    Args:
        model: Base model
        r: Rank
        lora_alpha: Alpha scaling
        lora_dropout: Dropout
        target_modules: Target module names

    Returns:
        Model with LoRA
    """
    config = LoRAConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    manager = LoRAManager(model, config)
    return manager.apply_lora()


def merge_lora(model: nn.Module):
    """Merge LoRA into model."""
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.merge()


__all__ = [
    "LoRAConfig",
    "LoRALayer",
    "LoRAWrapper",
    "LoRAManager",
    "apply_lora",
    "merge_lora",
]