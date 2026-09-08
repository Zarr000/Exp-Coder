"""Optimizer configuration for Expera AI."""

from typing import Dict, Any
import torch.optim as optim


def get_optimizer(
    model_parameters,
    config: Dict[str, Any],
) -> optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model_parameters: Model parameters to optimize
        config: Optimizer configuration dictionary
        
    Returns:
        PyTorch optimizer instance
    """
    optimizer_name = config.get("name", "adamw").lower()
    lr = config.get("learning_rate", 3e-4)
    weight_decay = config.get("weight_decay", 0.01)
    betas = config.get("betas", [0.9, 0.95])
    eps = config.get("eps", 1e-8)
    
    # Separate parameters for weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model_parameters:
        if param.ndim < 2 or "bias" in name or "layer_norm" in name or "layernorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    if optimizer_name == "adamw":
        optimizer = optim.AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
        )
    elif optimizer_name == "adam":
        optimizer = optim.Adam(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
        )
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            param_groups,
            lr=lr,
            momentum=config.get("momentum", 0.9),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer