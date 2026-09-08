"""Learning rate scheduler for Expera AI."""

import math
from typing import Dict, Any, Optional
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    """
    Cosine learning rate schedule with linear warmup.
    
    Schedule:
        For step < warmup_steps:
            lr = max_lr * (step / warmup_steps)
        For step >= warmup_steps:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int = 2000,
        total_steps: int = 100000,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            scale = step / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (
                1 + math.cos(math.pi * progress)
            )
            return [base_lr * scale for base_lr in self.base_lrs]


def get_scheduler(
    optimizer,
    config: Dict[str, Any],
) -> _LRScheduler:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: PyTorch optimizer
        config: Scheduler configuration dictionary
        
    Returns:
        PyTorch learning rate scheduler
    """
    scheduler_name = config.get("name", "cosine_with_warmup").lower()
    warmup_steps = config.get("warmup_steps", 2000)
    total_steps = config.get("num_training_steps", 100000)
    min_lr_ratio = config.get("min_lr_ratio", 0.1)
    
    if scheduler_name == "cosine_with_warmup":
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        )
    elif scheduler_name == "linear":
        from torch.optim.lr_scheduler import LinearLR
        scheduler = LinearLR(optimizer, total_iters=total_steps)
    elif scheduler_name == "constant":
        from torch.optim.lr_scheduler import ConstantLR
        scheduler = ConstantLR(optimizer)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler