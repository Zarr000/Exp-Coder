"""
Temperature-based dataset mixing for Expera AI.

Supports:
1. Static weights: fixed mixing proportions
2. Temperature sampling: weight_i^(1/T) for sharpness control
3. Curriculum schedules: shift weights over training steps
4. Dynamic loss-based reweighting: adjust based on per-domain loss
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random


@dataclass
class MixConfig:
    """Configuration for a single dataset in the mix."""
    name: str
    weight: float
    temperature: float = 1.0
    min_weight: float = 0.01
    max_weight: float = 0.9


class DatasetMixer:
    """
    Mixes multiple datasets with configurable strategies.
    
    Temperature scaling: weight_i = weight_i^(1/T)
    - T=1: proportional to original weights
    - T<1: sharpens distribution (favors high-weight datasets)
    - T>1: flattens distribution (more uniform)
    """

    def __init__(
        self,
        configs: List[MixConfig],
        global_temperature: float = 1.0,
        curriculum_schedule: Optional[List[Tuple[int, Dict[str, float]]]] = None,
        loss_reweight_alpha: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.configs = {c.name: c for c in configs}
        self.global_temperature = global_temperature
        self.curriculum_schedule = curriculum_schedule or []
        self.loss_reweight_alpha = loss_reweight_alpha
        self._rng = random.Random(seed)
        
        # Track per-dataset loss for dynamic reweighting
        self._loss_history: Dict[str, List[float]] = {c.name: [] for c in configs}
        self._current_weights: Dict[str, float] = {}
        self._update_weights(step=0)
    
    def _update_weights(self, step: int) -> None:
        """Update mixing weights based on current step."""
        weights = {}
        
        # Apply curriculum schedule
        for schedule_step, schedule_weights in self.curriculum_schedule:
            if step >= schedule_step:
                for name, w in schedule_weights.items():
                    if name in self.configs:
                        self.configs[name].weight = w
        
        # Compute temperature-scaled weights
        for name, config in self.configs.items():
            if config.temperature > 0:
                w = config.weight ** (1.0 / config.temperature)
            else:
                w = config.weight
            weights[name] = w
        
        # Apply global temperature
        if self.global_temperature != 1.0:
            for name in weights:
                weights[name] = weights[name] ** (1.0 / self.global_temperature)
        
        # Apply loss-based reweighting
        if self.loss_reweight_alpha > 0:
            for name in weights:
                if len(self._loss_history[name]) > 0:
                    avg_loss = sum(self._loss_history[name][-100:]) / min(100, len(self._loss_history[name]))
                    weights[name] *= math.exp(self.loss_reweight_alpha * avg_loss)
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            for name in weights:
                weights[name] = weights[name] / total
                # Clamp to min/max
                weights[name] = max(self.configs[name].min_weight,
                                   min(self.configs[name].max_weight, weights[name]))
        
        self._current_weights = weights
    
    def sample(self, step: Optional[int] = None) -> str:
        """
        Sample a dataset name based on current mixing weights.

        Args:
            step: Current training step (for curriculum updates)

        Returns:
            Dataset name
        """
        if step is not None:
            self._update_weights(step)

        # Filter out zero-weight datasets, keeping original ordering
        names = []
        weights = []
        for n, w in self._current_weights.items():
            if w > 0:
                names.append(n)
                weights.append(w)

        if not names:
            return list(self._current_weights.keys())[0]

        # Normalize
        total = sum(weights)
        weights = [w / total for w in weights]

        return self._rng.choices(names, weights=weights, k=1)[0]
    
    def record_loss(self, dataset_name: str, loss: float) -> None:
        """Record loss for dynamic reweighting."""
        if dataset_name in self._loss_history:
            self._loss_history[dataset_name].append(loss)
    
    def get_weights(self) -> Dict[str, float]:
        """Get current mixing weights."""
        return dict(self._current_weights)