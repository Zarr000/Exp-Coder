"""
Training pipeline for Expera AI
"""

from .trainer import Trainer, TrainerConfig
from .optimizer import get_optimizer
from .scheduler import get_scheduler, CosineWarmupScheduler
from .loss import LanguageModelingLoss, CombinedLoss
from .checkpoint_manager import CheckpointManager
from .distributed_trainer import DistributedTrainer, DistributedConfig
from .evaluator import Evaluator, EvaluatorWithMetrics
from .metrics import MetricsCollector, TrainingMetrics, ValidationMetrics
from .callbacks import (
    Callback,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    LearningRateSchedulerCallback,
    CallbackList,
)

__all__ = [
    # Trainer
    "Trainer",
    "TrainerConfig",
    # Optimizer
    "get_optimizer",
    # Scheduler
    "get_scheduler",
    "CosineWarmupScheduler",
    # Loss
    "LanguageModelingLoss",
    "CombinedLoss",
    # Checkpointing
    "CheckpointManager",
    # Distributed
    "DistributedTrainer",
    "DistributedConfig",
    # Evaluation
    "Evaluator",
    "EvaluatorWithMetrics",
    # Metrics
    "MetricsCollector",
    "TrainingMetrics",
    "ValidationMetrics",
    # Callbacks
    "Callback",
    "EarlyStoppingCallback",
    "ModelCheckpointCallback",
    "LearningRateSchedulerCallback",
    "CallbackList",
]