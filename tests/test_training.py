"""
Tests for Training Pipeline.

Tests:
- Trainer
- CheckpointManager
- DistributedTrainer
- Evaluator
- MetricsCollector
- Callbacks
- Benchmarks
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, vocab_size: int = 32000, hidden_size: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        hidden = self.transformer(hidden)
        logits = self.lm_head(hidden)
        return logits


class TestTrainer:
    """Test Trainer."""

    def test_trainer_config(self):
        """Test TrainerConfig creation."""
        from src.training import TrainerConfig

        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        config = TrainerConfig(
            model=model,
            train_dataloader=[],
            optimizer=optimizer,
            max_steps=10,
            log_every=5,
        )

        assert config.max_steps == 10
        assert config.log_every == 5
        assert config.use_amp == True

    def test_trainer_initialization(self):
        """Test Trainer initialization."""
        from src.training import Trainer, TrainerConfig

        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        config = TrainerConfig(
            model=model,
            train_dataloader=[],
            optimizer=optimizer,
            max_steps=10,
        )

        trainer = Trainer(config)

        assert trainer.global_step == 0
        assert trainer.epoch == 0
        assert trainer.model is not None

    def test_trainer_with_device(self):
        """Test Trainer with different device."""
        from src.training import Trainer, TrainerConfig

        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        config = TrainerConfig(
            model=model,
            train_dataloader=[],
            optimizer=optimizer,
            device="cpu",
            max_steps=10,
        )

        trainer = Trainer(config)

        assert trainer.device.type == "cpu"


class TestCheckpointManager:
    """Test CheckpointManager."""

    def test_checkpoint_manager_init(self):
        """Test CheckpointManager initialization."""
        from src.training import CheckpointManager

        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        manager = CheckpointManager(
            model=model,
            optimizer=optimizer,
            save_dir="test_checkpoints",
            keep_last_n=2,
        )

        assert manager.save_dir.name == "test_checkpoints"
        assert manager.keep_last_n == 2

    def test_save_and_cleanup(self):
        """Test checkpoint save and cleanup."""
        from src.training import CheckpointManager

        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        manager = CheckpointManager(
            model=model,
            optimizer=optimizer,
            save_dir="test_checkpoints",
            keep_last_n=2,
        )

        # Save some checkpoints
        manager.save(step=100, metrics={"loss": 1.0})
        manager.save(step=200, metrics={"loss": 0.8})
        manager.save(step=300, metrics={"loss": 0.7})

        # Check latest step
        assert manager.get_latest_step() == 300

        # Check cleanup
        manager.cleanup()


class TestEvaluator:
    """Test Evaluator."""

    def test_evaluator_init(self):
        """Test Evaluator initialization."""
        from src.training import Evaluator

        model = SimpleModel()
        dataloader = []

        evaluator = Evaluator(
            model=model,
            dataloader=dataloader,
            device="cpu",
        )

        assert evaluator.model is not None

    def test_compute_perplexity(self):
        """Test perplexity computation."""
        from src.training import Evaluator

        model = SimpleModel()
        evaluator = Evaluator(model=model, dataloader=[])

        # Test with loss = 1.0
        perplexity = evaluator.compute_perplexity(1.0)
        assert perplexity > 0
        assert perplexity < 100


class TestMetricsCollector:
    """Test MetricsCollector."""

    def test_metrics_collector_init(self):
        """Test MetricsCollector initialization."""
        from src.training import MetricsCollector, TrainingMetrics

        collector = MetricsCollector(window_size=100)

        assert collector.window_size == 100
        assert len(collector.train_metrics) == 0

    def test_record_metrics(self):
        """Test recording metrics."""
        from src.training import MetricsCollector, TrainingMetrics

        collector = MetricsCollector()

        metrics = TrainingMetrics(
            step=1,
            loss=1.0,
            lr=1e-4,
            tokens_per_sec=1000.0,
            batch_time=0.1,
        )

        collector.record(metrics)

        assert len(collector.train_metrics) == 1

    def test_get_average(self):
        """Test getting averages."""
        from src.training import MetricsCollector, TrainingMetrics

        collector = MetricsCollector()

        for i in range(10):
            metrics = TrainingMetrics(
                step=i,
                loss=1.0 - i * 0.1,
                lr=1e-4,
                tokens_per_sec=1000.0,
                batch_time=0.1,
            )
            collector.record(metrics)

        avg = collector.get_average(window=5)

        assert "loss" in avg
        assert avg["loss"] > 0


class TestCallbacks:
    """Test Callbacks."""

    def test_early_stopping_init(self):
        """Test EarlyStoppingCallback initialization."""
        from src.training import EarlyStoppingCallback

        callback = EarlyStoppingCallback(
            patience=3,
            min_delta=0.01,
        )

        assert callback.patience == 3
        assert callback.min_delta == 0.01

    def test_early_stopping_check(self):
        """Test early stopping check."""
        from src.training import EarlyStoppingCallback

        callback = EarlyStoppingCallback(patience=2)

        # First validation
        callback.on_val_end(step=100, metrics={"val_loss": 1.0})
        assert not callback.should_stop

        # No improvement
        callback.on_val_end(step=200, metrics={"val_loss": 1.0})
        assert not callback.should_stop

        # Should trigger
        callback.on_val_end(step=300, metrics={"val_loss": 1.0})
        assert callback.should_stop

    def test_model_checkpoint_init(self):
        """Test ModelCheckpointCallback initialization."""
        from src.training import ModelCheckpointCallback

        callback = ModelCheckpointCallback(
            save_dir="test_checkpoints",
            monitor="val_loss",
        )

        assert callback.monitor == "val_loss"

    def test_callback_list(self):
        """Test CallbackList."""
        from src.training import Callback, CallbackList

        class TestCallback(Callback):
            def on_step_end(self, step, metrics):
                pass

        callbacks = CallbackList()
        callbacks.add(TestCallback())

        assert len(callbacks.callbacks) == 1


class TestDistributedTrainer:
    """Test DistributedTrainer."""

    def test_distributed_config(self):
        """Test DistributedConfig."""
        from src.training import DistributedConfig

        config = DistributedConfig(
            strategy="ddp",
            world_size=1,
        )

        assert config.strategy == "ddp"
        assert config.world_size == 1


class TestBenchmarks:
    """Test Benchmarks."""

    def test_benchmark_result_init(self):
        """Test BenchmarkResult."""
        from src.training.benchmarks import BenchmarkResult

        result = BenchmarkResult(
            name="test",
            batch_size=8,
            seq_len=1024,
            tokens_per_sec=10000.0,
        )

        assert result.name == "test"
        assert result.batch_size == 8

    def test_training_benchmark_init(self):
        """Test TrainingBenchmark initialization."""
        from src.training.benchmarks import TrainingBenchmark

        model = SimpleModel()
        benchmark = TrainingBenchmark(model=model)

        assert benchmark.model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])