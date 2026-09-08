"""
Evaluation for Expera AI pretraining.

Provides:
- Validation loop
- Test set evaluation
- Perplexity computation
- Zero-shot evaluation
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from .loss import LanguageModelingLoss


class Evaluator:
    """
    Evaluation for pretraining.

    Features:
    - Validation loop
    - Test set evaluation
    - Perplexity computation
    - Zero-shot evaluation
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
        loss_fn: Optional[nn.Module] = None,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device(device)
        self.loss_fn = loss_fn or LanguageModelingLoss()
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype

    def evaluate(
        self,
        num_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation.

        Args:
            num_batches: Number of batches to evaluate (None for all)

        Returns:
            Dictionary with metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0
        num_batches_evaluated = 0

        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if num_batches and i >= num_batches:
                    break

                # Move batch to device
                if isinstance(batch, dict):
                    batch = {
                        k: v.to(self.device) if torch.is_tensor(v) else v
                        for k, v in batch.items()
                    }
                else:
                    batch = [
                        b.to(self.device) if torch.is_tensor(b) else b
                        for b in batch
                    ]

                # Forward
                with autocast(
                    enabled=self.use_amp,
                    dtype=self.amp_dtype,
                ):
                    if isinstance(batch, dict):
                        inputs = batch.get("input_ids", batch.get("inputs"))
                        labels = batch.get("labels", inputs)
                    else:
                        inputs, labels = batch[0], batch[1]

                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels)

                # Compute metrics
                batch_size = inputs.shape[0]
                seq_len = inputs.shape[1]
                tokens = batch_size * seq_len

                total_loss += loss.item() * tokens
                total_tokens += tokens
                num_batches_evaluated += 1

        self.model.train()

        # Compute average
        avg_loss = total_loss / max(1, total_tokens)
        perplexity = self.compute_perplexity(avg_loss)

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "num_batches": num_batches_evaluated,
            "num_tokens": total_tokens,
        }

    def compute_perplexity(self, loss: float) -> float:
        """
        Compute perplexity from loss.

        Args:
            loss: Cross-entropy loss

        Returns:
            Perplexity
        """
        return torch.exp(torch.tensor(loss)).item()

    def evaluate_code(
        self,
        code_samples: List[str],
        tokenizer: Any,
    ) -> Dict[str, float]:
        """
        Evaluate on code samples.

        Args:
            code_samples: List of code strings
            tokenizer: Tokenizer

        Returns:
            Code metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for code in code_samples:
                # Tokenize
                tokens = tokenizer.encode(code, add_special_tokens=True)
                input_ids = torch.tensor([tokens]).to(self.device)

                # Forward
                outputs = self.model(input_ids)
                loss = self.loss_fn(outputs, input_ids)

                total_loss += loss.item() * (input_ids.shape[1] - 1)
                total_tokens += input_ids.shape[1] - 1

        self.model.train()

        avg_loss = total_loss / max(1, total_tokens)
        perplexity = self.compute_perplexity(avg_loss)

        return {
            "code_loss": avg_loss,
            "code_perplexity": perplexity,
            "num_samples": len(code_samples),
        }

    def evaluate_zero_shot(
        self,
        tasks: List[str],
        few_shot_examples: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, float]:
        """
        Run zero-shot evaluation on tasks.

        Args:
            tasks: List of task names
            few_shot_examples: Few-shot examples for each task

        Returns:
            Zero-shot metrics
        """
        # Placeholder - real implementation would use lm-evaluation-harness
        return {
            "tasks_evaluated": len(tasks),
            "average_score": 0.0,
        }


class EvaluatorWithMetrics:
    """
    Extended evaluator with additional metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device(device)
        self.loss_fn = LanguageModelingLoss()

    def evaluate_detailed(
        self,
        num_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Detailed evaluation with extensive metrics.

        Args:
            num_batches: Number of batches

        Returns:
            Detailed metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches_evaluated = 0

        all_losses: List[float] = []

        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if num_batches and i >= num_batches:
                    break

                # Move batch to device
                if isinstance(batch, dict):
                    batch = {
                        k: v.to(self.device) if torch.is_tensor(v) else v
                        for k, v in batch.items()
                    }
                else:
                    batch = [
                        b.to(self.device) if torch.is_tensor(b) else b
                        for b in batch
                    ]

                # Get inputs and labels
                if isinstance(batch, dict):
                    inputs = batch.get("input_ids", batch.get("inputs"))
                    labels = batch.get("labels", inputs)
                else:
                    inputs, labels = batch[0], batch[1]

                # Forward
                outputs = self.model(inputs)

                # Get predictions
                logits = outputs[:, :-1, :]
                targets = labels[:, 1:]

                # Compute loss
                loss = self.loss_fn(outputs, labels)

                # Compute accuracy
                predictions = logits.argmax(dim=-1)
                correct = (predictions == targets).float()
                mask = (targets != self.loss_fn.loss_fn.ignore_index).float()

                batch_correct = (correct * mask).sum().item()
                batch_tokens = mask.sum().item()

                total_loss += loss.item()
                total_correct += batch_correct
                total_tokens += batch_tokens
                num_batches_evaluated += 1

                all_losses.append(loss.item())

        self.model.train()

        # Compute metrics
        avg_loss = total_loss / max(1, num_batches_evaluated)
        perplexity = self.compute_perplexity(avg_loss)
        accuracy = total_correct / max(1, total_tokens)

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "accuracy": accuracy,
            "num_batches": num_batches_evaluated,
            "num_tokens": total_tokens,
            "loss_std": self._compute_std(all_losses),
        }

    def compute_perplexity(self, loss: float) -> float:
        """Compute perplexity from loss."""
        return torch.exp(torch.tensor(loss)).item()

    def _compute_std(self, values: List[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5