"""Metrics computation for Expera AI."""

from typing import Dict, List, Optional
import torch


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss."""
    return float(torch.exp(torch.tensor(loss)))


def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """Compute token prediction accuracy."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    predictions = torch.argmax(shift_logits, dim=-1)
    mask = shift_labels != ignore_index
    
    correct = (predictions[mask] == shift_labels[mask]).sum().item()
    total = mask.sum().item()
    
    return correct / total if total > 0 else 0.0


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss: float,
) -> Dict[str, float]:
    """Compute comprehensive metrics."""
    return {
        "loss": loss,
        "perplexity": compute_perplexity(loss),
        "accuracy": compute_accuracy(logits, labels),
    }