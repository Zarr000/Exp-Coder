"""
Loss functions for Expera AI training

Implements:
1. Causal Language Modeling Loss (standard cross-entropy)
2. Optional auxiliary losses for memory and reasoning modules
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModelingLoss(nn.Module):
    """
    Causal Language Modeling Loss.
    
    Computes cross-entropy loss for autoregressive language modeling.
    The loss is computed only on the target tokens (shifted by 1 position).
    
    Mathematical Formulation:
        L = -∑ log P(x_t | x_1, ..., x_{t-1})
    
    where x_t is the target token at position t.
    """
    
    def __init__(
        self,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ):
        """
        Initialize language modeling loss.
        
        Args:
            label_smoothing: Label smoothing factor
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
        )
        
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute language modeling loss.
        
        Args:
            logits: Model output logits (batch_size, seq_len, vocab_size)
            labels: Target token IDs (batch_size, seq_len)
            
        Returns:
            Scalar loss value
        """
        # Shift logits and labels for next-token prediction
        # logits: [:, :-1, :] -> predict next token
        # labels: [:, 1:] -> target is next token
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Flatten for cross-entropy
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Compute loss
        loss = self.loss_fn(shift_logits, shift_labels)
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function with optional auxiliary losses.
    
    Supports:
    - Main language modeling loss
    - Memory regularization loss
    - Reasoning consistency loss
    - Auxiliary task losses
    """
    
    def __init__(
        self,
        lm_weight: float = 1.0,
        memory_weight: float = 0.1,
        reasoning_weight: float = 0.1,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ):
        """
        Initialize combined loss.
        
        Args:
            lm_weight: Weight for language modeling loss
            memory_weight: Weight for memory regularization
            reasoning_weight: Weight for reasoning consistency
            label_smoothing: Label smoothing factor
            ignore_index: Index to ignore
        """
        super().__init__()
        self.lm_weight = lm_weight
        self.memory_weight = memory_weight
        self.reasoning_weight = reasoning_weight
        
        self.lm_loss = LanguageModelingLoss(
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
        )
        
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        memory_loss: Optional[torch.Tensor] = None,
        reasoning_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            logits: Model output logits
            labels: Target token IDs
            memory_loss: Optional memory regularization loss
            reasoning_loss: Optional reasoning consistency loss
            
        Returns:
            Combined loss value
        """
        # Main language modeling loss
        loss = self.lm_weight * self.lm_loss(logits, labels)
        
        # Auxiliary losses
        if memory_loss is not None:
            loss = loss + self.memory_weight * memory_loss
            
        if reasoning_loss is not None:
            loss = loss + self.reasoning_weight * reasoning_loss
            
        return loss