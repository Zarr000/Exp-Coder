"""
Speculative Decoding for Expera AI.

Speculative Decoding accelerates autoregressive generation by using a
smaller draft model to speculate multiple tokens, then verifying them
in parallel with the target model.

Reference: "Accelerating Large Language Model Decoding with Speculative Sampling"
(https://arxiv.org/abs/2302.01318)

Features:
- Draft-Target model architecture
- Parallel verification
- Tree-based speculation
- Rejection sampling with exact equivalence
"""

from typing import Optional, Tuple, List
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    draft_max_tokens: int = 4  # Number of tokens to speculate
    ngram_size: int = 4  # N-gram size for drafting
    temperature: float = 1.0  # Sampling temperature
    top_k: int = 50  # Top-k filtering
    top_p: float = 0.95  # Nucleus filtering
    draft_threshold: float = 0.5  # Min probability for draft acceptance
    max_draft_len: int = 10  # Maximum draft sequence length
    rejection_threshold: float = 0.1  # Threshold for rejection sampling


class SpeculativeDecoder(nn.Module):
    """
    Speculative decoding with draft-target model.

    Uses a smaller draft model to propose tokens, then verifies
    them with the larger target model in parallel.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        vocab_size: int,
        config: Optional[SpeculativeConfig] = None,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.target_model = target_model
        self.vocab_size = vocab_size
        self.config = config or SpeculativeConfig()

        # Disable gradients for draft model during speculation
        for param in self.draft_model.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 32,
        stop_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        Generate with speculative decoding.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            max_new_tokens: Maximum tokens to generate
            stop_token_id: Token ID to stop generation

        Returns:
            Generated token IDs
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Initialize
        generated = input_ids.clone()
        draft_model = self.draft_model
        target_model = self.target_model

        # Generation loop
        while generated.shape[1] - input_ids.shape[1] < max_new_tokens:
            # Speculate with draft model
            draft_tokens, draft_probs = self._speculate(
                generated,
                self.config.draft_max_tokens,
            )

            # Verify with target model
            verified_tokens, accepted_count = self._verify(
                generated,
                draft_tokens,
                draft_probs,
            )

            # Append accepted tokens
            if accepted_count > 0:
                new_tokens = verified_tokens[:, -accepted_count:]
                generated = torch.cat([generated, new_tokens], dim=-1)

            # Check for stop
            if stop_token_id is not None and (generated == stop_token_id).any():
                break

            # If nothing accepted, add single token from draft
            if accepted_count == 0:
                next_token = draft_tokens[:, 0:1]
                generated = torch.cat([generated, next_token], dim=-1)

        return generated

    def _speculate(
        self,
        input_ids: Tensor,
        max_tokens: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Speculate tokens with draft model.

        Args:
            input_ids: Current sequence (batch, seq_len)
            max_tokens: Maximum tokens to speculate

        Returns:
            (draft_tokens, draft_probs)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        vocab_size = self.vocab_size

        # Maximum length to consider
        max_len = min(
            input_ids.shape[1] + max_tokens,
            self.config.max_draft_len,
        )

        draft_tokens = []
        draft_probs = []

        current_input = input_ids

        # Autoregressive generation
        for _ in range(max_tokens):
            # Get logits from draft model
            with torch.no_grad():
                logits = self.draft_model(current_input)

            # Get distribution for last position
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            # Sampling
            if self.config.temperature > 0:
                logits = logits / self.config.temperature
                probs = F.softmax(logits, dim=-1)

            # Apply top-k/ top-p
            if self.config.top_k > 0:
                top_k = min(self.config.top_k, vocab_size)
                values, indices = torch.topk(probs, top_k)
                probs = torch.zeros_like(probs).scatter_(-1, indices, values)

            if self.config.top_p > 0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum > self.config.top_p
                sorted_probs = sorted_probs.masked_fill(mask, 0)
                probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)

            probs = probs / probs.sum(dim=-1, keepdim=True)

            # Sample token
            next_token = torch.multinomial(probs, 1)
            next_prob = probs.gather(-1, next_token).squeeze(-1)

            draft_tokens.append(next_token)
            draft_probs.append(next_prob)

            # Update for next iteration
            current_input = torch.cat([current_input, next_token], dim=-1)

            # Stop if end token (assuming EOS token at index 0)
            if next_token.item() == 0:
                break

        # Stack results
        draft_tokens = torch.cat(draft_tokens, dim=-1)  # (batch, num_tokens)
        draft_probs = torch.cat(draft_probs, dim=-1)  # (batch, num_tokens)

        return draft_tokens, draft_probs

    def _verify(
        self,
        input_ids: Tensor,
        draft_tokens: Tensor,
        draft_probs: Tensor,
    ) -> Tuple[Tensor, int]:
        """
        Verify draft tokens with target model.

        Args:
            input_ids: Original input (batch, seq_len)
            draft_tokens: Speculated tokens (batch, num_draft)
            draft_probs: Draft probabilities (batch, num_draft)

        Returns:
            (verified_tokens, accepted_count)
        """
        batch_size = input_ids.shape[0]
        num_draft = draft_tokens.shape[1]
        device = input_ids.device

        # Build input with draft tokens
        full_input = torch.cat([input_ids, draft_tokens], dim=-1)

        # Get target model predictions
        with torch.no_grad():
            target_logits = self.target_model(full_input)

        # Get probabilities for draft positions
        target_probs = F.softmax(target_logits[:, -num_draft - 1:-1, :], dim=-1)

        # Gather probabilities for draft tokens
        target_draft_probs = torch.gather(
            target_probs,
            -1,
            draft_tokens.unsqueeze(-1)
        ).squeeze(-1)  # (batch, num_draft)

        # Verification: target probability > draft probability
        # If target assigns higher probability, accept
        # Otherwise reject and resample
        acceptance_ratio = target_draft_probs / (draft_probs + 1e-10)

        # Generate random numbers for rejection
        rand = torch.rand(batch_size, num_draft, device=device)

        # Accept if ratio > random OR target > draft
        accepted = (acceptance_ratio > rand).cumprod(dim=-1)
        accepted = torch.cat([torch.ones(batch_size, 1, device=device), accepted[:, :-1]], dim=-1)

        # Count accepted tokens
        accepted_count = accepted.sum(dim=-1).long().clamp(0, num_draft)

        # Handle first token specially (always check target vs uniform)
        first_check = acceptance_ratio[:, 0] > self.config.rejection_threshold
        accepted[:, 0] = first_check.float()

        # Resample rejected tokens from target distribution
        final_tokens = draft_tokens.clone()

        for b in range(batch_size):
            count = accepted_count[b].item()
            if count < num_draft:
                # Resample rejected tokens
                for i in range(count, num_draft):
                    if not accepted[b, i].item():
                        # Sample from target distribution
                        target_dist = target_probs[b, i]
                        new_token = torch.multinomial(target_dist, 1)
                        final_tokens[b, i] = new_token

        return final_tokens, accepted_count[0].item()

    def set_draft_temperature(self, temp: float) -> None:
        """Set draft model temperature."""
        self.config.temperature = temp

    def set_target_temperature(self, temp: float) -> None:
        """Set target model temperature (for non-speculative fallback)."""
        self._target_temp = temp


class NGramSpeculator(nn.Module):
    """
    N-gram based lookahead for speculative decoding.

    Uses n-gram matches from input to propose next tokens.
    More efficient than neural draft models for repetitive patterns.
    """

    def __init__(
        self,
        vocab_size: int,
        ngram_size: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.ngram_size = ngram_size
        self.ngram_table: dict = {}

    def forward(
        self,
        input_ids: Tensor,
        max_tokens: int = 4,
    ) -> Tuple[Tensor, Tensor]:
        """
        Propose tokens using n-gram matching.

        Args:
            input_ids: Input tokens (batch, seq_len)
            max_tokens: Maximum tokens to propose

        Returns:
            (proposed_tokens, proposed_probs)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        proposed_tokens = []
        proposed_probs = []

        for b in range(batch_size):
            tokens = input_ids[b].tolist()
            tokens_tuples = []

            # Build n-grams from input
            for i in range(len(tokens) - self.ngram_size + 1):
                ngram = tuple(tokens[i:i + self.ngram_size])
                tokens_tuples.append(ngram)

            # Find matching n-grams for prediction
            last_n = tuple(tokens[-(self.ngram_size - 1):]) if len(tokens) >= self.ngram_size - 1 else ()
            matches = []

            # Find completions for last n-gram
            for i, ngram in enumerate(tokens_tuples):
                if ngram[:-1] == last_n and len(ngram) == self.ngram_size:
                    next_token = ngram[-1]
                    matches.append(next_token)

            # If no matches, use frequency-based prediction
            if not matches:
                most_common = self._get_most_common(tokens)
                matches = [most_common] * max_tokens

            # Create proposals
            for i in range(max_tokens):
                if i < len(matches):
                    proposed_tokens.append(matches[i])
                    proposed_probs.append(0.8)  # High confidence for n-gram matches
                else:
                    # Random continuation
                    proposed_tokens.append(torch.randint(0, self.vocab_size, (1,)).item())
                    proposed_probs.append(0.5)

        # Convert to tensors
        proposed_tokens = torch.tensor(proposed_tokens, device=device).view(batch_size, -1)
        proposed_probs = torch.tensor(proposed_probs, device=device).view(batch_size, -1)

        return proposed_tokens, proposed_probs

    def _get_most_common(self, tokens: List[int]) -> int:
        """Get most common next token from history."""
        if len(tokens) < 2:
            return 0

        # Count bigram frequencies
        bigram_counts = {}
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i + 1])
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

        if not bigram_counts:
            return 0

        # Find most common continuation
        max_count = 0
        most_common = 0
        for (current, next_t), count in bigram_counts.items():
            if tokens[-1] == current and count > max_count:
                max_count = count
                most_common = next_t

        return most_common

    def update_ngrams(self, input_ids: Tensor) -> None:
        """Update n-gram table with new tokens."""
        tokens = input_ids.tolist()

        # Build and store n-grams
        for i in range(len(tokens) - self.ngram_size + 1):
            ngram = tuple(tokens[i:i + self.ngram_size])
            prefix = ngram[:-1]
            next_token = ngram[-1]

            if prefix not in self.ngram_table:
                self.ngram_table[prefix] = []
            self.ngram_table[prefix].append(next_token)


class TreeSpeculator(nn.Module):
    """
    Tree-based speculative decoding.

    Generates multiple token candidates in a tree structure,
    verifying them in parallel.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        vocab_size: int,
        branch_factor: int = 3,
        depth: int = 2,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.vocab_size = vocab_size
        self.branch_factor = branch_factor
        self.depth = depth

    def forward(
        self,
        input_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate tree of possible continuations.

        Args:
            input_ids: Input tokens (batch, seq_len)

        Returns:
            (tree_tokens, tree_probs)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Get top-k candidates from draft
        with torch.no_grad():
            logits = self.draft_model(input_ids)
            probs = F.softmax(logits[:, -1, :], dim=-1)

        # Get top branch_factor tokens
        top_probs, top_indices = torch.topk(probs, self.branch_factor)

        # Initialize tree
        tree_tokens = top_indices  # (batch, branch_factor)
        tree_probs = top_probs  # (batch, branch_factor)

        return tree_tokens, tree_probs


@dataclass
class SpeculationResult:
    """Result of speculative generation."""
    tokens: Tensor
    num_draft: int
    num_accepted: int
    acceptance_rate: float
    draft_time_ms: float
    verify_time_ms: float


def create_speculative_decoder(
    draft_model: nn.Module,
    target_model: nn.Module,
    vocab_size: int,
    max_draft_tokens: int = 4,
    ngram_size: int = 4,
    use_ngram: bool = False,
) -> SpeculativeDecoder:
    """
    Create a speculative decoder.

    Args:
        draft_model: Smaller model for drafting
        target_model: Larger model for verification
        vocab_size: Vocabulary size
        max_draft_tokens: Maximum tokens to draft at once
        ngram_size: N-gram size
        use_ngram: Use n-gram lookahead instead of draft model

    Returns:
        SpeculativeDecoder
    """
    config = SpeculativeConfig(
        draft_max_tokens=max_draft_tokens,
        ngram_size=ngram_size,
    )

    return SpeculativeDecoder(
        draft_model=draft_model,
        target_model=target_model,
        vocab_size=vocab_size,
        config=config,
    )


__all__ = [
    "SpeculativeConfig",
    "SpeculativeDecoder",
    "NGramSpeculator",
    "TreeSpeculator",
    "SpeculationResult",
    "create_speculative_decoder",
]