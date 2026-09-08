"""
Expera AI - Complete Model Architecture

This is the main model class that combines all components:
1. Token embeddings with RoPE support
2. Stack of transformer blocks
3. Final layer norm
4. Language modeling head (output projection + softmax)

Architecture:
    Input IDs → Embeddings → N×TransformerBlocks → LayerNorm → LM Head → Logits

The model supports:
- Autoregressive text generation
- KV caching for efficient generation
- Gradient checkpointing for memory efficiency
- Multiple model sizes (tiny, small, base, large, xlarge)
"""

import math
from typing import Optional, Tuple, List, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import EmbeddingModule
from .transformer_block import TransformerBlock


class ExperaModel(nn.Module):
    """
    Expera AI - Main model class.
    
    A decoder-only transformer model with:
    - Pre-LayerNorm architecture
    - Rotary Position Embeddings (RoPE)
    - Grouped Query Attention (GQA)
    - SwiGLU/GELU feed-forward networks
    - KV caching for efficient generation
    - Memory bank integration (optional)
    - Reasoning module integration (optional)
    """
    
    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        max_position_embeddings: int = 2048,
        activation: str = "gelu",
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        layer_norm_eps: float = 1e-5,
        padding_idx: int = None,
        std: float = 0.02,
    ):
        """
        Initialize Expera AI model.
        
        Args:
            vocab_size: Vocabulary size
            hidden_size: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            num_kv_heads: Number of key-value heads (GQA)
            head_dim: Dimension per head
            intermediate_size: FFN intermediate dimension
            max_position_embeddings: Maximum sequence length
            activation: FFN activation function
            dropout: Dropout probability
            attention_dropout: Attention dropout
            use_rope: Whether to use RoPE
            rope_theta: RoPE theta parameter
            layer_norm_eps: Layer norm epsilon
            padding_idx: Padding token ID
            std: Weight initialization standard deviation
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        
        # Default intermediate size (4 * hidden_size)
        if intermediate_size is None:
            if activation == "swiglu":
                # SwiGLU uses ~8/3 * hidden_size (since it has 3 projections)
                intermediate_size = int(8 * hidden_size / 3)
            else:
                intermediate_size = 4 * hidden_size
                
        self.intermediate_size = intermediate_size
        
        # Embedding module
        self.embed_tokens = EmbeddingModule(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            padding_idx=padding_idx,
            dropout=dropout,
            std=std,
            rope_theta=rope_theta,
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                activation=activation,
                dropout=dropout,
                attention_dropout=attention_dropout,
                use_rope=use_rope,
                rope_theta=rope_theta,
                max_position_embeddings=max_position_embeddings,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Language modeling head (output projection)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights between embedding and LM head (like GPT-2)
        self.lm_head.weight = self.embed_tokens.token_embedding.embedding.weight
        
        # Initialize weights
        self._init_weights(std)
        
    def _init_weights(self, std: float) -> None:
        """
        Initialize model weights.
        
        Args:
            std: Standard deviation for initialization
        """
        # Initialize all nn.Linear and nn.Embedding weights
        self.apply(lambda module: self._init_module_weights(module, std))
        
    def _init_module_weights(self, module: nn.Module, std: float) -> None:
        """
        Initialize weights for a single module.
        
        Args:
            module: Module to initialize
            std: Standard deviation
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
                
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            position_ids: Position IDs (batch_size, seq_len)
            past_key_values: List of K,V caches for each layer
            use_cache: Whether to return K,V caches
            return_dict: Whether to return dict or tuple
            
        Returns:
            Dictionary or tuple of (logits, past_key_values, hidden_states)
        """
        batch_size, seq_len = input_ids.shape
        
        # Generate position IDs if not provided
        if position_ids is None:
            past_len = 0
            if past_key_values is not None:
                past_len = past_key_values[0][0].size(2)
            position_ids = torch.arange(
                past_len, past_len + seq_len,
                dtype=torch.long,
                device=input_ids.device,
            ).unsqueeze(0).expand(batch_size, -1)
            
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids, position_ids)
        
        # Initialize past key values if None
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
            
        # Create causal attention mask
        if attention_mask is None:
            # Create causal mask
            total_len = hidden_states.size(1)
            if past_key_values[0] is not None:
                total_len += past_key_values[0][0].size(2)
            attention_mask = self._create_causal_mask(
                seq_len=hidden_states.size(1),
                total_len=total_len,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            
        # Store all hidden states if needed
        all_hidden_states = [hidden_states] if not return_dict else None
        
        # Pass through transformer layers
        new_past_key_values = []
        for i, layer in enumerate(self.layers):
            hidden_states, present_kv = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[i],
                use_cache=use_cache,
            )
            new_past_key_values.append(present_kv)
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
                
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        if return_dict:
            return {
                "logits": logits,
                "past_key_values": new_past_key_values if use_cache else None,
                "hidden_states": all_hidden_states,
            }
        else:
            return (logits, new_past_key_values if use_cache else None)
    
    def _create_causal_mask(
        self,
        seq_len: int,
        total_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create causal attention mask for autoregressive generation.

        Query row ``i`` (global position ``past + i``) may attend every
        column ``j <= past + i``.

        Args:
            seq_len: Current sequence length (query length)
            total_len: Total sequence length (query + past)
            dtype: Data type
            device: Device

        Returns:
            Causal mask (1, 1, seq_len, total_len) with 0.0 (attend) and
            -inf (masked).
        """
        past_len = total_len - seq_len
        rows = torch.arange(seq_len, device=device).unsqueeze(1)
        cols = torch.arange(total_len, device=device).unsqueeze(0)
        mask = torch.where(
            cols <= rows + past_len,
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), float("-inf"), dtype=dtype, device=device),
        )
        return mask.unsqueeze(0).unsqueeze(0)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to sample or use greedy decoding
            eos_token_id: End-of-sequence token ID
            
        Returns:
            Generated token IDs (batch_size, seq_len + generated)
        """
        for _ in range(max_new_tokens):
            # Forward pass with cache
            outputs = self.forward(
                input_ids=input_ids[:, -1:] if input_ids.size(1) > 1 else input_ids,
                use_cache=True,
                past_key_values=outputs.get("past_key_values") if _ > 0 else None,
                return_dict=True,
            ) if _ > 0 or input_ids.size(1) > 1 else self.forward(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
            )
            
            # Get logits for last token
            next_token_logits = outputs["logits"][:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(input_ids.size(0)):
                    for token_id in set(input_ids[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty
                        
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(
                    next_token_logits, top_k
                )[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
                
            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
                
            # Sample next token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Stop if EOS token is generated
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
                
        return input_ids
    
    def get_num_params(self) -> Dict[str, int]:
        """
        Get parameter count breakdown.
        
        Returns:
            Dictionary of parameter counts
        """
        params = {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        
        # Per-component breakdown
        component_params = {
            "embeddings": sum(
                p.numel() for p in self.embed_tokens.token_embedding.parameters()
            ),
            "transformer_layers": sum(
                p.numel() for p in self.layers.parameters()
            ),
            "lm_head": sum(
                p.numel() for p in self.lm_head.parameters()
            ),
        }
        params.update(component_params)
        
        return params
    
    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, "
            f"hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, "
            f"num_heads={self.num_heads}, "
            f"intermediate_size={self.intermediate_size}"
        )
