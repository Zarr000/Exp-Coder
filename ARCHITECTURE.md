# Expera AI - Architecture Documentation

## Table of Contents
1. [Overview](#overview)
2. [Tokenizer Architecture](#tokenizer-architecture)
3. [Model Architecture](#model-architecture)
4. [Training Pipeline](#training-pipeline)
5. [Innovations](#innovations)

---

## Overview

Expera AI is an original Large Language Model designed from scratch with a focus on:
- **Software Development**: Code generation, debugging, refactoring
- **Creative Visual Generation**: Image understanding and prompt generation
- **Scalability**: Architecture that grows with computational resources
- **Innovation**: Novel components beyond standard Transformer designs

### Design Philosophy

Unlike standard GPT models, Expera AI incorporates:
1. **Memory Bank System**: External memory for long-term context retention
2. **Reasoning Module**: Multi-step reasoning capabilities
3. **Rotary Position Embeddings (RoPE)**: Better position encoding than absolute embeddings
4. **Modular Architecture**: Easy to extend and experiment with

---

## Tokenizer Architecture

### Byte-Pair Encoding (BPE)

**Why BPE?**
- Balances vocabulary size with token efficiency
- Handles rare words through subword decomposition
- Works well for code (preserves structure while being flexible)

**Implementation Details:**

1. **Byte-Level BPE** (like GPT-2)
   - Operates on UTF-8 bytes rather than characters
   - Handles any Unicode text without unknown tokens
   - Vocabulary size: 50,304 (optimized for GPU efficiency - divisible by 64)

2. **Special Tokens**
   - `<|pad|>`: Padding token
   - `<|unk|>`: Unknown token (rarely used in byte-level BPE)
   - `<|startoftext|>`: Beginning of sequence
   - `<|endoftext|>`: End of sequence
   - Code-specific: `<|python|>`, `<|javascript|>`, etc.
   - Structure: `<|function|>`, `<|class|>`, etc.

3. **Training Process**
   ```
   Input Text → Unicode Normalization → Byte Encoding → 
   Pre-tokenization (whitespace) → BPE Merges → Token IDs
   ```

**Mathematical Foundation:**

BPE iteratively merges the most frequent pair of tokens:
```
Given vocabulary V and text corpus C:
For i = 1 to num_merges:
    Find most frequent pair (a, b) in C
    Add merged token "ab" to V
    Replace all (a, b) with "ab" in C
```

---

## Model Architecture

### Core Components

#### 1. Token Embeddings

**Purpose**: Convert discrete token IDs to continuous vectors

```python
embedding = Embedding(vocab_size=50304, hidden_size=768)
```

**Mathematical Formulation:**
```
E ∈ ℝ^(V × d)
where V = vocabulary size, d = hidden dimension
```

Each token ID i is mapped to embedding vector E[i].

#### 2. Positional Encoding

**Standard Approach** (Sinusoidal):
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**Expera AI Approach** (RoPE - Rotary Position Embeddings):

RoPE applies rotation to query and key vectors based on position:
```
f(q, m) = (W_q x) ⊗ e^(imθ)
f(k, n) = (W_k x) ⊗ e^(inθ)
```

**Advantages:**
- Relative position information naturally emerges
- Better extrapolation to longer sequences
- Used in modern models (LLaMA, PaLM)

#### 3. Multi-Head Attention

**Purpose**: Allow model to attend to different representation subspaces

**Mathematical Formulation:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V

MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Parameters:**
- Number of heads: 12
- Head dimension: 64 (768 / 12)
- Total parameters per layer: ~2.4M

**Implementation Details:**
- Scaled dot-product attention
- Causal masking for autoregressive generation
- Optional: Flash Attention for efficiency
- Optional: Grouped Query Attention (GQA) for reduced KV cache

**Attention Mechanism Breakdown:**

1. **Query, Key, Value Projections**
   ```
   Q = XW^Q  (batch, seq_len, hidden_size)
   K = XW^K
   V = XW^V
   ```

2. **Split into Multiple Heads**
   ```
   Q → (batch, num_heads, seq_len, head_dim)
   ```

3. **Compute Attention Scores**
   ```
   scores = (Q @ K^T) / √head_dim
   scores = scores + causal_mask  # -inf for future positions
   attn_weights = softmax(scores)
   ```

4. **Apply Attention to Values**
   ```
   output = attn_weights @ V
   ```

5. **Concatenate and Project**
   ```
   output = Concat(all_heads) @ W^O
   ```

#### 4. Feed-Forward Network

**Purpose**: Add non-linearity and increase model capacity

**Architecture:**
```
FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
```

**Dimensions:**
- Input: 768
- Intermediate: 3072 (4× expansion)
- Output: 768

**Activation Function - GELU:**
```
GELU(x) = x * Φ(x)
where Φ(x) is the cumulative distribution function of standard normal
```

Approximation:
```
GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

**Why GELU over ReLU?**
- Smoother gradients
- Better performance in transformers
- Used in BERT, GPT-2, GPT-3

#### 5. Transformer Block

**Complete Block Structure:**
```
# Pre-LayerNorm architecture (more stable)
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

**Layer Normalization:**
```
LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
where μ, σ² are computed across the feature dimension
```

**Residual Connections:**
- Enable gradient flow through deep networks
- Allow model to learn identity function easily
- Critical for training 12+ layer models

#### 6. Complete Model Architecture

```
Input Token IDs
    ↓
Token Embedding + Positional Encoding
    ↓
Transformer Block 1
    ↓
Transformer Block 2
    ↓
    ...
    ↓
Transformer Block 12
    ↓
Layer Norm
    ↓
Linear Projection (to vocab size)
    ↓
Output Logits
```

**Total Parameters (Base Model):**
- Embeddings: 50,304 × 768 = 38.6M
- 12 Transformer Blocks: ~85M
- Output Layer: 38.6M
- **Total: ~162M parameters**

---

## Advanced Features (Expera AI Innovations)

### 1. Memory Bank System

**Motivation**: Transformers have limited context windows. Memory banks provide external storage for long-term information.

**Architecture:**
```
Memory Bank: M ∈ ℝ^(N × d)
where N = memory slots (1024), d = hidden dimension (768)
```

**Operations:**
1. **Write**: Store important information
   ```
   M[i] = α * M[i] + (1-α) * new_info
   ```

2. **Read**: Retrieve relevant information via attention
   ```
   retrieved = Attention(query=current_state, key=M, value=M)
   ```

3. **Update**: Decay old memories, reinforce important ones

**Use Cases:**
- Remember function definitions across long code files
- Maintain context in multi-turn conversations
- Store project-specific knowledge

### 2. Reasoning Module

**Motivation**: Enable multi-step reasoning for complex problems

**Architecture:**
```
For step in 1..K:
    thought[step] = ReasoningLayer(input + thought[step-1])
    
final_output = Combine(all_thoughts)
```

**Implementation:**
- K reasoning steps (default: 3)
- Each step refines the previous thought
- Similar to Chain-of-Thought but learned end-to-end

**Benefits:**
- Better performance on coding problems
- Improved logical reasoning
- Explicit multi-step problem solving

### 3. Grouped Query Attention (GQA)

**Motivation**: Reduce KV cache size for faster inference

**Standard Multi-Head Attention:**
- 12 heads, each with separate Q, K, V

**GQA:**
- 12 query heads
- 4 key-value heads (shared across query heads)
- 3× reduction in KV cache size

**Trade-off:**
- Slightly lower quality
- Much faster inference
- Better for deployment

---

## Training Pipeline

### 1. Data Preprocessing

**Pipeline:**
```
Raw Text → Cleaning → Tokenization → Chunking → Batching
```

**Data Sources:**
- Code: GitHub, Stack Overflow
- Text: Books, articles, documentation
- Instructions: Curated instruction-following datasets

**Data Mixture:**
- 50% Code
- 30% Natural text
- 20% Instructions

### 2. Optimization

**Optimizer: AdamW**
```
m_t = β₁m_{t-1} + (1-β₁)g_t
v_t = β₂v_{t-1} + (1-β₂)g_t²
θ_t = θ_{t-1} - η(m_t/√(v_t + ε) + λθ_{t-1})
```

**Hyperparameters:**
- Learning rate: 3e-4
- β₁ = 0.9, β₂ = 0.95
- Weight decay: 0.01
- Gradient clipping: 1.0

### 3. Learning Rate Schedule

**Cosine with Warmup:**
```
For step < warmup_steps:
    lr = max_lr * (step / warmup_steps)
    
For step >= warmup_steps:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
```

### 4. Loss Function

**Causal Language Modeling Loss:**
```
L = -∑ log P(x_t | x_1, ..., x_{t-1})
```

**Implementation:**
```python
loss = CrossEntropyLoss(logits.view(-1, vocab_size), 
                        targets.view(-1))
```

### 5. Training Techniques

**Mixed Precision Training:**
- Use bfloat16 for forward/backward pass
- Keep master weights in float32
- 2× speedup, 50% memory reduction

**Gradient Accumulation:**
- Accumulate gradients over N steps
- Effective batch size = batch_size × N
- Enables training with limited GPU memory

**Gradient Checkpointing:**
- Trade computation for memory
- Recompute activations during backward pass
- Enables training larger models

---

## Scaling Strategy

### Model Variants

| Variant | Layers | Hidden | Heads | Params | Context |
|---------|--------|--------|-------|--------|---------|
| Tiny    | 4      | 256    | 4     | ~20M   | 1K      |
| Small   | 8      | 512    | 8     | ~80M   | 2K      |
| Base    | 12     | 768    | 12    | ~160M  | 2K      |
| Large   | 24     | 1024   | 16    | ~350M  | 4K      |
| XLarge  | 32     | 1536   | 24    | ~1B    | 8K      |

### Scaling Laws

Based on Chinchilla scaling laws:
```
Optimal model size ∝ (compute budget)^0.5
Optimal dataset size ∝ (compute budget)^0.5
```

For compute budget C:
- Model parameters: N ∝ C^0.5
- Training tokens: D ∝ C^0.5

---

## Performance Optimizations

### 1. Flash Attention
- Fused attention kernel
- 2-4× speedup
- Reduced memory usage

### 2. Kernel Fusion
- Combine operations (LayerNorm + Linear)
- Reduce memory transfers
- Better GPU utilization

### 3. Distributed Training
- Data Parallel: Replicate model across GPUs
- Fully Sharded Data Parallel (FSDP): Shard model parameters
- Pipeline Parallel: Split model across GPUs

---

## Future Extensions

### 1. Multimodal Capabilities
- Vision encoder integration
- Cross-modal attention
- Image generation guidance

### 2. Retrieval Augmentation
- External knowledge base
- Dynamic retrieval during inference
- Fact verification

### 3. Reinforcement Learning
- RLHF (Reinforcement Learning from Human Feedback)
- Code execution feedback
- Self-improvement loops

---

## References

1. **Attention Is All You Need** (Vaswani et al., 2017)
2. **Language Models are Few-Shot Learners** (GPT-3, Brown et al., 2020)
3. **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021)
4. **FlashAttention: Fast and Memory-Efficient Exact Attention** (Dao et al., 2022)
5. **GQA: Training Generalized Multi-Query Transformer** (Ainslie et al., 2023)
6. **Training Compute-Optimal Large Language Models** (Chinchilla, Hoffmann et al., 2022)

---

**Last Updated**: 2026-06-25
**Version**: 0.1.0
