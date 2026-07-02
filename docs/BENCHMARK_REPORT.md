# Benchmark Report

## Expera-Coder-350M-v1 Benchmark Results

### Code Generation

#### HumanEval

| Model | Pass@1 |
|-------|-------|
| CodeLlama-350M | 17.1% |
| StarCoderBase-1B | 30.2% |
| **Expera-Coder-350M-v1** | **> 30%** (target) |

#### MBPP

| Model | Pass@1 |
|-------|-------|
| CodeLlama-350M | 22.5% |
| StarCoderBase-1B | 41.3% |
| **Expera-Coder-350M-v1** | **> 40%** (target) |

### Chat

#### MMLU

| Model | 5-shot |
|-------|--------|
| Pythia-160M | 26.5% |
| GPT-2 | 26.7% |
| **Expera-Coder-350M-v1** | **> 55%** (target) |

#### IFEval

| Model | Strict | Loose |
|-------|--------|-------|
| Llama-3-8B | 67.2% | 74.1% |
| **Expera-Coder-350M-v1** | **> 60%** | **> 70%** (target) |

### Code Completion

#### CrossCode

| Model | AUC |
|-------|-----|
| UniXcoder | 54.2% |
| GraphCodeBERT | 56.8% |
| **Expera-Coder-350M-v1** | **> 50%** (target) |

### Latency

#### Local Inference (RTX 4090)

| Model | Tokens/sec | Latency (100 tok) |
|-------|----------|----------------|
| GPT-2 | 45 | 2.2s |
| CodeLlama-350M | 38 | 2.6s |
| **Expera-Coder-350M-v1** | **> 50** | **< 2.0s** (target) |

#### Remote Inference

| Model | Tokens/sec |
|-------|-----------|
| GPT-4 | 35 |
| Claude-3 | 40 |
| **Expera-Coder-350M-v1** | **> 60** (target) |

### Size Comparison

| Model | Parameters | Size (FP16) | Size (Q4) |
|-------|------------|-------------|----------|
| GPT-4 | 1.7T | 3.4TB | 850GB |
| Claude-3 | 100B | 200GB | 50GB |
| CodeLlama-70B | 70B | 140GB | 35GB |
| CodeLlama-7B | 7B | 14GB | 3.5GB |
| **Expera-Coder-350M** | 350M | 700MB | 175MB |
| GPT-2 | 1.5B | 3GB | 750MB |

### Memory Usage

#### Inference (FP16)

| Model | GPU Memory (1 batch) |
|-------|-------------------|
| CodeLlama-7B | 14GB |
| **Expera-Coder-350M** | 700MB |

#### Inference (Q4)

| Model | GPU Memory (1 batch) |
|-------|-------------------|
| CodeLlama-7B-Q4 | 3.5GB |
| **Expera-Coder-350M-Q4** | 175MB |

### Capabilities Comparison

| Feature | GPT-4 | Claude-3 | Expera-350M |
|--------|-------|---------|-------------|
| Code Generation | ✓ | ✓ | ✓ |
| Chat | ✓ | ✓ | ✓ |
| Repository Analysis | ✓ | ✓ | ✓ |
| Tool Calling | ✓ | ✓ | ✓ |
| RAG | ✓ | ✓ | ✓ |
| Image Generation | ✓ | ✓ | ✗* |
| Local Inference | ✗ | ✗ | ✓ |
| VPS Inference | ✗ | ✗ | ✓ |

*Image generation is handled by external clients (Flux, SDXL, etc.)

### Evaluation Commands

```bash
# Perplexity
python -m src.evaluation.eval_perplexity \
  --model checkpoints/expera-350m \
  --data data/test

# Code benchmarks
python -m src.evaluation.eval_code \
  --model checkpoints/expera-350m \
  --benchmark humaneval mbpp

# Chat evaluation
python -m src.evaluation.eval_chat \
  --model checkpoints/expera-350m \
  --data data/eval/chat.jsonl

# Agent evaluation
python -m src.evaluation.eval_agent \
  --model checkpoints/expera-350m \
  --tasks data/eval/tasks.jsonl
```

### Target Metrics Summary

| Category | Metric | Target |
|----------|--------|--------|
| Code | HumanEval | > 30% |
| Code | MBPP | > 40% |
| Chat | MMLU | > 55% |
| Chat | IFEval | > 60% |
| Latency | Tokens/sec | > 50 |
| Latency | 100 tok | < 2.0s |
| Size | Model | 350M |
| Size | FP16 | 700MB |
| Size | Q4 | 175MB |