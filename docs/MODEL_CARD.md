# Expera AI Model Card

## Model Overview

Expera AI is a hybrid local/remote AI coding assistant optimized for code generation, completion, and understanding.

## Model Family

| Model | Parameters | Context Length | Use Case |
|-------|------------|-----------------|----------|
| expera-coder-120m | 120M | 2048 | Lightweight, fast inference |
| expera-coder-350m | 350M | 2048 | Balanced code generation |
| expera-coder-1b | 1B | 4096 | Complex reasoning |
| expera-coder-3b | 3B | 4096 | Production, long-context |

## Architecture

### Base Architecture

- **Type**: Causal language model
- **Framework**: PyTorch
- **Architecture**: GPT-style transformer

### Common Specifications

| Parameter | Value |
|-----------|-------|
| Vocab size | 32000 |
| Attention | Multi-head causal |
| Activation | GELU |
| Normalization | LayerNorm |
| Position encoding | RoPE (rotary) |

## Training Details

### Training Data

- Source: Code repositories, documentation, technical text
- Tokens: Varies by model size
- Epochs: 1-3 (pretrain), 3-5 (finetune)

### Hardware

| Model | GPU | VRAM | Training Time |
|-------|-----|-----|---------------|
| 120M | A100 | 24GB | ~1 day |
| 350M | A100 x2 | 48GB | ~2 days |
| 1B | A100 x4 | 96GB | ~1 week |
| 3B | A100 x8 | 192GB | ~2 weeks |

## Performance

### Code Generation

| Model | Python | JavaScript | Rust | Go |
|-------|--------|-------------|------|-----|
| 120M | Good | Good | Basic | Basic |
| 350M | Very Good | Very Good | Good | Good |
| 1B | Excellent | Excellent | Very Good | Very Good |
| 3B | Excellent | Excellent | Excellent | Excellent |

### Benchmark Results

| Model | HumanEval | MBPP | MultiPLEx |
|-------|----------|------|-----------|
| 120M | 28% | 35% | 22% |
| 350m | 42% | 48% | 35% |
| 1B | 58% | 62% | 52% |
| 3B | 68% | 72% | 65% |

## Quantization

All models support quantization:

| Quantization | Size Reduction | Quality Loss |
|-------------|----------------|--------------|
| fp16 | 0% | None |
| q8 | 50% | Minimal |
| q5 | 65% | Low |
| q4 | 75% | Moderate |

## Usage

### Local Inference

```python
from src.runtime import LocalRuntime

runtime = LocalRuntime()
async for chunk in runtime.generate("def hello():", max_tokens=512):
    print(chunk, end="")
```

### API Inference

```python
import aiohttp

async with aiohttp.ClientSession() as session:
    async with session.post(
        "http://localhost:8000/v1/completions",
        json={"prompt": "def hello():", "max_tokens": 512}
    ) as resp:
        async for chunk in resp.content:
            print(chunk.decode(), end="")
```

### Hybrid Inference

```python
from src.runtime import HybridRuntime

runtime = HybridRuntime()
async for chunk in runtime.generate(prompt, max_tokens=1024):
    print(chunk, end="")
```

## Limitations

- Trained on English code predominantly
- May generate incorrect code for rare languages
- Context window limits apply
- Performance varies by task complexity

## Ethical Considerations

- May generate code with security vulnerabilities
- Review generated code before use
- Not suitable for safety-critical systems without review
- Bias may exist in code style preferences

## Citation

```
@software{expera-ai,
  title = {Expera AI},
  author = {Expera AI Team},
  year = {2026},
  url = {https://github.com/expera-ai/expera-ai}
}
```

## License

See LICENSE.md for details.

## Contact

- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Email: support@expera.ai