# Expera-Coder-120M Training Plan

## Model Configuration

| Parameter | Value |
|------------|-------|
| Model | Expera-Coder-120M |
| Parameters | 120M |
| Layers | 12 |
| Heads | 8 |
| KV Heads | 4 |
| Hidden Size | 768 |
| Intermediate | 1536 |
| Context Length | 2048 |
| Vocab Size | 50304 |

## Architecture

- RMSNorm (pre-Norm)
- SwiGLU Activation
- FlashAttention
- Grouped Query Attention (GQA)
- RoPE (θ=10000)
- YaRN extended context
- KV Cache enabled

## Training Config

```yaml
context_length: 2048
micro_batch_size: 1
gradient_accumulation: auto
bf16: true
activation_checkpointing: true
optimizer: AdamW
lr: 8e-4
warmup: 10%
```

## Hardware Estimate (single A100-40GB)

| Metric | Value |
|-------|-------|
| VRAM Usage | ~3.2 GB |
| Tokens/sec | ~45K |
| Steps/day | ~3900 |
| Total Days | ~7 |

## Dataset Mix

| Dataset | Tokens | % |
|---------|--------|---|
| Code (The Stack) | 2.96B | 31% |
| Code (CodeSearchNet) | 0.8B | 8% |
| Code (Project CodeNet) | 0.56B | 6% |
| Text (FineWeb) | 5.12B | 54% |
| Instructions | 0.12B | 1% |

## Training Strategy

1. Phase 1: Code-focused pretraining (4B tokens)
2. Phase 2: General finetuning on text (2B tokens)
3. Phase 3: Instruction tuning (300M tokens)

## Tokenization

Using custom SentencePiece unigram tokenizer:
- Vocab: 50,304
- Byte fallback: enabled
- Split digits: enabled
- Coverage: 99.95%

## Compression Ratios

| Dataset Type | Avg Tokens/Doc |
|--------------|---------------|
| Python | 512 |
| JavaScript | 480 |
| Java | 640 |
| Text | 400 |
| Instructions | 256 |