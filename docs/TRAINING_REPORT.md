# Training Report

## Expera-Coder-350M-v1 Training Summary

### Training Pipeline Overview

```
Dataset Pipeline → Pretraining → Instruction Tuning → Preference Optimization → Coding Specialization → Quantization
```

### 1. Dataset Pipeline

#### Code Datasets
- The Stack (350K files, 86 languages)
- StarCoderData (1B tokens)
- CodeSearchNet (2M examples)
- Python, JavaScript, Go, Rust, Java

#### Chat Datasets
- OpenHermes-2.5 (100K)
- UltraChat (1.2M)
- Code Alpaca (52K)
- SlimOrca (1.3M)

#### Image Datasets
- JourneyDB (500K)
- DiffusionDB (300K)
- LAION metadata

#### Quality Filtering
- Deduplication (minhash)
- Language balancing
- Quality filters (length, code structure)

### 2. Pretraining

#### Configuration
```yaml
model:
  vocab_size: 32000
  embedding_dim: 1024
  num_layers: 24
  num_heads: 16

training:
  batch_size: 64
  seq_length: 2048
  learning_rate: 5e-5
  epochs: 3
  precision: bf16
  gradient_checkpointing: true
```

#### Training Command
```bash
torchrun --nproc_per_node=8 scripts/train/train_350m.py \
  --data data/mixture \
  --output checkpoints/expera-350m \
  --epochs 3
```

#### Expected Metrics
- Initial perplexity: ~15.0
- Final perplexity: ~8.5
- Training time (8x A100): ~4 hours
- GPU memory: ~8GB per device

### 3. Instruction Tuning

#### Formats Supported
- Alpaca format
- ShareGPT format
- OpenAI format
- HuggingFace format

#### SFT Configuration
```yaml
training:
  learning_rate: 3e-5
  batch_size: 16
  gradient_accumulation: 4
  epochs: 3
  warmup_steps: 100
```

### 4. Preference Optimization

#### DPO Configuration
```yaml
dpo:
  beta: 0.1
  learning_rate: 5e-6
  batch_size: 8
  max_length: 2048
```

#### ORPO Configuration
```yaml
orpo:
  lambda: 0.01
  learning_rate: 1e-5
```

### 5. Coding Specialization

#### Language-Specific Finetuning
- Python: 3 epochs
- JavaScript: 2 epochs
- Web Dev: 2 epochs
- Debugging: 3 epochs

### 6. Evaluation Benchmarks

| Benchmark | Target Score |
|-----------|------------|
| HumanEval | > 30% |
| MBPP | > 40% |
| MMLU | > 55% |
| IFEval | > 60% |

### 7. Quantization

| Format | Size Reduction | Quality Loss |
|--------|--------------|-------------|
| FP16 | 50% | Minimal |
| INT8 | 75% | Low |
| Q4 | 87.5% | Medium |
| GGUF Q4 | 87.5% | Medium |

### Training Recipes

#### Quick Start (Local)
```bash
# 1. Download datasets
python scripts/datasets/download_code_datasets.py
python scripts/datasets/download_chat_datasets.py

# 2. Build mixture
python scripts/datasets/build_mixture.py

# 3. Pretrain
torchrun --nproc_per_node=1 scripts/train/train_350m.py \
  --data data/mixture \
  --output checkpoints/expera-350m

# 4. SFT
python scripts/sft/sft_chat.py \
  --model checkpoints/expera-350m \
  --data data/sft/chat \
  --output checkpoints/expera-350m-sft

# 5. Export
python scripts/quantize/export_fp16.py \
  --model checkpoints/expera-350m-sft
```

#### Full Pipeline (Multi-GPU)
```bash
# Distributed training
torchrun --nproc_per_node=8 scripts/train/train_350m.py \
  --data data/mixture \
  --output checkpoints/expera-350m \
  --epochs 3 \
  --batch-size 32

# Preference optimization
python -m src.alignment.dpo \
  --model checkpoints/expera-350m-sft \
  --data data/preferences \
  --output checkpoints/expera-350m-dpo
```

### Troubleshooting

#### Out of Memory
- Reduce batch size
- Enable gradient checkpointing
- Use mixed precision (bf16)
- Reduce sequence length

#### Slow Training
- Increase GPU count
- Enable Flash Attention
- Use data loading workers

#### Poor Quality
- Increase dataset size
- Adjust learning rate
- More epochs
- Better quality data

## Timeline

- Dataset curation: 1 hour
- Pretraining: 4 hours (8x A100)
- SFT: 2 hours
- DPO: 1 hour
- Evaluation: 1 hour
- Total: ~9 hours