# Expera AI Training Guide

## Training Overview

Expera AI supports custom training and fine-tuning through a unified training pipeline.

## Training Modes

### 1. Pretraining

Train from scratch on raw text data.

```python
from src.training.config import TrainingConfig, TrainingMode
from src.training.pipeline import TrainingPipeline

config = TrainingConfig(
    model_name="expera-coder-120m",
    mode=TrainingMode.PRETRAIN,
    data_path="data/pretrain/",
    output_path="checkpoints/expera_coder_120m",
    epochs=3,
    batch_size=32,
    learning_rate=1e-4,
)

pipeline = TrainingPipeline(config)
await pipeline.train()
```

### 2. Fine-tuning

Fine-tune on specific domain data.

```python
config = TrainingConfig(
    model_name="expera-coder-120m",
    mode=TrainingMode.FINETUNE,
    data_path="data/code/",
    output_path="checkpoints/expera_coder_120m_ft",
    epochs=5,
    batch_size=16,
    learning_rate=5e-5,
    warmup_steps=100,
)
```

### 3. DPO (Direct Preference Optimization)

Train with preference data.

```python
config = TrainingConfig(
    model_name="expera-coder-350m",
    mode=TrainingMode.DPO,
    preference_data_path="data/preferences/",
    output_path="checkpoints/expera_coder_350m_dpo",
    epochs=3,
    batch_size=8,
    learning_rate=1e-6,
)
```

## Data Formats

### Pretraining Data

Plain text files, one document per line:
```
First document content here...
Second document content here...
```

### Fine-tuning Data

JSONL format:
```json
{"prompt": "Write a function to sort a list:", "completion": "def quicksort(arr): ..."}
```

### Preference Data

JSONL format with preferred/rejected:
```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

## Training Configuration

| Parameter | Description | Default |
|-----------|-------------|----------|
| model_name | Model to train | required |
| mode | Training mode | required |
| data_path | Training data path | required |
| output_path | Output checkpoint path | required |
| epochs | Number of epochs | 3 |
| batch_size | Batch size | 32 |
| learning_rate | Learning rate | 1e-4 |
| warmup_steps | Warmup steps | 500 |
| max_seq_length | Max sequence length | 2048 |
| gradient_accumulation | Gradient accumulation | 1 |
| precision | Mixed precision | bf16 |

## GPU Requirements

| Model Size | VRAM Required | Recommended GPU |
|------------|---------------|------------------|
| 120M | 2GB | RTX 3060 |
| 350M | 4GB | RTX 4070 |
| 1B | 16GB | A100 |
| 3B | 24GB | A100 x2 |

## Evaluation

Run evaluation after training:

```python
from src.training.pipeline import evaluate_model

results = await evaluate_model(
    model_path="checkpoints/expera_coder_120m",
    eval_data_path="data/eval/",
    metrics=["code_accuracy", "perplexity"],
)

print(f"Perplexity: {results['perplexity']}")
print(f"Code Accuracy: {results['code_accuracy']}")
```

## Checkpointing

The pipeline automatically:
- Saves checkpoints every N steps
- Keeps only the best checkpoint
- Supports resuming from checkpoint

## Monitoring

- TensorBoard logs in `outputs/logs/`
- Training metrics in `outputs/metrics/`
- GPU utilization tracked

## Common Issues

### Out of Memory

- Reduce batch_size
- Enable gradient checkpointing
- Use gradient accumulation

### Poor Convergence

- Check learning rate
- Verify data format
- Increase warmup steps

### Slow Training

- Enable mixed precision (bf16)
- Use DataLoader with multiple workers
- Optimize data preprocessing

## Production Training

For large-scale training, use distributed training:

```python
config = TrainingConfig(
    ...
    distributed=True,
    world_size=8,
    gradient_accumulation=4,
)
```

## Scripts

Training scripts are available in `scripts/`:
- `train.sh` - Basic training
- `finetune.sh` - Fine-tuning
- `evaluate.sh` - Evaluation