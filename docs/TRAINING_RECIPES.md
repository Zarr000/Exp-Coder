# Training Recipes for Expera AI

Training configurations for different hardware setups.

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

###Prepare data
```bash
python scripts/download_datasets.py --dataset the-stack --output data/raw/
python scripts/prepare_code_dataset.py --input data/raw/the-stack/ --output data/processed/code/
```

## Hardware Configurations

### RTX 4050 6GB (Laptop)

**Model:** Expera-Tiny (33M)

```bash
# Pretraining
python scripts/pretrain.py \
    --config config/expera_tiny.yaml \
    --train_file data/processed/code/train.jsonl \
    --val_file data/processed/code/val.jsonl \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3
```

- Batch size: 4
- Sequence length: 1024
- Estimated time: 4-6 hours
- Memory: ~4GB VRAM

### RTX 4060 8GB (Laptop)

**Model:** Expera-Small (160M)

```bash
python scripts/pretrain.py \
    --config config/expera_small.yaml \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3
```

- Batch size: 2
- Sequence length: 2048
- Estimated time: 12-18 hours
- Memory: ~7GB VRAM

### RTX 4070 Ti 12GB

**Model:** Expera-Small (160M)

```bash
python scripts/pretrain.py \
    --config config/expera_small.yaml \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3
```

- Batch size: 4
- Sequence length: 2048
- Estimated time: 6-8 hours
- Memory: ~10GB VRAM

### RTX 4090 24GB

**Model:** Expera-Base (760M)

```bash
python scripts/pretrain.py \
    --config config/expera_base.yaml \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3
```

- Batch size: 2
- Sequence length: 2048
- Estimated time: 8-12 hours
- Memory: ~20GB VRAM

### A100 40GB (Cloud)

**Model:** Expera-Base (760M)

```bash
python scripts/pretrain.py \
    --config config/expera_base.yaml \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3
```

- Batch size: 4
- Sequence length: 2048
- Estimated time: 4-6 hours
- Memory: ~35GB VRAM

### A100 80GB x 4 (Cloud Cluster)

**Model:** Expera-Large (2.7B) - coming soon

## Training Scripts

### Pretraining
```bash
# Basic pretraining
python scripts/pretrain.py --config config/expera_tiny.yaml

# With custom settings
python scripts/pretrain.py \
    --config config/expera_tiny.yaml \
    --learning_rate 1e-3 \
    --num_train_epochs 5 \
    --warmup_ratio 0.2
```

### Instruction Tuning
```bash
# SFT training
python scripts/sft_train.py \
    --config config/expera_tiny.yaml \
    --train_file data/processed/instruction/train.jsonl \
    --val_file data/processed/instruction/val.jsonl
```

### Coding Specialization
```bash
# Fine-tune for code
python scripts/finetune_coder.py \
    --base_model checkpoints/expera-small \
    --train_file data/processed/code/train.jsonl \
    --output checkpoints/expera-coder
```

## Optimizations

### Gradient Checkpointing
Reduces memory by recomputing activations:
```bash
--gradient_checkpointing true
```

### Mixed Precision
Use BF16 on supported hardware:
```bash
--bf16 true
```

### Flash Attention
Enable for faster training:
```bash
--use_flash_attention true
```

### LoRA
For efficient fine-tuning:
```bash
--use_lora true \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.05
```

## Monitoring

### Loss tracking
Check `logs/{model}/loss.csv` for training curves.

### Weights & Biases
```bash
export WANDB_API_KEY=your_key
python scripts/pretrain.py --report_to wandb
```

## Troubleshooting

### OOM Errors
- Reduce batch size
- Enable gradient checkpointing
- Reduce sequence length

### Slow Training
- Check GPU utilization with `nvidia-smi`
- Enable Flash Attention
- Reduce logging frequency

### Poor Quality
- Increase learning rate slightly
- Train for more epochs
- Add more data