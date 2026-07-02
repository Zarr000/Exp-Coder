#!/usr/bin/env python3
"""
SFT Training (Supervised Fine-Tuning).

Fine-tunes model on instruction datasets with chat templates:
- ExperaChat (general chat)
- ExperaCoder (code-focused)
- ExperaVision (multimodal)

Usage:
    python scripts/sft_train.py --config config/expera_small.yaml --train_file data/processed/instruction/train.jsonl
    python scripts/sft_train.py --model checkpoints/expera-small --template expera_coder --output checkpoints/expera-coder-sft
"""

import argparse
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Chat Templates
CHAT_TEMPLATES = {
    "expera_chat": {
        "system": "A chat between a curious user and an AI assistant. The assistant is helpful, creative, witty, and friendly.",
        "user": "{% for message in messages %}{% if message.role == 'user' %}User: {{message.content}}\n{% endif %}{% endfor %}User: {prompt}",
        "assistant": "Assistant: {response}",
    },
    "expera_coder": {
        "system": "You are ExperaCoder, an AI coding assistant. You help users write, debug, and understand code.",
        "user": "{% for message in messages %}{% if message.role == 'user' %}User: {{message.content}}\n{% endif %}{% endfor %}User: {prompt}",
        "assistant": "Assistant: {response}",
    },
    "expera_vision": {
        "system": "You are ExperaVision, an AI assistant that can understand and analyze images. Describe what you see.",
        "user": "{% for message in messages %}{% if message.role == 'user' %}[Image: {{message.image}}]\n{{message.content}}{% endif %}{% endfor %}User: {prompt}",
        "assistant": "Assistant: {response}",
    },
}


def format_conversation(
    messages: List[Dict],
    template: str = "expera_chat",
) -> str:
    """Format conversation with template."""
    template = CHAT_TEMPLATES.get(template, CHAT_TEMPLATES["expera_chat"])

    system = template["system"]
    user_template = template["user"]
    assistant_template = template["assistant"]

    # Build messages
    output = f"System: {system}\n\n"

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            output += f"User: {content}\n"
        elif role == "assistant":
            output += f"Assistant: {content}\n"

    return output


def tokenize_conversation(
    record: Dict,
    tokenizer,
    template: str = "expera_chat",
    max_length: int = 2048,
) -> Dict:
    """Tokenize conversation."""
    conversations = record.get("conversations", [])

    if not conversations:
        # Try text field
        text = record.get("text", "")
        if not text:
            return None

        # Already formatted
        return tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    # Format conversation
    formatted = format_conversation(conversations, template)

    return tokenizer(
        formatted,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def load_dataset(
    file_path: Path,
    tokenizer,
    template: str = "expera_chat",
    max_length: int = 2048,
) -> Dataset:
    """Load instruction dataset."""
    file_path = Path(file_path)

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                continue

    # Tokenize
    def tokenize_fn(example):
        tokens = tokenize_conversation(example, tokenizer, template, max_length)
        if tokens:
            return {
                "input_ids": tokens.input_ids,
                "attention_mask": tokens.attention_mask,
                "labels": tokens.input_ids,
            }
        return None

    ds = Dataset.from_list(records)
    ds = ds.map(
        tokenize_fn,
        remove_columns=ds.column_names,
    )

    return ds


def train(
    model_path: str,
    train_file: str,
    val_file: Optional[str] = None,
    output_dir: str = "checkpoints/sft",
    template: str = "expera_chat",
    max_length: int = 2048,
    learning_rate: float = 5e-5,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    warmup_ratio: float = 0.1,
    lr_scheduler_type: str = "cosine",
    fp16: bool = False,
    bf16: bool = True,
    logging_steps: int = 25,
    save_steps: int = 500,
    eval_steps: int = 200,
) -> None:
    """Train model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)

    # Load datasets
    train_ds = load_dataset(Path(train_file), tokenizer, template, max_length)
    logger.info(f"Train dataset: {len(train_ds)} samples")

    val_ds = None
    if val_file:
        val_ds = load_dataset(Path(val_file), tokenizer, template, max_length)
        logger.info(f"Val dataset: {len(val_ds)} samples")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=eval_steps,
        fp16=fp16,
        bf16=bf16,
        save_total_limit=3,
        load_best_model_at_end=True if val_ds else False,
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training")
    trainer.train()

    # Save
    logger.info(f"Saving to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Training")
    parser.add_argument("--config", type=str, help="Config YAML file")
    parser.add_argument("--model", type=str, help="Base model path")
    parser.add_argument("--train_file", type=str, required=True, help="Train JSONL file")
    parser.add_argument("--val_file", type=str, help="Validation JSONL file")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft", help="Output directory")
    parser.add_argument("--template", type=str, default="expera_chat", choices=list(CHAT_TEMPLATES.keys()), help="Chat template")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Train batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Eval batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler")
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    parser.add_argument("--bf16", action="store_true", help="Use BF16")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config:
        # Load from config
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)

        # Override with args
        model_path = args.model or config.get("model_path", config.get("base_model", ""))
    else:
        if not args.model:
            logger.error("Please specify --model or --config")
            return
        model_path = args.model

    train(
        model_path=model_path,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        template=args.template,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=args.fp16,
        bf16=args.bf16 or (not args.fp16),
    )


if __name__ == "__main__":
    main()