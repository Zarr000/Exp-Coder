#!/usr/bin/env python3
"""
Code Fine-tuning Specialization.

Fine-tunes model specifically for code tasks:
- Code completion
- Code generation
- Bug fixing
- Code explanation

Usage:
    python scripts/finetune_coder.py --base_model checkpoints/expera-small --train_file data/processed/code/train.jsonl
    python scripts/finetune_coder.py --base_model checkpoints/expera-tiny --output checkpoints/expera-coder --lora
"""

import argparse
import json
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


# Code formatting templates
CODE_PROMPTS = {
    "completion": "Complete the following code:\n{code}\n\nCompleted:",
    "generation": "Write code to solve this problem:\n{prompt}\n\nCode:",
    "bugfix": "Find and fix bugs in this code:\n{code}\n\nFixed code:",
    "explain": "Explain this code:\n{code}\n\nExplanation:",
    "refactor": "Refactor this code:\n{code}\n\nRefactored:",
}


def format_code_sample(
    record: Dict,
    task: str = "generation",
) -> str:
    """Format code sample for training."""
    prompt_template = CODE_PROMPTS.get(task, CODE_PROMPTS["generation"])

    code = record.get("content", record.get("code", ""))
    if not code:
        return ""

    if task == "generation":
        # Use as prompt
        prompt = record.get("prompt", "")
        return prompt_template.format(prompt=prompt) + "\n" + code
    elif task == "bugfix":
        return CODE_PROMPTS["bugfix"].format(code=code) + "\n" + code
    elif task == "explain":
        return CODE_PROMPTS["explain"].format(code=code) + "\n" + record.get("explanation", "Code explanation...")
    elif task == "completion":
        # Extract partial code
        return CODE_PROMPTS["completion"].format(code=code) + "\n" + code
    else:
        return code


def load_code_dataset(
    file_path: Path,
    tokenizer,
    max_length: int = 2048,
    tasks: List[str] = ["generation"],
) -> Dataset:
    """Load code dataset."""
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
        # Format text
        text = example.get("content", "")
        if not text:
            return None

        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        return {
            "input_ids": tokens.input_ids,
            "attention_mask": tokens.attention_mask,
            "labels": tokens.input_ids,
        }

    ds = Dataset.from_list(records)
    ds = ds.map(
        tokenize_fn,
        remove_columns=ds.column_names,
    )

    return ds


def train(
    base_model: str,
    train_file: str,
    val_file: Optional[str] = None,
    output_dir: str = "checkpoints/expera-coder",
    max_length: int = 2048,
    learning_rate: float = 1e-4,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    warmup_ratio: float = 0.1,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> None:
    """Fine-tune for code."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    logger.info(f"Loading model from {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)

    # LoRA
    if use_lora:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"Using LoRA with r={lora_r}, alpha={lora_alpha}")

    model.to(device)

    # Load datasets
    train_ds = load_code_dataset(Path(train_file), tokenizer, max_length)
    logger.info(f"Train dataset: {len(train_ds)} samples")

    val_ds = None
    if val_file:
        val_ds = load_code_dataset(Path(val_file), tokenizer, max_length)
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
        lr_scheduler_type="cosine",
        logging_dir=f"{output_dir}/logs",
        logging_steps=25,
        save_steps=500,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=200,
        bf16=True,
        save_total_limit=3,
        load_best_model_at_end=True if val_ds else False,
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
    logger.info("Starting code fine-tuning")
    trainer.train()

    # Save
    logger.info(f"Saving to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune for code")
    parser.add_argument("--base_model", type=str, required=True, help="Base model path")
    parser.add_argument("--train_file", type=str, required=True, help="Train JSONL file")
    parser.add_argument("--val_file", type=str, help="Validation JSONL file")
    parser.add_argument("--output_dir", type=str, default="checkpoints/expera-coder", help="Output directory")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Train batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Eval batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    return parser.parse_args()


def main():
    args = parse_args()

    train(
        base_model=args.base_model,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )


if __name__ == "__main__":
    main()