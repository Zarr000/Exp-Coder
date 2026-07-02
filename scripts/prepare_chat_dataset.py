#!/usr/bin/env python3
"""Prepare chat dataset for training."""

import json
from pathlib import Path

# Format template - WITHOUT special tokens that tokenizer doesn't know
SYSTEM_PROMPT = "You are Expera AI."
USER_PREFIX = "User: "
ASSISTANT_PREFIX = "Assistant: "

def format_example(instruction, response, system_prompt=None):
    """Format chat example into training format."""
    system = system_prompt or SYSTEM_PROMPT
    # Plain format - no special tokens that tokenizer can't recognize
    prompt = f"{system}\n{USER_PREFIX}{instruction}\n{ASSISTANT_PREFIX}{response}"
    return prompt

def main():
    """Process chat dataset."""
    # Load all chat datasets
    dataset_files = [
        "datasets/chat/basic_chat.json",
        "datasets/chat/coding_chat.json",
        "datasets/chat/reasoning_chat.json",
        "datasets/chat/instruction_chat.json",
    ]

    examples = []
    for input_file in dataset_files:
        if Path(input_file).exists():
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                examples.extend(data)
                print(f"Loaded {len(data)} from {input_file}")

    print(f"Total: {len(examples)} examples")

    # Format for training
    formatted = []
    for ex in examples:
        instruction = ex.get("instruction", "").strip()
        response = ex.get("response", "").strip()
        if instruction and response:
            formatted.append(format_example(instruction, response))

    print(f"Formatted {len(formatted)} examples")

    # Split train/valid (95/5)
    split = int(len(formatted) * 0.95)
    train_data = formatted[:split]
    valid_data = formatted[split:]

    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}")

    # Save as JSONL
    output_dir = Path("data/processed/chat")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "train.jsonl"
    valid_file = output_dir / "valid.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(valid_file, "w", encoding="utf-8") as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved to {train_file} and {valid_file}")

if __name__ == "__main__":
    main()