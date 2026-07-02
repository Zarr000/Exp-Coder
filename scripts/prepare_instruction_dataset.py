#!/usr/bin/env python3
"""
Instruction Dataset Preparation.

Prepares instruction/chat datasets for tuning:
- Converts to chat format
- Filters quality
- Deduplicates
- Generates metadata

Usage:
    python scripts/prepare_instruction_dataset.py --input data/raw/openassistant/ --output data/processed/instruction/
    python scripts/prepare_instruction_dataset.py --input data/raw/sharegpt/ --format sharegpt --output data/processed/chat/
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
import hashlib
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Chat templates
CHAT_TEMPLATES = {
    "default": {
        "system": "A chat between a curious user and an assistant. The assistant is helpful, creative, and witty.",
        "user": "USER: {prompt}",
        "assistant": "ASSISTANT: {response}",
        "sep": "\n\n",
    },
    "expera": {
        "system": "{% if messages[0] and messages[0].role == 'system' %}{{messages[0].content}}{% else %}A chat between a curious user and an AI assistant.{% endif %}",
        "user": "{% for message in messages %}{% if message.role == 'user' %}USER: {{message.content}}\n{% endif %}{% endfor %}USER: {prompt}",
        "assistant": "ASSISTANT: {response}",
        "sep": "\n\n",
    },
}


def calculate_hash(text: str) -> str:
    """Calculate hash for deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def detect_format(record: Dict) -> str:
    """Detect dataset format."""
    # Check for OpenAssistant format
    if "parent_id" in record or "message_id" in record:
        return "openassistant"

    # Check for ShareGPT format
    if "conversations" in record:
        return "sharegpt"

    # Check for Alpaca format
    if "instruction" in record and "input" in record:
        return "alpaca"

    # Check for UltraChat format
    if "dialogs" in record:
        return "ultrachat"

    # Check for standard format
    if "prompt" in record and "response" in record:
        return "standard"

    if "input" in record and "output" in record:
        return "standard"

    return "unknown"


def convert_alpaca(record: Dict) -> Optional[Dict]:
    """Convert Alpaca format."""
    instruction = record.get("instruction", "")
    input_text = record.get("input", "")

    if instruction and input_text:
        prompt = f"{instruction}\n\nInput: {input_text}"
    elif instruction:
        prompt = instruction
    else:
        return None

    output = record.get("output", "")
    if not output:
        return None

    return {
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": output},
        ],
        "source": "alpaca",
    }


def convert_openassistant(record: Dict) -> Optional[Dict]:
    """Convert OpenAssistant format to conversations."""
    # Skip non-assistant messages
    if record.get("role") != "assistant":
        return None

    content = record.get("content", "")
    if not content:
        return None

    return {
        "conversations": [
            {"role": "assistant", "content": content},
        ],
        "source": "openassistant",
    }


def convert_sharegpt(record: Dict) -> Optional[Dict]:
    """Convert ShareGPT format."""
    conversations = record.get("conversations", [])
    if not conversations:
        return None

    # Convert to standard format
    converted = []
    for msg in conversations:
        role = msg.get("from", "")
        content = msg.get("value", "")

        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"

        if role and content:
            converted.append({"role": role, "content": content})

    if len(converted) < 2:
        return None

    return {
        "conversations": converted,
        "source": "sharegpt",
    }


def convert_ultrachat(record: Dict) -> Optional[Dict]:
    """Convert UltraChat format."""
    dialogs = record.get("dialogs", [])
    if not dialogs:
        return None

    converted = []
    for msg in dialogs:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role and content:
            converted.append({"role": role, "content": content})

    if len(converted) < 2:
        return None

    return {
        "conversations": converted,
        "source": "ultrachat",
    }


def convert_to_standard(record: Dict, format_name: str = "standard") -> Optional[Dict]:
    """Convert any format to standard."""
    if format_name == "alpaca":
        return convert_alpaca(record)
    elif format_name == "openassistant":
        return convert_openassistant(record)
    elif format_name == "sharegpt":
        return convert_sharegpt(record)
    elif format_name == "ultrachat":
        return convert_ultrachat(record)
    elif format_name == "standard":
        return {
            "conversations": [
                {"role": "user", "content": record.get("prompt", record.get("input", ""))},
                {"role": "assistant", "content": record.get("response", record.get("output", ""))},
            ],
            "source": "standard",
        }

    return None


def validate_conversation(
    conversations: List[Dict],
    min_turns: int = 1,
    min_length: int = 10,
    max_length: int = 8192,
) -> Dict:
    """Validate conversation quality."""
    if not conversations:
        return {"valid": False, "reason": "Empty conversation"}

    # Check for required roles
    roles = [c.get("role") for c in conversations]
    if "user" not in roles and "assistant" not in roles:
        return {"valid": False, "reason": "Missing user/assistant roles"}

    # Check minimum turns
    user_turns = roles.count("user")
    if user_turns < min_turns:
        return {"valid": False, "reason": f"Too few turns: {user_turns}"}

    # Check length
    total_length = sum(len(c.get("content", "")) for c in conversations)
    if total_length < min_length:
        return {"valid": False, "reason": f"Too short: {total_length}"}

    if total_length > max_length:
        return {"valid": False, "reason": f"Too long: {total_length}"}

    return {"valid": True, "length": total_length}


def format_conversation(
    conversations: List[Dict],
    template: str = "default",
) -> str:
    """Format conversation with template."""
    template = CHAT_TEMPLATES.get(template, CHAT_TEMPLATES["default"])

    # Build messages
    messages = []
    for msg in conversations:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role and content:
            messages.append({"role": role, "content": content})

    if template["system"]:
        system = template["system"]
    else:
        system = messages[0].get("content", "") if messages and messages[0].get("role") == "system" else ""

    # Format output
    output = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            output += template["user"].format(prompt=content) + template["sep"]
        elif role == "assistant":
            output += template["assistant"].format(response=content) + template["sep"]
        elif role == "system":
            continue

    return output.strip()


def apply_chat_template(
    record: Dict,
    template_name: str = "expera",
) -> Optional[Dict]:
    """Apply chat template to conversation."""
    conversations = record.get("conversations", [])
    if not conversations:
        return None

    # Format with template
    formatted = format_conversation(conversations, template_name)

    return {
        "text": formatted,
        "conversations": conversations,
        "source": record.get("source", "unknown"),
    }


def process_jsonl(
    input_path: Path,
    output_path: Path,
    format_name: str = "auto",
    template: str = "expera",
    min_turns: int = 1,
    min_length: int = 10,
    max_length: int = 8192,
) -> Dict:
    """Process JSONL instruction dataset."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "valid": 0, "invalid": 0, "formats": Counter(), "sources": Counter()}

    output_file = output_path / f"{input_path.stem}_processed.jsonl"

    # Track hashes for deduplication
    seen_hashes: Set[str] = set()

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            stats["processed"] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid"] += 1
                continue

            # Detect format if auto
            if format_name == "auto":
                format_name = detect_format(record)

            # Convert format
            converted = convert_to_standard(record, format_name)
            if not converted:
                stats["invalid"] += 1
                continue

            stats["formats"][format_name] += 1
            stats["sources"][converted.get("source", "unknown")] += 1

            # Validate
            conversations = converted.get("conversations", [])
            validation = validate_conversation(conversations, min_turns, min_length, max_length)
            if not validation.get("valid"):
                stats["invalid"] += 1
                continue

            # Check for duplicates
            content_hash = calculate_hash(json.dumps(conversations))
            if content_hash in seen_hashes:
                stats["invalid"] += 1
                continue
            seen_hashes.add(content_hash)

            # Apply chat template
            formatted = apply_chat_template(converted, template)

            output_record = {
                "conversations": conversations,
                "text": formatted.get("text", ""),
                "source": converted.get("source", "unknown"),
                "length": validation.get("length", 0),
                "hash": content_hash,
            }

            outfile.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            stats["valid"] += 1

            if stats["processed"] % 10000 == 0:
                logger.info(f"Processed {stats['processed']}, valid: {stats['valid']}")

    logger.info(f"Completed: {stats}")
    return stats


def merge_datasets(
    input_paths: List[Path],
    output_path: Path,
    weights: Optional[List[float]] = None,
) -> Dict:
    """Merge multiple instruction datasets."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if weights is None:
        weights = [1.0] * len(input_paths)

    output_file = output_path / "merged.jsonl"

    all_records = []

    for input_path in input_paths:
        input_path = Path(input_path)
        if not input_path.exists():
            logger.warning(f"Input not found: {input_path}")
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(line)

    import random
    random.shuffle(all_records)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_records))

    return {
        "total": len(all_records),
        "sources": len(input_paths),
    }


def filter_by_source(
    input_path: Path,
    output_path: Path,
    source: str,
) -> Dict:
    """Filter by data source."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "filtered": 0}

    output_file = output_path / f"{source}.jsonl"

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            try:
                record = json.loads(line)
                if record.get("source") == source:
                    outfile.write(line + "\n")
                    stats["filtered"] += 1
            except json.JSONDecodeError:
                continue

    logger.info(f"Filtered {stats['filtered']}/{stats['total']} for {source}")
    return stats


def split_train_val(
    input_path: Path,
    output_dir: Path,
    val_split: float = 0.01,
) -> Dict:
    """Split into train and validation."""
    import random

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(line)

    random.shuffle(records)

    val_size = int(len(records) * val_split)
    val_records = records[:val_size]
    train_records = records[val_size:]

    train_file = output_dir / "train.jsonl"
    val_file = output_dir / "val.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        f.write("\n".join(train_records))

    with open(val_file, "w", encoding="utf-8") as f:
        f.write("\n".join(val_records))

    return {
        "train": len(train_records),
        "val": len(val_records),
    }


def generate_report(input_path: Path) -> Dict:
    """Generate instruction dataset statistics."""
    stats = {
        "total_samples": 0,
        "sources": Counter(),
        "total_turns": 0,
        "avg_length": 0,
    }
    lengths = []

    for file_path in Path(input_path).glob("*.jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    stats["total_samples"] += 1
                    stats["sources"][record.get("source", "unknown")] += 1

                    conversations = record.get("conversations", [])
                    stats["total_turns"] += len(conversations)

                    text = record.get("text", "") or record.get("conversations", [])
                    if isinstance(text, str):
                        lengths.append(len(text))
                    else:
                        lengths.append(sum(len(c.get("content", "")) for c in text))
                except json.JSONDecodeError:
                    continue

    if lengths:
        stats["avg_length"] = sum(lengths) / len(lengths)
        stats["median_length"] = sorted(lengths)[len(lengths) // 2]

    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare instruction datasets for training")
    parser.add_argument("--input", "-i", type=str, help="Input file or directory")
    parser.add_argument("--output", "-o", type=str, help="Output directory")
    parser.add_argument("--format", type=str, default="auto", help="Input format (auto/alpaca/sharegpt/openassistant)")
    parser.add_argument("--template", type=str, default="expera", help="Chat template")
    parser.add_argument("--min-turns", type=int, default=1, help="Minimum conversation turns")
    parser.add_argument("--min-length", type=int, default=10, help="Minimum total length")
    parser.add_argument("--max-length", type=int, default=8192, help="Maximum total length")
    parser.add_argument("--merge", nargs="+", help="Merge multiple datasets")
    parser.add_argument("--source", type=str, help="Filter by source")
    parser.add_argument("--split", action="store_true", help="Split into train/val")
    parser.add_argument("--val-split", type=float, default=0.01, help="Validation split")
    parser.add_argument("--report", action="store_true", help="Generate report")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.report:
        stats = generate_report(Path(args.input))
        logger.info(f"Dataset stats: {stats}")
        return

    if args.merge:
        # Merge datasets
        input_paths = [Path(p) for p in args.merge]
        merge_stats = merge_datasets(input_paths, Path(args.output))
        logger.info(f"Merged: {merge_stats}")
        return

    if not args.input or not args.output:
        logger.error("Please specify --input and --output")
        return

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.source:
        filter_stats = filter_by_source(input_path, output_path, args.source)
        logger.info(f"Filter: {filter_stats}")
    else:
        stats = process_jsonl(
            input_path,
            output_path,
            format_name=args.format,
            template=args.template,
            min_turns=args.min_turns,
            min_length=args.min_length,
            max_length=args.max_length,
        )
        logger.info(f"Processing complete: {stats}")

    if args.split:
        split_stats = split_train_val(
            output_path / f"{input_path.stem}_processed.jsonl",
            output_path,
            args.val_split,
        )
        logger.info(f"Split: {split_stats}")


if __name__ == "__main__":
    main()