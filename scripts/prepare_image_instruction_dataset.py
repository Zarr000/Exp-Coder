#!/usr/bin/env python3
"""
Image Instruction Dataset Preparation.

Prepares vision-language instruction datasets:
- Converts to multimodal format
- Filters quality
- Generates image prompts
- Creates paired data

Usage:
    python scripts/prepare_image_instruction_dataset.py --input data/raw/llava/ --output data/processed/vision/
    python scripts/prepare_image_instruction_dataset.py --input data/raw/coco/ --format coco --output data/processed/coco/
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


# Image captioning prompts
IMAGE_PROMPTS = {
    "describe": "Describe this image in detail.",
    "caption": "Provide a brief caption for this image.",
    "explain": "Explain what's happening in this image.",
    "analyze": "Analyze the visual elements in this image.",
    "prompt_sd": "Create a prompt for Stable Diffusion based on this image.",
}


def calculate_hash(text: str) -> str:
    """Calculate hash."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def detect_format(record: Dict) -> str:
    """Detect dataset format."""
    # LLaVA format
    if "image" in record and ("conversations" in record or "messages" in record):
        return "llava"

    # COCO format
    if "images" in record:
        return "coco"

    # Caption format
    if "caption" in record:
        return "caption"

    # VQA format
    if "question" in record and "answer" in record:
        return "vqa"

    # Standard ImageText
    if "image_url" in record or "image_path" in record:
        return "imagetext"

    return "unknown"


def convert_llava(record: Dict) -> Optional[Dict]:
    """Convert LLaVA format."""
    conversations = record.get("conversations", record.get("messages", []))
    if not conversations:
        return None

    image = record.get("image", "")
    image_path = record.get("image_path", "")

    if not image and not image_path:
        return None

    # Convert to standard format
    converted = []
    for msg in conversations:
        role = msg.get("from", msg.get("role", ""))
        content = msg.get("value", msg.get("content", ""))

        if role == "gpt" or role == "assistant":
            role = "assistant"
        elif role == "human" or role == "user":
            role = "user"

        if role and content:
            converted.append({"role": role, "content": content})

    if len(converted) < 1:
        return None

    return {
        "conversations": converted,
        "image": image or image_path,
        "source": "llava",
    }


def convert_coco(record: Dict) -> Optional[Dict]:
    """Convert COCO format."""
    images = record.get("images", [])
    if not images:
        return None

    annotations = record.get("annotations", [])

    results = []
    for img in images:
        img_id = img.get("id", img.get("image_id", ""))

        # Find captions for this image
        caps = [a.get("caption", "") for a in annotations if a.get("image_id") == img_id]

        if caps:
            # Use first caption
            results.append({
                "image": img.get("file_name", ""),
                "captions": caps,
                "conversations": [
                    {"role": "user", "content": IMAGE_PROMPTS["describe"]},
                    {"role": "assistant", "content": caps[0]},
                ],
                "source": "coco",
            })

    return results if results else None


def convert_caption(record: Dict) -> Optional[Dict]:
    """Convert caption format."""
    caption = record.get("caption", "")
    image = record.get("image", record.get("image_url", record.get("image_path", ""))))

    if not caption or not image:
        return None

    return {
        "conversations": [
            {"role": "user", "content": IMAGE_PROMPTS["describe"]},
            {"role": "assistant", "content": caption},
        ],
        "image": image,
        "source": "caption",
    }


def convert_vqa(record: Dict) -> Optional[Dict]:
    """Convert VQA format."""
    question = record.get("question", "")
    answer = record.get("answer", record.get("answers", []))
    image = record.get("image", record.get("image_url", ""))

    if not question or not answer:
        return None

    # Handle multiple answers
    if isinstance(answer, list):
        # Use majority answer or first
        answer = answer[0] if answer else ""

    if not answer or not image:
        return None

    return {
        "conversations": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "image": image,
        "source": "vqa",
    }


def convert_standard(record: Dict) -> Optional[Dict]:
    """Convert standard format."""
    image = record.get("image", record.get("image_url", record.get("image_path", "")))
    text = record.get("text", record.get("prompt", "")))

    if not image or not text:
        return None

    return {
        "conversations": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": record.get("response", "")},
        ],
        "image": image,
        "source": "standard",
    }


def generate_sd_prompt(conversations: List[Dict]) -> Optional[str]:
    """Generate Stable Diffusion prompt from description."""
    # Find description from conversations
    for msg in conversations:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            # Extract key elements for SD
            content = content.strip()
            if len(content) > 10:
                return f"{content}, high quality, detailed, 4k"

    return None


def process_image_url(image_url: str) -> Dict:
    """Process image URL/path."""
    url = image_url.strip()

    # Determine if local or remote
    if url.startswith("http://") or url.startswith("https://"):
        return {"type": "url", "url": url}
    elif url.startswith("s3://"):
        return {"type": "s3", "path": url}
    elif Path(url).exists():
        return {"type": "local", "path": str(Path(url).absolute())}
    else:
        return {"type": "unknown", "path": url}


def validate_conversation(
    conversations: List[Dict],
    min_length: int = 5,
    max_length: int = 2048,
) -> Dict:
    """Validate conversation."""
    if not conversations:
        return {"valid": False, "reason": "Empty conversation"}

    # Check required roles
    roles = [c.get("role") for c in conversations]
    if not any(r in roles for r in ["user", "assistant"]):
        return {"valid": False, "reason": "Missing roles"}

    # Check length
    total_length = sum(len(c.get("content", "")) for c in conversations)
    if total_length < min_length:
        return {"valid": False, "reason": f"Too short: {total_length}"}

    if total_length > max_length:
        return {"valid": False, "reason": f"Too long: {total_length}"}

    return {"valid": True, "length": total_length}


def process_jsonl(
    input_path: Path,
    output_path: Path,
    format_name: str = "auto",
    min_length: int = 5,
    max_length: int = 2048,
) -> Dict:
    """Process JSONL vision dataset."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "valid": 0, "invalid": 0, "formats": Counter()}

    output_file = output_path / f"{input_path.stem}_processed.jsonl"

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

            # Detect format
            if format_name == "auto":
                format_name = detect_format(record)

            # Convert format
            if format_name == "llava":
                converted = convert_llava(record)
            elif format_name == "caption":
                converted = convert_caption(record)
            elif format_name == "vqa":
                converted = convert_vqa(record)
            elif format_name == "imagetext":
                converted = convert_standard(record)
            else:
                converted = convert_standard(record)

            if not converted:
                stats["invalid"] += 1
                continue

            stats["formats"][format_name] += 1

            # Validate
            conversations = converted.get("conversations", [])
            validation = validate_conversation(conversations, min_length, max_length)
            if not validation.get("valid"):
                stats["invalid"] += 1
                continue

            # Process image reference
            image_ref = converted.get("image", "")
            image_info = process_image_url(image_ref)

            # Generate SD prompt
            sd_prompt = generate_sd_prompt(conversations)

            output_record = {
                "conversations": conversations,
                "image": image_ref,
                "image_type": image_info.get("type"),
                "sd_prompt": sd_prompt,
                "source": converted.get("source", "unknown"),
                "length": validation.get("length", 0),
                "hash": calculate_hash(json.dumps(conversations)),
            }

            outfile.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            stats["valid"] += 1

            if stats["processed"] % 5000 == 0:
                logger.info(f"Processed {stats['processed']}, valid: {stats['valid']}")

    logger.info(f"Completed: {stats}")
    return stats


def process_directory(
    input_dir: Path,
    output_dir: Path,
    image_extensions: Set[str] = {".jpg", ".jpeg", ".png", ".gif", ".webp"},
) -> Dict:
    """Process directory of images with captions."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0}

    output_file = output_dir / "images_processed.jsonl"

    # Find all images
    all_files = list(input_dir.rglob("*"))
    images = [f for f in all_files if f.suffix.lower() in image_extensions]

    # Look for caption files
    caption_files = {}
    for cap_file in input_dir.rglob("*.json") + input_dir.rglob("*.txt"):
        caption_files[cap_file.stem] = cap_file

    with open(output_file, "w", encoding="utf-8") as outfile:
        for img_path in images:
            stats["images"] += 1

            # Look for caption
            caption = ""
            if img_path.stem in caption_files:
                cap_path = caption_files[img_path.stem]
                if cap_path.suffix == ".json":
                    with open(cap_path, "r") as f:
                        data = json.load(f)
                        caption = data.get("caption", data.get("description", ""))
                else:
                    caption = cap_path.read_text()

            if not caption:
                continue

            record = {
                "conversations": [
                    {"role": "user", "content": IMAGE_PROMPTS["describe"]},
                    {"role": "assistant", "content": caption},
                ],
                "image": str(img_path.relative_to(input_dir)),
                "image_type": "local",
                "source": "directory",
                "hash": calculate_hash(str(img_path)),
            }

            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Completed: {stats}")
    return stats


def merge_vision_datasets(
    input_paths: List[Path],
    output_path: Path,
) -> Dict:
    """Merge vision datasets."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    all_records = []

    for input_path in input_paths:
        input_path = Path(input_path)
        if not input_path.exists():
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(line)

    import random
    random.shuffle(all_records)

    output_file = output_path / "merged.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_records))

    return {"total": len(all_records), "datasets": len(input_paths)}


def generate_report(input_path: Path) -> Dict:
    """Generate vision dataset statistics."""
    stats = {
        "total_samples": 0,
        "sources": Counter(),
        "image_types": Counter(),
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
                    stats["image_types"][record.get("image_type", "unknown")] += 1

                    conversations = record.get("conversations", [])
                    lengths.append(sum(len(c.get("content", "")) for c in conversations))
                except json.JSONDecodeError:
                    continue

    if lengths:
        stats["avg_length"] = sum(lengths) / len(lengths)

    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare vision instruction datasets")
    parser.add_argument("--input", "-i", type=str, help="Input file or directory")
    parser.add_argument("--output", "-o", type=str, help="Output directory")
    parser.add_argument("--format", type=str, default="auto", help="Input format")
    parser.add_argument("--min-length", type=int, default=5, help="Minimum length")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum length")
    parser.add_argument("--merge", nargs="+", help="Merge datasets")
    parser.add_argument("--report", action="store_true", help="Generate report")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.report:
        stats = generate_report(Path(args.input))
        logger.info(f"Dataset stats: {stats}")
        return

    if args.merge:
        input_paths = [Path(p) for p in args.merge]
        merge_stats = merge_vision_datasets(input_paths, Path(args.output))
        logger.info(f"Merged: {merge_stats}")
        return

    if not args.input or not args.output:
        logger.error("Please specify --input and --output")
        return

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        stats = process_directory(input_path, output_path)
    else:
        stats = process_jsonl(
            input_path,
            output_path,
            format_name=args.format,
            min_length=args.min_length,
            max_length=args.max_length,
        )

    logger.info(f"Processing complete: {stats}")


if __name__ == "__main__":
    main()