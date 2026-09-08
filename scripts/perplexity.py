#!/usr/bin/env python3
"""
Perplexity Evaluation.

Measures model perplexity on text/code datasets.

Usage:
    python scripts/perplexity.py --model checkpoints/expera-small/ --input data/processed/code/val.jsonl
    python scripts/perplexity.py --model checkpoints/expera-tiny/ --input data/processed/text/ --batch 32
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_path: Path, device: str = "auto"):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # Setup tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()

    return model, tokenizer, device


def calculate_perplexity(
    model,
    tokenizer,
    text: str,
    device: str = "cpu",
    max_length: int = 2048,
) -> float:
    """Calculate perplexity for a single text."""
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    perplexity = math.exp(loss.item())
    return perplexity


def calculate_batch_perplexity(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cpu",
    max_length: int = 2048,
    batch_size: int = 8,
) -> List[float]:
    """Calculate perplexity for batch of texts."""
    model.eval()
    perplexities = []

    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

        ppl = math.exp(loss.item())
        perplexities.append(ppl)

    return perplexities


def evaluate_jsonl(
    model,
    tokenizer,
    input_path: Path,
    device: str = "cpu",
    max_length: int = 2048,
    batch_size: int = 8,
    text_field: str = "text",
) -> Dict:
    """Evaluate perplexity on JSONL file."""
    input_path = Path(input_path)

    texts = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                text = record.get(text_field, record.get("content", ""))
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                continue

    logger.info(f"Evaluating {len(texts)} texts")

    perplexities = calculate_batch_perplexity(
        model, tokenizer, texts, device, max_length, batch_size
    )

    avg_ppl = sum(perplexities) / len(perplexities) if perplexities else 0
    min_ppl = min(perplexities) if perplexities else 0
    max_ppl = max(perplexities) if perplexities else 0

    return {
        "perplexity": avg_ppl,
        "perplexity_std": (sum((p - avg_ppl) ** 2 for p in perplexities) / len(perplexities)) ** 0.5,
        "min_perplexity": min_ppl,
        "max_perplexity": max_ppl,
        "num_samples": len(texts),
    }


def evaluate_by_language(
    model,
    tokenizer,
    input_path: Path,
    device: str = "cpu",
    max_length: int = 2048,
) -> Dict:
    """Evaluate perplexity by language."""
    input_path = Path(input_path)

    by_language: Dict[str, List[float]] = {}

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                text = record.get("text", record.get("content", ""))
                if not text:
                    continue

                lang = record.get("language", "unknown")

                ppl = calculate_perplexity(model, tokenizer, text, device, max_length)

                if lang not in by_language:
                    by_language[lang] = []

                by_language[lang].append(ppl)

            except json.JSONDecodeError:
                continue

    results = {}
    for lang, ppls in by_language.items():
        avg = sum(ppls) / len(ppls)
        results[lang] = {
            "perplexity": avg,
            "count": len(ppls),
        }

    return results


def compare_models(
    model_paths: List[Path],
    input_path: Path,
    device: str = "cpu",
) -> Dict:
    """Compare perplexity across models."""
    results = {}

    for model_path in model_paths:
        model, tokenizer, _ = load_model(model_path, device)

        ppl_result = evaluate_jsonl(
            model, tokenizer, input_path, device
        )

        results[str(model_path)] = ppl_result

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate perplexity")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model path")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--text-field", type=str, default="text", help="Text field name")
    parser.add_argument("--by-language", action="store_true", help="Group by language")
    parser.add_argument("--compare", nargs="+", help="Compare multiple models")
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = Path(args.model)
    input_path = Path(args.input)

    if args.compare:
        results = compare_models(
            [Path(p) for p in args.compare],
            input_path,
            device,
        )
    else:
        model, tokenizer, _ = load_model(model_path, device)

        if args.by_language:
            results = evaluate_by_language(
                model, tokenizer, input_path, device, args.max_length
            )
        else:
            results = evaluate_jsonl(
                model, tokenizer, input_path, device, args.max_length, args.batch_size, args.text_field
            )

    logger.info(f"Results: {results}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results written to {args.output}")


if __name__ == "__main__":
    main()