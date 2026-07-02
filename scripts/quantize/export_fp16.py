"""
Export FP16 Quantization.

Converts model to FP16 for reduced memory.

Usage:
    python scripts/quantize/export_fp16.py --model checkpoints/expera-350m --output checkpoints/expera-350m-fp16
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def export_fp16(
    model_path: str,
    output_path: str,
) -> None:
    """Export model to FP16."""
    logger.info(f"Loading model from {model_path}")

    # Load in fp16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # Save
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Calculate size
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 2 / 1_000_000

    logger.info(f"Model saved: {total_params:,} parameters, {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export FP16 model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    export_fp16(args.model, args.output)


if __name__ == "__main__":
    main()