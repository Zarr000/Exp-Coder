"""
Export INT8 Quantization.

Converts model to INT8 for reduced memory.

Usage:
    python scripts/quantize/export_int8.py --model checkpoints/expera-350m --output checkpoints/expera-350m-int8
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def export_int8(
    model_path: str,
    output_path: str,
    threshold: float = 6.0,
) -> None:
    """Export model to INT8 using dynamic quantization."""
    logger.info(f"Loading model from {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # Apply dynamic quantization
    logger.info("Applying INT8 quantization")

    # Quantize linear layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if module.out_features >= threshold:
                module.weight.data = torch.quantize_per_channel(
                    module.weight.data,
                    scales=torch.ones(module.out_features),
                    zero_points=torch.zeros(module.out_features, dtype=torch.int8),
                    dtype=torch.quint8,
                ).dequantize()

    # Save
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Calculate size
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 1 / 1_000_000

    logger.info(f"Model saved: {total_params:,} parameters, ~{size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export INT8 model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=6.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    export_int8(args.model, args.output, args.threshold)


if __name__ == "__main__":
    main()