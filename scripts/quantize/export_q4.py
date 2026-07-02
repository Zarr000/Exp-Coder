"""
Export Q4 Quantization.

Converts model to Q4 for reduced memory.

Usage:
    python scripts/quantize/export_q4.py --model checkpoints/expera-350m --output checkpoints/expera-350m-q4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class Q4Quantizer:
    """Quantize to Q4."""

    BITS = 4
    GROUP_SIZE = 128

    def quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to Q4."""
        # Flatten for quantization
        original_shape = tensor.shape
        flat = tensor.flatten()

        # Reshape for groups
        num_groups = (flat.numel() + self.GROUP_SIZE - 1) // self.GROUP_SIZE
        padded = torch.zeros(num_groups * self.GROUP_SIZE)
        padded[: flat.numel()] = flat

        # Get scales
        reshaped = padded.view(num_groups, self.GROUP_SIZE)
        scales = reshaped.abs().max(dim=1).values / ((2**self.BITS - 1) / 2)

        # Quantize
        quantized = (reshaped / scales.unsqueeze(1)).round().to(torch.int8)

        # Store quantized + scales
        return quantized.view(-1)

    def dequantize_tensor(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        shape: torch.Size,
    ) -> torch.Tensor:
        """Dequantize Q4 tensor."""
        flat = quantized.flatten()[: scales.numel() * self.GROUP_SIZE]
        reshaped = flat.view(-1, self.GROUP_SIZE)
        dequantized = (reshaped.float() * scales.unsqueeze(1)).flatten()
        return dequantized[: shape.numel()].view(shape)


def export_q4(
    model_path: str,
    output_path: str,
) -> None:
    """Export model to Q4."""
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

    # Quantize
    logger.info("Applying Q4 quantization")
    quantizer = Q4Quantizer()

    # Save quantized state
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = {}
    for name, param in model.state_dict().items():
        if isinstance(param, torch.nn.Parameter):
            param = param.data

        if param.dtype in (torch.float32, torch.float16):
            # Quantize
            quantized = quantizer.quantize_tensor(param.float())
            state_dict[name] = quantized

            # Save scales separately
            scales_name = f"{name}.scale"
            scales = param.flatten().view(-1, quantizer.GROUP_SIZE).abs().max(dim=1).values
            scales = scales / ((2**quantizer.BITS - 1) / 2)
            state_dict[scales_name] = scales

    # Save
    logger.info(f"Saving to {output_path}")
    torch.save(state_dict, output_dir / "quantized.pt")

    # Save config
    config = {
        "quantization": "q4",
        "bits": quantizer.BITS,
        "group_size": quantizer.GROUP_SIZE,
    }
    with open(output_dir / "quantize_config.json", "w") as f:
        json.dump(config, f)

    tokenizer.save_pretrained(output_path)

    # Calculate size
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 0.5 / 1_000_000  # 4 bits = 0.5 bytes

    logger.info(f"Model saved: {total_params:,} parameters, ~{size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export Q4 model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    export_q4(args.model, args.output)


if __name__ == "__main__":
    main()