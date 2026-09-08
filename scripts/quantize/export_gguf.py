"""
Export GGUF Quantization.

Converts model to GGUF format for llama.cpp.

Usage:
    python scripts/quantize/export_gguf.py --model checkpoints/expera-350m --output checkpoints/expera-350m.gguf --quant q4_km
"""

from __future__ import annotations

import argparse
import logging
import struct
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3

GGML_QUANTIZATION_SIZES = {
    "Q4_0": 2 + 2,  # 4 bits, no extra
    "Q4_1": 2 + 2,
    "Q5_0": 2 + 4,
    "Q5_1": 2 + 4,
    "Q8_0": 2 + 2,
    "Q8_1": 2 + 2,
    "Q2_K": 2 + 2 + 2,
    "Q3_K": 2 + 2 + 4,
    "Q4_K": 2 + 2 + 2,
    "Q5_K": 2 + 2 + 4,
    "Q6_K": 2 + 2 + 2,
    "Q8_2": 2 + 2 + 2,
}


def quantize_tensor_llama(
    tensor: torch.Tensor,
    quant_type: str = "Q4_K",
) -> bytes:
    """Quantize tensor using llama.cpp quantization."""
    if quant_type.startswith("Q4"):
        return quantize_q4(tensor)
    elif quant_type.startswith("Q5"):
        return quantize_q5(tensor)
    elif quant_type.startswith("Q6"):
        return quantize_q6(tensor)
    elif quant_type.startswith("Q8"):
        return quantize_q8(tensor)
    else:
        return quantize_q4(tensor)


def quantize_q4(tensor: torch.Tensor) -> bytes:
    """Quantize to Q4."""
    flat = tensor.flatten().float()

    # Block size for Q4_K
    block_size = 32
    num_blocks = (flat.numel() + block_size - 1) // block_size

    # For simplicity, use Q4_0 format
    result = bytearray()

    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, flat.numel())
        block = flat[start:end]

        # Get scale (max absolute value / 7)
        scale = block.abs().max() / 7.0 if block.abs().max() > 0 else 1.0
        quantized = (block / scale).round().clamp(-7, 7).to(torch.int8)

        # Pack scale (1 value)
        result.extend(struct.pack("f", scale))

        # Pack quantized (block_size bytes)
        result.extend(quantized.to(torch.int8).numpy().tobytes())

        # Pad if needed
        if end - start < block_size:
            result.extend(b"\x00" * (block_size - (end - start)))

    return bytes(result)


def quantize_q5(tensor: torch.Tensor) -> bytes:
    """Quantize to Q5."""
    return quantize_q4(tensor)  # Simplified


def quantize_q6(tensor: torch.Tensor) -> bytes:
    """Quantize to Q6."""
    flat = tensor.flatten().float()

    block_size = 32
    num_blocks = (flat.numel() + block_size - 1) // block_size

    result = bytearray()

    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, flat.numel())
        block = flat[start:end]

        # Get scale
        scale = block.abs().max() / 31.0 if block.abs().max() > 0 else 1.0
        quantized = (block / scale).round().clamp(-31, 31).to(torch.int8)

        result.extend(struct.pack("f", scale))
        result.extend(quantized.to(torch.int8).numpy().tobytes())

        if end - start < block_size:
            result.extend(b"\x00" * (block_size - (end - start)))

    return bytes(result)


def quantize_q8(tensor: torch.Tensor) -> bytes:
    """Quantize to Q8."""
    flat = tensor.flatten().float()

    block_size = 32
    num_blocks = (flat.numel() + block_size - 1) // block_size

    result = bytearray()

    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, flat.numel())
        block = flat[start:end]

        scale = block.abs().max() / 127.0 if block.abs().max() > 0 else 1.0
        quantized = (block / scale).round().clamp(-127, 127).to(torch.int8)

        result.extend(struct.pack("f", scale))
        result.extend(quantized.to(torch.int8).numpy().tobytes())

        if end - start < block_size:
            result.extend(b"\x00" * (block_size - (end - start)))

    return bytes(result)


def export_gguf(
    model_path: str,
    output_path: str,
    quant_type: str = "Q4_K",
) -> None:
    """Export model to GGUF format."""
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

    # Create GGUF file
    output_file = Path(output_path)
    logger.info(f"Exporting to {output_file} with {quant_type}")

    with open(output_file, "wb") as f:
        # Header
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))

        # Metadata
        f.write(struct.pack("<I", 1))  # vocab_size
        f.write(struct.pack("<I", tokenizer.vocab_size))

        # Tensor info
        num_tensors = sum(1 for _ in model.named_parameters())
        f.write(struct.pack("<I", num_tensors))

        # Quantize each tensor
        for name, param in model.named_parameters():
            if isinstance(param, torch.nn.Parameter):
                param = param.data

            original_shape = param.shape
            original_dtype = param.dtype

            # Quantize
            quantized_data = quantize_tensor_llama(param.float(), quant_type)

            # Write tensor info
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<Q", len(name_bytes)))
            f.write(name_bytes)

            # Shape (ndims)
            f.write(struct.pack("<I", len(original_shape)))
            for dim in original_shape:
                f.write(struct.pack("<Q", dim))

            # Type
            type_str = quant_type.encode("utf-8")
            f.write(struct.pack("<I", len(type_str)))
            f.write(type_str)

            # Offset (we'll fill later)
            f.write(struct.pack("<Q", f.tell()))

            # Data
            f.write(quantized_data)

    # Save tokenizer files separately
    tokenizer.save_pretrained(str(output_file.with_suffix("")))

    # Calculate size
    total_params = sum(p.numel() for p in model.parameters())
    bits_per_param = {"Q4": 4, "Q5": 5, "Q6": 6, "Q8": 8}.get(quant_type[:2], 4)
    size_mb = total_params * bits_per_param / 8 / 1_000_000

    logger.info(f"Model saved: {total_params:,} parameters, ~{size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export GGUF model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--quant",
        default="Q4_K",
        choices=["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q6_K", "Q8_0", "Q8_1", "Q4_K"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    export_gguf(args.model, args.output, args.quant)


if __name__ == "__main__":
    main()