#!/usr/bin/env python3
"""
Model Export and Conversion.

Exports model for release:
- safetensors format
- ONNX conversion
- Quantization
- Tile format for inference

Usage:
    python scripts/export_model.py --input checkpoints/expera-small --output release/expera-coder-120m
    python scripts/export_model.py --input checkpoints/expera-tiny --quantize int8 --output release/expera-tiny-33m
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def export_safetensors(
    input_path: Path,
    output_path: Path,
) -> None:
    """Export model in safetensors format."""
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Exporting {input_path} to safetensors")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(str(input_path))
    tokenizer = AutoTokenizer.from_pretrained(str(input_path))

    # Save model weights
    output_path.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    save_file(state_dict, str(output_path / "model.safetensors"))

    # Save config
    config = model.config.to_dict()
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save tokenizer
    tokenizer.save_pretrained(str(output_path))

    logger.info(f"Exported to {output_path}")


def export_onnx(
    input_path: Path,
    output_path: Path,
    opset: int = 14,
) -> None:
    """Export model to ONNX format."""
    import torch.onnx

    from transformers import AutoModelForCausalLM

    logger.info(f"Exporting {input_path} to ONNX")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(str(input_path))
    model.eval()

    # Create dummy input
    dummy_input = torch.randint(1, 1000, (1, 128))

    # Export
    output_path.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path / "model.onnx"),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=opset,
    )

    logger.info(f"Exported to {output_path}")


def quantize_model(
    input_path: Path,
    output_path: Path,
    quantization: str = "int8",
) -> None:
    """Quantize model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Quantizing {input_path} to {quantization}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        str(input_path),
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(input_path))

    output_path.mkdir(parents=True, exist_ok=True)

    if quantization == "int8":
        # Dynamic quantization
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
    elif quantization == "int4":
        # For int4, use bitsandbytes
        try:
            from bitsandbytes import quantize_fp4

            model = quantize_fp4(model)
        except ImportError:
            logger.warning("bitsandbytes not installed, using int8 instead")
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )

    # Save
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    logger.info(f"Quantized model saved to {output_path}")


def create_tile_model(
    input_path: Path,
    output_path: Path,
    tile_size: int = 512,
) -> None:
    """Create tile model for fast inference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Creating tile model from {input_path}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(str(input_path))
    tokenizer = AutoTokenizer.from_pretrained(str(input_path))

    output_path.mkdir(parents=True, exist_ok=True)

    # Export to tile format
    try:
        from tile import export_model
        export_model(model, tokenizer, str(output_path), tile_size)
    except ImportError:
        logger.warning("tile not installed, saving HF format")

    # Save normally
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Save metadata
    metadata = {
        "model_type": "llama",
        "tile_size": tile_size,
        "input_path": str(input_path),
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Tile model saved to {output_path}")


def export_gguf(
    input_path: Path,
    output_path: Path,
    quantization: str = "q4_0",
) -> None:
    """Export to GGUF format for llama.cpp."""
    logger.info(f"Exporting to GGUF format ({quantization})")

    output_path.mkdir(parents=True, exist_ok=True)

    # This would use llama.cpp to convert - placeholder
    # Real implementation would call llama.cpp convert.py

    logger.info(f"GGUF model would be saved to {output_path}")


def create_model_card(
    output_path: Path,
    model_info: Dict,
) -> None:
    """Create model cardREADME.md."""
    card = f"""# {model_info.get('name', 'Expera Model')}

{model_info.get('description', 'Expera AI Model')}

## Model Details

- **Parameters**: {model_info.get('parameters', 'Unknown')}
- **Architecture**: {model_info.get('architecture', 'LLaMA')}
- **Context Length**: {model_info.get('context_length', 2048')}
- **Vocabulary Size**: {model_info.get('vocab_size', 32000')}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{output_path}")
tokenizer = AutoTokenizer.from_pretrained("{output_path}")
```

## Quantization

Available quantization formats:
- f16 (default)
- int8
- int4

## License

{model_info.get('license', 'Apache 2.0')}
"""

    with open(output_path / "README.md", "w") as f:
        f.write(card)

    logger.info(f"Created model card at {output_path}/README.md")


def parse_args():
    parser = argparse.ArgumentParser(description="Export model")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input model path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--format", choices=["safetensors", "onnx", "gguf", "tile"], default="safetensors", help="Export format")
    parser.add_argument("--quantize", choices=["int8", "int4", "f16"], help="Quantization")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--tile_size", type=int, default=512, help="Tile size")
    parser.add_argument("--model_name", type=str, help="Model name for card")
    parser.add_argument("--description", type=str, help="Model description")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.format == "safetensors":
        export_safetensors(input_path, output_path)
    elif args.format == "onnx":
        export_onnx(input_path, output_path, args.opset)
    elif args.format == "tile":
        create_tile_model(input_path, output_path, args.tile_size)
    elif args.format == "gguf":
        export_gguf(input_path, output_path, args.quantize or "q4_0")

    if args.quantize:
        quant_path = output_path.parent / f"{output_path.name}_{args.quantize}"
        quantize_model(input_path, quant_path, args.quantize)

    # Create model card
    model_info = {
        "name": args.model_name or output_path.name,
        "description": args.description or "Expera AI Model",
    }
    create_model_card(output_path, model_info)


if __name__ == "__main__":
    main()