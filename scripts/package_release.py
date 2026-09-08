#!/usr/bin/env python3
"""
Release Packaging.

Creates release packages:
- Model weights
- Config files
- Tokenizer
- Documentation

Usage:
    python scripts/package_release.py --input release/expera-coder-120m --output release/Expera-Coder-120M
    python scripts/package_release.py --input release/expera-small --version 1.0.0 --output release/
"""

import argparse
import json
import logging
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_model(path: Path) -> bool:
    """Validate model directory."""
    required = ["config.json", "tokenizer.json"]

    for f in required:
        if not (path / f).exists():
            logger.warning(f"Missing required file: {f}")
            return False

    return True


def create_release_package(
    input_path: Path,
    output_dir: Path,
    name: str,
    version: str = "1.0.0",
    include_quantized: bool = True,
) -> Path:
    """Create release package."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Package name
    package_name = f"{name}-v{version}"
    package_path = output_dir / package_name

    logger.info(f"Creating release: {package_name}")

    # Copy model files
    package_path.mkdir(parents=True, exist_ok=True)

    for item in input_path.iterdir():
        # Skip large files initially
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            if size_mb > 500 and "quantized" not in item.name.lower():
                logger.info(f"Skipping large file: {item.name} ({size_mb:.1f}MB)")
                continue

        dest = package_path / item.name
        shutil.copy2(item, dest)
        logger.info(f"Copied: {item.name}")

    # Create CHANGELOG
    changelog = f"""# Changelog

## v{version} ({version})

- Initial release
"""

    with open(package_path / "CHANGELOG.md", "w") as f:
        f.write(changelog)

    # Create manifest
    manifest = {
        "name": name,
        "version": version,
        "model_path": str(input_path),
        "files": [f.name for f in package_path.iterdir() if f.is_file()],
    }

    with open(package_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Create tarball
    tar_path = output_dir / f"{package_name}.tar.gz"

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(package_path, arcname=package_name)

    logger.info(f"Created: {tar_path}")

    return tar_path


def create_docker_image(
    model_path: Path,
    output_dir: Path,
    tag: str = "latest",
) -> str:
    """Create Docker image definition."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dockerfile = f"""FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "-m", "src.server.app"]
"""

    dockerfile_path = output_dir / "Dockerfile"
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile)

    return str(dockerfile_path)


def create_api_spec(
    model_path: Path,
    output_path: Path,
) -> None:
    """Create OpenAPI spec for inference."""
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Expera AI API",
            "version": "1.0.0",
            "description": "Inference API for Expera AI models",
        },
        "servers": [
            {"url": "http://localhost:8000", "description": "Local development"},
        ],
        "paths": {
            "/v1/chat/completions": {
                "post": {
                    "summary": "Chat completion",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "messages": {"type": "array"},
                                        "model": {"type": "string"},
                                        "max_tokens": {"type": "integer"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "/v1/completions": {
                "post": {
                    "summary": "Text completion",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "prompt": {"type": "string"},
                                        "max_tokens": {"type": "integer"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    }

    with open(output_path / "api_spec.json", "w") as f:
        json.dump(spec, f, indent=2)

    logger.info(f"Created API spec at {output_path}")


def create_aws_lambda_package(
    model_path: Path,
    output_dir: Path,
) -> None:
    """Create AWS Lambda deployment package."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lambda handler
    handler = """import json
import boto3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "model"

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def handler(event, context):
    global model, tokenizer

    if model is None:
        load_model()

    prompt = event.get("prompt", "")
    max_tokens = event.get("max_tokens", 512)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": response}
"""

    with open(output_dir / "handler.py", "w") as f:
        f.write(handler)

    # Requirements
    requirements = """transformers>=4.30.0
torch>=2.0.0
"""

    with open(output_dir / "requirements.txt", "w") as f:
        f.write(requirements)

    logger.info(f"Created Lambda package at {output_dir}")


def create_huggingface_card(
    model_path: Path,
    output_path: Path,
    model_name: str,
    license: str = "apache-2.0",
    language: str = "en",
    license_link: str = "https://choosealicense.com/licenses/apache-2.0/",
) -> None:
    """Create HuggingFace model card."""
    card = f"""---
license: {license}
license_link: {license_link}
language: {language}
---
# {model_name}

Expera AI model for code and text generation.

## Model Details

- **Architecture**: LLaMA-based
- **Task**: Causal language modeling
- **Fine-tuned for**: Code generation

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")
```
"""

    with open(output_path / "README.md", "w") as f:
        f.write(card)

    logger.info(f"Created model card at {output_path}/README.md")


def parse_args():
    parser = argparse.ArgumentParser(description="Create release package")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input model path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--name", type=str, help="Package name")
    parser.add_argument("--version", type=str, default="1.0.0", help="Version")
    parser.add_argument("--include_quantized", action="store_true", help="Include quantized versions")
    parser.add_argument("--format", choices=["tar", "zip", "docker", "lambda", "hf"], default="tar", help="Package format")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if args.format == "tar":
        name = args.name or input_path.name
        package = create_release_package(
            input_path,
            output_dir,
            name,
            args.version,
            args.include_quantized,
        )
        logger.info(f"Package: {package}")

    elif args.format == "docker":
        create_docker_image(input_path, output_dir)

    elif args.format == "lambda":
        create_aws_lambda_package(input_path, output_dir)

    elif args.format == "hf":
        create_huggingface_card(input_path, output_dir, args.name or input_path.name)


if __name__ == "__main__":
    main()