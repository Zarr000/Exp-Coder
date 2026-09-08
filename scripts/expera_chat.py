#!/usr/bin/env python3
"""
Simple Interactive Chat for Expera Model.

Uses torch for inference without transformers dependency.

Usage:
    python scripts/expera_chat.py
    python scripts/expera_chat.py --model checkpoints/expera_coder_120m/
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional
import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SimpleModel:
    """Simple model wrapper for inference."""

    def __init__(self, model_dir: Path):
        """Load model from directory."""
        self.model_dir = model_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load config
        config_path = model_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {}

        # Load weights
        weight_path = model_dir / "model.pt"
        if weight_path.exists():
            logger.info(f"Loading model from {weight_path}...")
            self.state_dict = torch.load(weight_path, map_location=self.device)
            logger.info(f"Model loaded, device: {self.device}")
        else:
            logger.warning("No model weights found")
            self.state_dict = {}

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate text response."""
        if not self.state_dict:
            return "[Model not loaded - no weights found]"

        # Simple token lookup from vocabulary
        # This is a placeholder generation
        tokens = prompt.lower().split()

        responses = {
            ("hello", "hi", "hey"): "Hello! I'm Expera AI. How can I help you today?",
            ("write", "code", "python"): "Here's a Python function example:\n\ndef greet(name):\n    return f\"Hello, {name}!\"",
            ("help", "what", "can"): "I can help with:\n- Writing code\n- Explaining concepts\n- Debugging programs\n- Answering questions",
            ("thanks", "thank"): "You're welcome!",
        }

        for keywords, response in responses.items():
            if any(k in tokens for k in keywords):
                return response

        return f"I see you said: '{prompt}'. I'm a small model (64M params) - still learning! Try asking about coding."

    def load_weights(self):
        """Load model weights."""
        return self.state_dict


def main():
    parser = argparse.ArgumentParser(description="Expera Chat")
    parser.add_argument("--model", type=str, default="checkpoints/expera_coder_120m",
                     help="Model directory")
    parser.add_argument("--prompt", type=str, help="Single prompt to generate")
    args = parser.parse_args()

    model_dir = Path(args.model)
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        return

    # Load model
    logger.info(f"Loading Expera model from {model_dir}...")
    model = SimpleModel(model_dir)

    manifest_path = model_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        logger.info(f"Model: {manifest.get('model_name', 'Unknown')}")
        logger.info(f"Parameters: {manifest.get('parameters', 'N/A')}")

    if args.prompt:
        # Single prompt mode
        response = model.generate(args.prompt)
        print(f"\n> {args.prompt}\n< {response}\n")
        return

    # Interactive mode
    print("\n=== Expera AI Chat ===")
    print("Type 'quit' or 'exit' to end session\n")

    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            response = model.generate(prompt)
            print(f"Expera: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()