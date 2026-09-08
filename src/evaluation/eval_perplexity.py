"""
Perplexity Evaluation.

Measures model perplexity on text.

Usage:
    python -m src.evaluation.eval_perplexity --model checkpoints/expera-350m --data data/test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class PerplexityResult:
    """Perplexity result."""

    perplexity: float
    loss: float
    num_tokens: int
    num_batches: int


class PerplexityEvaluator:
    """Evaluates model perplexity."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 8,
    ):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None

    async def load(self) -> None:
        """Load model and tokenizer."""
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            logger.warning("Using CPU")

        logger.info(f"Loading model from {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        self.model.eval()

    async def compute_perplexity(
        self,
        texts: list[str],
        stride: int = 512,
    ) -> PerplexityResult:
        """Compute perplexity on texts."""
        await self.load()

        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        for text in texts:
            encodings = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.model.config.max_position_embeddings,
            )

            input_ids = encodings.input_ids.to(self.device)
            labels = input_ids.clone()

            # Compute loss with sliding window
            seq_len = input_ids.shape[1]
            num_chunks = max(1, (seq_len - stride) // stride)

            for i in range(num_chunks):
                start = i * stride
                end = min(start + stride + 1, seq_len)

                chunk_input = input_ids[:, start:end]
                chunk_labels = labels[:, start:end]

                with torch.no_grad():
                    outputs = self.model(chunk_input)
                    logits = outputs.logits

                    # Shift for next-token prediction
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = chunk_labels[..., 1:].contiguous()

                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction="mean",
                    )

                total_loss += loss.item()
                total_tokens += end - start - 1
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

        return PerplexityResult(
            perplexity=perplexity,
            loss=avg_loss,
            num_tokens=total_tokens,
            num_batches=num_batches,
        )

    async def evaluate_file(
        self,
        file_path: Path,
    ) -> PerplexityResult:
        """Evaluate on file."""
        if file_path.suffix == ".jsonl":
            texts = []
            with open(file_path) as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        text = item.get("content", item.get("text", ""))
                        if text:
                            texts.append(text)
                    except:
                        continue
        else:
            text = file_path.read_text()
            texts = [text]

        return await self.compute_perplexity(texts)

    async def evaluate_directory(
        self,
        dir_path: Path,
    ) -> list[PerplexityResult]:
        """Evaluate on directory."""
        results = []

        for file_path in dir_path.rglob("*.txt"):
            result = await self.evaluate_file(file_path)
            results.append(result)

        for file_path in dir_path.rglob("*.jsonl"):
            result = await self.evaluate_file(file_path)
            results.append(result)

        return results


async def main():
    parser = argparse.ArgumentParser(description="Perplexity evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--text", help="Text to evaluate")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    evaluator = PerplexityEvaluator(
        model_path=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )

    if args.data:
        if args.data.is_dir():
            results = await evaluator.evaluate_directory(args.data)
            for result in results:
                print(f"PPL: {result.perplexity:.2f}, Loss: {result.loss:.4f}")
        else:
            result = await evaluator.evaluate_file(args.data)
            print(f"Perplexity: {result.perplexity:.2f}")
            print(f"Loss: {result.loss:.4f}")
            print(f"Tokens: {result.num_tokens}")

    elif args.text:
        result = await evaluator.compute_perplexity([args.text])
        print(f"Perplexity: {result.perplexity:.2f}")

    else:
        print("Provide --data or --text")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())