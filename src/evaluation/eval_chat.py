"""
Chat Evaluation.

Evaluates chat/completion quality.

Usage:
    python -m src.evaluation.eval_chat --model checkpoints/expera-350m --data data/eval/chat.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    """Chat evaluation result."""

    prompt: str
    reference: str
    generated: str
    similarity: float = 0.0
    fluency: float = 0.0
    relevance: float = 0.0
    overall: float = 0.0


@dataclass
class ChatBenchmarkResult:
    """Chat benchmark result."""

    name: str
    total: int
    avg_similarity: float
    avg_fluency: float
    avg_relevance: float
    avg_overall: float
    results: list[ChatResult] = None


class ChatEvaluator:
    """Evaluates chat quality."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        temperature: float = 0.7,
    ):
        self.model_path = model_path
        self.device = device
        self.temperature = temperature
        self.model = None
        self.tokenizer = None

    async def load(self) -> None:
        """Load model."""
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

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

    async def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate response."""
        if not self.model:
            await self.load()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
            )

        generated = outputs[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute word overlap similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _compute_fluency(self, text: str) -> float:
        """Compute fluency score based on basic heuristics."""
        words = text.split()

        if not words:
            return 0.0

        # Check for basic coherence
        score = 1.0

        # Penalize very short or empty responses
        if len(text) < 10:
            score *= 0.5

        # Penalize repeated words
        if len(words) != len(set(words)):
            score *= 0.9

        return min(1.0, score)

    def _compute_relevance(self, prompt: str, response: str) -> float:
        """Compute relevance to prompt."""
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())

        if not prompt_words:
            return 0.0

        # Check for overlap with prompt content
        overlap = prompt_words & response_words

        # Check for common question words that should be addressed
        question_words = {"what", "how", "why", "when", "where", "who"}
        has_question = question_words & prompt_words

        if has_question:
            # Stronger penalty for ignoring question
            return min(1.0, len(overlap) / len(prompt_words) * 2)

        return min(1.0, len(overlap) / len(prompt_words))

    async def evaluate_single(
        self,
        prompt: str,
        reference: str = "",
    ) -> ChatResult:
        """Evaluate a single prompt-response pair."""
        # Generate
        generated = await self.generate(prompt)

        # Compute metrics
        similarity = self._compute_similarity(generated, reference) if reference else 0.5
        fluency = self._compute_fluency(generated)
        relevance = self._compute_relevance(prompt, generated)
        overall = (similarity + fluency + relevance) / 3

        return ChatResult(
            prompt=prompt,
            reference=reference,
            generated=generated,
            similarity=similarity,
            fluency=fluency,
            relevance=relevance,
            overall=overall,
        )

    async def evaluate_file(
        self,
        file_path: Path,
    ) -> ChatBenchmarkResult:
        """Evaluate on chat test file."""
        await self.load()

        results = []
        with open(file_path) as f:
            for line in f:
                item = json.loads(line)
                prompt = item.get("prompt", item.get("instruction", ""))
                reference = item.get("reference", item.get("output", ""))

                result = await self.evaluate_single(prompt, reference)
                results.append(result)

        if not results:
            return ChatBenchmarkResult(
                name=file_path.stem,
                total=0,
                avg_similarity=0.0,
                avg_fluency=0.0,
                avg_relevance=0.0,
                avg_overall=0.0,
            )

        return ChatBenchmarkResult(
            name=file_path.stem,
            total=len(results),
            avg_similarity=sum(r.similarity for r in results) / len(results),
            avg_fluency=sum(r.fluency for r in results) / len(results),
            avg_relevance=sum(r.relevance for r in results) / len(results),
            avg_overall=sum(r.overall for r in results) / len(results),
            results=results,
        )

    async def evaluate_directory(
        self,
        dir_path: Path,
    ) -> list[ChatBenchmarkResult]:
        """Evaluate on directory of chat files."""
        results = []

        for file_path in dir_path.glob("*.jsonl"):
            result = await self.evaluate_file(file_path)
            results.append(result)

        return results


async def main():
    parser = argparse.ArgumentParser(description="Chat evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    evaluator = ChatEvaluator(
        model_path=args.model,
        temperature=args.temperature,
    )

    if args.data:
        if args.data.is_dir():
            results = await evaluator.evaluate_directory(args.data)
            for result in results:
                print(f"\n{result.name}:")
                print(f"  Overall: {result.avg_overall:.3f}")
                print(f"  Similarity: {result.avg_similarity:.3f}")
                print(f"  Fluency: {result.avg_fluency:.3f}")
                print(f"  Relevance: {result.avg_relevance:.3f}")
        else:
            result = await evaluator.evaluate_file(args.data)
            print(f"\n{result.name}:")
            print(f"  Overall: {result.avg_overall:.3f}")
            print(f"  Similarity: {result.avg_similarity:.3f}")
            print(f"  Fluency: {result.avg_fluency:.3f}")
            print(f"  Relevance: {result.avg_relevance:.3f}")

    else:
        print("Provide --data")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())