"""
Code Evaluation.

Evaluates code generation on benchmarks like HumanEval, MBPP.

Usage:
    python -m src.evaluation.eval_code --model checkpoints/expera-350m --benchmark humaneval
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class CodeResult:
    """Code evaluation result."""

    task_id: str
    passed: bool
    error: Optional[str] = None
    output: Optional[str] = None
    expected: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Benchmark result."""

    name: str
    total: int
    passed: int
    pass_rate: float
    results: list[CodeResult] = None


class CodeEvaluator:
    """Evaluates code generation."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        temperature: float = 0.2,
        max_new_tokens: int = 512,
    ):
        self.model_path = model_path
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
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

    async def generate(self, prompt: str) -> str:
        """Generate code."""
        if not self.model:
            await self.load()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
            )

        generated = outputs[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def extract_code(self, generated: str) -> str:
        """Extract code from generated text."""
        # Try to find code block
        if "```" in generated:
            match = re.search(r"```(?:\w+)?\n(.*?)```", generated, re.DOTALL)
            if match:
                return match.group(1)

        # Try to find function definition
        match = re.search(r"def\s+\w+\([^)]*\):", generated)
        if match:
            start = match.start()
            # Find matching indentation
            lines = generated[start:].split("\n")
            code_lines = [lines[0]]
            base_indent = len(lines[0]) - len(lines[0].lstrip())

            for line in lines[1:]:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    if indent >= base_indent:
                        code_lines.append(line)
                    else:
                        break
                else:
                    code_lines.append(line)

            return "\n".join(code_lines)

        return generated

    async def execute_code(self, code: str, input_data: str = "") -> tuple[bool, str]:
        """Execute code and check output."""
        import tempfile
        import subprocess

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            # Run with input
            result = subprocess.run(
                ["python", temp_path],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=10,
            )

            success = result.returncode == 0
            output = result.stdout + result.stderr

            return success, output

        except subprocess.TimeoutExpired:
            return False, "Timeout"

        except Exception as e:
            return False, str(e)

        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def evaluate_humaneval(
        self,
        data_path: Optional[Path] = None,
    ) -> BenchmarkResult:
        """Evaluate on HumanEval."""
        await self.load()

        # Load HumanEval data
        if data_path is None:
            data_path = Path("data/benchmarks/humaneval.jsonl")

        if not data_path.exists():
            logger.warning(f"HumanEval data not found: {data_path}")
            return BenchmarkResult(name="humaneval", total=0, passed=0, pass_rate=0.0)

        tasks = []
        with open(data_path) as f:
            for line in f:
                tasks.append(json.loads(line))

        results = []

        for task in tasks:
            task_id = task.get("task_id", "")
            prompt = task.get("prompt", "")
            canonical_solution = task.get("canonical_solution", "")
            test = task.get("test", "")

            # Generate
            generated = await self.generate(prompt)
            code = self.extract_code(generated)

            # Test
            full_code = code + "\n" + test
            passed, output = await self.execute_code(full_code)

            results.append(
                CodeResult(
                    task_id=task_id,
                    passed=passed,
                    output=output,
                    expected=canonical_solution,
                )
            )

        passed_count = sum(1 for r in results if r.passed)

        return BenchmarkResult(
            name="humaneval",
            total=len(results),
            passed=passed_count,
            pass_rate=passed_count / len(results) if results else 0.0,
            results=results,
        )

    async def evaluate_mbpp(
        self,
        data_path: Optional[Path] = None,
    ) -> BenchmarkResult:
        """Evaluate on MBPP."""
        await self.load()

        if data_path is None:
            data_path = Path("data/benchmarks/mbpp.jsonl")

        if not data_path.exists():
            logger.warning(f"MBPP data not found: {data_path}")
            return BenchmarkResult(name="mbpp", total=0, passed=0, pass_rate=0.0)

        tasks = []
        with open(data_path) as f:
            for line in f:
                tasks.append(json.loads(line))

        results = []

        for task in tasks:
            task_id = task.get("task_id", "")
            prompt = task.get("prompt", "")
            test = task.get("test", "")

            # Generate
            generated = await self.generate(prompt)
            code = self.extract_code(generated)

            # Test
            full_code = code + "\n" + test
            passed, output = await self.execute_code(full_code)

            results.append(
                CodeResult(
                    task_id=task_id,
                    passed=passed,
                    output=output,
                )
            )

        passed_count = sum(1 for r in results if r.passed)

        return BenchmarkResult(
            name="mbpp",
            total=len(results),
            passed=passed_count,
            pass_rate=passed_count / len(results) if results else 0.0,
            results=results,
        )

    async def evaluate_multi(
        self,
        benchmarks: list[str],
    ) -> list[BenchmarkResult]:
        """Evaluate on multiple benchmarks."""
        results = []

        for name in benchmarks:
            if name == "humaneval":
                result = await self.evaluate_humaneval()
            elif name == "mbpp":
                result = await self.evaluate_mbpp()
            else:
                logger.warning(f"Unknown benchmark: {name}")
                continue

            results.append(result)

        return results


async def main():
    parser = argparse.ArgumentParser(description="Code evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", nargs="+", default=["humaneval", "mbpp"])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    evaluator = CodeEvaluator(
        model_path=args.model,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    )

    results = await evaluator.evaluate_multi(args.benchmark)

    for result in results:
        print(f"\n{result.name}:")
        print(f"  Total: {result.total}")
        print(f"  Passed: {result.passed}")
        print(f"  Pass Rate: {result.pass_rate * 100:.1f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())