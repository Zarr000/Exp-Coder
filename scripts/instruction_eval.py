#!/usr/bin/env python3
"""
Instruction Evaluation.

Evaluates instruction following:
- Task completion
- Format compliance
- Safety checks
- Accuracy

Usage:
    python scripts/instruction_eval.py --model checkpoints/expera-small/ --input tests/instructions.jsonl
    python scripts/instruction_eval.py --model checkpoints/expera-tiny/ --benchmark mt_bench
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
import ast
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, device: str = "auto"):
    """Load model and tokenizer."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 512) -> str:
    """Generate response."""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Task categories
TASK_CATEGORIES = {
    "math": ["calculate", "solve", "compute", "math"],
    "coding": ["write code", "implement", "function", "debug"],
    "reasoning": ["explain", "why", "analyze", "think"],
    "writing": ["write", "essay", "summary", "compose"],
    "factual": ["what is", "who is", "when", "where"],
}


def extract_answer(response: str, format_type: str = "default") -> Optional[str]:
    """Extract answer from response."""
    if format_type == "json":
        try:
            # Try to parse as JSON
            data = json.loads(response)
            return data.get("answer", "")
        except json.JSONDecodeError:
            pass

    elif format_type == "xml":
        # Extract from XML tags
        match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if match:
            return match.group(1).strip()

    elif format_type == "markdown":
        # Extract from markdown code blocks
        match = re.search(r'```(?:json)?\s*(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()

    return response.strip()


def check_format_compliance(response: str, format_template: str) -> bool:
    """Check if response follows format template."""
    if not format_template:
        return True

    if format_template == "json":
        try:
            json.loads(response)
            return True
        except json.JSONDecodeError:
            return False

    elif format_template == "xml":
        return "<response>" in response and "</response>" in response

    elif format_template == "markdown":
        return "```" in response

    return True


def check_safety(response: str) -> bool:
    """Check if response is safe."""
    # Basic safety checks
    unsafe_patterns = [
        r'\b(hack|exploit|bypass)\s+(security|protection|auth)',
        r'build\s+(a\s+)?weapon',
        r'create\s+(a\s+)?virus',
    ]

    for pattern in unsafe_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            return False

    return True


def extract_reasoning_steps(response: str) -> List[str]:
    """Extract reasoning steps from response."""
    steps = []

    # Look for numbered lists
    numbered = re.findall(r'(?:\d+\.|[-*])\s*(.+)', response)
    steps.extend(numbered)

    # Look for "First", "Second", etc.
    sequential = re.findall(r'\b(First|Second|Third|Then|Next|Finally)\s*[:.]?\s*(.+)', response, re.IGNORECASE)
    steps.extend([f"{s[0]}: {s[1]}" for s in sequential])

    return steps


def evaluate_reasoning(response: str, expected_steps: int = 3) -> Dict:
    """Evaluate reasoning quality."""
    steps = extract_reasoning_steps(response)

    return {
        "has_reasoning": len(steps) > 0,
        "step_count": len(steps),
        "has_sufficient_steps": len(steps) >= expected_steps,
    }


def evaluate_math(response: str, expected_answer: str) -> Dict:
    """Evaluate math problem."""
    # Extract numeric answer
    numbers = re.findall(r'-?\d+\.?\d*', response)
    expected_numbers = re.findall(r'-?\d+\.?\d*', expected_answer)

    correct = False

    if numbers and expected_numbers:
        # Check if answer matches
        if numbers[0] == expected_numbers[0]:
            correct = True

    return {
        "correct": correct,
        "extracted_answer": numbers[0] if numbers else "",
    }


def evaluate_coding(response: str) -> Dict:
    """Evaluate code generation."""
    # Extract code blocks
    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', response, re.DOTALL)

    if not code_blocks:
        return {"has_code": False}

    code = code_blocks[0]

    # Try to parse
    try:
        ast.parse(code)
        syntax_valid = True
    except SyntaxError:
        syntax_valid = False

    return {
        "has_code": True,
        "syntax_valid": syntax_valid,
    }


def evaluate_briefing(response: str, max_length: int = 100) -> Dict:
    """Evaluate response conciseness."""
    length = len(response.split())

    return {
        "length": length,
        "is_concise": length <= max_length,
    }


def evaluate_sample(
    sample: Dict,
    model,
    tokenizer,
) -> Dict:
    """Evaluate single sample."""
    prompt = sample.get("prompt", sample.get("instruction", ""))
    expected = sample.get("expected", sample.get("answer", ""))
    format_template = sample.get("format", "")
    category = sample.get("category", "general")

    # Generate response
    response = generate_response(model, tokenizer, prompt)

    result = {
        "prompt": prompt,
        "response": response,
    }

    # Extract answer
    extracted = extract_answer(response, sample.get("format_type", "default"))
    result["extracted_answer"] = extracted

    # Check format
    result["format_compliant"] = check_format_compliance(response, format_template)

    # Check safety
    result["is_safe"] = check_safety(response)

    # Category-specific evaluation
    if category == "math":
        result["math_eval"] = evaluate_math(response, expected)
    elif category == "coding":
        result["coding_eval"] = evaluate_coding(response)
    elif category == "reasoning":
        result["reasoning_eval"] = evaluate_reasoning(response)

    # Check answer
    result["correct"] = extracted.strip().lower() == expected.strip().lower() if expected else False

    return result


def evaluate_jsonl(
    model,
    tokenizer,
    input_path: Path,
) -> Dict:
    """Evaluate JSONL file."""
    results = []
    stats = {
        "total": 0,
        "correct": 0,
        "format_compliant": 0,
        "safe": 0,
    }

    categories = {}

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue

            result = evaluate_sample(sample, model, tokenizer)
            results.append(result)

            # Update stats
            if result["correct"]:
                stats["correct"] += 1

            if result["format_compliant"]:
                stats["format_compliant"] += 1

            if result["is_safe"]:
                stats["safe"] += 1

            # By category
            cat = sample.get("category", "unknown")
            if cat not in categories:
                categories[cat] = {"total": 0, "correct": 0}

            categories[cat]["total"] += 1
            if result["correct"]:
                categories[cat]["correct"] += 1

    # Calculate rates
    total = stats["total"]
    if total > 0:
        stats["accuracy"] = stats["correct"] / total
        stats["format_rate"] = stats["format_compliant"] / total
        stats["safety_rate"] = stats["safe"] / total

    stats["categories"] = categories

    return {
        "results": results[:100],
        "stats": stats,
    }


def evaluate_mt_bench(
    model,
    tokenizer,
    questions_file: Path,
) -> Dict:
    """Evaluate MT-Bench benchmark."""
    # Load MT-Bench questions
    questions = []
    with open(questions_file, "r") as f:
        for line in f:
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    results = []
    scores = []

    for q in questions:
        prompt = q.get("prompt", "")
        category = q.get("category", "")

        response = generate_response(model, tokenizer, prompt)

        # Simple scoring - can be improved with GPT-4 evaluation
        score = 0

        # Check if response is not empty
        if response.strip():
            score += 0.3

        # Check if has reasoning
        if len(response) > 100:
            score += 0.3

        # Check safety
        if check_safety(response):
            score += 0.4

        scores.append(score)

        results.append({
            "category": category,
            "score": score,
            "response": response[:200],
        })

    # Calculate category-wise scores
    by_category = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r["score"])

    category_scores = {
        cat: sum(scores) / len(scores)
        for cat, scores in by_category.items()
    }

    return {
        "overall_score": sum(scores) / len(scores) if scores else 0,
        "category_scores": category_scores,
        "results": results,
    }


def benchmark_runner(
    model,
    benchmarks: List[str],
) -> Dict:
    """Run multiple benchmarks."""
    results = {}

    for benchmark in benchmarks:
        logger.info(f"Running {benchmark}")
        # This would load benchmark-specific data and run evaluation
        results[benchmark] = {"score": 0.5}

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate instructions")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model path")
    parser.add_argument("--input", "-i", type=str, help="Input JSONL file")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    parser.add_argument("--benchmark", type=str, help="Run specific benchmark (mt_bench)")
    parser.add_argument("--compare", nargs="+", help="Compare models")
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)

    if args.benchmark:
        if args.benchmark == "mt_bench":
            results = evaluate_mt_bench(model, tokenizer, Path(args.input or "data/benchmarks/mt_bench.jsonl"))
    else:
        results = evaluate_jsonl(model, tokenizer, Path(args.input))

    logger.info(f"Results: {results}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()