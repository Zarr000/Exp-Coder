#!/usr/bin/env python3
"""
Code Evaluation.

Evaluates code generation capability:
- Pass@K metrics
- Code execution
- Syntax validation

Usage:
    python scripts/code_eval.py --model checkpoints/expera-small/ --input tests/code_samples.jsonl
    python scripts/code_eval.py --model checkpoints/expera-tiny/ --run-tests --timeout 30
"""

import argparse
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_python_syntax(code: str) -> bool:
    """Check Python syntax."""
    try:
        import ast
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def check_javascript_syntax(code: str) -> bool:
    """Check JavaScript syntax using node."""
    try:
        result = subprocess.run(
            ["node", "--check", "-e", code],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_syntax(code: str, language: str = "python") -> bool:
    """Check syntax for language."""
    if language == "python":
        return check_python_syntax(code)
    elif language in ("javascript", "typescript"):
        return check_javascript_syntax(code)

    return True


def execute_code(code: str, language: str = "python", timeout: int = 30) -> Dict:
    """Execute code and return results."""
    result = {
        "success": False,
        "output": "",
        "error": "",
        "return_code": -1,
    }

    if language != "python":
        result["error"] = f"Unsupported language: {language}"
        return result

    # Create temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        proc = subprocess.run(
            ["python", temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        result["success"] = proc.returncode == 0
        result["output"] = proc.stdout
        result["error"] = proc.stderr
        result["return_code"] = proc.returncode

    except subprocess.TimeoutExpired:
        result["error"] = "Timeout"
    except Exception as e:
        result["error"] = str(e)
    finally:
        Path(temp_path).unlink()

    return result


def extract_code_blocks(text: str) -> List[str]:
    """Extract code blocks from markdown."""
    blocks = []

    # Extract fenced code blocks
    pattern = r'```(?:\w+)?\n(.*?)```'
    blocks.extend(re.findall(pattern, text, re.DOTALL))

    # Extract inline code that looks like complete code
    if "def " in text or "class " in text or "import " in text:
        # Find the largest code-like block
        lines = text.split("\n")
        code_lines = []
        in_code = False

        for line in lines:
            if line.strip().startswith("```"):
                in_code = not in_code
                continue

            if in_code or line.startswith("    ") or line.startswith("\t"):
                code_lines.append(line)

        if code_lines:
            blocks.append("\n".join(code_lines))

    return blocks


def generate_code(prompt: str, model, tokenizer, max_tokens: int = 512) -> str:
    """Generate code from prompt."""
    input_text = f"Write code to solve this problem:\n{prompt}\n\n```python\n"

    inputs = tokenizer(input_text, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.2,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract code from response
    code = generated[len(input_text):]

    # Clean up code block markers
    if "```" in code:
        code = code.split("```")[0]

    return code.strip()


def evaluate_sample(
    prompt: str,
    expected_output: str,
    model,
    tokenizer,
    language: str = "python",
    max_tokens: int = 512,
    timeout: int = 30,
) -> Dict:
    """Evaluate a single sample."""
    # Generate code
    generated_code = generate_code(prompt, model, tokenizer, max_tokens)

    # Check syntax
    syntax_valid = check_syntax(generated_code, language)

    result = {
        "generated_code": generated_code,
        "syntax_valid": syntax_valid,
        "execution": None,
        "output_match": False,
    }

    if syntax_valid and language == "python":
        # Execute generated code
        exec_result = execute_code(generated_code, language, timeout)
        result["execution"] = exec_result

        # Check output matching
        if exec_result["success"]:
            # Compare outputs
            expected = expected_output.strip()
            actual = exec_result["output"].strip()

            # Simple string match (can be improved)
            result["output_match"] = expected == actual

    return result


def evaluate_jsonl(
    model,
    tokenizer,
    input_path: Path,
    language: str = "python",
    max_tokens: int = 512,
    timeout: int = 30,
) -> Dict:
    """Evaluate JSONL file."""
    input_path = Path(input_path)

    results = []
    syntax_valid_count = 0
    execution_success_count = 0
    output_match_count = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt = record.get("prompt", record.get("question", ""))
            expected = record.get("expected_output", record.get("answer", ""))

            if not prompt:
                continue

            result = evaluate_sample(
                prompt, expected, model, tokenizer, language, max_tokens, timeout
            )

            results.append(result)

            if result["syntax_valid"]:
                syntax_valid_count += 1

            if result["execution"] and result["execution"]["success"]:
                execution_success_count += 1

            if result["output_match"]:
                output_match_count += 1

    total = len(results)

    return {
        "total": total,
        "syntax_valid": syntax_valid_count,
        "syntax_valid_rate": syntax_valid_count / total if total > 0 else 0,
        "execution_success": execution_success_count,
        "execution_success_rate": execution_success_count / total if total > 0 else 0,
        "output_match": output_match_count,
        "output_match_rate": output_match_count / total if total > 0 else 0,
        "results": results[:10],  # Limit stored results
    }


def calculate_pass_at_k(results: List[Dict], k: int = 1) -> float:
    """Calculate Pass@K metric."""
    if not results:
        return 0.0

    correct = sum(1 for r in results if r.get("output_match", False))
    return correct / len(results)


def evaluate_humaneval(
    model,
    tokenizer,
    input_path: Path,
) -> Dict:
    """Evaluate HumanEval benchmark."""
    # This is a placeholder - real implementation would use the actual HumanEval dataset
    return evaluate_jsonl(model, tokenizer, input_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate code generation")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model path")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    parser.add_argument("--language", type=str, default="python", help="Language")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--timeout", type=int, default=30, help="Execution timeout")
    parser.add_argument("--run-tests", action="store_true", help="Run code tests")
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)

    input_path = Path(args.input)

    results = evaluate_jsonl(
        model, tokenizer, input_path,
        args.language, args.max_tokens, args.timeout
    )

    logger.info(f"Results: {results}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()