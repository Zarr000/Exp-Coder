#!/usr/bin/env python3
"""
Human Evaluation Framework.

Collects human preferences for model outputs:
- Pairwise comparison
- Rating scales
- Feedback collection

Usage:
    python scripts/human_eval.py --model checkpoints/expera-small/ --output eval/human/
    python scripts/human_eval.py --input eval/samples.jsonl --compare
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional
import uuid
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Evaluation criteria
CRITERIA = {
    "helpfulness": "How helpful is the response?",
    "accuracy": "How accurate is the information?",
    "clarity": "How clear and easy to understand?",
    "coherence": "How coherent is the response?",
    "safety": "How safe and appropriate?",
}


def generate_evaluation_prompt(
    prompt: str,
    num_outputs: int = 2,
) -> List[Dict]:
    """Generate evaluation samples."""
    # This would use the model to generate responses
    # Placeholder - return prompt for evaluation
    samples = []

    for i in range(num_outputs):
        samples.append({
            "prompt": prompt,
            "response": f"Response {i+1} to: {prompt[:50]}...",
            "model": f"model_{i+1}",
        })

    return samples


def create_comparison_sample(sample_id: str, outputs: List[Dict]) -> Dict:
    """Create comparison sample."""
    sample = {
        "id": sample_id,
        "timestamp": datetime.now().isoformat(),
        "criteria": list(CRITERIA.keys()),
        "outputs": outputs,
        "comparisons": [],
    }

    # Create pairwise comparisons
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            sample["comparisons"].append({
                "output_a": i,
                "output_b": j,
                "winner": None,  # "a", "b", or "tie"
            })

    return sample


def record_preference(
    sample: Dict,
    comparison_idx: int,
    winner: str,
) -> Dict:
    """Record preference."""
    sample["comparisons"][comparison_idx]["winner"] = winner
    return sample


def calculate_win_rate(results: List[Dict]) -> Dict:
    """Calculate win rate."""
    if not results:
        return {"model_a": 0, "model_b": 0, "tie": 0}

    wins = {"a": 0, "b": 0, "tie": 0}

    for r in results:
        for comp in r.get("comparisons", []):
            winner = comp.get("winner")
            if winner:
                wins[winner] += 1

    total = sum(wins.values())
    if total > 0:
        wins = {k: v / total for k, v in wins.items()}

    return wins


def generate_evaluation_dataset(
    model_a,
    model_b,
    input_path: Path,
    num_samples: int = 100,
) -> List[Dict]:
    """Generate evaluation dataset."""
    input_path = Path(input_path)

    samples = []

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[:num_samples]

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        prompt = record.get("prompt", record.get("instruction", ""))

        if not prompt:
            continue

        sample_id = str(uuid.uuid4())[:8]

        # Generate outputs from both models
        # This is a placeholder - actual implementation would call models
        outputs = [
            {"model": "model_a", "text": f"Response A for: {prompt[:30]}"},
            {"model": "model_b", "text": f"Response B for: {prompt[:30]}"},
        ]

        sample = create_comparison_sample(sample_id, outputs)
        samples.append(sample)

    # Shuffle to prevent order bias
    random.shuffle(samples)

    return samples


def evaluate_interactive(samples: List[Dict]) -> List[Dict]:
    """Interactive evaluation."""
    results = []

    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i+1}/{len(samples)} ---")
        print(f"Prompt: {sample['outputs'][0]['text'][:100]}")
        print("\nOutput A:")
        print(sample["outputs"][0]["text"])
        print("\nOutput B:")
        print(sample["outputs"][1]["text"])

        choice = input("\nWhich is better? (a/b/tie/q): ").lower()

        if choice == "q":
            break

        if choice in ["a", "b", "tie"]:
            sample["comparisons"][0]["winner"] = choice
            results.append(sample)

    return results


def save_results(
    results: List[Dict],
    output_path: Path,
) -> Dict:
    """Save evaluation results."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save results
    results_file = output_path / "results.jsonl"
    with open(results_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Calculate statistics
    stats = {
        "total_samples": len(results),
        "win_rates": calculate_win_rate(results),
    }

    # Save stats
    stats_file = output_path / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def create_evaluation_app(
    samples: List[Dict],
    output_dir: Path,
) -> str:
    """Create evaluation app HTML."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html = """<!DOCTYPE html>
<html>
<head>
    <title>Human Evaluation</title>
    <style>
        body { font-family: sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }
        .sample { border: 1px solid #ccc; padding: 20px; margin: 20px 0; }
        .output { background: #f5f5f5; padding: 15px; margin: 10px 0; }
        button { padding: 10px 20px; margin: 5px; cursor: pointer; }
        .selected { background: #4CAF50; color: white; }
    </style>
</head>
<body>
    <h1>Human Evaluation</h1>
    <div id="app"></div>
    <script>
        const samples = """ + json.dumps(samples) + """;
        let current = 0;

        function render() {
            const app = document.getElementById('app');
            const sample = samples[current];

            app.innerHTML = `
                <div class="sample">
                    <h2>Sample ${current + 1}/${samples.length}</h2>
                    <div class="output">
                        <h3>Output A</h3>
                        <p>${sample.outputs[0].text}</p>
                        <button onclick="vote('a')">Select A</button>
                    </div>
                    <div class="output">
                        <h3>Output B</h3>
                        <p>${sample.outputs[1].text}</p>
                        <button onclick="vote('b')">Select B</button>
                    </div>
                    <button onclick="vote('tie')">Tie</button>
                </div>
            `;
        }

        function vote(winner) {
            samples[current].comparisons[0].winner = winner;
            current++;

            if (current < samples.length) {
                render();
            } else {
                document.getElementById('app').innerHTML = '<h2>Evaluation Complete!</h2>';
                saveResults();
            }
        }

        function saveResults() {
            localStorage.setItem('eval_results', JSON.stringify(samples));
        }

        render();
    </script>
</body>
</html>"""

    app_file = output_dir / "evaluation.html"
    with open(app_file, "w") as f:
        f.write(html)

    return str(app_file)


def parse_args():
    parser = argparse.ArgumentParser(description="Human evaluation")
    parser.add_argument("--model-a", type=str, help="Model A path")
    parser.add_argument("--model-b", type=str, help="Model B path")
    parser.add_argument("--input", "-i", type=str, help="Input JSONL file")
    parser.add_argument("--output", "-o", type=str, help="Output directory")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--create-app", action="store_true", help="Create evaluation app")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.create_app and args.input:
        # Create evaluation app
        input_path = Path(args.input)
        samples = []

        with open(input_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    sample = create_comparison_sample(
                        str(uuid.uuid4())[:8],
                        [{"text": record.get("text", "")}]
                    )
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue

        output_dir = Path(args.output or "eval")
        app_path = create_evaluation_app(samples, output_dir)
        logger.info(f"Created: {app_path}")
        return

    if args.interactive:
        # Generate samples
        samples = generate_evaluation_prompt("Sample prompt", 2)
        results = evaluate_interactive(samples)

        if args.output:
            results = save_results(results, Path(args.output))

    logger.info("Evaluation setup complete")


if __name__ == "__main__":
    main()