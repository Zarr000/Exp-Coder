#!/usr/bin/env python3
"""
Dataset Report Generator.

Generates comprehensive reports on datasets:
- Statistics
- Quality metrics
- Distribution analysis
- Comparison

Usage:
    python scripts/dataset_report.py --input data/processed/ --output docs/dataset_report.md
    python scripts/dataset_report.py --input data/ --format json --output stats.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter, defaultdict
import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def scan_directory(input_path: Path) -> Dict:
    """Scan directory for dataset files."""
    input_path = Path(input_path)

    files = {"jsonl": [], "json": [], "parquet": []}

    for pattern in ["*.jsonl", "*.json", "*.parquet"]:
        files["jsonl" if "jsonl" in pattern else "json" if "json" in pattern else "parquet"].extend(
            list(input_path.rglob(pattern))
        )

    return files


def analyze_jsonl(file_path: Path) -> Dict:
    """Analyze JSONL file."""
    stats = {
        "file": str(file_path),
        "records": 0,
        "total_size": 0,
        "languages": Counter(),
        "sources": Counter(),
        "avg_length": 0,
        "avg_quality": 0,
    }

    lengths = []
    quality_scores = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            stats["records"] += 1
            stats["total_size"] += len(line)

            try:
                record = json.loads(line)

                # Language
                if "_language" in record:
                    stats["languages"][record["_language"]] += 1

                # Source
                if "source" in record:
                    stats["sources"][record["source"]] += 1

                # Length
                text = record.get("text", "") or record.get("content", "") or record.get("code", "")
                if text:
                    lengths.append(len(text))

                # Quality
                if "_quality" in record:
                    quality_scores.append(record["_quality"].get("overall_quality", 0))

            except json.JSONDecodeError:
                continue

    if lengths:
        stats["avg_length"] = sum(lengths) / len(lengths)
        stats["median_length"] = sorted(lengths)[len(lengths) // 2]

    if quality_scores:
        stats["avg_quality"] = sum(quality_scores) / len(quality_scores)

    stats["languages"] = dict(stats["languages"])
    stats["sources"] = dict(stats["sources"])

    return stats


def generate_summary(input_path: Path) -> Dict:
    """Generate dataset summary."""
    files = scan_directory(input_path)

    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "directory": str(input_path),
        "files": {
            "jsonl": len(files["jsonl"]),
            "json": len(files["json"]),
            "parquet": len(files["parquet"]),
        },
        "datasets": {},
    }

    total_records = 0

    for jsonl_file in files["jsonl"]:
        name = jsonl_file.stem.replace("_processed", "").replace("_filtered", "").replace("_scored", "")
        stats = analyze_jsonl(jsonl_file)
        summary["datasets"][name] = stats
        total_records += stats["records"]

    summary["total_records"] = total_records

    # Overall statistics
    all_languages = Counter()
    all_sources = Counter()

    for dataset_stats in summary["datasets"].values():
        all_languages.update(dataset_stats.get("languages", {}))
        all_sources.update(dataset_stats.get("sources", {}))

    summary["combined"] = {
        "languages": dict(all_languages),
        "sources": dict(all_sources),
    }

    return summary


def generate_markdown_report(summary: Dict) -> str:
    """Generate markdown report."""
    md = []

    md.append("# Dataset Report")
    md.append("")
    md.append(f"Generated: {summary['timestamp']}")
    md.append("")
    md.append(f"Directory: `{summary['directory']}`")
    md.append("")
    md.append(f"Total records: **{summary['total_records']:,}**")
    md.append("")

    # Files overview
    md.append("## Files Overview")
    md.append("")
    md.append(f"- JSONL files: {summary['files']['jsonl']}")
    md.append(f"- JSON files: {summary['files']['json']}")
    md.append(f"- Parquet files: {summary['files']['parquet']}")
    md.append("")

    # Combined statistics
    if "combined" in summary:
        md.append("## Combined Statistics")
        md.append("")

        if summary["combined"].get("languages"):
            md.append("### Languages")
            md.append("")
            for lang, count in sorted(summary["combined"]["languages"].items(), key=lambda x: -x[1]):
                pct = count / summary["total_records"] * 100
                md.append(f"- {lang}: {count:,} ({pct:.1f}%)")
            md.append("")

        if summary["combined"].get("sources"):
            md.append("### Sources")
            md.append("")
            for source, count in sorted(summary["combined"]["sources"].items(), key=lambda x: -x[1])[:10]:
                md.append(f"- {source}: {count:,}")
            md.append("")

    # Per-dataset stats
    md.append("## Datasets")
    md.append("")

    for name, stats in summary["datasets"].items():
        md.append(f"### {name}")
        md.append("")
        md.append(f"- Records: {stats['records']:,}")
        md.append(f"- Avg length: {stats.get('avg_length', 0):.0f}")
        md.append(f"- Median length: {stats.get('median_length', 0):,}")

        if stats.get("avg_quality"):
            md.append(f"- Avg quality: {stats['avg_quality']:.2f}")

        md.append("")

    return "\n".join(md)


def generate_html_report(summary: Dict) -> str:
    """Generate HTML report."""
    html = []

    html.append("""<!DOCTYPE html>
<html>
<head>
    <title>Dataset Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1, h2, h3 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f5f5f5; }
        .stat { display: inline-block; margin: 10px 20px 10px 0; }
        .stat-value { font-size: 24px; font-weight: bold; }
        .stat-label { color: #666; }
    </style>
</head>
<body>
    <h1>Dataset Report</h1>
    <p>Generated: {timestamp}</p>
    <p>Directory: {directory}</p>

    <div class="stat">
        <div class="stat-value">{total_records:,}</div>
        <div class="stat-label">Total Records</div>
    </div>
""".format(**summary))

    # Datasets table
    html.append("    <h2>Datasets</h2>")
    html.append("    <table>")
    html.append("    <tr><th>Dataset</th><th>Records</th><th>Avg Length</th><th>Quality</th></tr>")

    for name, stats in summary["datasets"].items():
        html.append(f"    <tr><td>{name}</td><td>{stats['records']:,}</td><td>{stats.get('avg_length', 0):.0f}</td><td>{stats.get('avg_quality', 'N/A')}</td></tr>")

    html.append("    </table>")
    html.append("</body></html>")

    return "\n".join(html)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate dataset reports")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory")
    parser.add_argument("--output", "-o", type=str, help="Output file")
    parser.add_argument("--format", choices=["markdown", "html", "json"], default="markdown", help="Output format")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)

    logger.info(f"Analyzing: {input_path}")
    summary = generate_summary(input_path)

    if args.verbose:
        logger.info(f"Found {summary['total_records']:,} records")

    if args.format == "json":
        output = json.dumps(summary, indent=2)
    elif args.format == "html":
        output = generate_html_report(summary)
    else:
        output = generate_markdown_report(summary)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        logger.info(f"Report written to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()