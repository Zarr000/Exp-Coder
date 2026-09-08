#!/usr/bin/env python3
"""
Code Dataset Preparation.

Prepares code datasets for training:
- Parses and validates code files
- Extracts programming language
- Filters by file size and quality
- Generates metadata

Usage:
    python scripts/prepare_code_dataset.py --input data/raw/the-stack/ --output data/processed/code/
    python scripts/prepare_code_dataset.py --input data/raw/code-search-net/ --languages python javascript
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
import hashlib
import re
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Supported languages and extensions
LANGUAGE_EXTENSIONS = {
    "python": {".py", ".pyw", ".ipynb"},
    "javascript": {".js", ".mjs", ".cjs"},
    "typescript": {".ts", ".tsx"},
    "java": {".java"},
    "cpp": {".cpp", ".cc", ".cxx", ".hpp", ".h"},
    "c": {".c", ".h"},
    "csharp": {".cs"},
    "go": {".go"},
    "rust": {".rs"},
    "ruby": {".rb"},
    "php": {".php"},
    "swift": {".swift"},
    "kotlin": {".kt", ".kts"},
    "scala": {".scala"},
    "julia": {".jl"},
    "r": {".r", ".R"},
    "sql": {".sql"},
    "bash": {".sh", ".bash"},
    "powershell": {".ps1"},
    "html": {".html", ".htm"},
    "css": {".css", ".scss", ".sass", ".less"},
    "json": {".json"},
    "yaml": {".yaml", ".yml"},
    "xml": {".xml"},
    "markdown": {".md", ".markdown"},
}

# Language detection patterns
LANGUAGE_PATTERNS = {
    "python": [r"^import\s+\w+", r"^from\s+\w+\s+import", r"def\s+\w+\s*\(", r"class\s+\w+\s*[:\(]"],
    "javascript": [r"^const\s+\w+\s*=", r"^let\s+\w+\s*=", r"^function\s+\w+\s*\(", r"export\s+"],
    "typescript": [r":\s*(string|number|boolean|any)\s*[;=]", r"interface\s+\w+\s*{", r"type\s+\w+\s*="],
    "java": [r"^public\s+class\s+", r"^import\s+java\.", r"System\.out\.print"],
    "cpp": [r"#include\s*<", r"^std::", r"cout\s*<<", r"int\s+main\s*\("],
    "go": [r"^package\s+\w+", r"^import\s+\(", r"func\s+\w+\s*\(", r"\.go$"],
    "rust": [r"^fn\s+\w+", r"^let\s+mut\s+", r"impl\s+\w+", r"use\s+std::"],
}


def detect_language(content: str, filename: str) -> Optional[str]:
    """Detect programming language from content and filename."""
    # Check extension first
    ext = Path(filename).suffix.lower()
    for lang, extensions in LANGUAGE_EXTENSIONS.items():
        if ext in extensions:
            return lang

    # Fall back to content patterns
    for lang, patterns in LANGUAGE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, content, re.MULTILINE):
                return lang

    return None


def calculate_complexity(code: str) -> int:
    """Calculate cyclomatic complexity estimate."""
    complexity = 1
    keywords = ["if", "elif", "else", "for", "while", "and", "or", "case", "switch", "catch", "?"]

    for keyword in keywords:
        complexity += len(re.findall(rf'\b{keyword}\b', code))

    return max(1, complexity)


def extract_functions(code: str, language: str) -> List[Dict]:
    """Extract function/method definitions."""
    functions = []

    if language == "python":
        pattern = r'def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[\w\[\],\s]+)?:'
    elif language in ("javascript", "typescript"):
        pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\(|(\w+)\s*\([^)]*\)\s*(?:=>))'
    elif language == "java":
        pattern = r'(?:public|private|protected)\s+(?:static\s+)?[\w<>[\],\s]+\s+(\w+)\s*\('
    elif language == "go":
        pattern = r'func\s+(?:\([^)]+\)\s+)?(\w+)\s*\('
    elif language == "rust":
        pattern = r'fn\s+(\w+)\s*\('
    else:
        pattern = r'[^#]*\b(\w+)\s*\([^)]*\)\s*\{'

    for match in re.finditer(pattern, code):
        func_name = match.group(1) if match.group(1) else "anonymous"
        functions.append({
            "name": func_name,
            "start": match.start(),
        })

    return functions


def extract_imports(code: str, language: str) -> List[str]:
    """Extract import/require statements."""
    imports = []

    if language == "python":
        pattern = r'^(?:from\s+([\w.]+)\s+)?import\s+([\w.]+(?:\s*as\s+\w+)?)'
    elif language in ("javascript", "typescript"):
        pattern = r"^(?:import\s+(?:\{[^}]*\}|\*\s+as\s+\w+|[\w]+)\s+from\s+)?['\"]([^'\"]+)['\"]"
    elif language == "java":
        pattern = r'^import\s+([\w.]+\..+);'
    elif language == "go":
        pattern = r'^import\s+(?:\(\s*)?["\']([^"\']+)["\']'
    else:
        return imports

    for match in re.finditer(pattern, code, re.MULTILINE):
        if match.lastindex:
            module = match.group(match.lastindex)
            imports.append(module)

    return imports[:50]  # Limit imports


def calculate_checksum(content: str) -> str:
    """Calculate SHA256 checksum."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def validate_code(
    content: str,
    language: str,
    min_lines: int = 5,
    max_lines: int = 50000,
    min_quality: float = 0.3,
) -> Dict:
    """Validate code quality."""
    lines = content.split("\n")

    if len(lines) < min_lines:
        return {"valid": False, "reason": f"Too few lines: {len(lines)}"}

    if len(lines) > max_lines:
        return {"valid": False, "reason": f"Too many lines: {len(lines)}"}

    # Check for minimum code quality indicators
    if language == "python":
        quality_indicators = len(re.findall(r'\b(def|class|import|from|if|for|while)\b', content))
    elif language in ("javascript", "typescript"):
        quality_indicators = len(re.findall(r'\b(function|const|let|var|import|export|if|for|while)\b', content))
    elif language == "java":
        quality_indicators = len(re.findall(r'\b(public|private|class|import|if|for|while|return)\b', content))
    else:
        quality_indicators = len(content) // 100

    quality_score = quality_indicators / max(len(lines), 1)

    if quality_score < min_quality:
        return {"valid": False, "reason": f"Low quality: {quality_score:.2f}"}

    # Check for suspicious patterns
    suspicious = ["eval(", "exec(", "os.system(", "subprocess.call(", "shell=True"]
    for pattern in suspicious:
        if pattern in content.lower():
            return {"valid": False, "reason": f"Suspicious pattern: {pattern}"}

    return {"valid": True, "quality_score": quality_score, "complexity": calculate_complexity(content)}


def process_file(file_path: Path, min_lines: int = 5, max_lines: int = 50000) -> Optional[Dict]:
    """Process a single code file."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"Failed to read {file_path}: {e}")
        return None

    language = detect_language(content, file_path.name)
    if not language:
        return None

    validation = validate_code(content, language, min_lines, max_lines)
    if not validation.get("valid"):
        return None

    lines = content.split("\n")

    return {
        "file_path": str(file_path),
        "language": language,
        "content": content,
        "checksum": calculate_checksum(content),
        "lines": len(lines),
        "quality_score": validation.get("quality_score", 0.5),
        "complexity": validation.get("complexity", 1),
        "functions": extract_functions(content, language),
        "imports": extract_imports(content, language),
    }


def process_jsonl(input_path: Path, output_path: Path, languages: Optional[Set[str]] = None, min_lines: int = 5, max_lines: int = 50000) -> Dict:
    """Process JSONL dataset."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "valid": 0, "invalid": 0, "languages": Counter()}

    output_file = output_path / f"{input_path.stem}_processed.jsonl"

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            stats["processed"] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid"] += 1
                continue

            # Extract content
            content = record.get("content", record.get("code", record.get("text", "")))
            if not content:
                stats["invalid"] += 1
                continue

            # Detect language
            language = detect_language(content, record.get("repo_name", "unknown"))

            if languages and language not in languages:
                stats["invalid"] += 1
                continue

            if not language:
                stats["invalid"] += 1
                continue

            # Validate
            validation = validate_code(content, language, min_lines, max_lines)
            if not validation.get("valid"):
                stats["invalid"] += 1
                continue

            # Process record
            processed = {
                "content": content,
                "language": language,
                "checksum": calculate_checksum(content),
                "lines": len(content.split("\n")),
                "quality_score": validation.get("quality_score", 0.5),
                "complexity": validation.get("complexity", 1),
                "repo_name": record.get("repo_name", ""),
                "license": record.get("license", ""),
            }

            outfile.write(json.dumps(processed, ensure_ascii=False) + "\n")
            stats["valid"] += 1
            stats["languages"][language] += 1

            if stats["processed"] % 10000 == 0:
                logger.info(f"Processed {stats['processed']}, valid: {stats['valid']}")

    logger.info(f"Completed: {stats}")
    return stats


def process_directory(
    input_dir: Path,
    output_dir: Path,
    languages: Optional[Set[str]] = None,
    min_lines: int = 5,
    max_lines: int = 50000,
    extensions: Optional[Set[str]] = None,
) -> Dict:
    """Process directory of code files."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "valid": 0, "invalid": 0, "languages": Counter()}

    if extensions is None:
        extensions = {ext for exts in LANGUAGE_EXTENSIONS.values() for ext in exts}

    all_files = list(input_dir.rglob("*"))
    code_files = [f for f in all_files if f.suffix.lower() in extensions and f.is_file()]

    output_file = output_dir / "code_dataset.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for file_path in code_files:
            stats["processed"] += 1

            processed = process_file(file_path, min_lines, max_lines)
            if not processed:
                stats["invalid"] += 1
                continue

            if languages and processed["language"] not in languages:
                stats["invalid"] += 1
                continue

            record = {
                "content": processed["content"],
                "language": processed["language"],
                "checksum": processed["checksum"],
                "lines": processed["lines"],
                "quality_score": processed["quality_score"],
                "complexity": processed["complexity"],
                "file_path": processed["file_path"],
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["valid"] += 1
            stats["languages"][processed["language"]] += 1

            if stats["processed"] % 1000 == 0:
                logger.info(f"Processed {stats['processed']}, valid: {stats['valid']}")

    logger.info(f"Completed: {stats}")
    return stats


def filter_by_language(
    input_path: Path,
    output_path: Path,
    target_language: str,
) -> Dict:
    """Filter dataset by language."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "filtered": 0}

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path / f"{target_language}.jsonl", "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1
            record = json.loads(line)

            if record.get("language") == target_language:
                outfile.write(line + "\n")
                stats["filtered"] += 1

    logger.info(f"Filtered {stats['filtered']}/{stats['total']} for {target_language}")
    return stats


def generate_report(input_path: Path) -> Dict:
    """Generate dataset statistics report."""
    stats = {"total_files": 0, "languages": Counter(), "total_lines": 0, "avg_quality": 0}
    quality_scores = []

    for file_path in Path(input_path).glob("*.jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                stats["total_files"] += 1
                record = json.loads(line)
                stats["languages"][record.get("language", "unknown")] += 1
                stats["total_lines"] += record.get("lines", 0)
                quality_scores.append(record.get("quality_score", 0))

    if quality_scores:
        stats["avg_quality"] = sum(quality_scores) / len(quality_scores)

    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare code datasets for training")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory or file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--languages", nargs="+", help="Filter by languages")
    parser.add_argument("--min-lines", type=int, default=5, help="Minimum lines per file")
    parser.add_argument("--max-lines", type=int, default=50000, help="Maximum lines per file")
    parser.add_argument("--format", choices=["jsonl", "directory"], default="jsonl", help="Input format")
    parser.add_argument("--report", action="store_true", help="Generate report only")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.report:
        stats = generate_report(Path(args.input))
        logger.info(f"Dataset stats: {stats}")
        return

    input_path = Path(args.input)
    output_path = Path(args.output)

    languages = set(args.languages) if args.languages else None

    if args.format == "jsonl":
        stats = process_jsonl(input_path, output_path, languages, args.min_lines, args.max_lines)
    else:
        stats = process_directory(input_path, output_path, languages, args.min_lines, args.max_lines)

    logger.info(f"Processing complete: {stats}")


if __name__ == "__main__":
    main()