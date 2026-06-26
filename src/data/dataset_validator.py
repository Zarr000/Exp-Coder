"""
Dataset Validation Utilities for Expera AI.

Provides validation for:
1. JSONL schema and structure
2. Tokenization quality and coverage
3. Data integrity (checksums, missing files)
4. Consistency (cross-shard deduplication checks)
5. Quality thresholds

All validators produce detailed reports with pass/fail/warning status.
"""

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Iterator
from enum import Enum


class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue found."""
    message: str
    severity: ValidationSeverity
    location: Optional[str] = None  # file/line reference
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Complete validation result for a dataset."""
    dataset_name: str
    passed: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    checks_performed: int = 0
    
    def add_issue(self, message: str, severity: ValidationSeverity,
                  location: Optional[str] = None,
                  details: Optional[Dict[str, Any]] = None) -> None:
        """Add a validation issue."""
        self.issues.append(ValidationIssue(message, severity, location, details))
        if severity == ValidationSeverity.ERROR:
            self.passed = False
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        errors = sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)
        warnings = sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)
        return (
            f"Validation of '{self.dataset_name}': "
            f"{'PASSED' if self.passed else 'FAILED'} "
            f"({self.checks_performed} checks, {errors} errors, {warnings} warnings)"
        )


class DatasetValidator:
    """
    Multi-faceted dataset validator.
    
    Performs configurable validation checks:
    - Schema validation for JSONL/Parquet
    - Tokenization coverage and OOV rates
    - Checksum integrity
    - Quality threshold verification
    - Cross-shard consistency
    """
    
    def __init__(
        self,
        required_fields: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
        quality_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.required_fields = required_fields or ["text"]
        self.tokenizer = tokenizer
        self.quality_thresholds = quality_thresholds or {
            "min_doc_length": 10,
            "max_doc_length": 1000000,
            "max_empty_lines_ratio": 0.5,
        }
    
    def validate_jsonl(
        self,
        path: str,
        sample_size: int = 10000,
        max_issues: int = 50,
    ) -> ValidationResult:
        """
        Validate a JSONL file's structure and content.
        
        Args:
            path: Path to JSONL file
            sample_size: Number of lines to sample for content validation
            max_issues: Stop after this many issues
            
        Returns:
            ValidationResult with all findings
        """
        result = ValidationResult(dataset_name=Path(path).stem)
        filepath = Path(path)
        
        if not filepath.exists():
            result.add_issue(f"File not found: {path}", ValidationSeverity.ERROR)
            return result
        
        if filepath.suffix not in ('.jsonl', '.json', '.gz'):
            result.add_issue(
                f"Unexpected file extension: {filepath.suffix}",
                ValidationSeverity.WARNING,
                location=path,
            )
        
        # Validate structure by sampling
        issues = 0
        line_count = 0
        valid_lines = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line_count += 1
                    line = line.strip()
                    if not line:
                        issues += 1
                        if issues <= max_issues:
                            result.add_issue(
                                f"Empty line at line {line_count}",
                                ValidationSeverity.WARNING,
                                location=f"{path}:{line_count}",
                            )
                        continue
                    
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        issues += 1
                        if issues <= max_issues:
                            result.add_issue(
                                f"Invalid JSON at line {line_count}: {e}",
                                ValidationSeverity.ERROR,
                                location=f"{path}:{line_count}",
                            )
                        continue
                    
                    if not isinstance(data, dict):
                        issues += 1
                        if issues <= max_issues:
                            result.add_issue(
                                f"Line {line_count} is not a JSON object",
                                ValidationSeverity.ERROR,
                                location=f"{path}:{line_count}",
                            )
                        continue
                    
                    # Check required fields
                    for field in self.required_fields:
                        if field not in data:
                            issues += 1
                            if issues <= max_issues:
                                result.add_issue(
                                    f"Missing required field '{field}' at line {line_count}",
                                    ValidationSeverity.ERROR,
                                    location=f"{path}:{line_count}",
                                )
                            break
                    
                    # Check text field quality
                    if "text" in data and isinstance(data["text"], str):
                        text = data["text"]
                        if len(text) < self.quality_thresholds["min_doc_length"]:
                            issues += 1
                            if issues <= max_issues:
                                result.add_issue(
                                    f"Text too short ({len(text)} chars) at line {line_count}",
                                    ValidationSeverity.WARNING,
                                    location=f"{path}:{line_count}",
                                    details={"length": len(text)},
                                )
                        elif len(text) > self.quality_thresholds["max_doc_length"]:
                            result.add_issue(
                                f"Text too long ({len(text)} chars) at line {line_count}",
                                ValidationSeverity.INFO,
                                location=f"{path}:{line_count}",
                            )
                    
                    valid_lines += 1
                    
                    # Stop sampling if we've seen enough
                    if line_count >= sample_size and issues >= max_issues:
                        break
                    
        except UnicodeDecodeError as e:
            result.add_issue(
                f"File encoding error: {e}",
                ValidationSeverity.ERROR,
                location=path,
            )
        except Exception as e:
            result.add_issue(
                f"Unexpected error reading file: {e}",
                ValidationSeverity.ERROR,
                location=path,
            )
        
        result.checks_performed = line_count
        
        if line_count == 0:
            result.add_issue("File is empty", ValidationSeverity.ERROR, location=path)
        
        if valid_lines == 0 and line_count > 0:
            result.add_issue("No valid JSON lines found", ValidationSeverity.ERROR, location=path)
        
        return result
    
    def validate_checksums(
        self,
        file_checksums: Dict[str, str],
        algorithm: str = "sha256",
    ) -> ValidationResult:
        """
        Validate file checksums.
        
        Args:
            file_checksums: Dict mapping file paths to expected checksums
            algorithm: Hash algorithm to use
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(dataset_name="checksum_validation")
        
        for filepath_str, expected_hash in file_checksums.items():
            filepath = Path(filepath_str)
            
            if not filepath.exists():
                result.add_issue(
                    f"File not found: {filepath}",
                    ValidationSeverity.ERROR,
                    location=filepath_str,
                )
                continue
            
            try:
                hasher = hashlib.new(algorithm)
                with open(filepath, 'rb') as f:
                    for chunk in iter(lambda: f.read(65536), b''):
                        hasher.update(chunk)
                actual_hash = hasher.hexdigest()
                
                if actual_hash != expected_hash:
                    result.add_issue(
                        f"Checksum mismatch for {filepath}: "
                        f"expected {expected_hash}, got {actual_hash}",
                        ValidationSeverity.ERROR,
                        location=filepath_str,
                        details={"expected": expected_hash, "actual": actual_hash},
                    )
            except Exception as e:
                result.add_issue(
                    f"Error computing checksum for {filepath}: {e}",
                    ValidationSeverity.ERROR,
                    location=filepath_str,
                )
        
        result.checks_performed = len(file_checksums)
        return result
    
    def validate_tokenization(
        self,
        texts: Iterator[str],
        sample_size: int = 1000,
    ) -> ValidationResult:
        """
        Validate tokenization quality.
        
        Args:
            texts: Iterator over text samples
            sample_size: Number of samples to check
            tokenizer: Tokenizer instance
            
        Returns:
            ValidationResult
        """
        if self.tokenizer is None:
            return ValidationResult(
                dataset_name="tokenization",
                passed=False,
                issues=[ValidationIssue(
                    "No tokenizer provided for validation",
                    ValidationSeverity.ERROR,
                )],
            )
        
        result = ValidationResult(dataset_name="tokenization_validation")
        
        total_tokens = 0
        total_chars = 0
        samples_checked = 0
        oov_tokens = 0
        max_compression_ratio = 0.0
        min_compression_ratio = float('inf')
        
        for i, text in enumerate(texts):
            if i >= sample_size:
                break
            
            if not isinstance(text, str) or not text.strip():
                continue
            
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                total_tokens += len(tokens)
                total_chars += len(text)
                samples_checked += 1
                
                # Check for OOV tokens
                for tid in tokens:
                    if tid == self.tokenizer.token_to_id(self.tokenizer.special_tokens["unk_token"]):
                        oov_tokens += 1
                
                # Compression ratio
                ratio = len(text) / max(1, len(tokens))
                max_compression_ratio = max(max_compression_ratio, ratio)
                min_compression_ratio = min(min_compression_ratio, ratio)
                
            except Exception as e:
                result.add_issue(
                    f"Tokenization error at sample {i}: {e}",
                    ValidationSeverity.WARNING,
                    details={"sample_index": i},
                )
        
        avg_compression = total_chars / max(1, total_tokens)
        
        if samples_checked == 0:
            result.add_issue("No valid samples for tokenization", ValidationSeverity.ERROR)
        else:
            if avg_compression < 1.0:
                result.add_issue(
                    f"Very low compression ratio: {avg_compression:.2f} chars/token",
                    ValidationSeverity.WARNING,
                    details={"avg_compression": avg_compression},
                )
            
            oov_rate = oov_tokens / max(1, total_tokens)
            if oov_rate > 0.01:
                result.add_issue(
                    f"High OOV rate: {oov_rate:.2%}",
                    ValidationSeverity.WARNING,
                    details={"oov_rate": oov_rate},
                )
        
        result.checks_performed = samples_checked
        return result
    
    def validate_quality_thresholds(
        self,
        stats: Dict[str, float],
    ) -> ValidationResult:
        """
        Validate that quality metrics meet thresholds.
        
        Args:
            stats: Dictionary of quality metric names to values
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(dataset_name="quality_thresholds")
        
        for metric, threshold in self.quality_thresholds.items():
            if metric in stats:
                actual = stats[metric]
                if isinstance(threshold, tuple):
                    lo, hi = threshold
                    if actual < lo or actual > hi:
                        result.add_issue(
                            f"Metric '{metric}' = {actual} outside range [{lo}, {hi}]",
                            ValidationSeverity.ERROR if actual < lo else ValidationSeverity.WARNING,
                            details={metric: actual, "threshold": threshold},
                        )
                elif actual > threshold:
                    result.add_issue(
                        f"Metric '{metric}' = {actual} exceeds threshold {threshold}",
                        ValidationSeverity.WARNING,
                        details={metric: actual, "threshold": threshold},
                    )
        
        result.checks_performed = len(self.quality_thresholds)
        return result