"""
Language Detection for Expera AI.

Detects both programming languages and natural languages from text content.
Uses:
- Fast heuristics for code (file extensions, shebang, AST patterns)
- Character-level n-gram features for natural language
- Configurable confidence thresholds
- Batch processing support
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# Programming language detection patterns
SHEBANG_PATTERNS = {
    r'^#!.*python': 'python',
    r'^#!.*node': 'javascript',
    r'^#!.*ruby': 'ruby',
    r'^#!.*perl': 'perl',
    r'^#!.*bash': 'bash',
    r'^#!.*sh': 'bash',
    r'^#!.*zsh': 'bash',
    r'^#!.*java': 'java',
    r'^#!.*lua': 'lua',
    r'^#!.*racket': 'racket',
}

FILE_EXTENSION_MAP = {
    '.py': 'python', '.pyw': 'python', '.pyx': 'python',
    '.js': 'javascript', '.jsx': 'javascript', '.mjs': 'javascript',
    '.ts': 'typescript', '.tsx': 'typescript',
    '.java': 'java', '.kt': 'kotlin', '.scala': 'scala',
    '.c': 'c', '.h': 'c', '.cpp': 'cpp', '.hpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp',
    '.cs': 'csharp', '.vb': 'vb',
    '.go': 'go',
    '.rs': 'rust',
    '.rb': 'ruby',
    '.php': 'php', '.phtml': 'php',
    '.swift': 'swift',
    '.sql': 'sql',
    '.html': 'html', '.htm': 'html', '.xhtml': 'html',
    '.css': 'css', '.scss': 'scss', '.less': 'less',
    '.sh': 'bash', '.bash': 'bash', '.zsh': 'bash',
    '.pl': 'perl', '.pm': 'perl',
    '.lua': 'lua',
    '.r': 'r', '.R': 'r',
    '.m': 'matlab',
    '.jl': 'julia',
    '.dart': 'dart',
    '.lisp': 'lisp', '.clj': 'clojure', '.cljs': 'clojure',
    '.erl': 'erlang',
    '.ex': 'elixir', '.exs': 'elixir',
    '.hs': 'haskell',
    '.ml': 'ocaml',
    '.vue': 'vue',
    '.yaml': 'yaml', '.yml': 'yaml',
    '.json': 'json',
    '.xml': 'xml', '.svg': 'xml',
    '.md': 'markdown', '.mdx': 'markdown',
    '.tex': 'latex',
    '.dockerfile': 'dockerfile',
    '.tf': 'terraform',
}

# Language-specific keyword signatures for content-based detection
LANGUAGE_SIGNATURES: Dict[str, List[str]] = {
    'python': [r'import\s+\w+', r'from\s+\w+\s+import', r'def\s+\w+\s*\(', r'class\s+\w+\s*:', r'if __name__'],
    'javascript': [r'function\s+\w+\s*\(', r'const\s+\w+\s*=', r'let\s+\w+\s*=', r'var\s+\w+\s*=', r'=>', r'console\.log', r'document\.', r'window\.', r'addEventListener', r'\$\('],
    'typescript': [r'interface\s+\w+', r'type\s+\w+\s*=', r':\s*(string|number|boolean)\b'],
    'java': [r'public\s+(class|static|void)', r'private\s+\w+', r'import\s+java\.'],
    'cpp': [r'#include\s*[<"]', r'using namespace', r'std::', r'int main\s*\('],
    'go': [r'func\s+\w+', r'package\s+\w+', r'import\s+\('],
    'rust': [r'fn\s+\w+', r'let\s+mut', r'impl\s+\w+'],
}

# Natural language detection via character n-grams (simplified)
# Maps common character sequences to languages
LANGUAGE_NGRAM_SIGNATURES = {
    'en': set('theandthatyouforarenotwithhavewillthisfromtheybeenmore'),
    'zh': set('的了不是人有在中国年大和会主为生发工时'),
    'ja': set('のたにあをがでといつものかられんし'),
    'de': set('der die und den das mit auf für ist nicht'),
    'fr': set('les des dans pour une sur pas avec comme'),
    'es': set('que los las del para por con una como más'),
}


@dataclass
class LanguageResult:
    """Result of language detection."""
    language: str
    confidence: float
    method: str  # 'shebang', 'extension', 'content', 'ngram', 'unknown'
    is_code: bool


class LanguageDetector:
    """
    Detect programming and natural languages from text.
    
    Detection priority:
    1. File extension (if available)
    2. Shebang line (for scripts)
    3. Content signatures (keyword patterns)
    4. Character n-gram analysis (natural language)
    
    All methods return confidence scores for ambiguous cases.
    """
    
    def __init__(
        self,
        code_confidence_threshold: float = 0.3,
        nl_confidence_threshold: float = 0.2,
    ):
        self.code_threshold = code_confidence_threshold
        self.nl_threshold = nl_confidence_threshold
        
        # Compile regex patterns
        self.shebang_patterns = {
            re.compile(p, re.MULTILINE): lang
            for p, lang in SHEBANG_PATTERNS.items()
        }
        self.lang_signatures = {
            lang: [re.compile(p, re.IGNORECASE) for p in patterns]
            for lang, patterns in LANGUAGE_SIGNATURES.items()
        }
    
    def detect_from_filename(self, filename: str) -> LanguageResult:
        """Detect language from file extension."""
        ext = Path(filename).suffix.lower() if '.' in filename else ''
        if ext in FILE_EXTENSION_MAP:
            return LanguageResult(
                language=FILE_EXTENSION_MAP[ext],
                confidence=0.95,
                method='extension',
                is_code=True,
            )
        return LanguageResult(
            language='unknown',
            confidence=0.0,
            method='extension',
            is_code=False,
        )
    
    def detect_from_shebang(self, text: str) -> Optional[LanguageResult]:
        """Detect language from shebang line."""
        first_line = text.split('\n')[0].strip() if text else ''
        for pattern, lang in self.shebang_patterns.items():
            if pattern.match(first_line):
                return LanguageResult(
                    language=lang,
                    confidence=0.9,
                    method='shebang',
                    is_code=True,
                )
        return None
    
    def detect_from_content(self, text: str) -> Optional[LanguageResult]:
        """
        Detect programming language from code signatures.
        
        Counts matching patterns for each language and returns
        the best match above threshold.
        """
        if len(text) < 50:
            return None
        
        scores: Dict[str, float] = {}
        for lang, patterns in self.lang_signatures.items():
            matches = sum(1 for p in patterns if p.search(text))
            if matches > 0:
                scores[lang] = matches / len(patterns)
        
        if not scores:
            return None
        
        best = max(scores, key=scores.get)
        confidence = scores[best]
        if confidence >= self.code_threshold:
            return LanguageResult(
                language=best,
                confidence=confidence,
                method='content',
                is_code=True,
            )
        return None
    
    def detect_natural_language(self, text: str) -> Optional[LanguageResult]:
        """
        Detect natural language using character n-gram features.
        
        Uses simple character frequency matching for common languages.
        More accurate than random, suitable for routing.
        """
        if len(text) < 100:
            return None
        
        # Extract character n-gram frequencies (simplified)
        chars = text.lower()
        scores: Dict[str, float] = {}
        
        for lang, signature in LANGUAGE_NGRAM_SIGNATURES.items():
            # Count how many signature characters appear in text
            matched = sum(1 for c in signature if c in chars[:500])
            scores[lang] = matched / max(1, len(signature))
        
        if not scores:
            return LanguageResult(
                language='unknown',
                confidence=0.0,
                method='ngram',
                is_code=False,
            )
        
        best = max(scores, key=scores.get)
        confidence = scores[best]
        
        if confidence >= self.nl_threshold:
            return LanguageResult(
                language=best,
                confidence=confidence,
                method='ngram',
                is_code=False,
            )
        
        return LanguageResult(
            language='unknown',
            confidence=confidence,
            method='ngram',
            is_code=False,
        )
    
    def detect(self, text: str, filename: Optional[str] = None) -> LanguageResult:
        """
        Detect language using all available methods.
        
        Args:
            text: The text content to analyze
            filename: Optional filename (for extension-based detection)
            
        Returns:
            LanguageResult with best detected language
        """
        # 1. Try filename extension (fastest, most reliable)
        if filename:
            result = self.detect_from_filename(filename)
            if result.confidence > 0.9:  # High confidence from extension
                return result
        
        # 2. Try shebang
        result = self.detect_from_shebang(text)
        if result:
            return result
        
        # 3. Try code content signatures
        result = self.detect_from_content(text)
        if result:
            return result
        
        # 4. Try natural language
        result = self.detect_natural_language(text)
        if result:
            return result
        
        return LanguageResult(
            language='unknown',
            confidence=0.0,
            method='unknown',
            is_code=False,
        )
    
    def detect_batch(
        self,
        texts: List[Tuple[str, Optional[str]]],
        batch_size: int = 1000,
    ) -> List[LanguageResult]:
        """
        Detect languages for multiple texts in batch.
        
        Args:
            texts: List of (text, filename_or_None) tuples
            batch_size: Processing batch size
            
        Returns:
            List of LanguageResult objects
        """
        results = []
        for text, filename in texts:
            result = self.detect(text, filename)
            results.append(result)
        return results


from pathlib import Path
