"""
Code-Aware Tokenizer for Expera AI.

Provides code-specific tokenization:
- CamelCase splitting
- Snake_case splitting
- Keyword preservation
- Indentation preservation
- Comment preservation
"""

import regex as re
from dataclasses import dataclass
from typing import List, Optional, Set, Dict

from .bpe_tokenizer import BPETokenizer


@dataclass
class CodeTokenConfig:
    """Configuration for code-aware tokenization."""
    split_camel_case: bool = True
    split_snake_case: bool = True
    preserve_keywords: bool = True
    preserve_indentation: bool = True
    max_keyword_length: int = 50


# Common programming language keywords
PROGRAMMING_KEYWORDS = {
    "python": {
        "def", "class", "import", "from", "return", "if", "elif", "else",
        "for", "while", "try", "except", "finally", "with", "as", "pass",
        "break", "continue", "and", "or", "not", "in", "is", "None",
        "True", "False", "lambda", "yield", "global", "nonlocal", "assert",
        "raise", "del", "async", "await",
    },
    "javascript": {
        "function", "const", "let", "var", "return", "if", "else", "for",
        "while", "do", "switch", "case", "break", "continue", "try", "catch",
        "finally", "throw", "new", "this", "class", "extends", "super", "import",
        "export", "default", "async", "await", "typeof", "instanceof",
    },
    "java": {
        "public", "private", "protected", "class", "interface", "extends", "implements",
        "return", "if", "else", "for", "while", "do", "switch", "case", "break",
        "continue", "try", "catch", "finally", "throw", "new", "this", "super",
        "import", "package", "static", "final", "void", "int", "long", "double",
        "float", "boolean", "char", "byte", "short",
    },
    "cpp": {
        "int", "long", "double", "float", "char", "bool", "void", "class",
        "struct", "public", "private", "protected", "virtual", "override", "return",
        "if", "else", "for", "while", "do", "switch", "case", "break", "continue",
        "try", "catch", "throw", "new", "delete", "this", "template", "typename",
        "namespace", "using", "include", "define", "ifdef", "ifndef",
    },
    "go": {
        "func", "return", "if", "else", "for", "range", "switch", "case",
        "default", "break", "continue", "go", "chan", "select", "defer", "type",
        "struct", "interface", "map", "package", "import", "var", "const",
        "true", "false", "nil",
    },
    "rust": {
        "fn", "let", "mut", "const", "return", "if", "else", "match",
        "for", "while", "loop", "break", "continue", "use", "mod", "pub",
        "struct", "enum", "trait", "impl", "type", "where", "self", "Self",
        "async", "await", "move", "ref",
    },
}


class CodeAwareTokenizer:
    """
    Tokenizer optimized for code.

    Provides special handling for:
    - CamelCase identifiers (createUser -> create + User)
    - Snake_case identifiers (my_function -> my + function)
    - Programming keywords
    - Code structure (indentation, brackets)
    - Comments
    """

    def __init__(
        self,
        base_tokenizer: BPETokenizer,
        config: Optional[CodeTokenConfig] = None,
    ):
        """
        Initialize code-aware tokenizer.

        Args:
            base_tokenizer: Base BPE tokenizer
            config: Configuration options
        """
        self.base_tokenizer = base_tokenizer
        self.config = config or CodeTokenConfig()

        # Keywords per language
        self._keywords: Dict[str, Set[str]] = {
            lang: set(kw) for lang, kw in PROGRAMMING_KEYWORDS.items()
        }
        self._all_keywords = set().union(*self._keywords.values())

    def tokenize_code(
        self,
        code: str,
        language: Optional[str] = None,
    ) -> List[int]:
        """
        Tokenize code with special handling.

        Args:
            code: Source code
            language: Programming language (optional)

        Returns:
            List of token IDs
        """
        # Process code into tokens
        processed = self._preprocess_code(code, language)

        # Encode with base tokenizer
        return self.base_tokenizer.encode(processed, add_special_tokens=False)

    def _preprocess_code(
        self,
        code: str,
        language: Optional[str] = None,
    ) -> str:
        """Preprocess code before tokenization."""
        result = []

        # Split into tokens (preserving structure)
        # Match words, numbers, operators, punctuation, whitespace
        tokens = re.findall(
            r"[\p{Lu}][\p{Ll}]*|[\p{Ll}]+|[\p{N}]+|"
            r"[^{}\[\]()+*///=;,\s]+|\s+|[{}\[\]()+*///=;,]",
            code,
            re.IGNORECASE
        )

        for token in tokens:
            # Handle based on type
            if token.strip().startswith(("#", "//", "/*", "*/")):
                # Comments - preserve as-is
                result.append(token)
            elif re.match(r"^[\s\n]+$", token):
                # Whitespace - preserve
                result.append(token)
            elif token in self._all_keywords:
                # Keywords - preserve
                result.append(token)
            elif re.match(r"^[A-Z][a-z]+[A-Z]", token) and self.config.split_camel_case:
                # CamelCase
                result.extend(self._split_camel_case(token))
            elif "_" in token and token.islower() and self.config.split_snake_case:
                # snake_case
                result.extend(self._split_snake_case(token))
            else:
                result.append(token)

        return " ".join(result)

    def _split_camel_case(self, identifier: str) -> List[str]:
        """
        Split CamelCase identifier.

        Args:
            identifier: CamelCase identifier

        Returns:
            List of components
        """
        # Insert space before uppercase letters
        split = re.sub(r"([a-z])([A-Z])", r"\1 \2", identifier)
        return split.split()

    def _split_snake_case(self, identifier: str) -> List[str]:
        """
        Split snake_case identifier.

        Args:
            identifier: snake_case identifier

        Returns:
            List of components
        """
        return identifier.split("_")

    def extract_identifiers(self, code: str) -> List[str]:
        """
        Extract identifiers from code.

        Args:
            code: Source code

        Returns:
            List of identifiers
        """
        # Match valid identifiers
        identifiers = re.findall(r"\b[\p{L}_][\p{L}\p{N}_]*\b", code)
        return [i for i in identifiers if not i.isdigit()]

    def split_compound(self, identifier: str) -> List[str]:
        """
        Split compound identifier.

        Args:
            identifier: Compound identifier

        Returns:
            List of components
        """
        if not identifier:
            return []

        # Try CamelCase
        if re.match(r"^[A-Z][a-z]+[A-Z]", identifier):
            return self._split_camel_case(identifier)

        # Try snake_case
        if "_" in identifier and identifier.islower():
            return self._split_snake_case(identifier)

        # Return as-is
        return [identifier]

    def is_keyword(self, token: str, language: Optional[str] = None) -> bool:
        """
        Check if token is a keyword.

        Args:
            token: Token to check
            language: Specific language (or None for any)

        Returns:
            True if keyword
        """
        if language and language in self._keywords:
            return token in self._keywords[language]
        return token in self._all_keywords


def create_code_tokenizer(
    base_tokenizer: BPETokenizer,
    config: Optional[CodeTokenConfig] = None,
) -> CodeAwareTokenizer:
    """
    Create code-aware tokenizer.

    Args:
        base_tokenizer: Base BPE tokenizer
        config: Configuration

    Returns:
        CodeAwareTokenizer
    """
    return CodeAwareTokenizer(base_tokenizer, config)