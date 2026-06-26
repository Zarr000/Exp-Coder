"""Text preprocessing for Expera AI: cleaning, normalization, code handling."""

import re
from typing import Optional


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving structure."""
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)
    text = re.sub(r'\t', '    ', text)
    text = re.sub(r'\u00a0', ' ', text)
    text = re.sub(r'\ufeff', '', text)
    return text


def strip_control_chars(text: str) -> str:
    """Remove control characters except newlines and tabs."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)


def normalize_unicode(text: str, form: str = 'NFC') -> str:
    """Normalize Unicode to NFC/NFC/NFD/NFKD."""
    import unicodedata
    return unicodedata.normalize(form, text)


def remove_html_tags(text: str) -> str:
    """Remove HTML tags (but keep content)."""
    return re.sub(r'<[^>]+>', '', text)


def collapse_blank_lines(text: str, max_blanks: int = 2) -> str:
    """Collapse excessive blank lines."""
    lines = text.split('\n')
    result = []
    blank_count = 0
    for line in lines:
        if line.strip():
            result.append(line)
            blank_count = 0
        else:
            blank_count += 1
            if blank_count <= max_blanks:
                result.append(line)
    return '\n'.join(result)


def clean_text(text: str, remove_html: bool = False,
               max_blanks: int = 2) -> str:
    """Comprehensive text cleaning pipeline."""
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    text = strip_control_chars(text)
    if remove_html:
        text = remove_html_tags(text)
    text = collapse_blank_lines(text, max_blanks)
    return text.strip()


class TextPreprocessor:
    """Configurable text preprocessing pipeline."""

    def __init__(self, remove_html: bool = False,
                 max_blank_lines: int = 2,
                 unicode_form: str = 'NFC',
                 min_length: int = 10):
        self.remove_html = remove_html
        self.max_blank_lines = max_blank_lines
        self.unicode_form = unicode_form
        self.min_length = min_length

    def __call__(self, text: str) -> Optional[str]:
        cleaned = clean_text(text, self.remove_html, self.max_blank_lines)
        if len(cleaned) < self.min_length:
            return None
        return cleaned

    def process_batch(self, texts):
        return [self(t) for t in texts]