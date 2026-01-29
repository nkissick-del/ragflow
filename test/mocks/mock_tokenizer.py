"""
Mock stubs for heavy dependencies in unit testing.

This module provides lightweight mocks for modules that have complex
dependencies (infinity, opendal, etc.) that are difficult to install
on a local Mac development environment.

Usage:
    This is automatically loaded by conftest.py when running tests locally.
"""


class MockRagTokenizer:
    """Mock tokenizer that just returns the input."""

    def tokenize(self, line: str) -> str:
        """Return input as-is for testing."""
        return line if line else ""

    def fine_grained_tokenize(self, tks: str) -> str:
        """Return input as-is for testing."""
        return tks if tks else ""

    def tag(self, text: str) -> list[str]:
        """Mock tag function."""
        return []

    def freq(self, word: str) -> int:
        """Mock freq function."""
        return 0

    def _tradi2simp(self, text: str) -> str:
        """Mock traditional to simplified conversion."""
        return text

    def _strQ2B(self, text: str) -> str:
        """Mock full-width to half-width conversion."""
        return text


# Module-level exports matching rag.nlp.rag_tokenizer
tokenizer = MockRagTokenizer()
tokenize = tokenizer.tokenize
fine_grained_tokenize = tokenizer.fine_grained_tokenize
tag = tokenizer.tag
freq = tokenizer.freq
tradi2simp = tokenizer._tradi2simp
strQ2B = tokenizer._strQ2B


def is_chinese(s):
    """Mock Chinese detection."""
    return False


def is_number(s):
    """Mock number detection."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def is_alphabet(s):
    """Mock alphabet detection."""
    return s.isalpha() if s else False


def naive_qie(txt):
    """Mock naive segmentation."""
    return txt.split() if txt else []
