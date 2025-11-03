# === Module Docstring ===
"""
Utilities module within data_editor.

Focusses on functions for text and message analysis.

Pure, stateless functions used across:
- ``cleaners.py``
- ``features.py``
- ``core.py``

All functions accept inputs and return outputs — **no side effects**.
"""

from __future__ import annotations

# === Imports ===
import re
import string
from typing import List, Set

import emoji
import pandas as pd
from nltk.tokenize import word_tokenize


# === Global Constants ===
# Global: Punctuation to remove during word analysis (exclude @ and &)
BROAD_PUNCTUATION = "".join(set(string.punctuation) - {"@", "&"})


# === Text Length & Structure ===
def length_chat(text: str | None) -> int:
    """Return character length of cleaned message.

    Args:
        text: Message string or NaN.

    Returns:
        int: Number of characters (0 if NaN).

    Examples:
        >>> length_chat("Hi!")
        3
        >>> length_chat(None)
        0
    """
    return len(str(text)) if pd.notna(text) else 0


# === Emoji Analysis ===
def list_of_all_emojis(text: str, ignore_emojis: Set[str]) -> List[str]:
    """Extract all individual emojis from message.

    Args:
        text: Input message.
        ignore_emojis: Set of emoji codes to skip (e.g., skin tones).

    Returns:
        List[str]: List of emoji characters.

    Examples:
        >>> list_of_all_emojis("Hello world", {"world"})
        ['world']
    """
    if not isinstance(text, str):
        return []
    return [c for c in text if c in emoji.EMOJI_DATA and c not in ignore_emojis]


def list_of_connected_emojis(text: str, pattern: re.Pattern) -> List[str]:
    """Find sequences of 2+ connected emojis.

    Args:
        text: Input message.
        pattern: Pre-compiled regex for connected emojis.

    Returns:
        List[str]: List of emoji sequences.
    """
    if not isinstance(text, str):
        return []
    return pattern.findall(text)


def starts_with_emoji(text: str, ignore_emojis: Set[str]) -> bool:
    """Check if message starts with an emoji.

    Args:
        text: Input message.
        ignore_emojis: Set of emoji codes to ignore.

    Returns:
        bool: True if first character is a valid emoji.
    """
    if not isinstance(text, str) or not text:
        return False
    return text[0] in emoji.EMOJI_DATA and text[0] not in ignore_emojis


def emoji_starting_chat(text: str, ignore_emojis: Set[str]) -> str:
    """Return the first emoji in the message.

    Args:
        text: Input message.
        ignore_emojis: Set of emoji codes to ignore.

    Returns:
        str: First emoji or empty string.
    """
    if not isinstance(text, str) or not text:
        return ""
    for c in text:
        if c in emoji.EMOJI_DATA and c not in ignore_emojis:
            return c
    return ""


def ends_with_emoji(text: str, ignore_emojis: Set[str]) -> bool:
    """Check if message ends with an emoji.

    Args:
        text: Input message.
        ignore_emojis: Set of emoji codes to ignore.

    Returns:
        bool: True if last character is a valid emoji.
    """
    if not isinstance(text, str) or not text:
        return False
    return text[-1] in emoji.EMOJI_DATA and text[-1] not in ignore_emojis


def emoji_ending_chat(text: str, ignore_emojis: Set[str]) -> str:
    """Return the last emoji in the message.

    Args:
        text: Input message.
        ignore_emojis: Set of emoji codes to ignore.

    Returns:
        str: Last emoji or empty string.
    """
    if not isinstance(text, str) or not text:
        return ""
    for c in reversed(text):
        if c in emoji.EMOJI_DATA and c not in ignore_emojis:
            return c
    return ""


# === Punctuation Analysis ===
def count_punctuations(text: str, pattern: re.Pattern) -> int:
    """Count standalone punctuation marks.

    Args:
        text: Input message.
        pattern: Pre-compiled regex for punctuation.

    Returns:
        int: Number of punctuation marks.
    """
    if not isinstance(text, str):
        return 0
    return len(pattern.findall(text))


def has_punctuation(text: str, pattern: re.Pattern) -> bool:
    """Check if message contains any punctuation.

    Args:
        text: Input message.
        pattern: Pre-compiled punctuation regex.

    Returns:
        bool: True if punctuation exists.
    """
    return count_punctuations(text, pattern) > 0


def list_of_all_punctuations(text: str, pattern: re.Pattern) -> List[str]:
    """List all standalone punctuation marks.

    Args:
        text: Input message.
        pattern: Pre-compiled punctuation regex.

    Returns:
        List[str]: List of punctuation characters.
    """
    if not isinstance(text, str):
        return []
    return pattern.findall(text)


def list_of_connected_punctuations(text: str, pattern: re.Pattern) -> List[str]:
    """Find sequences of 2+ connected punctuation.

    Args:
        text: Input message.
        pattern: Pre-compiled connected punctuation regex.

    Returns:
        List[str]: List of punctuation sequences.
    """
    if not isinstance(text, str):
        return []
    return pattern.findall(text)


def ends_with_punctuation(text: str, pattern: re.Pattern) -> bool:
    """Check if message ends with punctuation.

    Args:
        text: Input message.
        pattern: Pre-compiled punctuation regex.

    Returns:
        bool: True if ends with punctuation.
    """
    if not isinstance(text, str) or not text:
        return False
    return pattern.match(text[-1]) is not None


def punctuation_ending_chat(text: str, pattern: re.Pattern) -> str:
    """Return the last punctuation mark.

    Args:
        text: Input message.
        pattern: Pre-compiled punctuation regex.

    Returns:
        str: Last punctuation or empty string.
    """
    if not isinstance(text, str) or not text:
        return ""
    for c in reversed(text):
        if pattern.match(c):
            return c
    return ""


# === Word & Text Analysis ===
def count_words(text: str, stopwords: Set[str]) -> int:
    """Count meaningful words after removing punctuation and stopwords.

    Args:
        text: Input message.
        stopwords: Set of Dutch stopwords.

    Returns:
        int: Number of valid words.
    """
    if not isinstance(text, str):
        return 0
    text = re.sub(f"[{re.escape(BROAD_PUNCTUATION)}]", " ", text)
    words = word_tokenize(text.lower())
    return len([w for w in words if w not in stopwords and w.isalpha()])


def avg_word_length(text: str, stopwords: Set[str]) -> float:
    """Compute average length of valid words.

    Args:
        text: Input message.
        stopwords: Set of Dutch stopwords.

    Returns:
        float: Average word length or 0.0.
    """
    if not isinstance(text, str):
        return 0.0
    text = re.sub(f"[{re.escape(BROAD_PUNCTUATION)}]", " ", text)
    words = word_tokenize(text.lower())
    words = [w for w in words if w not in stopwords and w.isalpha()]
    return sum(len(w) for w in words) / len(words) if words else 0.0


# === Capitalization & Numbers ===
def count_capitals(text: str) -> int:
    """Count uppercase letters.

    Args:
        text: Input message.

    Returns:
        int: Number of uppercase characters.
    """
    return sum(1 for c in text if c.isupper()) if isinstance(text, str) else 0


def has_capitals(text: str) -> bool:
    """Check for any uppercase letters.

    Args:
        text: Input message.

    Returns:
        bool: True if contains uppercase.
    """
    return isinstance(text, str) and any(c.isupper() for c in text)


def list_of_connected_capitals(text: str) -> List[str]:
    """Find sequences of 2+ uppercase words.

    Args:
        text: Input message.

    Returns:
        List[str]: List of uppercase word sequences.
    """
    if not isinstance(text, str):
        return []
    return re.findall(r"\b[A-Z]{2,}\b", text)


def starts_with_capital(text: str) -> bool:
    """Check if message starts with uppercase letter.

    Args:
        text: Input message.

    Returns:
        bool: True if first character is uppercase.
    """
    if not isinstance(text, str) or not text:
        return False
    return text[0].isupper()


def capitalized_words_ratio(text: str) -> float:
    """Ratio of words starting with capital letter.

    Args:
        text: Input message.

    Returns:
        float: Capitalized word ratio or 0.0.
    """
    if not isinstance(text, str):
        return 0.0
    words = word_tokenize(text)
    caps = [w for w in words if w and w[0].isupper()]
    return len(caps) / len(words) if words else 0.0


def count_number_characters(text: str) -> int:
    """Count digit characters.

    Args:
        text: Input message.

    Returns:
        int: Number of digits.
    """
    return sum(1 for c in text if c.isdigit()) if isinstance(text, str) else 0


def has_number_characters(text: str) -> bool:
    """Check for any digit characters.

    Args:
        text: Input message.

    Returns:
        bool: True if contains digits.
    """
    return isinstance(text, str) and any(c.isdigit() for c in text)


def count_numbers(text: str) -> int:
    """Count numeric expressions (e.g., 123, €50, 12.5%).

    Args:
        text: Input message.

    Returns:
        int: Number of numeric tokens.
    """
    if not isinstance(text, str):
        return 0
    pattern = re.compile(
        r"""
        (?:[\$€£¥]\s*)?       # currency + space
        \d+(?:[.,]\d+)*       # digits with comma/dot
        (?:%\s?)?             # optional percent
        [.,=]*                # trailing
    """,
        re.VERBOSE,
    )
    return len([m for m in pattern.finditer(text) if re.search(r"\d", m.group(0))])


# === Question Mark ===
def has_question_mark(text: str) -> bool:
    """Check for question mark.

    Args:
        text: Input message.

    Returns:
        bool: True if contains '?'.
    """
    return isinstance(text, str) and "?" in text


def ends_with_question_mark(text: str) -> bool:
    """Check if message ends with question mark.

    Args:
        text: Input message.

    Returns:
        bool: True if ends with '?'.
    """
    if not isinstance(text, str) or not text:
        return False
    return text[-1] == "?"


# === Link Detection ===
def has_link(text: str, url_pattern: re.Pattern) -> bool:
    """Check if message contains a URL.

    Args:
        text: Input message.
        url_pattern: Pre-compiled URL regex.

    Returns:
        bool: True if URL found.
    """
    return bool(url_pattern.search(text)) if isinstance(text, str) else False


# === CODING STANDARD (APPLY TO ALL CODE) ===
# - `# === Module Docstring ===` before """
# - Google-style docstrings
# - `# === Section Name ===` for all blocks
# - Inline: `# One space, sentence case`
# - Tags: `# TODO:`, `# NOTE:`, `# NEW: (YYYY-MM-DD)`, `# FIXME:`
# - Type hints in function signatures
# - Examples: with >>>
# - No long ----- lines
# - No mixed styles
# - Add markers #NEW at the end of the module capturing the latest changes.

# NEW: Created utilities.py with pure functions (2025-11-03)
# NEW: All functions accept shared patterns/stopwords as args (2025-11-03)
# NEW: Strict 1-blank-line rule enforced (2025-11-03)
# NEW: Full Google-style docstrings with examples (2025-11-03)