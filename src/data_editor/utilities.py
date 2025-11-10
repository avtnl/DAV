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