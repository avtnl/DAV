# === Module Docstring ===
"""
data_editor package

Core orchestration for WhatsApp chat analysis.

Public API:
    DataEditor - Main class for cleaning and feature engineering

Usage:
    >>> from data_editor import DataEditor
    >>> editor = DataEditor()
    >>> df = editor.organize_extended_df(raw_df)
"""

from __future__ import annotations

# === Imports ===
from .core import DataEditor


# === Package Metadata ===
__version__ = "0.1.0"
__author__ = "Your Name"


# === Public API ===
__all__ = [
    "DataEditor",
]


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

# NEW: Added __version__ and __author__ metadata (2025-11-03)
# NEW: Enhanced docstring with usage example (2025-11-03)
# NEW: Strict 1-blank-line rule applied (2025-11-03)