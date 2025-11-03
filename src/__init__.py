# === Module Docstring ===
"""
src package – root of the DAV project.

Makes the whole project importable as a package when run with:
    python -m src.main

Exposes high-level components for external use.
"""

from __future__ import annotations

# === Public API ===
# (Add any top-level exports here if needed, e.g., Pipeline, DataPreparation)
# from .scripts.pipeline import Pipeline
# from .data_preparation import DataPreparation

__all__ = [
    # "Pipeline",
    # "DataPreparation",
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

# NEW: Added src/__init__.py to make src/ a proper package (2025-11-03)
# NEW: Enables python -m src.main and absolute imports (2025-11-03)