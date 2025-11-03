# === Module Docstring ===
"""
Entry point for the WhatsApp Chat Analyzer pipeline.

Configures and runs selected analysis scripts via :class:`src.scripts.pipeline.Pipeline`.

Examples
--------
>>> from src.scripts.pipeline import Pipeline
>>> Pipeline.run(scripts=[1, 2, 5])
"""

# === Imports ===
from src.scripts.pipeline import Pipeline

# === Pipeline Configuration ===
# Order matters! Script0 (preprocessing) runs automatically first.
# Only include scripts you want to execute.
SCRIPTS = [1,2,3,4]

# === Main Execution ===
if __name__ == "__main__":
    Pipeline.run(scripts=SCRIPTS)

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
# - Add markers #NEW at the end of the module capturing the latest changes. There can be a list of more #NEW lines.

# NEW: Standardized main entry with Google docstring and SCRIPTS config (2025-10-31)
