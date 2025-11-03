# === Module Docstring ===
"""
Entry point for the WhatsApp Chat Analyzer pipeline.
"""

# === Imports ===
from src.scripts.pipeline import Pipeline


# === Pipeline Configuration ===
# Only include scripts you want to execute.
# SCRIPT_6_DETAILS is only applicable if Script6 is included in SCRIPTS.
# See script6.py for full configuration details.
SCRIPTS = [6]
SCRIPT_6_DETAILS = ["tsne", True, True, False, True, 1]  # Only used if 6 in SCRIPTS | by_group, draw_ellipses, use_emb, hybrid, model_id

# === Main Execution ===
if __name__ == "__main__":
    Pipeline.run(
        scripts=SCRIPTS,
        script_6_details=SCRIPT_6_DETAILS  # ← REQUIRED for Script6
    )

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
# NEW: script_validation(scripts, SCRIPT_6_DETAILS)
