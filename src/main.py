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
SCRIPTS = [1]
SCRIPT_6_DETAILS = ["pca", False, 0, 25, True, True, 1] # Only used if 6 in SCRIPTS | by_group, draw_ellipses, conf_level, use_embeddings, hybrid, model_id


# === Main Execution ===
if __name__ == "__main__":
    result = Pipeline.run(
        scripts=SCRIPTS,
        script_6_details=SCRIPT_6_DETAILS
    )

    # === Graceful Dashboard Exit (Script7 only) ===
    if 7 in SCRIPTS:
        print("\nDashboard: http://localhost:8501")
        print("Close browser and press ENTER to exit.")
        try:
            input()
        except KeyboardInterrupt:
            pass
        finally:
            print("Goodbye!\n")


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
# NEW: Added graceful exit for Script7 with ENTER prompt (2025-11-03)