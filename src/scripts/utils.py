# === Module Docstring ===
"""
Utility functions shared across scripts.

Currently provides data preparation for category-based visualizations.

Examples
--------
>>> from src.scripts.utils import prepare_category_data
>>> data = prepare_category_data(prep, df, logger)
"""

# === Imports ===
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging

    import pandas as pd


# === Category Data Prep ===
def prepare_category_data(
    data_preparation: DataPreparation,  # string forward ref - Ruff respects it
    df: pd.DataFrame | None,
    logger: logging.Logger,
) -> tuple[
    pd.DataFrame | None,
    dict[str, list[str]] | None,
    dict[str, list[str]] | None,
    dict[str, list[str]] | None,
    list[str] | None,
]:
    """
    Prepare inputs for category visualizations.

    Args:
        data_preparation: Instance of DataPreparation.
        df: Input DataFrame.
        logger: Logger instance.

    Returns:
        tuple
            (df_out, group_authors, non_anthony_group, anthony_group, sorted_groups)
            or (None, None, None, None, None) on failure.
    """
    if df is None or df.empty:
        logger.error("prepare_category_data: df is None or empty.")
        return None, None, None, None, None

    try:
        result = data_preparation.build_visual_categories(df)
        if result is None:
            logger.error("build_visual_categories returned None.")
            return None, None, None, None, None

        df_out, group_authors, non_anthony_group, anthony_group, sorted_groups = result

        if df_out is None or group_authors is None or sorted_groups is None:
            logger.error("build_visual_categories: missing required outputs.")
            return None, None, None, None, None
        else:
            return df_out, group_authors, non_anthony_group, anthony_group, sorted_groups

    except Exception:
        logger.exception("Error in prepare_category_data")
        return None, None, None, None, None


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

# NEW: Fixed TRY300 with explicit else (2025-11-01)
# NEW: String forward ref + respect-string-annotations for Ruff 0.14.3
