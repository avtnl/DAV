# scripts/utils.py
"""
Utility functions shared across scripts.

Currently provides data preparation for category-based visualizations.
"""

# === Imports ===
from typing import Tuple, Optional

import pandas as pd
from loguru import logger


# === Category Data Prep ===
def prepare_category_data(
    data_preparation, df: Optional[pd.DataFrame], logger
) -> Tuple[Optional[pd.DataFrame], Optional[dict], Optional[dict], Optional[dict], Optional[list]]:
    """
    Prepare inputs for category visualizations.

    Returns
    -------
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

        return df_out, group_authors, non_anthony_group, anthony_group, sorted_groups

    except Exception as e:
        logger.exception(f"Error in prepare_category_data: {e}")
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
# - Add markers #NEW at the end of the module capturing the latest changes. There can be a list of more #NEW lines.


# NEW: Added full return type annotation and Google docstring (2025-10-31)