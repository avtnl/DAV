# src/scripts/utils.py
from typing import Tuple, Optional
from loguru import logger
import pandas as pd

def prepare_category_data(data_preparation, df, logger) -> Tuple[pd.DataFrame, dict, dict, dict, list]:
    """
    Prepare category-related data (group authors, non-Anthony averages, etc.) for scripts requiring it.

    Args:
        data_preparation: Instance to call build_visual_categories.
        df: Input DataFrame.
        logger: Logger instance.

    Returns:
        tuple: (df, group_authors, non_anthony_group, anthony_group, sorted_groups)
               or (None, None, None, None, None) on failure.
    """
    try:
        result = data_preparation.build_visual_categories(df)
        if result is None:
            logger.error("build_visual_categories returned None.")
            return None, None, None, None, None
        
        df_out, group_authors, non_anthony_group, anthony_group, sorted_groups = result
        
        if df_out is None or group_authors is None or sorted_groups is None:
            logger.error("Failed to initialize required variables for category-dependent scripts.")
            return None, None, None, None, None
        
        return df_out, group_authors, non_anthony_group, anthony_group, sorted_groups

    except Exception as e:
        logger.exception(f"Error preparing category data: {e}")
        return None, None, None, None, None