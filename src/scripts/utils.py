import pandas as pd

def prepare_category_data(data_preparation, df, logger) -> tuple[pd.DataFrame, dict, dict, dict, list]:
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