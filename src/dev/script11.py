from pathlib import Path
from typing import List, Optional
from loguru import logger
import pandas as pd
from .base import BaseScript
from src.constants import DataFeed, PlotFeed, GroupByPeriod, DeleteAttributes, PlotType, Columns, Groups
from src.dev.plot_manager import DimReductionSettings, PMNoMessageContentSettings


class Script11(BaseScript):
    """
    Step 11 – Build PCA / t-SNE visualisations of *non-message* features
    (e.g. message length, emoji count, time-of-day, etc.) grouped by week /
    month / year.
    """

    def __init__(self, file_manager, data_editor, data_preparation,
                 plot_manager, processed_dir: Path, image_dir: Path,
                 settings: Optional = None):
        super().__init__(
            file_manager=file_manager,
            data_editor=data_editor,
            data_preparation=data_preparation,
            plot_manager=plot_manager,
            settings=settings
        )
        self.processed_dir = processed_dir
        self.image_dir = image_dir

    # ------------------------------------------------------------------ #
    # 1. Input validation
    # ------------------------------------------------------------------ #
    def _check_input(self, data_feed: str, plot_feed: str,
                     groupby_period: str, delete_specific_attributes: bool,
                     plot_type: str) -> bool:
        if data_feed not in ['non_redundant', 'redundant']:
            self.log_error(f"Invalid data_feed: {data_feed}")
            return False
        if data_feed == 'non_redundant':
            if plot_feed not in ['per_group', 'global', 'both']:
                self.log_error(f"Invalid plot_feed for non_redundant: {plot_feed}")
                return False
        else:  # redundant
            if plot_feed != 'per_group':
                self.log_error(f"Invalid plot_feed for redundant: {plot_feed}")
                return False
        if groupby_period not in ['week', 'month', 'year']:
            self.log_error(f"Invalid groupby_period: {groupby_period}")
            return False
        if delete_specific_attributes and (data_feed != 'non_redundant' or groupby_period != 'month'):
            self.log_error("delete_specific_attributes only allowed with non_redundant + month")
            return False
        if plot_type not in ['both', 'pca', 'tsne']:
            self.log_error(f"Invalid plot_type: {plot_type}")
            return False
        logger.info("Input parameters validated.")
        return True

    # ------------------------------------------------------------------ #
    # 2. Load the latest organized file
    # ------------------------------------------------------------------ #
    def _load_and_preprocess(self, data_feed: str) -> Optional[pd.DataFrame]:
        prefix = ("organized_data_no_redundancy"
                  if data_feed == 'non_redundant' else "organized_data_with_redundancy")
        latest_file = self.file_manager.find_latest_file(
            self.processed_dir, prefix=prefix, suffix=".csv"
        )
        if latest_file is None:
            return self.log_error(f"No {prefix}.csv found.")
        logger.info(f"Loading {latest_file}")

        df = pd.read_csv(latest_file,
                         parse_dates=['timestamp'],
                         date_parser=lambda x: pd.to_datetime(x, errors='coerce'))

        # add month / week columns
        df['month'] = self.data_editor.get_month(df)
        df['week']  = self.data_editor.get_week(df)

        required = [Columns.WHATSAPP_GROUP.value, Columns.YEAR.value,
                    Columns.MONTH.value, Columns.WEEK.value]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return self.log_error(f"Missing columns: {missing}")

        # drop the Tillies group
        df = df[df[Columns.WHATSAPP_GROUP.value] != Groups.TILLIES.value].copy()
        if df.empty:
            return self.log_error("DataFrame empty after dropping Tillies.")
        logger.info(f"Loaded {len(df)} rows (Tillies excluded).")
        return df

    # ------------------------------------------------------------------ #
    # 3. Optional attribute deletion + re-save
    # ------------------------------------------------------------------ #
    def _delete_and_save(self, df: pd.DataFrame,
                         delete_specific_attributes: bool) -> Optional[pd.DataFrame]:
        if not delete_specific_attributes:
            return df

        df = self.data_editor.delete_specific_attributes(df)
        if df is None:
            return self.log_error("delete_specific_attributes failed.")

        csv = self.file_manager.save_csv(df, self.processed_dir,
                                         prefix="organized_data_specific")
        parq = self.file_manager.save_parq(df, self.processed_dir,
                                           prefix="organized_data_specific")
        if csv is None or parq is None:
            return self.log_error("Failed to save after attribute deletion.")
        logger.info(f"Saved specific-attribute version – CSV: {csv}")
        return df

    # ------------------------------------------------------------------ #
    # 4. Main run()
    # ------------------------------------------------------------------ #
    def run(self):
        try:
            # ---- configuration (you can move these to config.toml later) ----
            data_feed   = DataFeed.NON_REDUNDANT.value
            plot_feed   = PlotFeed.GLOBAL.value
            groupby     = GroupByPeriod.MONTH.value
            delete_attr = DeleteAttributes.FALSE.value
            plot_type   = PlotType.TSNE.value

            if not self._check_input(data_feed, plot_feed, groupby,
                                     delete_attr, plot_type):
                return None

            df = self._load_and_preprocess(data_feed)
            if df is None:
                return None

            df = self._delete_and_save(df, delete_attr)
            if df is None:
                return None

            # ---- feature matrix ------------------------------------------------
            feature_df = self.data_preparation.build_visual_no_message_content(
                df, groupby_period=groupby
            )
            if feature_df is None:
                return self.log_error("build_visual_no_message_content returned None.")

            # ---- dimensionality-reduction plots --------------------------------
            figs = self.plot_manager.build_visual_no_message_content(
                feature_df,
                plot_type=plot_type,
                dr_settings=DimReductionSettings(),
                nmc_settings=PMNoMessageContentSettings(plot_type=plot_feed)
            )
            if figs is None:
                return self.log_error("build_visual_no_message_content produced no figures.")

            for info in figs:
                self.save_figure(info['fig'], self.image_dir, info['filename'])

            # ---- optional correlation heatmap (only for month grouping) -------
            if groupby == 'month':
                corr = self.data_preparation.compute_month_correlations(feature_df)
                if corr is None:
                    return self.log_error("compute_month_correlations failed.")
                fig_corr = self.plot_manager.plot_month_correlations(corr)
                if fig_corr is None:
                    return self.log_error("plot_month_correlations failed.")
                self.save_figure(fig_corr, self.image_dir, "month_correlations")

            logger.success("Script11 finished successfully.")
            return None

        except Exception as e:
            logger.exception(f"Script11 crashed: {e}")
            return None