# scripts/script4.py
import pandas as pd
from loguru import logger
from pathlib import Path
from typing import Optional
from .base import BaseScript
from src.constants import Groups, Columns
from src.plot_manager import ArcPlotSettings


class Script4(BaseScript):
    """Arc diagram for the MAAP group."""

    def __init__(self, file_manager, data_preparation, plot_manager,
                 image_dir: Path, tables_dir: Path, group_authors,
                 original_df=None, settings: Optional[ArcPlotSettings] = None):
        super().__init__(file_manager, data_preparation=data_preparation,
                         plot_manager=plot_manager,
                         settings=settings or ArcPlotSettings(), df=original_df)
        self.image_dir = image_dir
        self.tables_dir = tables_dir
        self.group_authors = group_authors

    # ------------------------------------------------------------------ #
    def _build_participation_table(self, df_group: pd.DataFrame, group: str):
        participation_df = self.data_preparation.build_visual_relationships_arc(
            df_group, self.group_authors.get(group, [])
        )
        if participation_df is None or participation_df.empty:
            self.log_error("participation table empty.")
            return None
        self.save_table(participation_df, self.tables_dir, f"participation_{group}")
        logger.info(f"Saved participation table for {group}")
        return participation_df

    # ------------------------------------------------------------------ #
    def run(self):
        group = Groups.MAAP.value
        df_group = (self.df[self.df[Columns.WHATSAPP_GROUP.value] == group].copy()
                    if self.df is not None else
                    self.data_preparation.df[self.data_preparation.df[Columns.WHATSAPP_GROUP.value] == group].copy())

        if df_group.empty:
            return self.log_error(f"No data for group '{group}'. Skipping arc diagram.")

        participation_df = self._build_participation_table(df_group, group)
        if participation_df is None:
            return self.log_error("Participation table creation failed.")

        fig = self.plot_manager.build_visual_relationships_arc(
            participation_df, group, self.settings
        )
        if fig is None:
            return self.log_error("Arc diagram plot failed.")
        return self.save_figure(fig, self.image_dir, f"network_interactions_{group}")