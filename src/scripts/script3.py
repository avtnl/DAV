# scripts/script3.py
from loguru import logger
from pathlib import Path
from typing import Optional
from .base import BaseScript
from src.constants import Groups, Columns
from src.plot_manager import DistributionPlotSettings


class Script3(BaseScript):
    """Emoji distribution for the MAAP group."""

    def __init__(self, file_manager, data_editor, data_preparation,
                 plot_manager, image_dir: Path, df,
                 settings: Optional[DistributionPlotSettings] = None):
        super().__init__(file_manager, data_editor=data_editor,
                         data_preparation=data_preparation,
                         plot_manager=plot_manager,
                         settings=settings or DistributionPlotSettings())
        self.image_dir = image_dir
        self.df = df

    def run(self):
        df_maap = self.df[self.df[Columns.WHATSAPP_GROUP.value] == Groups.MAAP.value].copy()
        if df_maap.empty:
            return self.log_error(f"No data for group '{Groups.MAAP.value}'. Skipping.")

        df_maap = self.data_editor.clean_for_deleted_media_patterns(df_maap)
        if df_maap is None:
            return self.log_error("clean_for_deleted_media_patterns failed.")

        df_maap, emoji_counts_df = self.data_preparation.build_visual_distribution(df_maap)
        if df_maap is None or emoji_counts_df is None:
            return self.log_error("build_visual_distribution failed.")

        logger.info(f"Unique emojis: {len(emoji_counts_df)}")
        fig = self.plot_manager.build_visual_distribution(emoji_counts_df, self.settings)
        if fig is None:
            return self.log_error("Failed to create emoji bar chart.")
        return self.save_figure(fig, self.image_dir, "emoji_counts_once")