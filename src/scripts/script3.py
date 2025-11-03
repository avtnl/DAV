# === script3.py ===
# === Module Docstring ===
"""
Distribution plot: Emoji distribution for the MAAP group (Script 3).

Filters the enriched DataFrame for the MAAP group, prepares the emoji-frequency
DataFrame via :meth:`src.data_preparation.DataPreparation.build_visual_distribution`,
and generates the bar + cumulative line chart via
:meth:`src.plot_manager.PlotManager.build_visual_distribution`.

Examples
--------
>>> script = Script3(file_manager, data_editor, data_preparation, plot_manager, image_dir, df)
>>> script.run()
PosixPath('images/emoji_counts_once.png')
"""

# === Imports ===
from pathlib import Path
import pandas as pd

from loguru import logger
from src.constants import Columns, Groups
from src.plot_manager import DistributionPlotSettings

from .base import BaseScript


# === Script 3 ===
class Script3(BaseScript):
    """Generate emoji distribution plot for MAAP group."""

    def __init__(
        self,
        file_manager,
        data_editor,
        data_preparation,
        plot_manager,
        image_dir: Path,
        df: pd.DataFrame,
        settings: DistributionPlotSettings | None = None,
    ) -> None:
        """
        Initialize Script3 with required components.

        Args:
            file_manager: FileManager (required by BaseScript).
            data_editor: DataEditor (required for emoji parsing).
            data_preparation: DataPreparation for distribution data.
            plot_manager: PlotManager for rendering.
            image_dir: Directory to save plot.
            df: Enriched DataFrame (required).
            settings: Plot settings (optional).
        """
        super().__init__(
            file_manager=file_manager,
            data_editor=data_editor,
            data_preparation=data_preparation,
            plot_manager=plot_manager,
            settings=settings or DistributionPlotSettings(),
            df=df,
        )
        self.image_dir = image_dir

    def run(self) -> Path | None:
        """
        Generate and save the emoji distribution plot.

        Returns:
            Path: Path to saved PNG file.
            None: If data missing or plot fails.
        """
        df_maap = self.df[self.df[Columns.WHATSAPP_GROUP.value] == Groups.MAAP.value].copy()
        if df_maap.empty:
            self.log_error(f"No data for group '{Groups.MAAP.value}'. Skipping.")
            return None

        distribution_data = self.data_preparation.build_visual_distribution(df_maap)
        if distribution_data is None:
            self.log_error("build_visual_distribution returned None.")
            return None

        logger.info(f"Unique emojis: {len(distribution_data.emoji_counts_df)}")
        fig = self.plot_manager.build_visual_distribution(
            distribution_data,
            self.settings
        )
        if fig is None:
            self.log_error("Failed to create emoji bar chart.")
            return None

        return self.save_figure(fig, self.image_dir, "emoji_counts_once")


# === CODING STANDARD ===
# - `# === Module Docstring ===` before """
# - Google-style docstrings
# - `# === Section Name ===` for all blocks
# - Inline: `# One space, sentence case`
# - Tags: `# TODO:`, `# NOTE:`, `# NEW: (YYYY-MM-DD)`, `# FIXME:`
# - Type hints in function signatures
# - Examples: with >>>
# - No long ----- lines
# - No mixed styles
# - Add markers #NEW at the end of the module

# NEW: Removed *args, **kwargs; df passed to super() (2025-11-03)
# NEW: Removed local self.df = df (2025-11-03)