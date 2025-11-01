# === Module Docstring ===
"""
Emoji distribution plot for the MAAP group.

Cleans media deletion patterns, counts emoji usage, and generates a bar + cumulative
line chart via :meth:`src.plot_manager.PlotManager.build_visual_distribution`.

Examples
--------
>>> script = Script3(...)
>>> script.run()
PosixPath('images/emoji_counts_once.png')
"""

# === Imports ===
from pathlib import Path

from loguru import logger
from src.constants import Columns, Groups
from src.plot_manager import DistributionPlotSettings

from .base import BaseScript


# === Script 3 ===
class Script3(BaseScript):
    """Emoji distribution for the MAAP group."""

    def __init__(
        self,
        file_manager,
        data_editor,
        data_preparation,
        plot_manager,
        image_dir: Path,
        df,
        settings: DistributionPlotSettings | None = None,
    ) -> None:
        super().__init__(
            file_manager,
            data_editor=data_editor,
            data_preparation=data_preparation,
            plot_manager=plot_manager,
            settings=settings or DistributionPlotSettings(),
        )
        self.image_dir = image_dir
        self.df = df

    def run(self) -> Path | None:
        """Generate and save emoji distribution plot."""
        df_maap = self.df[self.df[Columns.WHATSAPP_GROUP.value] == Groups.MAAP.value].copy()
        if df_maap.empty:
            self.log_error(f"No data for group '{Groups.MAAP.value}'. Skipping.")
            return None

        df_maap = self.data_editor.clean_for_deleted_media_patterns(df_maap)
        if df_maap is None:
            self.log_error("clean_for_deleted_media_patterns failed.")
            return None

        df_maap, emoji_counts_df = self.data_preparation.build_visual_distribution(df_maap)
        if df_maap is None or emoji_counts_df is None:
            self.log_error("build_visual_distribution failed.")
            return None

        logger.info(f"Unique emojis: {len(emoji_counts_df)}")
        fig = self.plot_manager.build_visual_distribution(emoji_counts_df, self.settings)
        if fig is None:
            self.log_error("Failed to create emoji bar chart.")
            return None
        return self.save_figure(fig, self.image_dir, "emoji_counts_once")


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


# NEW: Full refactor with Google docstring, type hints, and SEnum usage (2025-10-31)
