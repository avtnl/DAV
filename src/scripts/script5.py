# === Module Docstring ===
"""
Relationships plot: Bubble plot of words vs punctuation across multiple groups (Script 5).

Filters specified groups, computes averages, and renders bubble plot via
:meth:`src.plot_manager.PlotManager.build_visual_relationships_bubble`.

Examples
--------
>>> script = Script5(...)
>>> script.run()
PosixPath('images/bubble_plot_words_vs_punct.png')
"""

# === Imports ===
from pathlib import Path

import pandas as pd
from loguru import logger

from src.constants import Columns, Groups
from src.plot_manager import BubblePlotSettings

from .base import BaseScript


# === Script 5 ===
class Script5(BaseScript):
    """Bubble plot across multiple groups."""

    def __init__(
            self,
            file_manager,
            data_preparation,
            plot_manager,
            image_dir: Path,
            df: pd.DataFrame | None = None,
            settings: BubblePlotSettings | None = None,
        ) -> None:
            super().__init__(
                file_manager,
                data_preparation=data_preparation,
                plot_manager=plot_manager,
                settings=settings or BubblePlotSettings(),
                df=df,
            )
            self.image_dir = image_dir

    def run(self) -> Path | None:
        """Generate and save multi-group bubble plot."""
        groups = [Groups.MAAP.value, Groups.GOLFMATEN.value, Groups.DAC.value, Groups.TILLIES.value]
        df_groups = self.df[self.df[Columns.WHATSAPP_GROUP.value].isin(groups)].copy()
        if df_groups.empty:
            self.log_error(f"No data for groups {groups}. Skipping bubble plot.")
            return None

        data = self.data_preparation.build_visual_relationships_bubble(df_groups)
        if data is None or data.feature_df.empty:
            self.log_error("build_visual_relationships_bubble returned nothing.")
            return None

        fig = self.plot_manager.build_visual_relationships_bubble(data.feature_df, self.settings)
        if fig is None:
            self.log_error("Bubble plot creation failed.")
            return None
        return self.save_figure(fig, self.image_dir, "bubble_plot_words_vs_punct")


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

# NEW: Full refactor with Google docstring, return type, and SEnum (2025-10-31)
# NEW: Renamed BubbleNewPlotSettings → BubblePlotSettings (2025-11-03)
# NEW: Fixed df assignment – use BaseScript.df, removed self.df = df (2025-11-03)
