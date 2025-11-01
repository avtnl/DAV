# === Module Docstring ===
"""
Arc diagram of messaging interactions for the MAAP group.

Builds participation table and renders network arcs via
:meth:`src.plot_manager.PlotManager.build_visual_relationships_arc`.

Examples
--------
>>> script = Script4(...)
>>> script.run()
PosixPath('images/network_interactions_maap.png')
"""

# === Imports ===
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from src.constants import Columns, Groups
from src.plot_manager import ArcPlotSettings

from .base import BaseScript


# === Script 4 ===
class Script4(BaseScript):
    """Arc diagram for the MAAP group."""

    def __init__(
        self,
        file_manager,
        data_preparation,
        plot_manager,
        image_dir: Path,
        tables_dir: Path,
        group_authors: Any,
        original_df=None,
        settings: ArcPlotSettings | None = None,
    ) -> None:
        super().__init__(
            file_manager,
            data_preparation=data_preparation,
            plot_manager=plot_manager,
            settings=settings or ArcPlotSettings(),
            df=original_df,
        )
        self.image_dir = image_dir
        self.tables_dir = tables_dir
        self.group_authors = group_authors

    # === Private Helpers ===
    def _build_participation_table(self, df_group: pd.DataFrame, group: str) -> pd.DataFrame | None:
        """Build and save participation table."""
        participation_df = self.data_preparation.build_visual_relationships_arc(
            df_group, self.group_authors.get(group, [])
        )
        if participation_df is None or participation_df.empty:
            self.log_error("participation table empty.")
            return None
        self.save_table(participation_df, self.tables_dir, f"participation_{group}")
        logger.info(f"Saved participation table for {group}")
        return participation_df

    def run(self) -> Path | None:
        """Generate and save arc diagram."""
        group = Groups.MAAP.value
        df_group = (
            self.df[self.df[Columns.WHATSAPP_GROUP.value] == group].copy()
            if self.df is not None
            else self.data_preparation.df[
                self.data_preparation.df[Columns.WHATSAPP_GROUP.value] == group
            ].copy()
        )

        if df_group.empty:
            self.log_error(f"No data for group '{group}'. Skipping arc diagram.")
            return None

        participation_df = self._build_participation_table(df_group, group)
        if participation_df is None:
            self.log_error("Participation table creation failed.")
            return None

        fig = self.plot_manager.build_visual_relationships_arc(
            participation_df, group, self.settings
        )
        if fig is None:
            self.log_error("Arc diagram plot failed.")
            return None
        return self.save_figure(fig, self.image_dir, f"network_interactions_{group}")


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


# NEW: Full refactor with Google docstring, section comments, and SEnum (2025-10-31)
