# === script4.py ===
# === Module Docstring ===
"""
Relationship plot: Arc diagram of messaging interactions for the MAAP group (Script 4).

Bonus plot (not part of the 5-plot assignment).

Builds participation table and renders network arcs via
:meth:`src.plot_manager.PlotManager.build_visual_relationships_arc`.

Examples
--------
>>> script = Script4(file_manager, data_preparation, plot_manager, image_dir, tables_dir, df)
>>> script.run()
PosixPath('images/arc_diagram_maap.png')
"""

# === Imports ===
from pathlib import Path
import pandas as pd

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
        df: pd.DataFrame,
        settings: ArcPlotSettings | None = None,
    ) -> None:
        """
        Initialize Script4 with required components.

        Args:
            file_manager: FileManager for saving tables.
            data_preparation: DataPreparation for arc data.
            plot_manager: PlotManager for rendering.
            image_dir: Directory to save plot.
            tables_dir: Directory to save participation table.
            df: Enriched DataFrame (required).
            settings: Arc plot settings (optional).
        """
        super().__init__(
            file_manager=file_manager,
            data_preparation=data_preparation,
            plot_manager=plot_manager,
            settings=settings or ArcPlotSettings(),
            df=df,
        )
        self.image_dir = image_dir
        self.tables_dir = tables_dir

    # === Private Helpers ===
    def _build_participation_table(self, df_group: pd.DataFrame, group: str) -> pd.DataFrame | None:
        """
        Build and save participation table for a group.

        Args:
            df_group: Filtered DataFrame for the group.
            group: Group name (for logging).

        Returns:
            pd.DataFrame: Participation table or None if empty.
        """
        data = self.data_preparation.build_visual_relationships_arc(df_group)
        if data is None or data.participation_df.empty:
            self.log_error("participation table empty.")
            return None
        participation_df = data.participation_df
        self.file_manager.save_table(participation_df, self.tables_dir, f"participation_{group}")
        logger.info(f"Saved participation table for {group}")
        return participation_df

    def run(self) -> Path | None:
        """
        Generate and save the author interaction arc diagram.

        Returns:
            Path: Path to saved PNG file.
            None: If data missing or plot fails.
        """
        df_maap = self.df[self.df[Columns.WHATSAPP_GROUP.value] == Groups.MAAP.value].copy()
        if df_maap.empty:
            self.log_error(f"No data for group '{Groups.MAAP.value}'. Skipping.")
            return None

        arc_data = self.data_preparation.build_visual_relationships_arc(df_maap)
        if arc_data is None:
            self.log_error("build_visual_relationships_arc returned None.")
            return None

        logger.info(f"Arc diagram data: {len(arc_data.participation_df)} participation rows")

        fig = self.plot_manager.build_visual_relationships_arc(
            arc_data,
            self.settings
        )
        if fig is None:
            self.log_error("Failed to create arc diagram.")
            return None

        return self.save_figure(fig, self.image_dir, "arc_diagram_maap")


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
# - Add markers #NEW at the end of the module

# NEW: df passed to super(); no *args, **kwargs (2025-11-03)