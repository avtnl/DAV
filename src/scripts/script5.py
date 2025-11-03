# === script5.py ===
# === Module Docstring ===
"""
Relationships plot: Bubble plot of words vs punctuation across multiple groups (Script 5).

Filters specified groups, computes averages, and renders bubble plot via
:meth:`src.plot_manager.PlotManager.build_visual_relationships_bubble`.

Examples
--------
>>> script = Script5(file_manager, data_preparation, plot_manager, image_dir, df)
>>> script.run()
PosixPath('images/bubble_words_vs_punct.png')
"""

# === Imports ===
from pathlib import Path
import pandas as pd

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
        df: pd.DataFrame,
        settings: BubblePlotSettings | None = None,
    ) -> None:
        """
        Initialize Script5 with required components.

        Args:
            file_manager: FileManager (required by BaseScript).
            data_preparation: DataPreparation for bubble data.
            plot_manager: PlotManager for rendering.
            image_dir: Directory to save plot.
            df: Enriched DataFrame (required).
            settings: Bubble plot settings (optional).
        """
        super().__init__(
            file_manager=file_manager,
            data_preparation=data_preparation,
            plot_manager=plot_manager,
            settings=settings or BubblePlotSettings(),
            df=df,
        )
        self.image_dir = image_dir

    def run(self) -> Path | None:
        """
        Generate and save the words vs punctuation bubble plot.

        Returns:
            Path: Path to saved PNG file.
            None: If data missing or plot fails.
        """
        if self.df.empty:
            self.log_error("Input DataFrame is empty. Skipping.")
            return None

        bubble_data = self.data_preparation.build_visual_relationships_bubble(self.df)
        if bubble_data is None:
            self.log_error("build_visual_relationships_bubble returned None.")
            return None

        logger.info(f"Bubble plot data: {len(bubble_data.feature_df)} author-group rows")

        fig = self.plot_manager.build_visual_relationships_bubble(
            bubble_data,
            self.settings
        )
        if fig is None:
            self.log_error("Failed to create bubble plot.")
            return None

        return self.save_figure(fig, self.image_dir, "bubble_words_vs_punct")


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
# - Add markers #NEW at the end of the module capturing the latest changes.

# NEW: df passed to super(); no *args, **kwargs (2025-11-03)