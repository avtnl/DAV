# === Module Docstring ===
"""
Category plot: Messages per author per group (Script 1).

Uses enriched DataFrame, builds validated CategoryPlotData,
and generates grouped bar chart with AvT highlight.

Examples
--------
>>> script = Script1(file_manager, data_preparation, plot_manager, image_dir, tables_dir, df)
>>> script.run()
PosixPath('images/category_plot.png')
"""

# === Imports ===
from pathlib import Path

from loguru import logger
from src.constants import Columns, Groups
from src.plot_manager import CategoriesPlotSettings

from .base import BaseScript


# === Script 1 ===
class Script1(BaseScript):
    """Generate grouped bar chart of messages per author per group."""

    def __init__(
        self,
        file_manager,
        data_preparation,
        plot_manager,
        image_dir: Path,
        tables_dir: Path,
        df,
        settings: CategoriesPlotSettings | None = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize Script1 with required components.

        Args:
            file_manager: FileManager for saving tables.
            data_preparation: DataPreparation for building plot data.
            plot_manager: PlotManager for rendering.
            image_dir: Directory to save plot.
            tables_dir: Directory to save participation tables.
            df: Enriched DataFrame (required).
            settings: Plot settings (optional).
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.
        """
        super().__init__(
            file_manager=file_manager,
            data_preparation=data_preparation,
            plot_manager=plot_manager,
            settings=settings or CategoriesPlotSettings(),
            *args,
            **kwargs,
        )
        self.image_dir = image_dir
        self.tables_dir = tables_dir
        self.df = df

    def run(self) -> Path | None:
        """
        Generate and save category plot.

        Returns:
            Path: Path to saved PNG file.
            None: If DataFrame is missing or plot fails.
        """
        if self.df is None or self.df.empty:
            self.log_error("No DataFrame provided to Script1.")
            return None

        data = self.data_preparation.build_visual_categories(self.df)
        if data is None:
            self.log_error("build_visual_categories returned None.")
            return None

        logger.info(f"Category plot: {len(data.groups)} groups, {data.total_messages:,} messages")

        fig = self.plot_manager.build_visual_categories(data, self.settings)
        if fig is None:
            self.log_error("Failed to create category plot.")
            return None

        return self.save_figure(fig, self.image_dir, "category_plot")


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

# NEW: Fixed df injection – stored locally, not passed to BaseScript (2025-11-02)
# NEW: Uses keyword args in super().__init__() to avoid DataFrame truth error