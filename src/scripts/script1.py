# === Module Docstring ===
"""
Script1 - Total Messages by Group and Author

Builds and saves the category bar chart using validated CategoryPlotData.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from pydantic import BaseModel  # ADD THIS LINE

from src.plot_manager import CategoriesPlotSettings
from src.constants import Columns
from .base import BaseScript


class Script1Config(BaseModel):
    image_dir: Path
    settings: CategoriesPlotSettings


class Script1(BaseScript):
    def __init__(
        self,
        file_manager: Any,
        data_preparation: Any,
        plot_manager: Any,
        image_dir: Path,
        tables_dir: Path,
        config: CategoriesPlotSettings,
        df: pd.DataFrame | None = None,
    ) -> None:
        super().__init__(
            file_manager,
            data_preparation=data_preparation,
            plot_manager=plot_manager,
            settings=config,
            df=df,
        )
        self.image_dir = image_dir
        self.tables_dir = tables_dir
        self.config = Script1Config(image_dir=image_dir, settings=config)

    def run(self) -> None:
        if self.df is None:
            return self.log_error("No DataFrame provided.")

        data = self.data_preparation.build_visual_categories(self.df)
        if not data:
            return self.log_error("Failed to build category data.")

        fig = self.plot_manager.build_visual_categories(data, self.config.settings)
        if not fig:
            return self.log_error("Failed to create total messages bar chart.")

        saved = self.save_figure(fig, self.image_dir, "total_messages_by_group_author")
        if saved:
            logger.info(f"Saved category chart: {saved}")


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

# NEW: Updated __init__ to accept 6 args + df (2025-11-01)
# NEW: Added Script1Config for type safety