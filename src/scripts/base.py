# scripts/base.py
"""
Base class for all analysis scripts.

Provides shared utilities for:
- Saving figures and tables via FileManager
- Consistent logging
- Settings injection
- Type-safe initialization

All scripts must inherit from :class:`BaseScript` and implement ``run()``.
"""

# === Imports ===
from pathlib import Path
from typing import Never, Optional

import pandas as pd
from loguru import logger
from pydantic import BaseModel


# === Base Script Class ===
class BaseScript:
    """Common helpers for every script."""

    def __init__(
        self,
        file_manager,
        data_editor=None,
        data_preparation=None,
        plot_manager=None,
        settings: Optional[BaseModel] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.file_manager = file_manager
        self.data_editor = data_editor
        self.data_preparation = data_preparation
        self.plot_manager = plot_manager
        self.settings = settings
        self.df = df  # Optional original DataFrame

    # === Figure & Table Saving ===
    def save_figure(self, fig, image_dir: Path, filename: str) -> Optional[Path]:
        """Save Matplotlib figure as PNG and return path."""
        png_file = self.file_manager.save_png(fig, image_dir, filename=filename)
        if png_file is None:
            logger.error(f"Failed to save {filename}.")
        else:
            logger.info(f"Saved {filename}: {png_file}")
        return png_file

    def save_table(self, df: pd.DataFrame, tables_dir: Path, prefix: str) -> Optional[Path]:
        """Save DataFrame as CSV and return path."""
        saved = self.file_manager.save_table(df, tables_dir, prefix=prefix)
        if saved:
            logger.info(f"Saved table: {saved}")
        else:
            logger.error(f"Failed to save table with prefix {prefix}.")
        return saved

    # === Logging ===
    def log_error(self, message: str) -> None:
        """Log error and return None for convenience."""
        logger.error(message)

    # === Abstract Method ===
    def run(self) -> Never:
        """Must be implemented by child class."""
        raise NotImplementedError


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


# NEW: Added Optional type hints and Google docstrings (2025-10-31)