# === base.py ===
# === Module Docstring ===
"""
Base module for all analysis scripts.

Provides shared utilities:
- ``file_manager``, ``data_editor``, ``data_preparation``, ``plot_manager``
- Optional ``df`` (enriched DataFrame)
- ``log_error`` and ``save_figure`` helpers
"""

# === Imports ===
from pathlib import Path
from typing import Any

import pandas as pd  # ← ADDED: required for df: pd.DataFrame type hint
from loguru import logger


# === Base Script ===
class BaseScript:
    """Common functionality for all scripts."""

    def __init__(
        self,
        file_manager,
        data_editor=None,
        data_preparation=None,
        plot_manager=None,
        settings=None,
        df: pd.DataFrame | None = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize shared components.

        Args:
            file_manager: FileManager instance.
            data_editor: DataEditor instance (optional).
            data_preparation: DataPreparation instance (optional).
            plot_manager: PlotManager instance (optional).
            settings: Plot settings model (optional).
            df: Enriched DataFrame (optional).
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.
        """
        self.file_manager = file_manager
        self.data_editor = data_editor
        self.data_preparation = data_preparation
        self.plot_manager = plot_manager
        self.settings = settings
        self.df = df

    def log_error(self, msg: str) -> None:
        """Log an error message."""
        logger.error(msg)

    def save_figure(self, fig, image_dir: Path, name: str) -> Path:
        """
        Save a matplotlib figure to disk.

        Args:
            fig: Matplotlib Figure object.
            image_dir: Directory to save image.
            name: Base filename (without extension).

        Returns:
            Path to saved image.
        """
        image_dir.mkdir(parents=True, exist_ok=True)
        path = image_dir / f"{name}.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        logger.success(f"Plot saved: {path}")
        return path


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

# NEW: Added pandas import, full Google docstrings, and df parameter (2025-11-03)