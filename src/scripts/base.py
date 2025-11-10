# === base.py ===
# === Module Docstring ===
"""
Base module for all analysis scripts.

Provides shared utilities:
- ``file_manager``, ``data_editor``, ``data_preparation``, ``plot_manager``
- Optional ``df`` (enriched DataFrame)
- ``log_error`` and ``save_figure`` helpers

All scripts inherit from :class:`BaseScript` to ensure consistency.
"""

# === Imports ===
from pathlib import Path
from typing import Any
from abc import ABC, abstractmethod

import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
import warnings


# === Base Script ===
class BaseScript(ABC):
    """Abstract base class for all analysis scripts.

    Ensures consistent access to shared components and enforces ``run()`` method.
    """

    def __init__(
        self,
        file_manager,
        data_editor=None,
        data_preparation=None,
        plot_manager=None,
        settings=None,
        df: pd.DataFrame | None = None,
    ) -> None:
        """
        Initialize shared components.

        Args:
            file_manager: FileManager instance (required).
            data_editor: DataEditor instance (optional).
            data_preparation: DataPreparation instance (optional).
            plot_manager: PlotManager instance (optional).
            settings: Plot settings model (optional).
            df: Enriched DataFrame (optional).

        Raises:
            ValueError: If ``file_manager`` is missing.
        """
        if not file_manager:
            raise ValueError("file_manager is required")
        self.file_manager = file_manager
        self.data_editor = data_editor
        self.data_preparation = data_preparation
        self.plot_manager = plot_manager
        self.settings = settings
        self.df = df

    def log_error(self, msg: str) -> None:
        """Log an error message using loguru."""
        logger.error(msg)

    def save_figure(self, fig, image_dir: Path, name: str, add_timestamp: bool = True) -> Path:
        """Delegate figure saving to FileManager with emoji support."""
        plt.rcParams['font.family'] = 'Segoe UI Emoji'
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Glyph .* missing from font"
        )
        return self.file_manager.save_png(fig, image_dir, filename=name, add_timestamp=add_timestamp)

        @abstractmethod
        def run(self) -> Any | None:
            """Execute the script's main logic. Must be implemented by subclasses."""
            pass


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

# NEW: Made BaseScript abstract with @abstractmethod (2025-11-03)
# NEW: Added file_manager validation and class docstring (2025-11-03)
# NEW: Removed *args, **kwargs from __init__ (2025-11-03)