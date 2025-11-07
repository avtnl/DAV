# === script0.py ===
# === Module Docstring ===
"""
Preprocess raw WhatsApp chat exports (Script0).

Reads, cleans, combines, and saves data as Parquet + CSV.
Runs automatically if no cache exists.

Examples
--------
>>> script = Script0(file_manager, data_editor, data_preparation, processed_dir, config, image_dir)
>>> df = script.run()
>>> df.shape
(11302, 62)
"""

# === Imports ===
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
from loguru import logger

from .base import BaseScript


# === Script 0 ===
class Script0(BaseScript):
    """Read raw WhatsApp files, clean, combine and store CSV + Parquet."""

    def __init__(
        self,
        file_manager: Any,
        data_editor: Any,
        data_preparation: Any,
        processed_dir: Path,
        config: Dict[str, Any],
        image_dir: Path,
    ) -> None:
        """
        Initialize Script0 with required components.

        Args:
            file_manager: FileManager for loading raw data.
            data_editor: DataEditor for cleaning.
            data_preparation: DataPreparation for enrichment.
            processed_dir: Directory to save processed files.
            config: Full config dictionary from config.toml.
            image_dir: Directory for output images (unused here).
        """
        super().__init__(
            file_manager=file_manager,
            data_editor=data_editor,
            data_preparation=data_preparation,
        )
        self.processed_dir = processed_dir
        self.config = config
        self.image_dir = image_dir
        self.tables_dir = Path("tables")
        self.tables_dir.mkdir(exist_ok=True)

    def run(self) -> pd.DataFrame | None:
        """
        Execute preprocessing and return enriched DataFrame.

        Returns:
            pd.DataFrame: The enriched dataset.
            None: If preprocessing fails.

        Raises:
            Exception: Propagated and logged via loguru.
        """
        logger.info("Script0: Starting preprocessing...")

        try:
            result = self.file_manager.get_preprocessed_data(
                data_editor=self.data_editor,
                data_preparation=self.data_preparation,
                config=self.config,
                processed_dir=self.processed_dir,
            )

            if result is None or "df" not in result:
                self.log_error("Script0: FileManager returned None or no 'df'.")
                return None

            df = result["df"]
            logger.success("Script0: Preprocessing completed successfully.")
            return df  # ← Return DataFrame (fixed)

        except Exception as e:
            logger.exception(f"Script0: Preprocessing failed: {e}")
            return None


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

# NEW: Fixed typo: processed_processed_dir → processed_dir (2025-11-03)
# NEW: Removed *args, **kwargs; df not passed (not needed) (2025-11-03)
# NEW: (2025-11-04) – Simplified run to wrapper around get_preprocessed_data; Removed duplicate saving
# NEW: Changed run() to return DataFrame (2025-11-07)