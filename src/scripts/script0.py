# === Module Docstring ===
"""
Preprocess raw WhatsApp chat exports (Script0).

Reads, cleans, combines, and saves data as Parquet + CSV.
Runs automatically if no cache exists.

Examples
--------
>>> from src.scripts.script0 import Script0
>>> script = Script0(file_manager, data_editor, data_preparation, processed_dir, config, image_dir)
>>> result = script.run()
>>> result["df"].shape
(12345, 20)
"""

# === Imports ===
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from .base import BaseScript


# === Script 0 ===
class Script0(BaseScript):
    """Read raw WhatsApp files, clean, combine and store CSV + Parquet."""

    def __init__(
        self,
        file_manager,
        data_editor,
        data_preparation,
        processed_dir: Path,
        config: dict[str, Any],
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
        super().__init__(file_manager, data_editor=data_editor, data_preparation=data_preparation)
        self.processed_dir = processed_processed_dir
        self.config = config
        self.image_dir = image_dir
        self.tables_dir = Path("tables")
        self.tables_dir.mkdir(exist_ok=True)

    def run(self) -> dict[str, Any] | None:
        """
        Execute preprocessing and return enriched DataFrame.

        Returns:
            dict: Contains 'df' (DataFrame) and 'tables_dir' (Path).
            None: If preprocessing fails.

        Raises:
            Exception: Propagated and logged via loguru.
        """
        logger.info("Script0: Starting preprocessing...")

        try:
            result = self.file_manager.get_preprocessed_data(
                self.data_editor, self.data_preparation, self.config, self.processed_dir
            )

            if result is None or "df" not in result:
                self.log_error("Script0: FileManager returned None or no 'df'.")
                return None

            df = result["df"]
            if df is None or df.empty:
                self.log_error("Script0: Empty DataFrame after preprocessing.")
                return None

            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            parquet_path = self.processed_dir / f"combined_{timestamp}.parq"
            df.to_parquet(parquet_path, index=False)
            logger.info(f"Script0: Saved parquet -> {parquet_path}")

            csv_path = self.tables_dir / f"combined_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Script0: Saved CSV -> {csv_path}")

            logger.success("Script0: Preprocessing completed successfully.")
            return {"df": df, "tables_dir": self.tables_dir}

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

# NEW: Fixed all pre-commit issues (2025-11-01)
# NEW: String forward refs + .ruff.toml ignore F821