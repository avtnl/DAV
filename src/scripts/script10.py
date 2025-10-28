# scripts/script10.py
from pathlib import Path
from typing import Optional
from loguru import logger
import pandas as pd
from .base import BaseScript


class Script10(BaseScript):
    """
    Step 10 – Add a few convenience columns (year, etc.) and re-save the
    DataFrame as CSV + Parquet.  This is a lightweight “feature-engineering”
    script that prepares the data for the later PCA / t-SNE visualisations.
    """

    def __init__(self, file_manager, data_editor, data_preparation,
                 processed_dir: Path, tables_dir: Path,
                 settings: Optional = None):
        super().__init__(
            file_manager=file_manager,
            data_editor=data_editor,
            data_preparation=data_preparation,
            settings=settings
        )
        self.processed_dir = processed_dir
        self.tables_dir = tables_dir

    def run(self) -> Optional[Path]:
        """
        Returns the path of the saved CSV (or None on failure).
        """
        try:
            # ------------------------------------------------------------------
            # 1. Make sure we have a DataFrame to work with
            # ------------------------------------------------------------------
            if (self.data_preparation.df is None or
                self.data_preparation.df.empty):
                return self.log_error("DataPreparation has no DataFrame.")

            df = self.data_preparation.df.copy()

            # ------------------------------------------------------------------
            # 2. Basic cleaning / feature addition
            # ------------------------------------------------------------------
            df = self.data_editor.clean_author(df)
            df["year"] = self.data_editor.get_year(df)          # adds a 'year' column

            # ------------------------------------------------------------------
            # 3. Persist the enriched DataFrame
            # ------------------------------------------------------------------
            csv_path = self.file_manager.save_csv(
                df, self.processed_dir, prefix="organized_data_script_10"
            )
            parq_path = self.file_manager.save_parq(
                df, self.processed_dir, prefix="organized_data_script_10"
            )

            if csv_path is None or parq_path is None:
                return self.log_error("Failed to save organized DataFrame.")
            logger.info(f"Saved organized data – CSV: {csv_path}, Parquet: {parq_path}")

            return csv_path

        except Exception as e:
            logger.exception(f"Script10 crashed: {e}")
            return None