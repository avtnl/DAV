# src/scripts/script0.py
from pathlib import Path
from typing import Optional, Dict
from .base import BaseScript
from loguru import logger
import pandas as pd


class Script0(BaseScript):
    """Read raw WhatsApp files, clean, combine and store CSV + Parquet."""

    def __init__(self, file_manager, data_editor, data_preparation,
                 processed_dir: Path, config: dict, image_dir: Path):
        super().__init__(file_manager, data_editor=data_editor,
                         data_preparation=data_preparation)
        self.processed_dir = processed_dir
        self.config = config
        self.image_dir = image_dir
        self.tables_dir = Path("tables")
        self.tables_dir.mkdir(exist_ok=True)

    def run(self) -> Optional[Dict]:
        logger.info("Script0: Starting preprocessing...")

        # === Run preprocessing via FileManager ===
        result = self.file_manager.get_preprocessed_data(
            self.data_editor, self.data_preparation, self.config, self.processed_dir
        )

        if result is None or "df" not in result:
            return self.log_error("Script0: FileManager returned None or no 'df'.")

        df = result["df"]
        if df is None or df.empty:
            return self.log_error("Script0: Empty DataFrame after preprocessing.")

        # === Save Parquet ===
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        parquet_path = self.processed_dir / f"combined_{timestamp}.parq"
        try:
            df.to_parquet(parquet_path, index=False)
            logger.info(f"Script0: Saved parquet → {parquet_path}")
        except Exception as e:
            logger.error(f"Script0: Failed to save parquet: {e}")
            return None

        # === Save CSV for inspection ===
        csv_path = self.tables_dir / f"combined_{timestamp}.csv"
        try:
            df.to_csv(csv_path, index=False)
            logger.info(f"Script0: Saved CSV → {csv_path}")
        except Exception as e:
            logger.warning(f"Script0: Failed to save CSV: {e}")

        # === Return df and tables_dir ===
        logger.success("Script0: Preprocessing completed successfully.")
        return {
            "df": df,
            "tables_dir": self.tables_dir
        }