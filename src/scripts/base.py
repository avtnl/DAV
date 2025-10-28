# scripts/base.py
import pandas as pd
from loguru import logger
from pathlib import Path
from typing import Optional
from pydantic import BaseModel


class BaseScript:
    """Common helpers for every script."""
    def __init__(self, file_manager, data_editor=None, data_preparation=None,
                 plot_manager=None, settings: Optional[BaseModel] = None, df=None):
        self.file_manager = file_manager
        self.data_editor = data_editor
        self.data_preparation = data_preparation
        self.plot_manager = plot_manager
        self.settings = settings
        self.df = df                     # optional original DataFrame

    # ------------------------------------------------------------------ #
    # Helper methods (unchanged from your original BaseScript)
    # ------------------------------------------------------------------ #
    def save_figure(self, fig, image_dir: Path, filename: str) -> Optional[Path]:
        png_file = self.file_manager.save_png(fig, image_dir, filename=filename)
        if png_file is None:
            logger.error(f"Failed to save {filename}.")
        else:
            logger.info(f"Saved {filename}: {png_file}")
        return png_file

    def save_table(self, df: pd.DataFrame, tables_dir: Path, prefix: str) -> Optional[Path]:
        saved = self.file_manager.save_table(df, tables_dir, prefix=prefix)
        if saved:
            logger.info(f"Saved table: {saved}")
        else:
            logger.error(f"Failed to save table with prefix {prefix}.")
        return saved

    def log_error(self, message: str):
        logger.error(message)
        return None

    def run(self):
        raise NotImplementedError