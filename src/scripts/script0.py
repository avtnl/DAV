# scripts/script0.py
import pandas as pd
from loguru import logger
from pathlib import Path
from typing import Dict, Optional
from .base import BaseScript
from src.constants import Columns, Groups

class Script0(BaseScript):
    """Read raw WhatsApp files, clean, combine and store CSV + Parquet."""

    def __init__(self, file_manager, data_editor, data_preparation,
                 processed_dir: Path, config: dict, image_dir: Path):
        super().__init__(file_manager, data_editor=data_editor,
                         data_preparation=data_preparation)
        self.processed_dir = processed_dir
        self.config = config
        self.image_dir = image_dir  # Not used here, but kept for compatibility

    # ------------------------------------------------------------------ #
    def run(self) -> Optional[dict]:
        # Delegate all logic to FileManager to avoid overlap
        return self.file_manager.get_preprocessed_data(
            self.data_editor, self.data_preparation, self.config, self.processed_dir
        )