# scripts/script0.py
import pandas as pd
from loguru import logger
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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
        self.image_dir = image_dir
        self.dataframes: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ #
    def run(self) -> Optional[dict]:
        try:
            # 1. Load raw files + mapping
            datafiles, processed, group_map, parq_files = self.file_manager.read_csv()
            if not datafiles:
                return self.log_error("No valid data files were loaded. Exiting.")

            # 2. Pre-process / load pre-processed
            if self.config["preprocess"]:
                for datafile in datafiles:
                    if not datafile.exists():
                        logger.warning(f"{datafile} does not exist – skip.")
                        continue
                    df = self.data_editor.convert_timestamp(datafile)
                    df = self.data_editor.clean_author(df)
                    df[Columns.HAS_EMOJI.value] = df["message"].apply(self.data_editor.has_emoji)

                    # assign group name from parq_files mapping
                    for key, parq_name in parq_files.items():
                        if parq_name.replace(".parq", ".csv") == datafile.name:
                            df[Columns.WHATSAPP_GROUP.value] = group_map[key]
                            break
                    else:
                        df[Columns.WHATSAPP_GROUP.value] = Groups.UNKNOWN.value

                    logger.info(f"Processed {datafile.name} – {len(df)} rows")
                    csv_file = self.file_manager.save_csv(df, processed)
                    parq_file = self.file_manager.save_parq(df, processed)
                    logger.info(f"Saved CSV={csv_file}, Parquet={parq_file}")
                    self.dataframes[datafile.stem] = df
            else:
                # load already-processed files
                for key, group in group_map.items():
                    csv_path = processed / self.config[key]
                    csv_path = csv_path.with_suffix(".csv")
                    if not csv_path.exists():
                        logger.warning(f"{csv_path} missing – skip.")
                        continue
                    df = self.data_editor.convert_timestamp(csv_path)
                    df = self.data_editor.clean_author(df)
                    df[Columns.HAS_EMOJI.value] = df["message"].apply(self.data_editor.has_emoji)
                    df[Columns.WHATSAPP_GROUP.value] = group
                    self.dataframes[key] = df
                    logger.info(f"Loaded {csv_path} – {len(df)} rows")

            if not self.dataframes:
                return self.log_error("No DataFrames were created/loaded.")

            # 3. Concatenate + final cleaning
            df = self.data_editor.concatenate_df(self.dataframes)
            if df is None:
                return self.log_error("Concatenation failed.")

            df = self.data_editor.filter_group_names(df)
            if df is None:
                return self.log_error("Group-name filter failed.")
            df = self.data_editor.clean_for_deleted_media_patterns(df)
            if df is None:
                return self.log_error("Deleted-media cleaning failed.")

            # 4. Save combined files
            csv_file, parq_file = self.file_manager.save_combined_files(df, self.processed_dir)
            if csv_file is None or parq_file is None:
                return self.log_error("Failed to save combined files.")

            # 5. Final filter (again – as in original code)
            df = self.data_editor.filter_group_names(df)
            if df is None:
                return self.log_error("Final group filter failed.")

            # 6. Tables directory
            tables_dir = Path("tables")
            tables_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Script0 (pre-processing) finished.")
            return {"df": df, "tables_dir": tables_dir, "dataframes": self.dataframes}

        except Exception as e:
            logger.exception(f"Script0 crashed: {e}")
            return None