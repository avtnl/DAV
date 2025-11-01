# === Module Docstring ===
"""
WhatsApp Chat Analyzer - File Management Module

Handles file discovery, preprocessing coordination, loading, saving, and
enrichment of WhatsApp chat data. Integrates with ``DataEditor`` and
``preprocessor`` to maintain consistent data pipelines.

Key responsibilities:
    * Discover and load raw/preprocessed CSV/Parquet files
    * Coordinate preprocessing via ``preprocessor.main()``
    * Save outputs with timestamped filenames
    * Enrich and concatenate group-specific DataFrames
"""

# === Imports ===
import re
import shutil
import tomllib
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
import wa_analyzer.preprocess as preprocessor
from loguru import logger
from src.constants import Columns, Groups
from src.data_editor import DataEditor


# === FileManager Class ===
class FileManager:
    """Manages file I/O, preprocessing, and data persistence for WhatsApp chat analysis."""

    # === Timestamped File Discovery ===

    def find_name_csv(self, path: Path, timestamp: str) -> Path | None:
        pattern = r"whatsapp-(\d{8}-\d{6})\.csv"
        try:
            input_dt = datetime.strptime(timestamp, "%Y%m%d-%H%M%S")
            input_dt = pytz.timezone("Europe/Amsterdam").localize(input_dt)

            for file in path.glob("*.csv"):
                match = re.match(pattern, file.name)
                if match:
                    file_timestamp = match.group(1)
                    file_dt = datetime.strptime(file_timestamp, "%Y%m%d-%H%M%S")
                    file_dt = pytz.timezone("Europe/Amsterdam").localize(file_dt)
                    if file_dt > input_dt:
                        logger.info(f"Found matching file: {file}")
                        return file
                    logger.warning(f"Invalid timestamp in filename: {file.name}")

            logger.warning(f"No CSV file found in {path} with timestamp later than {timestamp}")
            return None
        except ValueError:
            logger.error(f"Invalid timestamp format: {timestamp}")
            return None

    def find_latest_file(
        self, processed_dir: Path, prefix: str = "organized_data", suffix: str = ".csv"
    ) -> Path | None:
        pattern = f"{prefix}-(\\d{{8}}-\\d{{6}}){re.escape(suffix)}"
        try:
            files = list(processed_dir.glob(f"{prefix}-*{suffix}"))
            if not files:
                logger.warning(
                    f"No files found with prefix '{prefix}' and suffix '{suffix}' in {processed_dir}"
                )
                return None

            latest_file = max(
                files,
                key=lambda f: datetime.strptime(
                    re.search(pattern, f.name).group(1), "%Y%m%d-%H%M%S"
                )
                if re.search(pattern, f.name)
                else datetime.min,
            )
            logger.info(f"Found latest file: {latest_file}")
            return latest_file
        except Exception as e:
            logger.error(f"Failed to find latest file: {e}")
            return None

    # === Configuration & Preprocessing Pipeline ===

    def read_csv(
        self,
    ) -> tuple[list[Path] | None, Path | None, dict[str, str] | None, dict[str, str] | None]:
        try:
            configfile = Path("config.toml").resolve()
            with configfile.open("rb") as f:
                config = tomllib.load(f)

            processed = Path(config["processed"])
            raw_dir = Path(config["raw"])
            preprocess = config["preprocess"]

            # Define raw file to group mapping
            raw_files = {
                "raw_1": ("current_1", "maap"),
                "raw_2a": ("current_2a", "golfmaten"),
                "raw_2b": ("current_2b", "golfmaten"),
                "raw_3": ("current_3", "dac"),
                "raw_4": ("current_4", "tillies"),
            }
            group_map = {current_key: group for _, (current_key, group) in raw_files.items()}

            if preprocess:
                datafiles: list[Path] = []
                parq_files: dict[str, str] = {}

                for raw_key, (current_key, _group) in raw_files.items():
                    if raw_key not in config:
                        logger.warning(f"Key {raw_key} not found in config.toml")
                        continue
                    raw_file = raw_dir / config[raw_key]
                    if not raw_file.exists():
                        logger.warning(f"Raw file {raw_file} does not exist")
                        continue

                    # Rename raw file to _chat.txt
                    chat_file = raw_dir / "_chat.txt"
                    shutil.copy(raw_file, chat_file)
                    logger.info(f"Copied {raw_file} to {chat_file}")

                    # Run preprocessor
                    now = datetime.now(tz=pytz.timezone("Europe/Amsterdam")).strftime(
                        "%Y%m%d-%H%M%S"
                    )
                    logger.info(f"Preprocessing with timestamp: {now}")
                    preprocessor.main(["--device", "ios"])

                    # Find the generated CSV file
                    datafile = self.find_name_csv(processed, now)
                    if datafile is None:
                        logger.error(f"No CSV file found after preprocessing {raw_file}")
                        continue

                    # Find the corresponding Parquet file
                    parq_file = datafile.with_suffix(".parq")
                    if not parq_file.exists():
                        logger.warning(f"Parquet file {parq_file} not found")
                        continue

                    datafiles.append(datafile)
                    parq_files[current_key] = parq_file.name
                    logger.info(f"Processed {raw_file} → CSV: {datafile}, Parquet: {parq_file}")

                    # Clean up _chat.txt
                    chat_file.unlink()
                    logger.info(f"Removed temporary {chat_file}")

                if not datafiles:
                    logger.error("No valid CSV files found after preprocessing")
                    return None, None, None, None
                return datafiles, processed, group_map, parq_files
            else:
                datafiles: list[Path] = []
                for current_key in group_map:
                    if current_key not in config:
                        logger.warning(f"Key {current_key} not found in config.toml")
                        continue
                    datafile = processed / config[current_key]
                    datafile = datafile.with_suffix(".csv")  # Convert .parq to .csv
                    if not datafile.exists():
                        logger.warning(f"CSV file {datafile} does not exist")
                        continue
                    datafiles.append(datafile)
                if not datafiles:
                    logger.error("No valid CSV files found in config")
                    return None, None, None, None
                return datafiles, processed, group_map, None
        except FileNotFoundError:
            logger.error("config.toml not found")
            return None, None, None, None
        except Exception as e:
            logger.exception(f"Failed to read config or preprocess: {e}")
            return None, None, None, None

    # === Data Loading & Enrichment ===

    def get_preprocessed_data(
        self,
        data_editor: DataEditor,
        data_preparation,
        config: dict,
        processed_dir: Path,
    ) -> dict | None:
        try:
            datafiles, processed, group_map, parq_files = self.read_csv()
            if not datafiles:
                logger.error("No valid data files")
                return None

            dataframes: dict[str, pd.DataFrame] = {}
            preprocess = config["preprocess"]

            if preprocess:
                for datafile in datafiles:
                    if not datafile.exists():
                        continue
                    df = data_editor.convert_timestamp(datafile)
                    df = data_editor.clean_author(df)
                    df[Columns.HAS_EMOJI] = df[Columns.MESSAGE].apply(data_editor.has_emoji)
                    for key, parq_name in parq_files.items():
                        if parq_name.replace(".parq", ".csv") == datafile.name:
                            df[Columns.WHATSAPP_GROUP] = group_map[key]
                            break
                    else:
                        df[Columns.WHATSAPP_GROUP] = Groups.UNKNOWN
                    dataframes[datafile.stem] = df
            else:
                for key, group in group_map.items():
                    csv_path = processed / config[key]
                    csv_path = csv_path.with_suffix(".csv")
                    if not csv_path.exists():
                        continue
                    df = data_editor.convert_timestamp(csv_path)
                    df = data_editor.clean_author(df)
                    df[Columns.HAS_EMOJI] = df[Columns.MESSAGE].apply(data_editor.has_emoji)
                    df[Columns.WHATSAPP_GROUP] = group
                    dataframes[key] = df

            if not dataframes:
                return None

            df = data_editor.concatenate_df(dataframes)
            df = data_editor.filter_group_names(df)
            df = data_editor.clean_for_deleted_media_patterns(df)
            df = data_editor.organize_extended_df(df)  # One call for full enrichment

            if df is None:
                return None

            _csv_file, _parq_file = self.save_combined_files(df, processed_dir)
            tables_dir = Path("tables")
            tables_dir.mkdir(parents=True, exist_ok=True)

            return {"df": df, "tables_dir": tables_dir, "dataframes": dataframes}
        except Exception as e:
            logger.exception(f"Preprocessing failed: {e}")
            return None

    def load_dataframe(
        self, datafile: Path, mapping: dict[str, str] | None = None
    ) -> pd.DataFrame | None:
        if mapping is None:
            mapping = {}
        try:
            df = pd.read_csv(datafile, parse_dates=[Columns.TIMESTAMP])
            if mapping:
                df = df.rename(columns=mapping)
                logger.info(f"Applied column mapping: {mapping}")
            else:
                logger.debug("No column mapping applied.")
            return df
        except Exception as e:
            logger.error(f"Failed to load DataFrame from {datafile}: {e}")
            return None

    # === File Saving Utilities ===

    def save_csv(self, df: pd.DataFrame, processed_dir: Path, prefix: str = "whatsapp") -> Path:
        now = datetime.now(tz=pytz.timezone("Europe/Amsterdam")).strftime("%Y%m%d-%H%M%S")
        logger.info(f"Generated timestamp: {now}")
        output = processed_dir / f"{prefix}-{now}.csv"
        logger.info(f"Saving CSV to: {output}")
        df.to_csv(output, index=False)
        return output

    def save_parq(self, df: pd.DataFrame, processed_dir: Path, prefix: str = "whatsapp") -> Path:
        now = datetime.now(tz=pytz.timezone("Europe/Amsterdam")).strftime("%Y%m%d-%H%M%S")
        logger.info(f"Generated timestamp: {now}")
        output = processed_dir / f"{prefix}-{now}.parq"
        logger.info(f"Saving Parquet to: {output}")
        df.to_parquet(output, index=False)
        return output

    def save_combined_files(
        self, df: pd.DataFrame, processed_dir: Path
    ) -> tuple[Path | None, Path | None]:
        try:
            csv_file = self.save_csv(df, processed_dir, prefix="whatsapp_all")
            parq_file = self.save_parq(df, processed_dir, prefix="whatsapp_all")
            logger.info(f"DataFrame saved as: {csv_file} and {parq_file}")
            return csv_file, parq_file
        except Exception as e:
            logger.exception(f"Failed to save DataFrame: {e}")
            return None, None

    def save_png(
        self, fig, image_dir: Path, filename: str = "yearly_bar_chart_combined"
    ) -> Path | None:
        try:
            now = datetime.now(tz=pytz.timezone("Europe/Amsterdam")).strftime("%Y%m%d-%H%M%S")
            output = image_dir / f"{filename}-{now}.png"
            image_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=300, bbox_inches="tight")
            logger.info(f"Saved bar chart: {output}")
            return output
        except Exception as e:
            logger.exception(f"Failed to save PNG to {output}: {e}")
            return None

    def save_table(self, df: pd.DataFrame, tables_dir: Path, prefix: str = "table") -> Path:
        now = datetime.now(tz=pytz.timezone("Europe/Amsterdam")).strftime("%Y%m%d-%H%M%S")
        logger.info(f"Generated timestamp: {now}")
        output = tables_dir / f"{prefix}-{now}.csv"
        logger.info(f"Saving table to: {output}")
        df.to_csv(output, index=True)
        return output

    # === Group Enrichment & Concatenation ===

    def enrich_all_groups(self, data_editor: DataEditor, processed_dir: Path) -> Path | None:
        group_map = {
            "maap": "maap",
            "golf": "golfmaten",
            "dac": "dac",
            "voorganger-golf": "golfmaten",
            "til": "tillies",
        }

        try:
            dfs: list[pd.DataFrame] = []
            for csv_file in processed_dir.glob("whatsapp-*-cleaned.csv"):
                name_part = csv_file.stem.split("-", 3)[-1]
                group_key = name_part.split("-cleaned")[0]
                group = group_map.get(group_key)

                if not group:
                    logger.warning(f"Unknown group in {csv_file.name}, skipping")
                    continue

                logger.info(f"Loading {csv_file.name} → group='{group}'")
                df = pd.read_csv(csv_file, parse_dates=[Columns.TIMESTAMP])
                df[Columns.WHATSAPP_GROUP] = group

                df = data_editor.organize_extended_df(df)
                if df is None:
                    logger.error(f"Failed to enrich {csv_file}")
                    continue

                dfs.append(df)

            if not dfs:
                logger.error("No dataframes to concatenate")
                return None

            combined = pd.concat(dfs, ignore_index=True)
            out_path = (
                processed_dir / f"whatsapp_all_enriched-{pd.Timestamp.now():%Y%m%d-%H%M%S}.csv"
            )
            combined.to_csv(out_path, index=False)
            logger.success(f"Enriched file saved: {out_path}")
            return out_path
        except Exception as e:
            logger.exception(f"Failed to enrich groups: {e}")
            return None


# NEW: Full standardization with Google-style docstrings, StrEnum, and type hints (2025-10-31)
