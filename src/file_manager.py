# === Module Docstring ===
"""
WhatsApp Chat Analyzer – File Management Module

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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytz
import tomllib
from loguru import logger

import wa_analyzer.preprocess as preprocessor
from src.constants import Columns, Groups
from src.data_editor import DataEditor


# === FileManager Class ===
class FileManager:
    """Manages file I/O, preprocessing, and data persistence for WhatsApp chat analysis."""

    # === Timestamped File Discovery ===

    def find_name_csv(self, path: Path, timestamp: str) -> Optional[Path]:
        """
        Find a CSV file in path with name 'whatsapp-YYYYMMDD-HHMMSS.csv' where
        the timestamp is later than the provided timestamp.

        Args:
            path: Directory to search (e.g., Path("data/processed"))
            timestamp: Timestamp in format 'YYYYMMDD-HHMMSS' (e.g., '20250924-221905')

        Returns:
            Path to the matching file, or ``None`` if no file is found.

        Examples:
            >>> fm = FileManager()
            >>> fm.find_name_csv(Path("data/processed"), "20250924-221905")
            PosixPath('data/processed/whatsapp-20250925-010203.csv')
        """
        pattern = r"whatsapp-(\d{8}-\d{6})\.csv"
        try:
            input_dt = datetime.strptime(timestamp, "%Y%m%d-%H%M%S")
            input_dt = pytz.timezone("Europe/Amsterdam").localize(input_dt)
        except ValueError:
            logger.error(f"Invalid timestamp format: {timestamp}")
            return None

        for file in path.glob("*.csv"):
            match = re.match(pattern, file.name)
            if match:
                file_timestamp = match.group(1)
                try:
                    file_dt = datetime.strptime(file_timestamp, "%Y%m%d-%H%M%S")
                    file_dt = pytz.timezone("Europe/Amsterdam").localize(file_dt)
                    if file_dt > input_dt:
                        logger.info(f"Found matching file: {file}")
                        return file
                except ValueError:
                    logger.warning(f"Invalid timestamp in filename: {file.name}")
                    continue

        logger.warning(f"No CSV file found in {path} with timestamp later than {timestamp}")
        return None

    def find_latest_file(
        self, processed_dir: Path, prefix: str = "organized_data", suffix: str = ".csv"
    ) -> Optional[Path]:
        """
        Find the latest file with the given prefix and suffix in the processed directory
        based on timestamp in filename.

        Args:
            processed_dir: Directory to search.
            prefix: Filename prefix (default: "organized_data").
            suffix: Filename suffix (default: ".csv").

        Returns:
            Path to the latest file, or ``None`` if no file is found.

        Examples:
            >>> fm = FileManager()
            >>> fm.find_latest_file(Path("data/processed"), "organized_data", ".csv")
            PosixPath('data/processed/organized_data-20251031-120000.csv')
        """
        pattern = f"{prefix}-(\\d{{8}}-\\d{{6}}){re.escape(suffix)}"
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

    # === Configuration & Preprocessing Pipeline ===

    def read_csv(
        self,
    ) -> Tuple[Optional[List[Path]], Optional[Path], Optional[Dict[str, str]], Optional[Dict[str, str]]]:
        """
        Read configuration and determine the CSV file(s) to process.

        If ``preprocess=True`` in config, rename raw files, run preprocessing,
        and keep Parquet filenames in memory.

        If ``preprocess=False``, use ``current_*`` keys from config.

        Returns:
            Tuple of:
                - List of CSV ``Path`` objects (or ``None``)
                - Processed directory ``Path`` (or ``None``)
                - Group mapping ``{current_key: group_name}`` (or ``None``)
                - Parquet mapping ``{current_key: parq_filename}`` (or ``None`` if not preprocessing)

        Examples:
            >>> fm = FileManager()
            >>> csvs, proc_dir, gmap, parqs = fm.read_csv()
        """
        configfile = Path("config.toml").resolve()
        try:
            with configfile.open("rb") as f:
                config = tomllib.load(f)
        except FileNotFoundError:
            logger.error("config.toml not found")
            return None, None, None, None

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
            datafiles: List[Path] = []
            parq_files: Dict[str, str] = {}

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
                try:
                    shutil.copy(raw_file, chat_file)
                    logger.info(f"Copied {raw_file} to {chat_file}")
                except Exception as e:
                    logger.error(f"Failed to copy {raw_file} to {chat_file}: {e}")
                    continue

                # Run preprocessor
                now = datetime.now(tz=pytz.timezone("Europe/Amsterdam")).strftime("%Y%m%d-%H%M%S")
                logger.info(f"Preprocessing with timestamp: {now}")
                try:
                    preprocessor.main(["--device", "ios"])
                except Exception as e:
                    logger.error(f"Preprocessing failed for {raw_file}: {e}")
                    continue

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
                try:
                    chat_file.unlink()
                    logger.info(f"Removed temporary {chat_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove {chat_file}: {e}")

            if not datafiles:
                logger.error("No valid CSV files found after preprocessing")
                return None, None, None, None
            return datafiles, processed, group_map, parq_files
        else:
            datafiles: List[Path] = []
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

    # === Data Loading & Enrichment ===

    def get_preprocessed_data(
        self,
        data_editor: DataEditor,
        data_preparation,
        config: dict,
        processed_dir: Path,
    ) -> Optional[dict]:
        """
        Load and enrich all group DataFrames, concatenate, and save combined files.

        Applies timestamp conversion, author cleaning, emoji detection,
        group assignment, and full ``organize_extended_df`` enrichment.

        Args:
            data_editor: Instance of ``DataEditor`` for transformations.
            data_preparation: Unused (legacy).
            config: Loaded configuration dictionary.
            processed_dir: Path to processed data directory.

        Returns:
            Dictionary with:
                - ``df``: Combined enriched DataFrame
                - ``tables_dir``: Directory for saving tables
                - ``dataframes``: Group-specific DataFrames

            or ``None`` on failure.

        Examples:
            >>> result = fm.get_preprocessed_data(editor, prep, cfg, Path("data/processed"))
        """
        try:
            datafiles, processed, group_map, parq_files = self.read_csv()
            if not datafiles:
                logger.error("No valid data files")
                return None

            dataframes: Dict[str, pd.DataFrame] = {}
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
        self, datafile: Path, mapping: Optional[Dict[str, str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load a DataFrame from a CSV file and apply column renaming if a mapping is provided.

        Args:
            datafile: Path to the CSV file.
            mapping: Dictionary for renaming columns (e.g., {'old_name': 'new_name'}).

        Returns:
            Loaded and potentially renamed DataFrame, or ``None`` on failure.

        Examples:
            >>> df = fm.load_dataframe(Path("data/processed/whatsapp-20251031.csv"))
        """
        if mapping is None:
            mapping = {}
        try:
            df = pd.read_csv(
                datafile, parse_dates=[Columns.TIMESTAMP]
            )  # Use StrEnum for column reference
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
        """
        Save the DataFrame to a CSV file with a unique timestamped filename.

        Args:
            df: DataFrame to save.
            processed_dir: Directory to save the file.
            prefix: Filename prefix (default: "whatsapp").

        Returns:
            Path to the saved CSV file.

        Examples:
            >>> path = fm.save_csv(df, Path("data/processed"), "mydata")
        """
        now = datetime.now(tz=pytz.timezone("Europe/Amsterdam")).strftime("%Y%m%d-%H%M%S")
        logger.info(f"Generated timestamp: {now}")
        output = processed_dir / f"{prefix}-{now}.csv"
        logger.info(f"Saving CSV to: {output}")
        df.to_csv(output, index=False)
        return output

    def save_parq(self, df: pd.DataFrame, processed_dir: Path, prefix: str = "whatsapp") -> Path:
        """
        Save the DataFrame to a Parquet file with a unique timestamped filename.

        Args:
            df: DataFrame to save.
            processed_dir: Directory to save the file.
            prefix: Filename prefix (default: "whatsapp").

        Returns:
            Path to the saved Parquet file.

        Examples:
            >>> path = fm.save_parq(df, Path("data/processed"), "mydata")
        """
        now = datetime.now(tz=pytz.timezone("Europe/Amsterdam")).strftime("%Y%m%d-%H%M%S")
        logger.info(f"Generated timestamp: {now}")
        output = processed_dir / f"{prefix}-{now}.parq"
        logger.info(f"Saving Parquet to: {output}")
        df.to_parquet(output, index=False)
        return output

    def save_combined_files(
        self, df: pd.DataFrame, processed_dir: Path
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Save the concatenated DataFrame to CSV and Parquet files with 'whatsapp_all-' prefix.

        Args:
            df: DataFrame to save.
            processed_dir: Directory to save the files.

        Returns:
            Tuple of (CSV Path, Parquet Path), or (None, None) if saving fails.

        Examples:
            >>> csv_path, parq_path = fm.save_combined_files(df, Path("data/processed"))
        """
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
    ) -> Optional[Path]:
        """
        Save a matplotlib figure to a PNG file with a unique timestamped filename.

        Args:
            fig: Matplotlib figure to save.
            image_dir: Directory to save the file (e.g., Path("img")).
            filename: Base filename (default: "yearly_bar_chart_combined").

        Returns:
            Path to the saved PNG file, or ``None`` if saving fails.

        Examples:
            >>> path = fm.save_png(fig, Path("img"), "myplot")
        """
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
        """
        Save the DataFrame to a CSV file with a unique timestamped filename.

        Args:
            df: DataFrame to save.
            tables_dir: Directory to save the file (e.g., Path("tables")).
            prefix: Filename prefix (default: "table").

        Returns:
            Path to the saved CSV file.

        Examples:
            >>> path = fm.save_table(df, Path("tables"), "summary")
        """
        now = datetime.now(tz=pytz.timezone("Europe/Amsterdam")).strftime("%Y%m%d-%H%M%S")
        logger.info(f"Generated timestamp: {now}")
        output = tables_dir / f"{prefix}-{now}.csv"
        logger.info(f"Saving table to: {output}")
        df.to_csv(output, index=True)
        return output

    # === Group Enrichment & Concatenation ===

    def enrich_all_groups(self, data_editor: DataEditor, processed_dir: Path) -> Optional[Path]:
        """
        Load all cleaned CSV files, assign ``whatsapp_group`` from filename,
        run ``organize_extended_df``, concatenate, and save enriched CSV.

        Args:
            data_editor: Instance of ``DataEditor`` for enrichment.
            processed_dir: Directory containing cleaned CSV files.

        Returns:
            Path to the saved enriched combined file, or ``None`` on failure.

        Examples:
            >>> path = fm.enrich_all_groups(editor, Path("data/processed"))
        """
        # Mapping: filename pattern → group name
        group_map = {
            "maap": "maap",
            "golf": "golfmaten",
            "dac": "dac",
            "voorganger-golf": "golfmaten",
            "til": "tillies",
        }

        dfs: List[pd.DataFrame] = []
        for csv_file in processed_dir.glob("whatsapp-*-cleaned.csv"):
            # Extract group key from filename
            name_part = csv_file.stem.split("-", 3)[-1]  # e.g., "maap-cleaned" → "maap"
            group_key = name_part.split("-cleaned")[0]
            group = group_map.get(group_key)

            if not group:
                logger.warning(f"Unknown group in {csv_file.name}, skipping")
                continue

            logger.info(f"Loading {csv_file.name} → group='{group}'")
            df = pd.read_csv(csv_file, parse_dates=[Columns.TIMESTAMP])
            df[Columns.WHATSAPP_GROUP] = group

            # Run full enrichment
            df = data_editor.organize_extended_df(df)
            if df is None:
                logger.error(f"Failed to enrich {csv_file}")
                continue

            dfs.append(df)

        if not dfs:
            logger.error("No dataframes to concatenate")
            return None

        combined = pd.concat(dfs, ignore_index=True)
        out_path = processed_dir / f"whatsapp_all_enriched-{pd.Timestamp.now():%Y%m%d-%H%M%S}.csv"
        combined.to_csv(out_path, index=False)
        logger.success(f"Enriched file saved: {out_path}")
        return out_path


# NEW: Full standardization with Google-style docstrings, StrEnum, and type hints (2025-10-31)