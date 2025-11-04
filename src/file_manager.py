# === Module Docstring ===
"""
File Management Module

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
from typing import Any, Dict, List, Tuple

import pandas as pd
import pytz
import wa_analyzer.preprocess as preprocessor
from loguru import logger

from .constants import (
    CONFIG_FILE,
    TEMP_CHAT_FILE,
    Columns,
    ConfigKeys,
    FileExtensions,
    FilePatterns,
    FilePrefixes,
    Groups,
    GROUP_MAP_FROM_CLEANED,
    PreprocessorArgs,
    RAW_FILE_MAPPING,
    ImageFilenames,
)
from .data_editor import DataEditor


# === File Manager Class ===
class FileManager:
    """Manages file I/O, preprocessing, and data persistence."""

    def __init__(self, processed_dir: Path | None = None) -> None:
        """
        Initialize FileManager with processed directory.

        Args:
            processed_dir: Directory for processed files (optional).
        """
        self.processed_dir = processed_dir or Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # === Get Latest Enriched DataFrame ===
    def get_latest_preprocessed_df(self) -> pd.DataFrame | None:
        """
        Load the most recent enriched CSV file matching the enriched pattern.

        Returns:
            pd.DataFrame or None if no file found or load failed.
        """
        try:
            pattern = f"{FilePrefixes.WHATSAPP_ALL_ENRICHED}-*{FileExtensions.CSV}"
            files = list(self.processed_dir.glob(pattern))
            if not files:
                logger.warning("No enriched CSV files found in processed directory.")
                return None

            latest_file = max(files, key=lambda p: p.stat().st_mtime)
            df = pd.read_csv(latest_file, parse_dates=[Columns.TIMESTAMP])
            logger.info(f"Loaded enriched data: {latest_file.name} | Shape: {df.shape}")
            return df

        except Exception as e:
            logger.exception(f"Failed to load enriched data: {e}")
            return None

    # === Find Latest File ===
    def find_latest_file(
        self,
        processed_dir: Path,
        prefix: str = FilePrefixes.ORGANIZED,
        suffix: str = FileExtensions.CSV,
    ) -> Path | None:
        """
        Find the most recent file matching the timestamped pattern.

        Args:
            processed_dir: Directory to search.
            prefix: File prefix (default: organized_data).
            suffix: File extension (default: .csv).

        Returns:
            Path to the latest file, or None if not found.
        """
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

    # === Get Preprocessed Data ===
    def get_preprocessed_data(
        self,
        data_editor: DataEditor,
        data_preparation: Any,
        config: Dict[str, Any],
        processed_dir: Path,
    ) -> Dict[str, pd.DataFrame] | None:
        """
        Get preprocessed and enriched DataFrame, with optional reuse.

        Args:
            data_editor: DataEditor instance.
            data_preparation: DataPreparation instance (unused here but passed for compatibility).
            config: Configuration dictionary.
            processed_dir: Processed directory.

        Returns:
            Dict with 'df' key containing enriched DataFrame, or None on failure.
        """
        preprocess = config.get(ConfigKeys.PREPROCESS, False)
        reuse = config.get(ConfigKeys.REUSE_WHATSAPP_ALL, True)

        # === REUSE LOGIC: Only when NOT preprocessing AND reuse explicitly enabled ===
        # This ensures:
        # - preprocess = true  → always fresh
        # - preprocess = false, reuse = true  → reuse if exists
        # - preprocess = false, reuse = false → force fresh combine
        if not preprocess and reuse:
            cached = self.get_latest_preprocessed_df()
            if cached is not None:
                logger.success("Re-using latest enriched file")
                return {"df": cached}
            else:
                logger.info("No enriched file found — will create fresh one")

        # === FRESH RUN: Either preprocessing or forced rebuild ===
        # Continue with loading current_* or running preprocessor...
        cleaned_paths_with_groups: List[Tuple[Path, str]] = []

        if preprocess:
            logger.info("Running full preprocess → loading cleaned parquets")
            cleaned_paths_with_groups = self._run_preprocessor_and_get_parquets(config, processed_dir)
        else:
            logger.info("Loading current_* parquets from config")
            cleaned_paths_with_groups = self._load_current_parquets(config, processed_dir)

        if not cleaned_paths_with_groups:
            logger.error("No cleaned Parquet files found")
            return None

        # Merge-first-then-enrich
        dfs = []
        for path, group in cleaned_paths_with_groups:
            try:
                df = pd.read_parquet(path)
                # Ensure minimal required columns
                required_cols = [Columns.TIMESTAMP, Columns.AUTHOR, Columns.MESSAGE]
                missing = [col for col in required_cols if col not in df.columns]
                if missing:
                    logger.error(f"Missing columns {missing} in {path}")
                    continue
                df[Columns.WHATSAPP_GROUP] = group
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to read {path}: {e}")
                continue

        if not dfs:
            logger.error("No valid DataFrames to combine")
            return None

        combined_raw = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined {len(dfs)} groups → {combined_raw.shape[0]} rows")

        enriched = data_editor.organize_extended_df(combined_raw)
        if enriched is None:
            logger.error("Enrichment failed")
            return None

        # Save enriched file
        csv_path, parq_path = self.save_combined_files(enriched, processed_dir)
        logger.success(f"Enriched file saved: {csv_path.name}")

        return {"df": enriched}

    # === Helper: Run Preprocessor and Get Parquets ===
    def _run_preprocessor_and_get_parquets(
        self, config: Dict[str, Any], processed_dir: Path
    ) -> List[Tuple[Path, str]]:
        """Run wa-analyzer on all raw_* files and return list of (output .parq path, group) tuples.

        Returns:
            List of (Path, group) tuples, or empty list on failure.
        """
        raw_dir = Path(config[ConfigKeys.RAW])
        paths_with_groups = []

        for idx, (raw_key, (current_key, group)) in enumerate(RAW_FILE_MAPPING.items(), start=1):
            if raw_key not in config:
                logger.warning(f"Key {raw_key} not found in config.toml")
                continue
            raw_file = raw_dir / config[raw_key]
            if not raw_file.exists():
                logger.warning(f"Raw file {raw_file} does not exist")
                continue

            # Copy to temp
            temp_chat = raw_dir / TEMP_CHAT_FILE
            shutil.copy(raw_file, temp_chat)

            # Run preprocessor
            now = datetime.now(tz=pytz.timezone("Europe/Amsterdam")).strftime("%Y%m%d-%H%M%S")
            logger.info(f"Preprocessing {raw_file.name} → {now}")
            preprocessor.main([PreprocessorArgs.DEVICE, PreprocessorArgs.IOS])

            # Find output Parquet
            parq_file = processed_dir / f"whatsapp-{now}{FileExtensions.PARQUET}"
            if not parq_file.exists():
                logger.error(f"Expected Parquet not found: {parq_file}")
                temp_chat.unlink(missing_ok=True)
                continue

            paths_with_groups.append((parq_file, group))
            temp_chat.unlink(missing_ok=True)
            logger.info(f"Generated: {parq_file.name} for group {group}")

        return paths_with_groups

    # === Helper: Load Current Parquets ===
    def _load_current_parquets(self, config: Dict[str, Any], processed_dir: Path) -> List[Tuple[Path, str]]:
        """Return list of (current_* Parquet path, group) tuples from config.

        Returns:
            List of (Path, group) tuples, or empty list if none found.
        """
        paths_with_groups = []
        for raw_key, (current_key, group) in RAW_FILE_MAPPING.items():
            if current_key not in config:
                logger.warning(f"Key {current_key} missing in config")
                continue
            path = processed_dir / config[current_key]
            if path.exists():
                paths_with_groups.append((path, group))
                logger.info(f"Loaded current: {path.name} for group {group}")
            else:
                logger.warning(f"current_* file not found: {path}")
        return paths_with_groups

    # === Save Combined Files ===
    def save_combined_files(
        self, df: pd.DataFrame, processed_dir: Path
    ) -> Tuple[Path | None, Path | None]:
        """
        Save enriched DataFrame as both CSV and Parquet.

        Args:
            df: Enriched DataFrame.
            processed_dir: Output directory.

        Returns:
            Tuple of (csv_path, parq_path).
        """
        try:
            csv_file = self.save_csv(df, processed_dir, prefix=FilePrefixes.WHATSAPP_ALL_ENRICHED)
            parq_file = self.save_parq(df, processed_dir, prefix=FilePrefixes.WHATSAPP_ALL_ENRICHED)
            logger.info(f"DataFrame saved as: {csv_file} and {parq_file}")
            return csv_file, parq_file
        except Exception as e:
            logger.exception(f"Failed to save DataFrame: {e}")
            return None, None

    # === Save CSV ===
    def save_csv(
        self, df: pd.DataFrame, output_dir: Path, prefix: str = FilePrefixes.ORGANIZED
    ) -> Path | None:
        """
        Save DataFrame as timestamped CSV.

        Args:
            df: DataFrame to save.
            output_dir: Output directory.
            prefix: File prefix (default: organized_data).

        Returns:
            Path to saved CSV or None.
        """
        try:
            now = datetime.now(tz=pytz.timezone("Europe/Amsterdam")).strftime("%Y%m%d-%H%M%S")
            output = output_dir / f"{prefix}-{now}{FileExtensions.CSV}"
            df.to_csv(output, index=False)
            logger.info(f"Saved CSV: {output}")
            return output
        except Exception as e:
            logger.exception(f"Failed to save CSV to {output}: {e}")
            return None

    # === Save Parquet ===
    def save_parq(
        self, df: pd.DataFrame, output_dir: Path, prefix: str = FilePrefixes.ORGANIZED
    ) -> Path | None:
        """
        Save DataFrame as timestamped Parquet.

        Args:
            df: DataFrame to save.
            output_dir: Output directory.
            prefix: File prefix (default: organized_data).

        Returns:
            Path to saved Parquet or None.
        """
        try:
            now = datetime.now(tz=pytz.timezone("Europe/Amsterdam")).strftime("%Y%m%d-%H%M%S")
            output = output_dir / f"{prefix}-{now}{FileExtensions.PARQUET}"
            df.to_parquet(output, index=False)
            logger.info(f"Saved Parquet: {output}")
            return output
        except Exception as e:
            logger.exception(f"Failed to save Parquet to {output}: {e}")
            return None

    # === Save PNG ===
    def save_png(
        self, fig: Any, image_dir: Path, filename: str = ImageFilenames.YEARLY_BAR_CHART
    ) -> Path | None:
        """
        Save a matplotlib figure as PNG with timestamp.

        Args:
            fig: Matplotlib figure.
            image_dir: Output directory.
            filename: Base filename.

        Returns:
            Path to saved image or None.
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

    # === Save Table ===
    def save_table(self, df: pd.DataFrame, tables_dir: Path, prefix: str = FilePrefixes.TABLE) -> Path:
        """
        Save DataFrame as indexed CSV table.

        Args:
            df: DataFrame to save.
            tables_dir: Output directory.
            prefix: File prefix.

        Returns:
            Path to saved table.
        """
        now = datetime.now(tz=pytz.timezone("Europe/Amsterdam")).strftime("%Y%m%d-%H%M%S")
        logger.info(f"Generated timestamp: {now}")
        output = tables_dir / f"{prefix}-{now}{FileExtensions.CSV}"
        logger.info(f"Saving table to: {output}")
        df.to_csv(output, index=True)
        return output

    # === Group Enrichment & Concatenation ===
    def enrich_all_groups(self, data_editor: DataEditor, processed_dir: Path) -> Path | None:
        """
        Enrich and concatenate all cleaned group CSVs into a single enriched file.

        Args:
            data_editor: DataEditor instance for enrichment.
            processed_dir: Directory containing cleaned CSVs.

        Returns:
            Path to the enriched output file, or None on failure.
        """
        try:
            dfs: List[pd.DataFrame] = []
            for csv_file in processed_dir.glob(FilePatterns.CLEANED_CSV):
                name_part = csv_file.stem.split("-", 3)[-1]
                group_key = name_part.split("-cleaned")[0]
                group = GROUP_MAP_FROM_CLEANED.get(group_key)

                if not group:
                    logger.warning(f"Unknown group key '{group_key}' in {csv_file.name}, assigning to UNKNOWN")
                    group = Groups.UNKNOWN

                logger.info(f"Loading {csv_file.name} → group='{group}'")
                df = pd.read_csv(csv_file, parse_dates=[Columns.TIMESTAMP])
                df[Columns.WHATSAPP_GROUP] = group

                enriched_df = data_editor.organize_extended_df(df)
                if enriched_df is None:
                    logger.error(f"Failed to enrich {csv_file}")
                    continue

                dfs.append(enriched_df)

            if not dfs:
                logger.error("No dataframes to concatenate")
                return None

            combined = pd.concat(dfs, ignore_index=True)
            out_path = (
                processed_dir / f"{FilePrefixes.WHATSAPP_ALL_ENRICHED}-{pd.Timestamp.now():%Y%m%d-%H%M%S}{FileExtensions.CSV}"
            )
            combined.to_csv(out_path, index=False)
            logger.success(f"Enriched file saved: {out_path}")
            return out_path
        except Exception as e:
            logger.exception(f"Failed to enrich groups: {e}")
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
# - Add markers #NEW at the end of the module

# NEW: Full refactor with constants.py integration, docstrings, blank lines, and GROUP_MAP_FROM_CLEANED retained (2025-11-03)
# NEW: (2025-11-04) – Refactored get_preprocessed_data for reuse_whatsapp_all; Implemented merge-first-then-enrich; Removed per-group emoji/group addition; Added helpers for parquets; Unified saving with enriched prefix
# NEW: (2025-11-04) – Fixed reuse logic to honor reuse_whatsapp_all = false when preprocess = false