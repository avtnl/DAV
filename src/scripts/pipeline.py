# === Module Docstring ===
"""
Central pipeline orchestrator.

Loads config, manages caching, initializes components, and runs selected scripts
in order. Automatically runs Script0 (preprocessing) if needed.

Examples
--------
>>> from src.scripts.pipeline import Pipeline
>>> Pipeline.run(scripts=[7, 1])
"""

# === Imports ===
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytz
from loguru import logger

from src.data_editor import DataEditor  # type: ignore[import-not-found]
from src.data_preparation import (  # type: ignore[import-not-found]
    DataPreparation,
    InteractionSettings,
    NoMessageContentSettings,
)
from src.dev.plot_manager import PlotManager
from src.file_manager import FileManager  # type: ignore[import-not-found]

from .script0 import Script0
from .script1 import Script1
from .script2 import Script2  # type: ignore[import-not-found]
from .script3 import Script3  # type: ignore[import-not-found]
from .script4 import Script4  # type: ignore[import-not-found]
from .script5 import Script5  # type: ignore[import-not-found]
from .utils import prepare_category_data


# === Pipeline Class ===
class Pipeline:
    """Orchestrates script execution with caching and logging."""

    # === Logging Setup ===
    @staticmethod
    def _setup_logging() -> Path:
        """Create timestamped log file and configure loguru."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now(tz=pytz.timezone("Europe/Amsterdam")).strftime("%Y%m%d-%H%M%S")
        log_file = log_dir / f"logfile-{timestamp}.log"

        logger.remove()
        logger.add(log_file, level="DEBUG")
        logger.add(sys.stderr, level="INFO", colorize=True)
        logger.info(f"Pipeline started. Log: {log_file}")
        return log_file

    # === Config Loading ===
    @staticmethod
    def _load_config() -> dict[str, Any]:
        """Load TOML config from root."""
        configfile = Path("config.toml").resolve()
        with configfile.open("rb") as f:
            return tomllib.load(f)

    # === Main Runner ===
    @staticmethod
    # ruff: noqa: C901, PLR0912, PLR0915
    def run(scripts: list[int] | None = None) -> None:
        """Execute scripts in order with preprocessing fallback."""
        try:
            # Load config
            config = Pipeline._load_config()
            processed_dir = Path(config["processed"])
            image_dir = Path(config["image"])

            # Use SCRIPTS from entry point (main)
            if scripts is None:
                scripts = [1]
                logger.info("No SCRIPTS supplied - using fallback: [1]")
            else:
                logger.info(f"Running user-defined SCRIPTS: {scripts}")

            # Setup logging
            Pipeline._setup_logging()

            # Core components
            file_manager = FileManager()
            data_editor = DataEditor()
            data_preparation = DataPreparation(
                data_editor=data_editor,
                int_settings=InteractionSettings(),
                nmc_settings=NoMessageContentSettings(),
            )
            plot_manager = PlotManager()

            df: pd.DataFrame | None = None
            tables_dir = Path("tables")
            tables_dir.mkdir(exist_ok=True)

            # Load or run preprocessing
            files = list(processed_dir.glob("combined_*.parq"))
            if files and 0 not in scripts:
                latest_file = max(files, key=lambda p: p.stat().st_mtime)
                try:
                    df = pd.read_parquet(latest_file)
                    logger.info(f"Loaded cached data: {latest_file.name}. DF shape: {df.shape}")
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to load cached data: {e}. Forcing Script0.")
                    scripts = [0, *scripts]
            else:
                logger.info("No cached parquet or Script0 requested → forcing Script0.")
                scripts = [0] + [s for s in scripts if s != 0]

            # Script registry
            script_registry = {
                0: (
                    Script0,
                    [file_manager, data_editor, data_preparation, processed_dir, config, image_dir],
                ),
                1: (Script1, [file_manager, plot_manager, image_dir]),
                2: (Script2, [file_manager, data_preparation, plot_manager, image_dir]),
                3: (
                    Script3,
                    [file_manager, data_editor, data_preparation, plot_manager, image_dir],
                ),
                4: (Script4, [file_manager, data_preparation, plot_manager, image_dir, tables_dir]),
                5: (Script5, [file_manager, data_preparation, plot_manager, image_dir]),
            }

            # Single execution loop
            instances: dict[int, Any] = {}

            for script_id in scripts:
                if script_id not in script_registry:
                    logger.warning(f"Script {script_id} not in registry. Skipping.")
                    continue

                cls, base_args = script_registry[script_id]
                args = base_args.copy()

                # Inject df for scripts that need it
                if script_id in {2, 3, 4, 5, 7, 10, 11}:
                    if df is None:
                        logger.error(f"Script {script_id} needs df, but preprocessing failed.")
                        continue
                    args.append(df)

                # Special handling for Script0
                if script_id == 0:
                    logger.info("Running Script0 (preprocessing)...")
                    try:
                        instance = cls(*args)
                        result = instance.run()
                        if result is None or "df" not in result:
                            logger.error("Script0 failed or didn't return 'df'. Aborting.")
                            return
                        df = result["df"]
                        tables_dir = result.get("tables_dir", tables_dir)
                        instances[script_id] = instance
                        logger.info(f"Script0 completed. DF shape: {df.shape}")
                    except Exception as e:  # noqa: BLE001
                        logger.exception(f"Script0 failed: {e}")
                        return
                    continue

                # Prepare category data
                if script_id in {1, 4, 5}:
                    try:
                        category_data = prepare_category_data(data_preparation, df, logger)
                        if category_data[0] is None:
                            logger.warning(
                                f"Category data failed for Script {script_id}. Skipping."
                            )
                            continue
                        df_out, group_authors, non_anthony, anthony, sorted_g = category_data
                        df = df_out

                        if script_id == 1:
                            args = [
                                file_manager,
                                plot_manager,
                                image_dir,
                                group_authors,
                                non_anthony,
                                anthony,
                                sorted_g,
                            ]
                        elif script_id in {4, 7}:
                            args.insert(-1, group_authors)
                    except Exception as e:  # noqa: BLE001
                        logger.exception(
                            f"Category data preparation failed for Script {script_id}: {e}"
                        )
                        continue

                # Instantiate
                try:
                    instance = cls(*args)
                    instances[script_id] = instance
                    logger.info(f"Initialized Script {script_id}")
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to initialize Script {script_id}: {e}")
                    continue

                # Run
                logger.info(f"Running Script {script_id}...")
                try:
                    instance.run()
                except Exception as e:  # noqa: BLE001
                    logger.exception(f"Script {script_id} failed: {e}")

            logger.success("Pipeline completed successfully.")
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Pipeline failed: {e}")


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

# NEW: Fixed pipeline.py with ruff/mypy ignores (2025-11-01)
