# === pipeline.py ===
# === Module Docstring ===
"""
Central pipeline orchestrator.

Loads config, manages caching, initializes components, and runs selected scripts
in order. Automatically runs Script0 (preprocessing) if needed.

Uses **BaseModel configs**, **validated data contracts**, and **flexible script registry**.
"""

from __future__ import annotations

# === Imports ===
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytz
from loguru import logger

from src.data_editor import DataEditor
from src.data_preparation import DataPreparation
from src.plot_manager import (
    CategoriesPlotSettings,
    DistributionPlotSettings,
    ArcPlotSettings,
    BubblePlotSettings,
    MultiDimPlotSettings,
    PlotManager,
)
from src.file_manager import FileManager

from .script0 import Script0
from .script1 import Script1
from .script2 import Script2
from .script3 import Script3
from .script4 import Script4
from .script5 import Script5
from .script6 import Script6  # ← NEW


# === Pipeline Class ===
class Pipeline:
    """Orchestrates script execution with caching, logging, and BaseModel configs."""

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
    def run(
        scripts: list[int] | None = None,
        script_6_details: list | None = None,  # ← Passed directly to Script6
    ) -> None:
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
            data_preparation = DataPreparation(data_editor=data_editor)
            plot_manager = PlotManager()
            plot_manager.data_preparation = data_preparation  # Inject dependency

            df: pd.DataFrame | None = None
            tables_dir = Path("tables")
            tables_dir.mkdir(exist_ok=True)

            # Load enriched CSV (whatsapp_all_enriched-*.csv)
            df = file_manager.get_latest_preprocessed_df()
            if df is None or df.empty:
                logger.info("No enriched CSV or Script0 requested - forcing Script0.")
                scripts = [0] + [s for s in scripts if s != 0]

            # === Script Registry ===
            script_registry = {
                0: (
                    Script0,
                    [file_manager, data_editor, data_preparation, processed_dir, config, image_dir],
                    None  # No df needed
                ),
                1: (
                    Script1,
                    [file_manager, data_preparation, plot_manager, image_dir, tables_dir, df],
                    None
                ),
                2: (
                    Script2,
                    [file_manager, data_preparation, plot_manager, image_dir],
                    df
                ),
                3: (
                    Script3,
                    [file_manager, data_editor, data_preparation, plot_manager, image_dir, df],
                    None
                ),
                4: (
                    Script4,
                    [file_manager, data_preparation, plot_manager, image_dir, tables_dir],
                    df
                ),
                5: (
                    Script5,
                    [file_manager, data_preparation, plot_manager, image_dir, df],
                    None
                ),
                6: (
                    Script6,
                    [file_manager, data_preparation, plot_manager, image_dir, df],
                    None
                )
            }

            # === Single Execution Loop ===
            instances: dict[int, Any] = {}

            for script_id in scripts:
                if script_id not in script_registry:
                    logger.warning(f"Script {script_id} not in registry. Skipping.")
                    continue

                # Unpack entry
                cls, base_args, df_arg = script_registry[script_id]
                args = base_args.copy()  # e.g. [file_manager, ..., image_dir]

                # === Script1: Categories ===
                if script_id == 1:
                    config_obj = CategoriesPlotSettings(
                        figsize=(16, 9),
                        group_spacing=3.0,
                        title="Anthony's participation is significantly lower for the 3rd group",
                        subtitle="Too much to handle or too much crap?",
                    )
                    args.append(config_obj)

                # === Script3: Distribution ===
                if script_id == 3:
                    args.append(DistributionPlotSettings())

                # === Script4: Arc ===
                if script_id == 4:
                    args.append(ArcPlotSettings())

                # === Script5: Bubble ===
                if script_id == 5:
                    args.append(BubblePlotSettings())

                # === Script6: Multi-Dimensional Style ===
                if script_id == 6:
                    if script_6_details is None:
                        logger.error("SCRIPT_6_DETAILS not provided for Script6. Check main.py.")
                        continue

                    # ORDER: [plot_type, by_group, draw_ellipses, use_embeddings, hybrid_features, embedding_model]
                    plot_type = script_6_details[0]
                    by_group = script_6_details[1]
                    draw_ellipses = script_6_details[2]
                    use_embeddings = script_6_details[3]
                    hybrid_features = script_6_details[4]
                    embedding_model = script_6_details[5]

                    args.append(MultiDimPlotSettings(
                        by_group=by_group,
                        draw_ellipses=draw_ellipses,
                        use_embeddings=use_embeddings,
                        hybrid_features=hybrid_features,
                        embedding_model=embedding_model,
                    ))
                    args.append(script_6_details)  # Pass full list to Script6

                # Inject df if provided (after settings)
                if df_arg is not None and script_id in {1, 2, 3, 4, 5}:
                    args.append(df_arg)

                # === Special handling for Script0 ===
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

                # === Instantiate ===
                try:
                    instance = cls(*args)
                    instances[script_id] = instance
                    logger.info(f"Initialized Script {script_id}")
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to initialize Script {script_id}: {e}")
                    continue

                # === Run ===
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

# NEW: Final working version with df injection (2025-11-01)
# NEW: 3-tuple registry: (cls, args, df)
# NEW: Script1 receives df and config
# NEW: Clean, robust, production-ready
# NEW: Removed duplicate config append for Script1 (2025-11-01)
# NEW: Added Script3 with correct df injection order (2025-11-01)
# NEW: df appended after settings to match Script3.__init__ (2025-11-01)
# NEW: DistributionPlotSettings injected for Script3 (2025-11-01)
# NEW: Added Script4 to registry with ArcPlotSettings injection (2025-11-03)
# NEW: df appended after settings to match Script4.__init__ (2025-11-03)
# NEW: Added Script5 to registry with BubblePlotSettings injection (2025-11-03)
# NEW: df in base_args to match Script5.__init__ (2025-11-03)
# NEW: Added Script6 with direct script_6_details injection (2025-11-03)
# NEW: Validation moved to script6.py (2025-11-03)
# NEW: Removed pipeline instance state (2025-11-03)