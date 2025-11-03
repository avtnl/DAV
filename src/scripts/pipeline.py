# === pipeline.py ===
# === Module Docstring ===
"""
Central pipeline orchestrator.

Loads config, manages caching, initializes components, and runs selected scripts
in order. Automatically runs Script0 (preprocessing) if needed.

Uses **BaseModel configs**, **validated data contracts**, and **flexible script registry**.
"""

# === Imports ===
from __future__ import annotations

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
from .script6 import Script6
from .script7 import Script7


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
    def run(
        scripts: list[int] | None = None,
        script_6_details: list | None = None,
    ) -> None:
        """Execute scripts in order with preprocessing fallback."""
        try:
            config = Pipeline._load_config()
            processed_dir = Path(config["processed"])
            image_dir = Path(config["image"])

            if scripts is None:
                scripts = [1]
                logger.info("No SCRIPTS supplied - using fallback: [1]")
            else:
                logger.info(f"Running user-defined SCRIPTS: {scripts}")

            Pipeline._setup_logging()

            file_manager = FileManager()
            data_editor = DataEditor()
            data_preparation = DataPreparation(data_editor=data_editor)
            plot_manager = PlotManager()
            plot_manager.data_preparation = data_preparation

            df: pd.DataFrame | None = file_manager.get_latest_preprocessed_df()
            tables_dir = Path("tables")
            tables_dir.mkdir(exist_ok=True)

            if df is None or df.empty:
                logger.info("No enriched CSV or Script0 requested - forcing Script0.")
                scripts = [0] + [s for s in scripts if s != 0]

            # === Script Registry ===
            script_registry = {
                0: (Script0, {
                    "file_manager": file_manager,
                    "data_editor": data_editor,
                    "data_preparation": data_preparation,
                    "processed_dir": processed_dir,
                    "config": config,
                    "image_dir": image_dir,
                }),
                1: (Script1, {
                    "file_manager": file_manager,
                    "data_preparation": data_preparation,
                    "plot_manager": plot_manager,
                    "image_dir": image_dir,
                    "tables_dir": tables_dir,
                    "df": df,
                    "settings": CategoriesPlotSettings(
                        figsize=(16, 9),
                        group_spacing=3.0,
                        title="Anthony's participation is significantly lower for the 3rd group",
                        subtitle="Too much to handle or too much crap?",
                    ),
                }),
                2: (Script2, {
                    "file_manager": file_manager,
                    "data_preparation": data_preparation,
                    "plot_manager": plot_manager,
                    "image_dir": image_dir,
                    "df": df,
                }),
                3: (Script3, {
                    "file_manager": file_manager,
                    "data_editor": data_editor,
                    "data_preparation": data_preparation,
                    "plot_manager": plot_manager,
                    "image_dir": image_dir,
                    "df": df,
                    "settings": DistributionPlotSettings(),
                }),
                4: (Script4, {
                    "file_manager": file_manager,
                    "data_preparation": data_preparation,
                    "plot_manager": plot_manager,
                    "image_dir": image_dir,
                    "tables_dir": tables_dir,
                    "df": df,
                    "settings": ArcPlotSettings(),
                }),
                5: (Script5, {
                    "file_manager": file_manager,
                    "data_preparation": data_preparation,
                    "plot_manager": plot_manager,
                    "image_dir": image_dir,
                    "df": df,
                    "settings": BubblePlotSettings(),
                }),
                6: (Script6, {
                    "file_manager": file_manager,
                    "data_preparation": data_preparation,
                    "plot_manager": plot_manager,
                    "image_dir": image_dir,
                    "df": df,
                    "settings": MultiDimPlotSettings(
                        by_group=script_6_details[1] if script_6_details else True,
                        draw_ellipses=script_6_details[2] if script_6_details else False,
                        use_embeddings=script_6_details[3] if script_6_details else True,
                        hybrid_features=script_6_details[4] if script_6_details else True,
                        embedding_model=script_6_details[5] if script_6_details else 3,
                    ),
                    "script_details": script_6_details or ["tsne", True, False, True, True, 3],
                }),
                7: (Script7, {
                    "file_manager": file_manager,
                    "image_dir": image_dir,
                }),
            }

            instances: dict[int, Any] = {}

            for script_id in scripts:
                if script_id not in script_registry:
                    logger.warning(f"Script {script_id} not in registry. Skipping.")
                    continue

                cls, kwargs = script_registry[script_id]

                if script_id == 0:
                    logger.info("Running Script0 (preprocessing)...")
                    try:
                        instance = cls(**kwargs)
                        result = instance.run()
                        if result is None or "df" not in result:
                            logger.error("Script0 failed or didn't return 'df'. Aborting.")
                            return
                        df = result["df"]
                        tables_dir = result.get("tables_dir", tables_dir)
                        instances[script_id] = instance
                        logger.info(f"Script0 completed. DF shape: {df.shape}")
                    except Exception as e:
                        logger.exception(f"Script0 failed: {e}")
                        return
                    continue

                try:
                    instance = cls(**kwargs)
                    instances[script_id] = instance
                    logger.info(f"Initialized Script {script_id}")
                except Exception as e:
                    logger.error(f"Failed to initialize Script {script_id}: {e}")
                    continue

                logger.info(f"Running Script {script_id}...")
                try:
                    instance.run()
                except Exception as e:
                    logger.exception(f"Script {script_id} failed: {e}")

            logger.success("Pipeline completed successfully.")
        except Exception as e:
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

# NEW: Used **kwargs in registry for clarity and safety (2025-11-03)
# NEW: Removed positional args entirely (2025-11-03)
# NEW: Added fallback for script_6_details (2025-11-03)