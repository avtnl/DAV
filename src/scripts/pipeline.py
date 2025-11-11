# === Module Docstring ===
"""
Central pipeline orchestrator.

- Loads config, manages caching, initializes components
  and runs selected scripts in order.
- Automatically runs Script0 (preprocessing) if needed.

Uses flexible script registry with keyword arguments for clean, safe execution.
"""

# === Imports ===
from __future__ import annotations

import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytz
from loguru import logger
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

from src.constants import Columns, Groups
from src.data_editor import DataEditor
from src.data_preparation import DataPreparation
from src.file_manager import FileManager
from src.plot_manager import (
    CategoriesPlotSettings,
    TimePlotSettings,
    DistributionPlotSettings,
    RelationshipsPlotSettings,
    MultiDimPlotSettings,
    PlotManager,
)
from .script0 import Script0  # PreProcess
from .script1 import Script1  # Categories
from .script2 import Script2  # Time
from .script3 import Script3  # Distribution
from .script4 import Script4  # Relationships
from .script5 import Script5  # Multi Dimensions
from .script6 import Script6  # Dashboards


# === Pipeline Class ===
class Pipeline:
    """Orchestrates script execution with caching, logging, and flexible registry."""

    # === Logging Setup ===
    @staticmethod
    def _setup_logging() -> Path:
        """Create timestamped log file and configure loguru.

        Returns:
            Path to the log file.
        """
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
    def _load_config() -> Dict[str, Any]:
        """Load TOML config from root.

        Returns:
            Config dictionary.
        """
        configfile = Path("config.toml").resolve()
        with configfile.open("rb") as f:
            return tomllib.load(f)

    # === Main Runner ===
    @staticmethod
    def run(
        scripts: List[int] | None = None,
        script_5_details: List[Any] | None = None,
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

            # === DO NOT LOAD DF HERE ===
            df: pd.DataFrame | None = None
            tables_dir = Path("tables")
            tables_dir.mkdir(exist_ok=True)

            # === RUN SCRIPT0 FIRST IF NEEDED ===
            if 0 in scripts or df is None:
                logger.info("Running Script0 to generate or refresh data...")
                scripts = [0] + [s for s in scripts if s != 0]

                # Run Script0 immediately to populate df
                instance = Script0(
                    file_manager=file_manager,
                    data_editor=data_editor,
                    data_preparation=data_preparation,
                    processed_dir=processed_dir,
                    config=config,
                    image_dir=image_dir,
                )
                df = instance.run()
                if not isinstance(df, pd.DataFrame):
                    logger.error("Script0 did not return a DataFrame")
                    sys.exit(1)
                logger.info(f"Script0 completed. DF shape: {df.shape}")

            # === Script Registry (df now available) ===
            script_registry = {
                1: (Script1, {
                    "file_manager": file_manager,
                    "data_preparation": data_preparation,
                    "plot_manager": plot_manager,
                    "image_dir": image_dir,
                    "tables_dir": tables_dir,
                    "df": df,
                    "settings": CategoriesPlotSettings()
                }),
                2: (Script2, {
                    "file_manager": file_manager,
                    "data_preparation": data_preparation,
                    "plot_manager": plot_manager,
                    "image_dir": image_dir,
                    "df": df,
                    "settings": TimePlotSettings()
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
                    "df": df,
                    "settings": RelationshipsPlotSettings(),
                }),
                5: (Script5, {
                    "file_manager": file_manager,
                    "data_preparation": data_preparation,
                    "plot_manager": plot_manager,
                    "image_dir": image_dir,
                    "df": df,
                    "settings": MultiDimPlotSettings(),
                    "script_details": script_5_details or ["tsne", True, 0, 75, True, True, 3],
                }),                
                6: (Script6, {
                    "file_manager": file_manager,
                    "image_dir": image_dir,
                }),
            }

            instances: Dict[int, Any] = {}

            for script_id in scripts:
                if script_id == 0:
                    continue  # Already run

                if script_id not in script_registry:
                    logger.warning(f"Script {script_id} not in registry. Skipping.")
                    continue

                cls, kwargs = script_registry[script_id]

                try:
                    instance = cls(**kwargs)
                    instances[script_id] = instance
                    logger.info(f"Initialized Script {script_id}")
                except Exception as e:
                    logger.error(f"Failed to initialize Script {script_id}: {e}")
                    continue

                logger.info(f"Running Script {script_id}...")
                try:
                    result = instance.run()

                    # === FULL DIAGNOSTIC: Compare decomposition models ===
                    if script_id == 2:
                        main_path, season_path = result  # Unpack tuple
                        if season_path:
                            try:
                                df_dac = df[df[Columns.WHATSAPP_GROUP.value] == Groups.DAC.value]
                                evidence = data_preparation._compute_seasonality_evidence(df_dac)

                                # === RECOMPUTE BOTH MODELS FOR COMPARISON ===
                                y = evidence.raw_series
                                idx = pd.date_range(start="2015-01-01", periods=len(y), freq="W-MON")

                                # 1. Raw + additive
                                ts_raw = pd.Series(y, index=idx)
                                decomp_raw = seasonal_decompose(ts_raw, model="additive", period=52)
                                sigma_raw = np.nanstd(decomp_raw.resid)

                                # 2. Log + additive
                                y_log = np.log1p(y)
                                ts_log = pd.Series(y_log, index=idx)
                                decomp_log = seasonal_decompose(ts_log, model="additive", period=52)
                                sigma_log = np.nanstd(decomp_log.resid)
                                sigma_back = np.expm1(sigma_log)

                                # === PRINT FULL TABLE ===
                                print(f"\n{'Model':<25} {'Residual σ':<15} {'Back to messages'}")
                                print(f"{'-'*55}")
                                print(f"{'Raw + additive':<25} {sigma_raw:<15.2f} {'—'}")
                                print(f"{'Log + additive':<25} {sigma_log:<15.3f} {sigma_back:>12.1f}")
                                print(f"{'Used in plot':<25} {np.nanstd(evidence.decomposition['resid']):<15.2f} {'← FINAL'}")
                                print(f"{'-'*55}\n")

                                logger.info(f"Residual comparison: Raw={sigma_raw:.2f}, Log={sigma_log:.3f}, Back={sigma_back:.1f}")

                            except Exception as e:
                                logger.warning(f"Could not compute full diagnostic: {e}")
                    # === END DIAGNOSTIC ===
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

