# src/scripts/pipeline.py
import sys
from pathlib import Path
from loguru import logger
from datetime import datetime
import pytz
import tomllib
from typing import List, Optional, Dict, Any
import pandas as pd

# Local imports
from .script0 import Script0
from .script1 import Script1
from .script2 import Script2
from .script3 import Script3
from .script4 import Script4
from .script5 import Script5
from .script7 import Script7
from .script10 import Script10
from .script11 import Script11
from .utils import prepare_category_data

# External imports
from src.file_manager import FileManager
from src.data_editor import DataEditor
from src.data_preparation import DataPreparation, InteractionSettings, NoMessageContentSettings
from src.plot_manager import PlotManager


class Pipeline:
    @staticmethod
    def _setup_logging() -> Path:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now(tz=pytz.timezone('Europe/Amsterdam')).strftime("%Y%m%d-%H%M%S")
        log_file = log_dir / f"logfile-{timestamp}.log"

        logger.remove()
        logger.add(log_file, level="DEBUG")
        logger.add(sys.stderr, level="INFO", colorize=True)
        logger.info(f"Pipeline started. Log: {log_file}")
        return log_file

    @staticmethod
    def _load_config() -> Dict[str, Any]:
        configfile = Path("config.toml").resolve()
        with configfile.open("rb") as f:
            return tomllib.load(f)

    @staticmethod
    def run(scripts: Optional[List[int]] = None):
        # === Load config ===
        config = Pipeline._load_config()
        processed_dir = Path(config["processed"])
        image_dir = Path(config["image"])

        # === Use SCRIPTS from main.py ===
        if scripts is None:
            scripts = [1]
            logger.info("No SCRIPTS supplied – using fallback: [1]")
        else:
            logger.info(f"Running user-defined SCRIPTS: {scripts}")

        # === Force Script0 if no .parq exists ===
        if not list(processed_dir.glob("combined_*.parq")) and 0 not in scripts:
            logger.info("No pre-processed parquet found → forcing Script0.")
            scripts = [0] + [s for s in scripts if s != 0]

        # === Setup logging ===
        Pipeline._setup_logging()

        # === Core Components ===
        file_manager = FileManager()
        data_editor = DataEditor()
        data_preparation = DataPreparation(
            data_editor=data_editor,
            int_settings=InteractionSettings(),
            nmc_settings=NoMessageContentSettings(),
        )
        plot_manager = PlotManager()

        df = None
        tables_dir = Path("tables")
        tables_dir.mkdir(exist_ok=True)

        # === SCRIPT REGISTRY (including Script0) ===
        script_registry = {
            0: (Script0, [file_manager, data_editor, data_preparation, processed_dir, config, image_dir]),
            1: (Script1, [file_manager, plot_manager, image_dir]),
            2: (Script2, [file_manager, data_preparation, plot_manager, image_dir]),
            3: (Script3, [file_manager, data_editor, data_preparation, plot_manager, image_dir]),
            4: (Script4, [file_manager, data_preparation, plot_manager, image_dir, tables_dir]),
            5: (Script5, [file_manager, data_preparation, plot_manager, image_dir]),
            7: (Script7, [file_manager, data_preparation, plot_manager, image_dir]),
            10: (Script10, [file_manager, data_editor, data_preparation, processed_dir, tables_dir]),
            11: (Script11, [file_manager, data_editor, data_preparation, plot_manager, processed_dir, image_dir]),
        }

        # === SINGLE EXECUTION LOOP: Script0 first, then others ===
        instances = {}

        # 1. Run Script0 first (if in scripts)
        if 0 in scripts:
            logger.info("Running Script0 (preprocessing)...")
            cls, args = script_registry[0]
            script0 = cls(*args)
            result = script0.run()
            if result is None or "df" not in result:
                logger.error("Script0 failed or didn't return 'df'. Aborting.")
                return
            df = result["df"]
            tables_dir = result.get("tables_dir", tables_dir)
            instances[0] = script0
            logger.info(f"Script0 completed. DF shape: {df.shape}")

        # 2. Run all other scripts (once each)
        for script_id in [s for s in scripts if s != 0]:
            if script_id not in script_registry:
                logger.warning(f"Script {script_id} not in registry. Skipping.")
                continue

            cls, base_args = script_registry[script_id]
            args = base_args.copy()  # Avoid mutating original

            # Inject df for scripts that need it (as last arg)
            if script_id in {2, 3, 4, 5, 7, 10, 11}:
                if df is None:
                    logger.error(f"Script {script_id} needs df, but Script0 failed.")
                    continue
                args.append(df)

            # Prepare/inject category data for scripts that need it
            if script_id in {1, 4, 5, 7, 10}:
                category_data = prepare_category_data(data_preparation, df, logger)
                if category_data[0] is None:
                    logger.warning(f"Category data failed for Script {script_id}. Skipping.")
                    continue
                df_out, group_authors, non_anthony, anthony, sorted_g = category_data
                df = df_out  # Update df if modified

                # Customize args for category-dependent scripts
                if script_id == 1:
                    args = [file_manager, plot_manager, image_dir, group_authors, non_anthony, anthony, sorted_g]
                elif script_id in {4, 7}:
                    args.insert(-1, group_authors)  # Insert before df

            # Instantiate
            try:
                instance = cls(*args)
                instances[script_id] = instance
                logger.info(f"Initialized Script {script_id}")
            except Exception as e:
                logger.error(f"Failed to initialize Script {script_id}: {e}")
                continue

            # Run (exactly once)
            logger.info(f"Running Script {script_id}...")
            try:
                instance.run()
            except Exception as e:
                logger.exception(f"Script {script_id} failed: {e}")

        logger.success("Pipeline completed successfully.")