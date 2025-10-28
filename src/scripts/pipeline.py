# src/scripts/pipeline.py
import sys
from pathlib import Path
from loguru import logger
from datetime import datetime
import pytz
import tomllib
from typing import List, Optional, Dict, Any
import pandas as pd

from .script0 import Script0
from .script1 import Script1
from .script2 import Script2
from .script3 import Script3
from .script4 import Script4
from .script5 import Script5
from .script7 import Script7
from .script10 import Script10
from .script11 import Script11

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
        logger.add(sys.stderr, level="INFO", colorize=True)  # ← LIVE OUTPUT
        logger.info(f"Pipeline started. Log: {log_file}")
        return log_file

    @staticmethod
    def _load_config() -> Dict[str, Any]:
        configfile = Path("config.toml").resolve()
        with configfile.open("rb") as f:
            return tomllib.load(f)

    @staticmethod
    def run(scripts: Optional[List[int]] = None, skip_preprocessing: bool = False):
        import sys
        scripts = scripts or [7, 1, 2, 3, 4, 5, 10, 11]

        Pipeline._setup_logging()
        config = Pipeline._load_config()
        image_dir = Path(config["image"])
        processed_dir = Path(config["processed"])

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
        tables_dir = None

        # === PREPROCESSING: RUN OR SKIP ===
        if not skip_preprocessing:
            logger.info("Running Script0: preprocessing raw data...")
            script0 = Script0(
                file_manager=file_manager,
                data_editor=data_editor,
                data_preparation=data_preparation,
                processed_dir=processed_dir,
                config=config,
                image_dir=image_dir
            )
            result = script0.run()
            if result is None:
                logger.error("Script0 failed. Cannot continue.")
                return
            df = result['df']
            tables_dir = result['tables_dir']
            logger.info("Preprocessing completed.")
        else:
            logger.info("Skipping preprocessing. Attempting to load cached data...")
            # Find latest combined parquet file
            pattern = processed_dir / "combined_*.parq"
            files = list(processed_dir.glob("combined_*.parq"))
            if not files:
                logger.error("No preprocessed data found. Run without skip_preprocessing first.")
                return
            latest_file = max(files, key=lambda p: p.stat().st_mtime)
            try:
                df = pd.read_parquet(latest_file)
                tables_dir = Path("tables")
                tables_dir.mkdir(exist_ok=True)
                logger.info(f"Loaded cached data: {latest_file.name}")
            except Exception as e:
                logger.error(f"Failed to load cached data: {e}")
                return

        # === PREPARE CATEGORY DATA IF NEEDED ===
        scripts_needing_categories = {1, 4, 5, 7}
        category_data = None
        if scripts_needing_categories.intersection(scripts):
            from .utils import prepare_category_data
            category_data = prepare_category_data(data_preparation, df, logger)
            if category_data[0] is None:
                logger.error("Failed to prepare category data.")
                return
            df, group_authors, non_anthony_group, anthony_group, sorted_groups = category_data
        else:
            group_authors = non_anthony_group = anthony_group = sorted_groups = None

        # === SCRIPT REGISTRY ===
        script_registry = {
            1: (Script1, [file_manager, plot_manager, image_dir, group_authors, non_anthony_group, anthony_group, sorted_groups]),
            2: (Script2, [file_manager, data_preparation, plot_manager, image_dir, df]),
            3: (Script3, [file_manager, data_editor, data_preparation, plot_manager, image_dir, df]),
            4: (Script4, [file_manager, data_preparation, plot_manager, image_dir, tables_dir, group_authors, df]),
            5: (Script5, [file_manager, data_preparation, plot_manager, image_dir, df]),
            7: (Script7, [file_manager, data_preparation, plot_manager, image_dir, group_authors, df]),
            10: (Script10, [file_manager, data_editor, data_preparation, processed_dir, tables_dir]),
            11: (Script11, [file_manager, data_editor, data_preparation, plot_manager, processed_dir, image_dir]),
        }

        # === EXECUTE SCRIPTS ===
        instances = {}
        for script_id in scripts:
            if script_id not in script_registry:
                logger.warning(f"Script {script_id} not in registry. Skipping.")
                continue
            if script_id in scripts_needing_categories and group_authors is None:
                logger.warning(f"Script {script_id} needs category data but it's missing. Skipping.")
                continue

            cls, args = script_registry[script_id]
            try:
                instances[script_id] = cls(*args)
                logger.info(f"Initialized Script {script_id}")
            except Exception as e:
                logger.error(f"Failed to initialize Script {script_id}: {e}")

        for script_id in scripts:
            if script_id in instances:
                logger.info(f"Running Script {script_id}...")
                try:
                    instances[script_id].run()
                except Exception as e:
                    logger.exception(f"Script {script_id} failed: {e}")

        logger.success("Pipeline completed successfully.")