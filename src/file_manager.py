import tomllib
import re
import pytz
import shutil
from pathlib import Path
from loguru import logger
from datetime import datetime
import wa_analyzer.preprocess as preprocessor
import pandas as pd
import matplotlib.pyplot as plt

class FileManager:
    def find_name_csv(self, path, timestamp):
        """
        Find a CSV file in path with name 'whatsapp-YYYYMMDD-HHMMSS.csv' where
        the timestamp is later than the provided timestamp.
        
        Args:
            path (Path): Directory to search (e.g., Path("data/processed"))
            timestamp (str): Timestamp in format 'YYYYMMDD-HHMMSS' (e.g., '20250924-221905')
        
        Returns:
            Path or None: Path to the matching file, or None if no file is found
        """
        pattern = r"whatsapp-(\d{8}-\d{6})\.csv"
        try:
            input_dt = datetime.strptime(timestamp, "%Y%m%d-%H%M%S")
            input_dt = pytz.timezone('Europe/Amsterdam').localize(input_dt)
        except ValueError:
            logger.error(f"Invalid timestamp format: {timestamp}")
            return None
        
        for file in path.glob("*.csv"):
            match = re.match(pattern, file.name)
            if match:
                file_timestamp = match.group(1)
                try:
                    file_dt = datetime.strptime(file_timestamp, "%Y%m%d-%H%M%S")
                    file_dt = pytz.timezone('Europe/Amsterdam').localize(file_dt)
                    if file_dt > input_dt:
                        logger.info(f"Found matching file: {file}")
                        return file
                except ValueError:
                    logger.warning(f"Invalid timestamp in filename: {file.name}")
                    continue
        
        logger.warning(f"No CSV file found in {path} with timestamp later than {timestamp}")
        return None
    
    def find_latest_file(self, processed_dir, prefix="organized_data", suffix=".csv"):
        """
        Find the latest file with the given prefix and suffix in the processed directory based on timestamp in filename.

        Args:
            processed_dir (Path): Directory to search.
            prefix (str): Filename prefix (default: "organized_data").
            suffix (str): Filename suffix (default: ".csv").

        Returns:
            Path or None: Path to the latest file, or None if no file is found.
        """
        pattern = f"{prefix}-(\\d{{8}}-\\d{{6}}){suffix}"
        files = list(processed_dir.glob(f"{prefix}-*{suffix}"))
        if not files:
            logger.warning(f"No files found with prefix '{prefix}' and suffix '{suffix}' in {processed_dir}")
            return None
        latest_file = max(files, key=lambda f: datetime.strptime(re.search(pattern, f.name).group(1), "%Y%m%d-%H%M%S") if re.search(pattern, f.name) else datetime.min)
        logger.info(f"Found latest file: {latest_file}")
        return latest_file

    def read_csv(self):
        """
        Read configuration and determine the CSV file(s) to process.
        If preprocess is True, rename raw files, preprocess, and keep Parquet filenames in memory.
        If preprocess is False, use current_* keys from config.
        
        Returns:
            tuple: (list of Path or None, Path or None, dict, dict or None) - 
                   List of CSV files, processed directory, group mapping, Parquet files mapping
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
            "raw_4": ("current_4", "tillies")
        }
        group_map = {current_key: group for _, (current_key, group) in raw_files.items()}
        
        if preprocess:
            datafiles = []
            parq_files = {}
            
            for raw_key, (current_key, group) in raw_files.items():
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
                now = datetime.now(tz=pytz.timezone('Europe/Amsterdam')).strftime("%Y%m%d-%H%M%S")
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
                logger.info(f"Processed {raw_file} -> CSV: {datafile}, Parquet: {parq_file}")
                
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
            datafiles = []
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
    
    def save_csv(self, df, processed_dir, prefix="whatsapp"):
        """
        Save the DataFrame to a CSV file with a unique timestamped filename.

        Args:
            df (pandas.DataFrame): DataFrame to save.
            processed_dir (Path): Directory to save the file (e.g., Path("data/processed")).
            prefix (str): Filename prefix (default: "whatsapp").

        Returns:
            Path: Path to the saved CSV file.
        """
        now = datetime.now(tz=pytz.timezone('Europe/Amsterdam')).strftime("%Y%m%d-%H%M%S")
        logger.info(f"Generated timestamp: {now}")
        output = processed_dir / f"{prefix}-{now}.csv"
        logger.info(f"Saving CSV to: {output}")
        df.to_csv(output, index=False)
        return output

    def save_parq(self, df, processed_dir, prefix="whatsapp"):
        """
        Save the DataFrame to a Parquet file with a unique timestamped filename.

        Args:
            df (pandas.DataFrame): DataFrame to save.
            processed_dir (Path): Directory to save the file (e.g., Path("data/processed")).
            prefix (str): Filename prefix (default: "whatsapp").

        Returns:
            Path: Path to the saved Parquet file.
        """
        now = datetime.now(tz=pytz.timezone('Europe/Amsterdam')).strftime("%Y%m%d-%H%M%S")
        logger.info(f"Generated timestamp: {now}")
        output = processed_dir / f"{prefix}-{now}.parq"
        logger.info(f"Saving Parquet to: {output}")
        df.to_parquet(output, index=False)
        return output

    def save_combined_files(self, df, processed_dir):
        """
        Save the concatenated DataFrame to CSV and Parquet files with 'whatsapp_all-' prefix.

        Args:
            df (pandas.DataFrame): DataFrame to save.
            processed_dir (Path): Directory to save the files.

        Returns:
            tuple: (Path, Path) - Paths to the saved CSV and Parquet files, or (None, None) if saving fails.
        """
        try:
            csv_file = self.save_csv(df, processed_dir, prefix="whatsapp_all")
            parq_file = self.save_parq(df, processed_dir, prefix="whatsapp_all")
            logger.info(f"DataFrame saved as: {csv_file} and {parq_file}")
            return csv_file, parq_file
        except Exception as e:
            logger.exception(f"Failed to save DataFrame: {e}")
            return None, None

    def save_png(self, fig, image_dir, filename="yearly_bar_chart_combined"):
        """
        Save a matplotlib figure to a PNG file with a unique timestamped filename.

        Args:
            fig (matplotlib.figure.Figure): Figure to save.
            image_dir (Path): Directory to save the file (e.g., Path("img")).
            filename (str): Base filename (default: "yearly_bar_chart_combined").

        Returns:
            Path or None: Path to the saved PNG file, or None if saving fails.
        """
        try:
            now = datetime.now(tz=pytz.timezone('Europe/Amsterdam')).strftime("%Y%m%d-%H%M%S")
            output = image_dir / f"{filename}-{now}.png"
            image_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=300, bbox_inches="tight")
            logger.info(f"Saved bar chart: {output}")
            return output
        except Exception as e:
            logger.exception(f"Failed to save PNG to {output}: {e}")
            return None

    def save_table(self, df, tables_dir, prefix="table"):
        """
        Save the DataFrame to a CSV file with a unique timestamped filename.

        Args:
            df (pandas.DataFrame): DataFrame to save.
            tables_dir (Path): Directory to save the file (e.g., Path("tables")).
            prefix (str): Filename prefix (default: "table").

        Returns:
            Path: Path to the saved CSV file.
        """
        now = datetime.now(tz=pytz.timezone('Europe/Amsterdam')).strftime("%Y%m%d-%H%M%S")
        logger.info(f"Generated timestamp: {now}")
        output = tables_dir / f"{prefix}-{now}.csv"
        logger.info(f"Saving table to: {output}")
        df.to_csv(output, index=True)
        return output