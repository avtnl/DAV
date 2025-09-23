from constants import DATE_TIME
import pandas as pd
from pathlib import Path
from datetime import datetime

class FileManager:
    """
    Utility class for managing file input and output operations.

    Provides methods for:
    - Reading CSV files (with special handling for '_amended_' files)
    - Writing CSV files to output and reports directories
    - Ensuring required output directories exist on initialization

    Attributes:
        input_base (str): Base directory for input files.
        output_base (str): Base directory for output files.
        reports_base (str): Base directory for report files.
    """

    def __init__(self, input_base: str, output_base: str, reports_base: str):
        """
        Initializes FileManager and ensures output directories exist.

        Args:
            input_base (str): Base path for input CSV files.
            output_base (str): Base path for general output CSV files.
            reports_base (str): Base path for report CSV files.
        """
        self.input_base = Path(input_base)
        self.output_base = Path(output_base)
        self.reports_base = Path(reports_base)
        self.plots_base = self.output_base / "Plots"

        # Ensure output directories exist
        self.plots_base.mkdir(parents=True, exist_ok=True)
        self.reports_base.mkdir(parents=True, exist_ok=True)


    def read_csv(self, filename: str) -> pd.DataFrame:
        """
        Reads a CSV file from either the input_base or output_base directory,
        depending on whether it's marked as amended.

        Args:
            filename (str): Target filename for the CSV.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        # Determine source directory based on filename convention
        if "_amended_" in filename:
            full_path = self.output_base / filename
            print(f"[{datetime.now().strftime('%H:%M:%S')}][FileManager] Reading AMENDED CSV: {full_path.as_posix()}")
        else:
            full_path = self.input_base / filename
            print(f"[{datetime.now().strftime('%H:%M:%S')}][FileManager] Reading INPUT CSV: {full_path.as_posix()}")

        # Check if file exists
        if not full_path.exists():
            msg = f"[FileManager] ERROR: File not found: {full_path.as_posix()}"
            print(msg)
            raise FileNotFoundError(msg)

        # Read the CSV
        return pd.read_csv(full_path)


    def write_csv(self, df: pd.DataFrame, filename: str) -> str:
        """
        Writes a DataFrame to a CSV file in the output_base directory.

        Args:
            df (pd.DataFrame): DataFrame to write.
            filename (str): Target filename for the CSV.
        """
        full_path = self.output_base / f"{filename}_{DATE_TIME}.csv"
        print(f"[{datetime.now().strftime('%H:%M:%S')}][FileManager] Writing file: {full_path.as_posix()}")
        df.to_csv(full_path, index=False)
        return full_path.name  # Return filename only


    def write_report_csv(self, df: pd.DataFrame, filename: str):
        """
        Writes a DataFrame to a CSV file in the reports_base directory,
        appending DATE_TIME to avoid Excel file locking issues.

        Args:
            df (pd.DataFrame): DataFrame to write.
            filename (str): Base filename for the report CSV (without extension).
        """
        full_path = self.reports_base / f"{filename}_{DATE_TIME}.csv"

        # Ensure the directory exists
        full_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{datetime.now().strftime('%H:%M:%S')}][FileManager] Writing REPORT file: {full_path.as_posix()}")
        df.to_csv(full_path, index=False)
