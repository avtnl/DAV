from datetime import datetime

import numpy as np
import pandas as pd
from data_preparation import DataPreparation
from file_manager import FileManager
from plot_manager import PlotManager
from re_utilities import (
    apply_basic_re,
    apply_datetime_fixes_using_re,
    apply_split_at_end_number_using_re,
    delete_us_space_using_re,
)

from constants import (
    AIRPORT_COORDINATES,
    FOGGY,
    HAZY,
    INCIDENT_INFO,
    RAINY,
    ROAD_NAMES,
    SNOWY,
    STORMY,
    SUNNY,
    WINDY,
)


class DataEditor:
    """
    Responsible for adding and transforming columns in the master DataFrame.
    All final columns that will be written to CSV should be added here.

    Responsibilities:
    - Clean and parse datetime columns (new_datetime_columns)
    - Generic conversion of datetime columns (convert_datetime_columns)
    - Compute and apply Rush Hour and Bank Holiday flags (add_rush_hour_and_bankholiday_flags)
    - Add duration interval columns (add_new_duration_columns)

    Note:
    - This class DOES edit the master DataFrame.
    - All "feature columns" to be included in final output should be applied here.
    """

    def __init__(
        self, file_manager: FileManager, data_preparation: DataPreparation
    ) -> None:  # data_preparation is the instance of DataPreparation to use
        self.fm = (
            file_manager  # Assigns the incoming file manager object to an instance variable namedfm
        )
        self.dp = data_preparation  # Assigns the incoming data_preparation object to an instance variable named dp

    def clean_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts and cleans datetime-related columns from the input DataFrame:
        - Strips leading/trailing whitespace.
        - Removes trailing fractional seconds (e.g. '.123').

        Args:
            df (pd.DataFrame): Source DataFrame containing datetime columns.

        Returns:
            pd.DataFrame: New DataFrame with cleaned 'Start_Time', 'End_Time', 'Weather_Timestamp' columns.
        """
        # Guard clause: ensure required columns exist
        required_columns = {"Start_Time", "End_Time", "Weather_Timestamp"}
        missing_columns = required_columns - set(df.columns)
        assert not missing_columns, f"Missing required columns: {missing_columns}"

        # Create new DataFrame with copied columns
        df_datetimes = df[["Start_Time", "End_Time", "Weather_Timestamp"]].copy()

        # Convert to string and clean whitespace + fractional seconds
        for column in df_datetimes.columns:
            df_datetimes[column] = df_datetimes[column].astype(str).str.strip()
            df_datetimes[column] = df_datetimes[column].apply(apply_datetime_fixes_using_re)

        return df_datetimes

    def label_source_and_severity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps the 'Source' and 'Severity' columns to human-readable labels.

        Args:
            df (pd.DataFrame): DataFrame containing 'Source' and 'Severity' columns.

        Returns:
            pd.DataFrame: The same DataFrame with added 'Source_Label' and 'Severity_Label' columns.
        """
        assert "Source" in df.columns, "'Source' column is missing"
        assert "Severity" in df.columns, "'Severity' column is missing"

        source_mapping = {"Source1": "MapQuest", "Source2": "Bing", "Source3": "MapQuest & Bing"}

        severity_mapping = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}

        df["Source"] = df["Source"].map(source_mapping).fillna("Unknown Source")
        df["Severity"] = df["Severity"].map(severity_mapping).fillna("Unknown Severity")

        return df

    def add_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized version:
        - Parses cleaned datetime columns
        - Adds derived features
        - Flags invalids
        ==========

        Parses cleaned datetime columns and adds derived features to the input DataFrame.

        Assumes datetime strings in df_datetimes have already been cleaned.

        Performs:
        - Preserves original raw datetime values in '_Original' columns.
        - Parses cleaned strings into pandas datetime objects.
        - Flags invalid datetime entries.
        - Adds derived features:
            - 'Start_Date_Only', 'Start_Time_Only'
            - 'Day_of_week', 'Hour_of_day', 'Month_of_year'
            - Duration columns (in minutes)

        Args:
            df (pd.DataFrame): Target DataFrame to be updated.
            df_datetimes (pd.DataFrame): Cleaned datetime columns.

        Returns:
            pd.DataFrame: Updated DataFrame with parsed datetimes and new features.
        """

        def format_timedelta_series(td_series: pd.Series) -> pd.Series:
            total_seconds = td_series.dt.total_seconds().fillna(0).astype(int)
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return (
                hours.astype(str).str.zfill(2)
                + ":"
                + minutes.astype(str).str.zfill(2)
                + ":"
                + seconds.astype(str).str.zfill(2)
            )

        print(
            f"[{datetime.now().strftime('%H:%M:%S')}][DataEditor] Starting to add new 'datetime'columns"
        )

        # Preserve original raw values
        df["Start_Time_Original"] = df["Start_Time"]
        df["End_Time_Original"] = df["End_Time"]
        df["Weather_Timestamp_Original"] = df["Weather_Timestamp"]

        # Convert to string and clean whitespace + fractional seconds
        for col in ["Start_Time", "End_Time", "Weather_Timestamp"]:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.replace(r"\.\d+$", "", regex=True)

        # Parse cleaned strings into datetimes
        df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
        df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")
        df["Weather_Timestamp"] = pd.to_datetime(df["Weather_Timestamp"], errors="coerce")

        # Add 3 new columns to easily identify invalid datafields
        df["Start_Invalid"] = df[
            "Start_Time"
        ].isna()  # isna() is a pandas method which returns True if NaT or NaN
        df["End_Invalid"] = df["End_Time"].isna()
        df["Timestamp_Invalid"] = df["Weather_Timestamp"].isna()

        # Derive fields
        df["Start_Date_Only"] = df["Start_Time"].dt.date
        df["Start_Time_Only"] = df["Start_Time"].dt.time
        df["Day_of_Week"] = df["Start_Time"].dt.strftime("%A")
        df["Hour_of_Day"] = df["Start_Time"].dt.hour
        df["Month"] = df["Start_Time"].dt.month
        df["Year"] = df["Start_Time"].dt.year

        # Durations (numeric)
        df["Duration_Start_to_End(min)"] = (
            df["End_Time"] - df["Start_Time"]
        ).dt.total_seconds() / 60.0
        df["Duration_Start_to_Timestamp(min)"] = (
            df["Weather_Timestamp"] - df["Start_Time"]
        ).dt.total_seconds().abs() / 60.0

        # Durations (formatted)
        df["Duration_Start_to_End(hh:mm:ss)"] = format_timedelta_series(
            df["End_Time"] - df["Start_Time"]
        )
        df["Duration_Start_to_Timestamp(hh:mm:ss)"] = format_timedelta_series(
            df["Weather_Timestamp"] - df["Start_Time"]
        )

        self.number_of_columns(df)

        return df

    def validate_uniqueness_of_key(
        self, df: pd.DataFrame, column_name: str, max_duplicates: int = 10
    ) -> pd.DataFrame:
        """
        Ensures uniqueness of the specified key column (e.g., 'ID') by appending digits to duplicates.
        Updates 'Modification_Notes' accordingly using methods ensure_column_exists() and update_modification_notes().

        Args:
            df (pd.DataFrame): The DataFrame to process.
            column_name (str): The column name to enforce uniqueness on.
            max_duplicates (int): Max allowed duplicate count per original key (default: 10).

        Returns:
            pd.DataFrame: The updated DataFrame with unique keys and modification notes.
        """
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}][DataEditor] Validating uniqueness of key: {column_name}"
        )

        # Sort the DataFrame by the target column
        df = df.sort_values(by=column_name).reset_index(drop=True)

        # Ensure the modification notes column exists
        df = self.ensure_column_exists(df, "Modification_Notes")

        # Track how many times each key has appeared
        key_counts = {}
        total_duplicates_found = 0

        # Iterate over each row to enforce uniqueness
        for i, original_id in enumerate(df[column_name]):
            if original_id not in key_counts:
                key_counts[original_id] = 0
            else:
                key_counts[original_id] += 1

            count = key_counts[original_id]

            # If a duplicate is found
            if count > 0:
                if count < max_duplicates:
                    total_duplicates_found += 1

                    # Append digit to make the ID unique
                    new_id = f"{original_id}{count}"
                    df.at[i, column_name] = new_id

                    # Update 'Modification_Notes'
                    existing_note = df.at[i, "Modification_Notes"]
                    df.at[i, "Modification_Notes"] = self.update_modification_notes(
                        existing_note, "ID changed"
                    )
                else:
                    print(
                        f"[Warning] More than {max_duplicates} duplicates found for ID: {original_id}"
                    )

        print(
            f"[{datetime.now().strftime('%H:%M:%S')}][DataEditor] Duplicates found and repaired: {total_duplicates_found}"
        )

        return df

    def add_zipcode5(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a new column 'Zipcode5' containing the first 5 digits of the 'Zipcode' column.
        If the 'Zipcode' is exactly 5 characters long, it copies the full value.
        If longer (typically 9 digits), it takes only the first 5 digits.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with the new 'Zipcode5' column added.
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataEditor] Adding column 'Zipcode5")

        # Ensure 'Zipcode' is string to safely slice
        df["Zipcode"] = df["Zipcode"].astype(str)

        # Add 'Zipcode5' column using slicing logic
        df["Zipcode5"] = df["Zipcode"].apply(lambda z: z if len(z) == 5 else z[:5])

        self.number_of_columns(df)

        return df

    def add_traffic_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scans the 'Description' column for the first match with any phrase in INCIDENT_INFO.
        Matching is case-insensitive. The matched phrase is written to a new column 'Traffic_Info'.

        Args:
            df (pd.DataFrame): Input DataFrame with a 'Description' column.

        Returns:
            pd.DataFrame: Updated DataFrame with a new 'Traffic_Info' column.
        """

        def find_first_match(text: str) -> str:
            """
            Returns the first matching phrase from INCIDENT_INFO found in the given text.
            Matching is case-insensitive and searches for substring presence.
            If no match is found, returns None.
            """
            for phrase in INCIDENT_INFO:
                if phrase in text:
                    return phrase
            return None

        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataEditor] Adding column 'traffic_info'")

        # Ensure Description is string and non-null
        df["Description"] = df["Description"].fillna("").astype(str)

        # Convert all descriptions to lowercase once
        descriptions_lower = df["Description"].str.lower()

        # Find first matching phrase for each description
        df["Traffic_Info"] = descriptions_lower.apply(find_first_match)

        self.number_of_columns(df)

        return df

    def add_rush_hour_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds'Rush_Hour' columns to the DataFrame (True = Accident started in rush hour).
        Rush hour intervals are defined in DataPreparation.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with 'Rush_Hour' columns
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataEditor] Adding column 'rush_hour_flag'")

        # Compute base Rush Hour
        df["Rush_Hour"] = DataPreparation.compute_rush_hour_flag(df)

        self.number_of_columns(df)

        return df

    def add_bankholiday_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds 'Bankholiday' columns to the DataFrame.
        Weekends and working day before and after a bankholiday will all have 'Bankholiday' = True.
        Updates the existing 'Rush_Hour' column, if feasible. Example normal rush-hour starts at 16:00.
        However on a last working day before a bankholiday this is preponed to 14:00.
        Bnakholidays and Bankholiday-weekends have 'Rush-Hour' = False.
        Bankholiday patterns are defined in DataPreparation.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with updated 'Rush_Hour' and 'Bankholiday' columns.
        """
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}][DataEditor] Adding column 'bankholiday_flag'"
        )

        # Compute Bankholiday + Rush_Hour adjustment
        bankholiday_flag = DataPreparation.compute_bankholiday_flag(df)

        df["Bankholiday"] = bankholiday_flag

        self.number_of_columns(df)

        return df

    def add_weather_condition_flags(self, df):
        """
        Adds boolean columns based on 'Weather_Condition' and temperature.

        Weather condition columns:
            - 'Rainy', 'Snowy', 'Foggy', 'Hazy', 'Windy', 'Stormy', 'Sunny'

        Temperature-based column:
            - 'Icy': True if temp <= 32.0, False if > 32.0, None if missing
        """
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}][DataEditor] Adding several 'weather_condition' columns"
        )

        # Ensure required columns exist
        if "Weather_Condition" not in df.columns:
            raise ValueError("'Weather_Condition' column not found in DataFrame.")
        if "Temperature(F)" not in df.columns:
            raise ValueError("'Temperature(F)' column not found in DataFrame.")

        # Clean weather condition strings
        df["Weather_Condition"] = df["Weather_Condition"].fillna("")
        condition_lower = df["Weather_Condition"].str.lower()

        # Weather condition flags
        df["Rainy"] = condition_lower.isin([w.lower() for w in RAINY])
        df["Snowy"] = condition_lower.isin([w.lower() for w in SNOWY])
        df["Foggy"] = condition_lower.isin([w.lower() for w in FOGGY])
        df["Hazy"] = condition_lower.isin([w.lower() for w in HAZY])
        df["Windy"] = condition_lower.isin([w.lower() for w in WINDY])
        df["Stormy"] = condition_lower.isin([w.lower() for w in STORMY])
        df["Sunny"] = condition_lower.isin([w.lower() for w in SUNNY])

        # Icy flag based on temperature
        df["Icy"] = df["Temperature(F)"].apply(
            lambda x: True
            if pd.notna(x) and x <= 32.0
            else False
            if pd.notna(x) and x > 32.0
            else None
        )

        self.number_of_columns(df)

        return df

    def add_period_of_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Made by DataPrepatation based on Start_Time (missing occurances) and four twilight/sunrise datafields.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with updated 'Rush_Hour' and 'Bankholiday' columns.
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataEditor] Adding column 'Period_of_Day'")

        df = self.dp.summarize_period_of_day(df)

        self.number_of_columns(df)

        return df

    def add_airport_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds airport latitude, longitude, and distance from accident start location to the airport.
        Texas (TX) only)

        Adds columns:
        - 'Airport_Lat'
        - 'Airport_Lng'
        - 'Distance_Start_Airport'

        Returns updated DataFrame.
        """
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}][DataEditor] Adding several 'airport' columns"
        )

        # Initialize columns
        df["Airport_Lat"] = np.nan
        df["Airport_Lng"] = np.nan
        df["Distance_Start_Airport"] = np.nan

        # Iterate through rows to populate airport coordinates and compute distance
        for i, row in df.iterrows():
            code = row.get("Airport_Code")
            lat, lng = AIRPORT_COORDINATES.get(code, (np.nan, np.nan))
            df.at[i, "Airport_Lat"] = lat
            df.at[i, "Airport_Lng"] = lng

            if (
                pd.notna(lat)
                and pd.notna(lng)
                and pd.notna(row.get("Start_Lat"))
                and pd.notna(row.get("Start_Lng"))
            ):
                df.at[i, "Distance_Start_Airport"] = self.dp.haversine(
                    row["Start_Lat"], row["Start_Lng"], lat, lng
                )

        self.number_of_columns(df)

        return df

    def check_for_street_in_description(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Checks whether the value in the 'Street' column appears in the 'Description' column
        for each row, and creates a new boolean column 'Street_in_Description'.

        Args:
            df (pd.DataFrame): The input DataFrame containing 'Street' and 'Description' columns.

        Returns:
            pd.DataFrame: The DataFrame with a new column 'Street_in_Description' (True/False).
        """
        assert "Street" in df.columns, "'Street' column is missing"
        assert "Description" in df.columns, "'Description' column is missing"
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}][DataPreparation] Checking for 'Street'in 'Description'"
        )

        street_in_description = []
        for i in range(len(df)):
            street = str(df.at[i, "Street"])
            description = str(df.at[i, "Description"])
            street_in_description.append(street in description)

        df["Street_in_Description"] = street_in_description

        return df

    def ensure_column_exists(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Ensures that the specified column exists in the DataFrame. If it does not exist, adds it with None values.

        Args:
            df (pd.DataFrame): The input DataFrame.
            column_name (str): The column to check and add if missing.

        Returns:
            pd.DataFrame: The DataFrame with the ensured column.
        """
        if column_name not in df.columns:
            df[column_name] = None

        return df

    def number_of_columns(self, df: pd.DataFrame) -> int:
        """
        Returns the number of columns in the given DataFrame and logs the count.

        Args:
            df (pd.DataFrame): The DataFrame to inspect.

        Returns:
            int: The number of columns in the DataFrame.
        """
        num_cols = len(df.columns)
        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataEditor] Returning: {num_cols} columns")

    def verify_id_alignment(self, df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
        """
        Verifies that 'ID' columns in both dataframes match exactly row by row.

        Args:
            df1 (pd.DataFrame): First dataframe (usually the master).
            df2 (pd.DataFrame): Second dataframe (with additional columns).

        Returns:
            bool: True if IDs match perfectly, False otherwise. Prints warnings if mismatches exist.
        """
        if "ID" not in df1.columns or "ID" not in df2.columns:
            print(
                "[DataEditor] WARNING: 'ID' column missing in one or both DataFrames. Skipping ID check."
            )
            return True  # Not critical

        if len(df1) != len(df2):
            print(f"[DataEditor] ERROR: Row count mismatch: df1 = {len(df1)}, df2 = {len(df2)}")
            print("[DataEditor] ID check skipped due to unequal lengths.")
            return False

        ids1 = df1["ID"].reset_index(drop=True)
        ids2 = df2["ID"].reset_index(drop=True)

        if ids1.equals(ids2):
            print(f"[{datetime.now().strftime('%H:%M:%S')}][DataEditor] ID alignment confirmed.")
            return True
        else:
            mismatches = (ids1 != ids2).sum()
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}][DataEditor] ID mismatch detected in {mismatches} rows!"
            )
            print("[DataEditor] WARNING: You may be merging on mismatched rows.")
            return False

    def add_columns(
        self, df: pd.DataFrame, df_with_additional_columns: pd.DataFrame, show_warnings=False
    ) -> pd.DataFrame:
        """
        Adds columns from df_with_additional_columns to df (master DataFrame).

        Checks that all columns in df_with_additional_columns match the length of df.

        Args:
            df (pd.DataFrame): Master DataFrame.
            df_with_additional_columns (pd.DataFrame): DataFrame with columns to add.
            show_warnings (bool): If True, print warnings for each column already existing in df.

        Returns:
            pd.DataFrame: Updated master DataFrame (with new columns added if checks pass).
        """
        # Check length
        if len(df) != len(df_with_additional_columns):
            print("[DataEditor] ERROR: Length mismatch!")
            print(f"    Master file rows: {len(df)}")
            print(f"    Columns to add rows: {len(df_with_additional_columns)}")
            print("[DataEditor] WARNING: Columns were NOT added.")
            return df  # return unchanged

        # If check passed, add columns
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}][DataEditor] Adding {len(df_with_additional_columns.columns)} columns to master file."
        )

        if df_with_additional_columns.empty:
            print("[DataEditor] WARNING: No new columns to add. Skipping.")
            return df

        print(
            f"[{datetime.now().strftime('%H:%M:%S')}][DataEditor] Columns to add: {df_with_additional_columns.columns.tolist()}"
        )

        for col in df_with_additional_columns.columns:
            if col in df.columns:
                if show_warnings:
                    print(f"[DataEditor] WARNING: Column '{col}' already exists. Overwriting.")
            df[col] = df_with_additional_columns[col].values

        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataEditor] Columns added successfully.")

        return df

    def change_column_name(
        self, df: pd.DataFrame, current_column_name: str, new_column_name: str
    ) -> pd.DataFrame:
        """
        Rename a column in the DataFrame with validation warnings.

        Parameters:
            df: The DataFrame whose column will be renamed.
            current_column_name: The existing column name to be changed.
            new_column_name: The new name for the column.

        Returns:
            A copy of the DataFrame with the renamed column, or the original DataFrame if validation fails.
        """
        if current_column_name not in df.columns:
            print(
                f"[ColumnRename] WARNING: Column '{current_column_name}' does not exist. Rename aborted."
            )
            return df

        if new_column_name in df.columns:
            print(
                f"[ColumnRename] WARNING: Column '{new_column_name}' already exists. Rename aborted."
            )
            return df

        print(
            f"[{datetime.now().strftime('%H:%M:%S')}][DataEditor] Column '{current_column_name}' has been renamed to '{new_column_name}'."
        )

        return df.rename(columns={current_column_name: new_column_name})

    def merge_columns(self, df: pd.DataFrame, list_of_column_triples: list) -> pd.DataFrame:
        """
        Manually merge two columns for each triple in the list:
        - If 'preferred' has a non-empty, non-null value → use it.
        - Else → use 'fallback'.

        Parameters:
            df (pd.DataFrame): The original DataFrame.
            list_of_column_triples (list): List of [preferred, fallback, merged] column name triples.

        Returns:
            pd.DataFrame: DataFrame with new merged columns added.
        """
        if not list_of_column_triples:
            print("[DataEditor] WARNING: No column triples provided. Nothing to merge.")
            return df

        print(
            f"[{datetime.now().strftime('%H:%M:%S')}][DataEditor] Processing {len(list_of_column_triples)} columns triples for merging"
        )

        for triple in list_of_column_triples:
            if not isinstance(triple, list) or len(triple) != 3:
                print(f"[DataEditor] WARNING: Invalid triple (must be list of 3): {triple}")
                continue

            preferred, fallback, merged = triple

            # Validate existence
            missing = [col for col in [preferred, fallback] if col not in df.columns]
            if missing:
                print(
                    f"[DataEditor] WARNING: Missing columns {missing} in triple: {triple}. Skipping."
                )
                continue

            # Start with fallback copy
            df_result = df[fallback].copy()

            # Manually replace only where preferred is NOT null or empty
            for idx in df.index:
                val = df.at[idx, preferred]
                if pd.notna(val) and str(val).strip() != "":
                    df_result.at[idx] = val

            df[merged] = df_result
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}][DataEditor] Merged '{preferred}' and '{fallback}' into '{merged}'."
            )

        self.number_of_columns(df)

        return df

    def filter_columns(self, df: pd.DataFrame, list_of_columns: list) -> pd.DataFrame:
        """
        Filters specified columns from the DataFrame if they exist, and returns a new DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame from which to filter columns.
            list_of_columns (list): List of column names to select.

        Returns:
            pd.DataFrame: New DataFrame containing only the specified columns that exist in the input.
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataEditor] Starting to filter column(s)")

        existing_columns = [col for col in list_of_columns if col in df.columns]

        for col in list_of_columns:
            if col in df.columns:
                print(f"[DataEditor] Column '{col}' included")
            else:
                print(f"[DataEditor] WARNING: Column '{col}' does not exist. Skipping.")

        if not existing_columns:
            print("[DataEditor] No valid columns found to filter. Returning empty DataFrame.")
            df_result = pd.DataFrame()
        else:
            df_result = df[existing_columns]

        if hasattr(self, "number_of_columns"):  # Guard if method exists
            self.number_of_columns(df_result)

        return df_result

    def delete_columns(self, df: pd.DataFrame, list_of_columns: list) -> pd.DataFrame:
        """
        Deletes specified columns from the DataFrame if they exist.

        Parameters:
            df (pd.DataFrame): The DataFrame from which to delete columns.
            list_of_columns (list): List of column names to delete.

        Returns:
            pd.DataFrame: Updated DataFrame with specified columns removed.
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataEditor] Starting to delete column(s)")
        for col in list_of_columns:
            if col in df.columns:
                df = df.drop(columns=[col])
                print(f"[DataEditor] Column '{col}' deleted.")
            else:
                print(f"[DataEditor] WARNING: Column '{col}' does not exist. Skipping.")

        self.number_of_columns(df)

        return df

    def move_column(self, df: pd.DataFrame, column_name: str, new_position: int) -> pd.DataFrame:
        """
        Moves a specified column to a new position in the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            column_name (str): The name of the column to move.
            new_position (int): The zero-based target position (e.g., 0 for front, -1 for last).

        Returns:
            pd.DataFrame: A new DataFrame with the column moved.

        Raises:
            ValueError: If the column name does not exist in the DataFrame.

        Prints:
            - Confirmation of column movement if successful.
            - Warning if the column is already at the desired position.
        """
        cols = df.columns.tolist()

        if column_name not in cols:
            raise ValueError(
                f"[DataPreparation] Column '{column_name}' does not exist in the DataFrame."
            )

        current_position = cols.index(column_name)

        # Normalize negative index (like -1 for last)
        if new_position < 0:
            new_position = len(cols) + new_position

        if current_position == new_position:
            print(
                f"[DataPreparation] Warning: Column '{column_name}' is already at position {new_position}. No change made."
            )
            return df

        cols.insert(new_position, cols.pop(current_position))
        print(
            f"[DataPreparation] Column '{column_name}' has been moved from position {current_position} to {new_position}"
        )
        return df[cols]

    def ensure_column_exists(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Ensures that the specified column exists in the DataFrame. If it does not exist, adds it with None values.

        Args:
            df (pd.DataFrame): The input DataFrame.
            column_name (str): The column to check and add if missing.

        Returns:
            pd.DataFrame: The DataFrame with the ensured column.
        """
        if column_name not in df.columns:
            df[column_name] = None

        return df

    def number_of_columns(self, df: pd.DataFrame) -> int:
        """
        Returns the number of columns in the given DataFrame and logs the count.

        Args:
            df (pd.DataFrame): The DataFrame to inspect.

        Returns:
            int: The number of columns in the DataFrame.
        """
        num_cols = len(df.columns)
        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataEditor] Returning: {num_cols} columns")
        return df

    def script_1(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This script:
        - applies some minor transformations to the dataframe (master-file)
        - adds various columns to the dataframe (master-file)

        This includes:
        - new_datetime_columns(): parsing and enriching datetime fields
        - add_rush_hour_and_bankholiday_flags(): computes and adds Rush_Hour and Bankholiday

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with all updates.
        """
        # Check uniqueness of key
        df = self.validate_uniqueness_of_key(df, "ID")

        # Retrieve some missing data NOTE: self.dp is calling DataPreparation
        df_missing_weather_data = self.dp.find_missing_data_weather(df)
        df = self.dp.retrieve_best_matches(df, df_missing_weather_data, "Weather")
        df_missing_period_of_day_data = self.dp.find_missing_data_period_of_day(df)
        df = self.dp.retrieve_best_matches(df, df_missing_period_of_day_data, "Day/Night")
        df = self.dp.find_missing_data_period_of_day_from_table(df)

        # df_date_times = self.clean_datetime_columns(df)
        # Clear 'Start_lat'+ 'Start_lng' and End_lat' + 'End_lng' (example: fractions)
        # df_datetimes = self.dp.clean_using_re(df_date_times, ['Start_Time', 'End_Time', 'Weather_Timestamp'], apply_datetime_fixes_using_re, overwrite=True)
        # df_datetimes = self.dp.clean_using_re(df_date_times.copy(), ['Start_Time', 'End_Time', 'Weather_Timestamp'], apply_datetime_fixes_using_re, overwrite=True)

        # df_datetimes = self.dp.clean_using_re(df, ['Start_Time', 'End_Time', 'Weather_Timestamp'], apply_datetime_fixes_using_re, overwrite=True)
        # output_words = f"C:/Users/avtnl/Documents/HU/Bestanden (output code)/CHECK.csv"
        # df.to_csv(output_words, index=False)

        # Add new columns
        df = self.label_source_and_severity(df)
        df = self.add_datetime_columns(df)
        df = self.add_zipcode5(df)
        df = self.add_traffic_info(df)
        df = self.add_period_of_day(df)
        df = self.add_rush_hour_flag(df)
        df = self.add_bankholiday_flag(df)
        df = self.add_weather_condition_flags(df)
        df = self.add_airport_columns(df)

        # Proof 'Distance(mi)'is based on Start (lat + lng) and End (lat + lng) and not total length of traffic jammed!
        df, df_summary = self.dp.compute_distances_and_generate_stats(
            df
        )  # self.dp is calling DataPreparation

        return df, df_summary

    def script_2(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This script:
        - deletes various columns from the dataframe (master-file)
        - moves various columns within the dataframe (master-file)
        - add various columns within df_structures to the dataframe (master-file)

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame reduced for delete column.
        """
        # Delete designated columns
        columns_to_delete = [
            "Sunrise_Sunset",
            "Civil_Twilight",
            "Nautical_Twilight",
            "Astronomical_Twilight",
            "Day_Period_By_Table",
            "Start_Time_Original",
            "End_Time_Original",
            "Weather_Timestamp_Original",
            "Start_Invalid",
            "End_Invalid",
            "Timestamp_Invalid",
            "Distance_Start_End",
            "Distances_Compared",
        ]
        df = self.delete_columns(df, columns_to_delete)

        df = self.move_column(df, "Traffic_Info", 10)
        df = self.move_column(df, "Zipcode5", 17)
        df = self.move_column(df, "Airport_Lat", 21)
        df = self.move_column(df, "Airport_Lng", 22)
        df = self.move_column(df, "Distance_Start_Airport", 23)
        df = self.move_column(df, "Period_of_Day", 47)

        # df_structures = self.fm.read_csv("Structures_Extracted_Details_11Jun2025_1234.csv")
        # df_structures = self.fm.read_csv("Structures_Extracted_Details_15Jun2025_1307.csv")

        return df

    def script_3(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This script:
        - transfers road-details to dataframe (master-file), just after having the columns of df_structures added

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with added columns
        """
        # Split road numbers from connected trailing using Regular Expression
        df = self.dp.clean_using_re(
            df, ["Primary_Location"], apply_split_at_end_number_using_re, overwrite=True
        )

        # output_words = f"C:/Users/avtnl/Documents/HU/Bestanden (output code)/NORMALIZED{DATE_TIME}.csv"
        # df_normilized.to_csv(output_words, index=False)

        df = self.dp.normalize_road_number(df, ["Primary_Location"], overwrite=True)

        df = self.dp.clean_using_re(
            df, ["Primary_Location"], delete_us_space_using_re, overwrite=True
        )

        df = self.dp.add_road_details(
            df, ["Primary_Location_Normalized"], ROAD_NAMES, overwrite=True
        )

        return df

    def script_4a(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This script:
        - changes various colum names within the dataframe (master-file)
        - deletes various columns from the dataframe (master-file)
        - moves various columns within the dataframe (master-file)

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Final DataFrame (end product)
        """
        # Change column names:
        df = self.change_column_name(
            df, "Distance_Start_Airport", "Distance_to_Weather_Station(mi)"
        )
        df = self.change_column_name(
            df, "Duration_Start_to_End(hh:mm:ss)", "Disruption_Period(hh:mm:ss)"
        )
        df = self.change_column_name(
            df, "Duration_Start_to_Timestamp(hh:mm:ss)", "Weather_Delay(hh:mm:ss)"
        )
        df = self.change_column_name(df, "Secondary_Location", "Proximity_Location")
        df = self.change_column_name(df, "Primary_Location_Normalized", "Location")
        df = self.change_column_name(df, "Numbered_Road", "Road_Number_Location")
        df = self.change_column_name(
            df, "Road_Type_Primary_Location_Normalized", "Road_Type_Location"
        )
        df = self.change_column_name(
            df, "Speed_Type_Primary_Location_Normalized", "Speed_Type_Location"
        )

        # Delete columns:
        columns_to_delete = [
            "Zipcode",
            "Start_Date_Only",
            "Start_Time_Only",
            "Primary_Structure",
            "Secondary_Structure",
            "Primary_Location",
        ]

        df = self.delete_columns(df, columns_to_delete)

        # Change column names:
        df = self.change_column_name(df, "Zipcode5", "Zipcode")

        # Move columns
        df = self.move_column(df, "Month", 4)
        df = self.move_column(df, "Year", 5)
        df = self.move_column(df, "Day_of_Week", 6)
        df = self.move_column(df, "Period_of_Day", 7)
        df = self.move_column(df, "Hour_of_Day", 8)
        df = self.move_column(df, "Disruption_Period(hh:mm:ss)", 10)
        df = self.move_column(df, "Rush_Hour", 11)
        df = self.move_column(df, "Bankholiday", 12)
        df = self.move_column(df, "Location", 21)
        df = self.move_column(df, "Road_Number_Location", 22)
        df = self.move_column(df, "Road_Type_Location", 23)
        df = self.move_column(df, "Speed_Type_Location", 24)
        df = self.move_column(df, "Proximity_Location", 25)
        df = self.move_column(df, "Weather_Delay(hh:mm:ss)", 37)
        df = self.move_column(df, "Rainy", 47)
        df = self.move_column(df, "Snowy", 48)
        df = self.move_column(df, "Foggy", 49)
        df = self.move_column(df, "Hazy", 50)
        df = self.move_column(df, "Windy", 51)
        df = self.move_column(df, "Stormy", 52)
        df = self.move_column(df, "Sunny", 53)
        df = self.move_column(df, "Icy", 54)

        return df

    def script_4b(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This script:
        - changes various colum names within the dataframe (master-file)
        - deletes various columns from the dataframe (master-file)
        - moves various columns within the dataframe (master-file)

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Final DataFrame (end product)
        """
        # Change column names:
        df = self.change_column_name(
            df, "Distance_Start_Airport", "Distance_to_Weather_Station(mi)"
        )
        df = self.change_column_name(
            df, "Duration_Start_to_End(hh:mm:ss)", "Disruption_Period(hh:mm:ss)"
        )
        df = self.change_column_name(
            df, "Duration_Start_to_Timestamp(hh:mm:ss)", "Weather_Delay(hh:mm:ss)"
        )
        df = self.change_column_name(df, "Secondary_Location", "Proximity_Location")
        df = self.change_column_name(df, "Primary_Location_Normalized", "Location")
        df = self.change_column_name(df, "Numbered_Road", "Road_Number_Location")
        df = self.change_column_name(
            df, "Road_Type_Primary_Location_Normalized", "Road_Type_Location"
        )
        df = self.change_column_name(
            df, "Speed_Type_Primary_Location_Normalized", "Speed_Type_Location"
        )

        # Delete columns:
        columns_to_delete = ["Zipcode", "Start_Date_Only", "Start_Time_Only", "Primary_Location"]

        df = self.delete_columns(df, columns_to_delete)

        # Change column names:
        df = self.change_column_name(df, "Zipcode5", "Zipcode")

        # Move columns
        df = self.move_column(df, "Month", 4)
        df = self.move_column(df, "Year", 5)
        df = self.move_column(df, "Day_of_Week", 6)
        df = self.move_column(df, "Period_of_Day", 7)
        df = self.move_column(df, "Hour_of_Day", 8)
        df = self.move_column(df, "Disruption_Period(hh:mm:ss)", 10)
        df = self.move_column(df, "Rush_Hour", 11)
        df = self.move_column(df, "Bankholiday", 12)
        df = self.move_column(df, "Location", 21)
        df = self.move_column(df, "Road_Number_Location", 22)
        df = self.move_column(df, "Road_Type_Location", 23)
        df = self.move_column(df, "Speed_Type_Location", 24)
        df = self.move_column(df, "Proximity_Location", 25)
        df = self.move_column(df, "Weather_Delay(hh:mm:ss)", 37)
        df = self.move_column(df, "Rainy", 47)
        df = self.move_column(df, "Snowy", 48)
        df = self.move_column(df, "Foggy", 49)
        df = self.move_column(df, "Hazy", 50)
        df = self.move_column(df, "Windy", 51)
        df = self.move_column(df, "Stormy", 52)
        df = self.move_column(df, "Sunny", 53)
        df = self.move_column(df, "Icy", 54)

        return df

    def script_5(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This script:
        - Copies the original 'Street' column for backup and transformation.
        - Applies regex-based cleaning and normalization to 'Street', similar to 'Primary_Location' (see script_3).
        - Adds road detail metadata (like road type and speed type) to 'Street'.
        - Renames selected columns to standard naming format.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with normalized 'Street' and enriched metadata columns.
        """
        # Column 'Street' is still original and needs to be prepared as 'Primary_Location' already faced
        df = self.check_for_street_in_description(df)

        # Make a copy of the orignal column 'Street', since 'Street' needs some transformation similar to 'Primary_Location'
        # Later on 'Copy_of_Street' will be renamed 'Street'again.
        df["Copy_of_Street"] = df["Street"].copy()

        # Column 'Street' is still original and needs to be prepared as 'Primary_Location' already faced
        df = self.dp.clean_using_re(df, ["Street"], apply_basic_re, overwrite=True)

        # Split road numbers from connected trailing using Regular Expression
        df = self.dp.clean_using_re(
            df, ["Street"], apply_split_at_end_number_using_re, overwrite=True
        )

        # output_words = f"C:/Users/avtnl/Documents/HU/Bestanden (output code)/NORMALIZED{DATE_TIME}.csv"
        # df_normilized.to_csv(output_words, index=False)

        df = self.dp.normalize_road_number(df, ["Street"], overwrite=True)

        df = self.dp.clean_using_re(df, ["Street"], delete_us_space_using_re, overwrite=True)

        df = self.dp.add_road_details(df, ["Street_Normalized"], ROAD_NAMES, overwrite=True)

        df = self.change_column_name(df, "Road_Type_Street_Normalized", "Road_Type_Street")
        df = self.change_column_name(df, "Speed_Type_Street_Normalized", "Speed_Type_Street")

        columns_to_delete = ["Street", "Street_Normalized"]
        df = self.delete_columns(df, columns_to_delete)

        df = self.change_column_name(df, "Copy_of_Street", "Street")
        df = self.change_column_name(df, "Numbered_Road", "Numbered_Road_Street")

        df = self.move_column(df, "Street", 20)
        df = self.move_column(df, "Street_in_Description", 21)
        df = self.move_column(df, "Numbered_Road_Street", 22)
        df = self.move_column(df, "Road_Type_Street", 23)
        df = self.move_column(df, "Speed_Type_Street", 24)

        PlotManager.plot_cities_compared(df)

        return df
