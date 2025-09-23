import re
from constants import BANKHOLIDAYS_SET, ROAD_NAMES as road_names, INCIDENT_INFO, DAY_TIMES
from constants import WEATHER_COLUMNS, DAY_NIGHT_COLUMNS
from typing import List, Tuple, Dict, Union
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from re_utilities import apply_basic_re, insert_using_re
# from tqdm import tqdm

class DataPreparation:
    """
    DataPreparation is typically used by DataEditor to compute derived values 
    for new columns or to update existing columns in a dataset.

    Designed for use with pandas DataFrames containing time and date information.

    Functions operate in a functional style and do not modify input DataFrames in place.

    Example use cases:
    - Identifying rush hour events in transport or traffic data.
    - Flagging records influenced by national holidays or scheduling patterns.
    """


    @staticmethod
    def compute_rush_hour_flag(df: pd.DataFrame) -> pd.Series:
        """
        Computes a boolean flag indicating whether each record occurs during rush hour.

        Rush hour periods are defined as:
        - Morning: 07:00 to 09:00 (inclusive start, exclusive end)
        - Evening: 16:00 to 19:00 (inclusive start, exclusive end)
        - Only applies on weekdays (Monday to Friday)

        Args:
            df (pd.DataFrame): DataFrame containing:
                - 'Start_Time_Only': datetime.time
                - 'Day_of_Week': string values like 'Monday', 'Tuesday', ..., 'Sunday'

        Returns:
            pd.Series: Boolean Series where True indicates the record is during rush hour.
        """
        # Guard clause: ensure required column exists
        assert 'Start_Time_Only' in df.columns, "'Start_Time_Only' column is missing"
        assert 'Day_of_Week' in df.columns, "'Day_of_Week' column is missing"

        morning_start = time(7, 0, 0)
        morning_end = time(9, 0, 0)
        evening_start = time(16, 0, 0)
        evening_end = time(19, 0, 0)

        is_weekday = ~df['Day_of_Week'].isin(['Saturday', 'Sunday'])

        is_morning_rush = df['Start_Time_Only'].between(morning_start, morning_end, inclusive='left')
        is_evening_rush = df['Start_Time_Only'].between(evening_start, evening_end, inclusive='left')

        rush_hour_flag = is_weekday & (is_morning_rush | is_evening_rush)

        return rush_hour_flag


    @staticmethod
    def compute_bankholiday_flag(df: pd.DataFrame) -> pd.Series:
        """
        Computes a boolean flag indicating whether each record is influenced by a bank holiday.

        Bank holiday influence includes:
        - The day before or after a bank holiday (with time windows).
        - Fridays, Saturdays, and Sundays preceding a Monday bank holiday.
        - Days surrounding a Friday bank holiday.
        - The bank holiday itself.

        Args:
            df (pd.DataFrame): DataFrame containing at least the following columns:
                - 'Start_Date_Only' (datetime.date): Date of the record.
                - 'Start_Time_Only' (datetime.time): Time of the record.
                - 'Day_of_Week' (str): Day of the week as a string (e.g., 'Monday').

        Returns:
            pd.Series: Boolean Series indicating bank holiday influence.
        """
        required_columns = {'Start_Date_Only', 'Start_Time_Only', 'Day_of_Week'}
        missing_columns = required_columns - set(df.columns)
        assert not missing_columns, f"Missing required columns: {missing_columns}"

        time_after_14 = time(14, 0)
        time_before_10 = time(10, 0)

        prev_day = df['Start_Date_Only'] - pd.Timedelta(days=1)
        next_day = df['Start_Date_Only'] + pd.Timedelta(days=1)

        bankholiday_flag = pd.Series(False, index=df.index)

        # Rule 1: Day before bank holiday (after 14:00)
        mask_day_before = next_day.isin(BANKHOLIDAYS_SET) & (df['Start_Time_Only'] >= time_after_14)
        bankholiday_flag |= mask_day_before

        # Rule 2: Day after bank holiday (before 10:00)
        mask_day_after = prev_day.isin(BANKHOLIDAYS_SET) & (df['Start_Time_Only'] <= time_before_10)
        bankholiday_flag |= mask_day_after

        # Rule 3: Friday-Sunday before a Monday bank holiday
        monday_bankholidays = df.loc[
            df['Start_Date_Only'].isin(BANKHOLIDAYS_SET) & (df['Day_of_Week'] == 'Monday'),
            'Start_Date_Only'
        ]
        friday_before = monday_bankholidays - pd.Timedelta(days=3)
        saturday_before = monday_bankholidays - pd.Timedelta(days=2)
        sunday_before = monday_bankholidays - pd.Timedelta(days=1)

        mask_friday = df['Start_Date_Only'].isin(friday_before) & (df['Start_Time_Only'] >= time_after_14)
        mask_saturday = df['Start_Date_Only'].isin(saturday_before)
        mask_sunday = df['Start_Date_Only'].isin(sunday_before)

        bankholiday_flag |= mask_friday
        bankholiday_flag |= mask_saturday
        bankholiday_flag |= mask_sunday

        # Rule 4: Surrounding days of a Friday bank holiday
        friday_bankholidays = df.loc[
            df['Start_Date_Only'].isin(BANKHOLIDAYS_SET) & (df['Day_of_Week'] == 'Friday'),
            'Start_Date_Only'
        ]
        thursday_before = friday_bankholidays - pd.Timedelta(days=1)
        saturday_after = friday_bankholidays + pd.Timedelta(days=1)
        sunday_after = friday_bankholidays + pd.Timedelta(days=2)
        monday_after = friday_bankholidays + pd.Timedelta(days=3)

        mask_thursday = df['Start_Date_Only'].isin(thursday_before) & (df['Start_Time_Only'] >= time_after_14)
        mask_saturday = df['Start_Date_Only'].isin(saturday_after)
        mask_sunday = df['Start_Date_Only'].isin(sunday_after)
        mask_monday = df['Start_Date_Only'].isin(monday_after) & (df['Start_Time_Only'] <= time_before_10)

        bankholiday_flag |= mask_thursday
        bankholiday_flag |= mask_saturday
        bankholiday_flag |= mask_sunday
        bankholiday_flag |= mask_monday

        # Rule 5: On the bank holiday itself
        mask_on_bankholiday = df['Start_Date_Only'].isin(BANKHOLIDAYS_SET)
        bankholiday_flag |= mask_on_bankholiday

        return bankholiday_flag


    def basic_validation(self, df: pd.DataFrame, list_of_columns: list) -> Tuple[List[str], pd.DataFrame]:
        """
        Validates a list of column names against the given DataFrame.

        Logs and reports any missing columns, and prepares an empty result DataFrame
        with the same index for downstream use.

        Args:
            df (pd.DataFrame): The source DataFrame to validate against.
            list_of_columns (list): List of column names to check for existence.

        Returns:
            Tuple[List[str], pd.DataFrame]:
                - A list of valid (existing) column names.
                - An empty DataFrame with the same index as the input DataFrame.
        """        
        df_result = pd.DataFrame(index=df.index)  # preserve index

        missing_columns = [col for col in list_of_columns if col not in df.columns]
        if missing_columns:
            print("[DataPreparation] WARNING: The following columns do NOT exist in master file and will be skipped:")
            for col in missing_columns:
                print(f"    - {col}")

        existing_columns = [col for col in list_of_columns if col in df.columns]
        if not existing_columns:
            print("[DataPreparation] No valid columns to process. Returning empty DataFrame.")

        return existing_columns, df_result


    def extract_added_columns(self, df: pd.DataFrame, original_number_of_columns: int) -> pd.DataFrame:
        """
        Return a DataFrame containing only the columns added after the original column count.
        
        Parameters:
            df: The full DataFrame with new columns.
            original_number_of_columns: The number of columns before additions.
        
        Returns:
            A new DataFrame with only the added columns.
        """
        added_column_count = len(df.columns) - original_number_of_columns
        if added_column_count <= 0:
            return pd.DataFrame(index=df.index)  # return empty with matching index
        else:
            return df.iloc[:, -added_column_count:]


    def clean_using_re(
        self,
        df: pd.DataFrame,
        list_of_columns: list,
        cleaning_function,
        overwrite: bool
    ) -> pd.DataFrame:
        """
        Cleans specified columns in a DataFrame using a cleaning function provided by Regular Expression.

        Args:
            df (pd.DataFrame): Input DataFrame.
            list_of_columns (list): List of column names to apply the cleaning function to.
            cleaning_function (callable): Function that takes a string and returns a cleaned string.
            overwrite (bool): If True, original columns are overwritten and full df is returned.
                            If False, only cleaned columns are added with '_cleaned' suffix.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataPreparation] Starting cleaning using Regular Expression for multiple column")

        original_number_of_columns = len(df.columns)
        existing_columns, df_result = self.basic_validation(df, list_of_columns)

        for col in existing_columns:
            print(f"[{datetime.now().strftime('%H:%M:%S')}][DataPreparation] Processing column: '{col}'")
            cleaned_series = []

            for idx, val in df[col].items():
                cleaned_val = cleaning_function(val)
                cleaned_series.append(cleaned_val)

            if overwrite:
                df[col] = cleaned_series  # Overwrite in original DataFrame
            else:
                df[f'{col}_cleaned'] = cleaned_series  # Add to separate result DataFrame

        if overwrite ==False:
            df_result = self.extract_added_columns(df, original_number_of_columns)
            num_cols = len(df_result.columns)
        else:
            num_cols = len(df.columns)
        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataPreparation] Cleaning complete;  returning: {num_cols} columns")

        return df if overwrite else df_result


    def clean_conform_dictionary( 
        self,
        df: pd.DataFrame,
        list_of_columns: List[str],
        list_of_words: Dict[str, Dict[str, Union[str, List[str]]]],
        action: str,
        overwrite: bool
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[str]]]:
        """
        Cleans and standardizes specified columns using a provided alias dictionary.

        This method replaces or removes word aliases from target columns based on a mapping
        dictionary. Optionally tracking which aliases were matched and how many.

        Args:
            df (pd.DataFrame): Input DataFrame to clean.
            list_of_columns (List[str]): Column names in the DataFrame to clean.
            list_of_words (Dict[str, Dict]): Dictionary with standard words as keys, and
                a dict containing 'aliases' (a list of alternative spellings or variations).
                Example:
                    {
                        "highway": { "aliases": ["hwy", "highwy"] },
                        "road": { "aliases": ["rd", "road"] }
                    }
            action (str): One of:
                - 'replace'            → Replace alias with its standard form.
                - 'remove'             → Remove aliases entirely.
                - 'replace & track'    → Replace aliases and track replacements.
                - 'remove & track'     → Remove aliases and track what was removed.
            overwrite (bool): 
                - If True, modify the input 'df' directly (adds new columns there).
                - If False, returns updated columns only (column names in list_of_columns) and
                            optionally 2 extra columns

        Returns:
            pd.DataFrame: 
                - If 'overwrite=True': returns modified original DataFrame.
                - If 'overwrite=False': returns cleaned copy of the DataFrame.

        Output Columns (added per target column):
            - '{col}_replaced' or '{col}_removed' depending on action
            Optionally (if 'track' in action):
                - '{col}_{action}': space-separated list of matched/replaced words
                - '{col}_count': number of matched words

        Notes:
            - Matching is case-insensitive and whitespace-insensitive.
            - Splitting is currently space-based.
            - This is useful for cleaning street/road types, abbreviations, or known alias sets.
        """
        allowed_actions = {'replace', 'remove', 'replace & track', 'remove & track'}
        if action not in allowed_actions:
            print(f"[DataPreparation] WARNING: Unknown action '{action}'. Supported actions are: {allowed_actions}")

        def build_alias_map(list_of_words: dict) -> dict:
            alias_map = {}
            for key, value in list_of_words.items():
                for alias in value['aliases']:
                    alias_map[alias] = key
            return alias_map

        def replace_or_remove_aliases(text, alias_map, action_type: str, use_regex: bool = False):
            if pd.isna(text):
                return text, ''
            if use_regex:
                words = apply_basic_re(text)
            else:
                words = str(text).lower().split()
            track_matches = []

            if 'replace' in action_type:
                replaced_words = []
                for word in words:
                    if word in alias_map:
                        track_matches.append(alias_map[word])  
                        replaced_words.append(alias_map[word])  # Replacement by key in Dict
                    else:
                        replaced_words.append(word)
                return ' '.join(replaced_words), ' '.join(track_matches)  # Space-based

            elif 'remove' in action_type:
                remaining_words = []
                for word in words:
                    if word in alias_map:
                        track_matches.append(alias_map[word])
                    else:
                        remaining_words.append(word)
                return ' '.join(remaining_words), ' '.join(track_matches)  # Space-based

            return text, ''

        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataPreparation] Starting cleaning using dictionary for multiple columns")
     
        alias_map = build_alias_map(list_of_words)
        existing_columns, df_result = self.basic_validation(df, list_of_columns)

        # Define suffix early, based on action
        if action == 'replace':
            suffix = '_replaced'
        elif action == 'remove':
            suffix = '_removed'
        elif action == 'replace & track':
            suffix = '_replaced_tracked'
        elif action == 'remove & track':
            suffix = '_removed_tracked'

        for col in existing_columns:
            print(f"[{datetime.now().strftime('%H:%M:%S')}][DataPreparation] Processing column: '{col}'")
            cleaned_series = []
            track_series = []
            count_series = []

            for idx, val in df[col].items():
                cleaned_val, matched_val = replace_or_remove_aliases(val, alias_map, action)
                cleaned_series.append(cleaned_val)
                if 'track' in action:
                    track_series.append(matched_val)
                    count = len(matched_val.split()) if matched_val else 0  # Count words matched
                    count_series.append(count)

            if overwrite:
                df[f'{col}{suffix}'] = cleaned_series
                if 'track' in action:
                    df[f'{col}_tracked'] = track_series
                    df[f'{col}_count'] = count_series
                    self.number_of_columns(df)
            else:
                df_result[f'{col}{suffix}'] = cleaned_series
                if 'track' in action:
                    df_result[f'{col}_tracked'] = track_series
                    df_result[f'{col}_count'] = count_series
                    self.number_of_columns(df_result)
                    
        return df if overwrite else df_result


    def fix_and_split_column(
        self,
        df: pd.DataFrame,
        column_name: str,
        list_of_words: Dict[str, Dict[str, Union[str, list]]],
        overwrite: bool
    ) -> pd.DataFrame:
        """
        Splits a text column into two parts based on a dictionary of alias words (e.g., directions).

        This function searches for the first matched alias word (based on `list_of_words`)
        in the given `column_name` and splits the text into:

        - {col}_part1: Includes all words up to and including the first matched alias.
                       Known direction aliases (e.g., 'n', 's', 'nb', etc.) are replaced
                       with their standard form (dictionary key).
        - {col}_part2: Contains all remaining words after the first match.
                       This is left unchanged for potential downstream processing.

        Only rows which matched target_number of alias words in {col}_count will be split.
        Other rows will receive `None` for both parts.

        Args:
            df (pd.DataFrame): The input DataFrame.
            column_name (str): The name of the column to process.
            list_of_words (dict): A mapping of standardized terms to a list of aliases.
                                Example:
                                {
                                    "North": {"aliases": ["n", "north", "nb"]},
                                    "South": {"aliases": ["s", "south", "sb"]},
                                    ...
                                }
            overwrite (bool): If True, adds new columns directly to the original `df`.
                            If False, returns only the new columns as a separate DataFrame.

        Returns:
            pd.DataFrame: 
                - If `overwrite=True`: the modified input DataFrame with 2 new columns.
                - If `overwrite=False`: a new DataFrame with only the added columns.

        Columns Added:
            - `{column_name}_part1`: cleaned direction phrase (up to and including first match)
            - `{column_name}_part2`: remaining text after first match

        Notes:
            - Matching is case-insensitive.
            - Replacements only apply to predefined direction_aliases
            - Matching logic uses `column_name_count` to limit split operations to relevant rows.
        """

        def build_alias_map(list_of_words: dict) -> dict:
            alias_map = {}
            for key, value in list_of_words.items():
                for alias in value['aliases']:
                    alias_map[alias.lower()] = key
            return alias_map

        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataPreparation] Start splitting string in column '{column_name}' on indicated position")

        direction_aliases = {'n', 'e', 's', 'w', 'ne', 'nw', 'se', 'sw', 'nb', 'sb', 'eb', 'wb'}
        target_number = 2  # Split will only apply to rows where {column_name}_count == target_number

        original_number_of_columns = len(df.columns)
        alias_map = build_alias_map(list_of_words)
        #existing_columns, df_result = self.basic_validation(df, column)
        #print(f'Processing for original column {existing_columns}')

        part1_col = f"{column_name}_part1"
        part2_col = f"{column_name}_part2"
        count_col = f"{column_name}_Count"
        if count_col not in df.columns:
            raise KeyError(f"Expected column '{count_col}' not found in DataFrame. Did you run preseding scripts?")

        part1_series = []
        part2_series = []

        for text, count in zip(df[column_name], df[count_col]):
            if pd.isna(text) or count != target_number:
                part1_series.append(None)
                part2_series.append(None)
                continue

            words = str(text).lower().split()
            split_index = -1

            # Find the first match (any alias)
            for i, word in enumerate(words):
                if word in alias_map:
                    split_index = i
                    break

            if split_index == -1:
                part1_series.append(text)
                part2_series.append('')
                continue

            # Build part1: replace only if word is in direction_aliases
            part1_words = []
            for i in range(split_index + 1):
                word = words[i]
                if word in direction_aliases:
                    part1_words.append(alias_map[word])
                else:
                    part1_words.append(word)

            part2_words = words[split_index + 1:]

            part1 = ' '.join(part1_words).strip() + ' ' if part1_words else ''
            part2 = ' '.join(part2_words).strip() if part2_words else ''

            part1_series.append(part1)
            part2_series.append(part2)

        df[part1_col] = pd.Series(part1_series, index=df.index).str.lower()
        df[part2_col] = part2_series
        self.number_of_columns(df)

        if overwrite ==False:
            df_result = self.extract_added_columns(df, original_number_of_columns)
            self.number_of_columns(df_result)

        return df if overwrite else df_result


    def merge_parts_of_column(
        self,
        df: pd.DataFrame,
        column_name: str,
        suffix: str,
        overwrite: bool
    ) -> pd.DataFrame:
        """
        Merges two split parts of a column into a single cleaned and combined text column.

        This method assumes the input DataFrame contains:
            - A direction-cleaned prefix column: `{column_name}_part1`
            - A suffix-cleaned column (typically preprocessed separately): `{column_name}_part2{suffix}`

        It joins both parts into a final cleaned version:
            - Trims whitespace
            - Handles None/NaN values
            - Converts result to lowercase

        Args:
            df (pd.DataFrame): The input DataFrame.
            column_name (str): The base name of the original column (e.g., "Street").
            suffix (str, optional): Suffix for the cleaned part2 column.
                                    
            overwrite (bool): 
                - If True: modifies the original DataFrame directly by adding the new column.
                - If False: returns a DataFrame containing only the newly added column.

        Returns:
            pd.DataFrame including {column_name}_combined}
                - If 'overwrite=True': returns the modified input DataFrame.
                - If 'overwrite=False`: returns a new DataFrame with only the added column.

        Raises:
            ValueError: If either '{column_name}_part1' or '{column_name}_part2{suffix}' is missing.

        Output Column:
            - '{column_name}_combined': lowercase string combining part1 and part2

        Notes:
            - The output column helps reassemble the text after applying direction parsing
              and alias cleaning in prior steps.
            - This function is typically used after fix_and_split_column() and
              'clean_conform_dictionary()' on part2.
        """        
        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataPreparation] Starting merge of relevant columns")
        
        original_number_of_columns = len(df.columns)

        part1_col = f'{column_name}_part1'
        cleaned_part2_col = f'{column_name}_part2_{suffix}'
        final_combined_col = f'{column_name}_combined'

        if part1_col not in df.columns:
            raise ValueError(f"Column '{part1_col}' not found in DataFrame.")
        if cleaned_part2_col not in df.columns:
            raise ValueError(f"Column '{cleaned_part2_col}' not found in DataFrame.")

        combined_series = []

        for part1, part2 in zip(df[part1_col], df[cleaned_part2_col]):
            # Handle None or NaN
            if pd.isna(part1) and pd.isna(part2):
                combined_series.append(None)
                continue
            if pd.isna(part1):
                combined_series.append(str(part2).strip())
                continue
            if pd.isna(part2):
                combined_series.append(str(part1).strip())
                continue

            part1_str = str(part1).strip()
            part2_str = str(part2).strip()

            if part1_str and part2_str:
                combined_series.append(f'{part1_str} {part2_str}')
            elif part1_str:
                combined_series.append(part1_str)
            else:
                combined_series.append(part2_str)

        df[final_combined_col] = pd.Series(combined_series, index=df.index).str.lower()       
        self.number_of_columns(df)

        if overwrite == False:
            df_result = self.extract_added_columns(df, original_number_of_columns)
            self.number_of_columns(df_result)

        return df if overwrite else df_result


    def normalize_road_number(self, df: pd.DataFrame, list_of_columns: list, overwrite: bool) -> pd.DataFrame:

        def format_road_number(text: str):
            if pd.isna(text):
                return text, None

            text = str(text).strip().lower()
            words = text.split()
            numbered_road = None  # will hold first matched road

            def scan_word(word: str):
                word = word.lower()
                if word.startswith('us') and word[2:].isdigit():
                    return 'us', word[2:]
                if word.startswith('tx') and word[2:].isdigit():
                    return 'tx', word[2:]
                if word.startswith('fw') and word[2:].isdigit():
                    return 'fw', word[2:]
                if word.startswith('fm') and word[2:].isdigit():
                    return 'fm', word[2:]
                if word.startswith('i') and word[1:].isdigit():
                    return 'i', word[1:]
                if word.startswith('u') and word[1:].isdigit():
                    return 'u', word[1:]
                return None

            def format_hit(prefix: str, number: str) -> str:
                prefix_map = {
                    'fm': 'FM-',
                    'i': 'I-',
                    'ih': 'IH-',
                    'rm': 'RM-',
                    'rr': 'RR-',
                    'tx': 'TX-',
                    'u': 'US-',
                    'us': 'US-'
                }
                return prefix_map.get(prefix, '') + number

            def format_hwy(double: list) -> str:
                if double[0] == 'highway':
                    return 'US-' + double[1]
                elif double[0] == 'interstate':
                    return 'I-' + double[1]
                else:
                    return double[0] + ' ' + double[1]

            # First pass: normalize patterns like 'fm1709', 'i35', etc.
            for i, word in enumerate(words):
                hit = scan_word(word)
                if hit:
                    prefix, number = hit
                    formatted = format_hit(prefix, number)
                    words[i] = formatted
                    if numbered_road is None:
                        numbered_road = formatted  # record first matched road

            # Second pass: detect patterns like 'highway 281'
            i = 0
            while i < len(words) - 1:
                word1, word2 = words[i], words[i + 1]
                if word1 in ['highway', 'interstate'] and word2.isdigit():
                    formatted = format_hwy([word1, word2])
                    words[i] = formatted
                    del words[i + 1]
                    if numbered_road is None:
                        numbered_road = formatted  # record first match
                else:
                    i += 1

            return ' '.join(words), numbered_road

        # Main of normalization logic
        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataPreparation] Normalizing road numbers for multiple columns")

        original_number_of_columns = len(df.columns)

        # df_result = df.copy()
        numbered_road_col = []  # new column to hold extracted numbered road per row

        missing_columns = [col for col in list_of_columns if col not in df.columns]
        if missing_columns:
            print("[DataPreparation] WARNING: The following columns do NOT exist in master file and will be skipped:")
            for col in missing_columns:
                print(f"    - {col}")

        existing_columns = [col for col in list_of_columns if col in df.columns]
        if not existing_columns:
            print("[DataPreparation] WARNING: No valid columns to process. Returning original DataFrame.")
            return df  # Was: df_result

        for col in existing_columns:
            print(f"[{datetime.now().strftime('%H:%M:%S')}][DataPreparation] Processing column: '{col}'")
            cleaned_values = []
            temp_numbered_roads = []

            for idx, val in df[col].items():
                cleaned_val, numbered_road = format_road_number(val)
                cleaned_values.append(cleaned_val)
                temp_numbered_roads.append(numbered_road)

            df[f'{col}_Normalized'] = cleaned_values  # Was: df_result

            # If multiple columns, track first non-null Numbered_Road
            if not numbered_road_col:
                numbered_road_col = temp_numbered_roads
            else:
                # Merge keeping first non-null
                numbered_road_col = [
                    existing if existing is not None else new
                    for existing, new in zip(numbered_road_col, temp_numbered_roads)
                ]

        df['Numbered_Road'] = numbered_road_col  # Was: df_result
        self.number_of_columns(df)

        if overwrite == False:
            df_result = self.extract_added_columns(df, original_number_of_columns)
            self.number_of_columns(df_result)

        # return df  # Was: df_result
        return df if overwrite else df_result


    def add_road_details(
        self,
        df: pd.DataFrame, 
        existing_cols: List[str], 
        road_dict: Dict[str, Dict[str, Union[str, List[str]]]],
        overwrite: bool
    ) -> pd.DataFrame:
        """
        Adds road classification details to specified text columns based on a road dictionary.

        For each column in `existing_cols`, the function checks for any matching alias from
        `road_dict` and assigns the corresponding 'road_type' and 'speed_type' values.

        Adds two new columns per input column:
            - 'Road_Type_{col}'
            - 'Speed_Type_{col}'

        Parameters:
            df (pd.DataFrame): Input DataFrame.
            existing_cols (List[str]): Columns to analyze.
            road_dict (Dict): Dictionary with aliases and corresponding road/speed types.

        Returns:
            pd.DataFrame: DataFrame with added road type and speed type columns.
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataPreparation] Adding road details to one or more columns")

        original_number_of_columns = len(df.columns)

        for col in existing_cols:
            print(f"[add_road_details] Processing column: '{col}'")

            road_type_col = []
            speed_type_col = []

            for val in df[col].fillna("").astype(str):
                road_type = None
                speed_type = None

                for key, props in road_dict.items():
                    aliases = props.get('aliases', [])
                    if any(alias.lower() in val.lower() for alias in aliases):
                        road_type = props.get('road_type')
                        speed_type = props.get('speed_type')
                        break  # take the first matching category

                road_type_col.append(road_type)
                speed_type_col.append(speed_type)

            df[f'Road_Type_{col}'] = road_type_col
            df[f'Speed_Type_{col}'] = speed_type_col
            self.number_of_columns(df)

        if overwrite == False:
            df_result = self.extract_added_columns(df, original_number_of_columns)
            self.number_of_columns(df_result)

        return df if overwrite else df_result


    def haversine(self, lat1, lon1, lat2, lon2):
        """
        Compute Haversine distance between two points (lat1, lon1) and (lat2, lon2) in miles.
        """
        from math import radians, sin, cos, sqrt, atan2

        R = 3958.8  # Earth radius in miles
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c

        return round(distance, 2)
    

    def compute_distances_and_generate_stats(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Computes:
        - 'Distance_Start_End' using haversine
        - 'Distances_Compared' = abs('Distance(mi)' - 'Distance_Start_End')
        
        Then summarizes:
        - Count None / Zero / Other, Min, Max, Mean, Median
        for: Start_Lat, End_Lat, Distance_Start_End, Distance(mi), Distances_Compared

        Returns:
            Tuple:
                - df_result: Updated DataFrame with computed columns
                - df_summary: Summary table with math results
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataPreparation] Starting examining distances")

        df['Distance_Start_End'] = np.nan
        df['Distances_Compared'] = None

        for i, row in df.iterrows():
            if pd.notna(row['Start_Lat']) and pd.notna(row['Start_Lng']) and \
               pd.notna(row['End_Lat']) and pd.notna(row['End_Lng']):
                distance = self.haversine(
                    row['Start_Lat'], row['Start_Lng'],
                    row['End_Lat'], row['End_Lng']
                )
                df.at[i, 'Distance_Start_End'] = distance
            else:
                distance = None

            reported = row.get('Distance(mi)', None)
            if pd.notna(reported) and distance is not None:
                df.at[i, 'Distances_Compared'] = abs(reported - distance)
            else:
                df.at[i, 'Distances_Compared'] = None

        # Generate content for report (summary)
        df_stats = {
            'Math': ['Count None', 'Count Zero', 'Count Other', 'Min', 'Max', 'Mean', 'Median'],
            'Start_Lat': [],
            'End_Lat': [],
            'Distance_Start_End': [],
            'Distance(mi)': [],
            'Distances_Compared': [],
        }

        def compute_stats(series):
            count_none = series.isna().sum()
            count_zero = (series == 0).sum()
            count_other = series.notna().sum() - count_zero
            min_val = round(series.min(skipna=True), 1)
            max_val = round(series.max(skipna=True), 1)
            mean_val = round(series.mean(skipna=True), 1)
            median_val = round(series.median(skipna=True), 1)
            return [count_none, count_zero, count_other, min_val, max_val, mean_val, median_val]

        for col in df_stats.keys():
            if col != 'Math':
                df_stats[col] = compute_stats(df[col])

        df_summary = pd.DataFrame(df_stats)

        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataPreparation] Examining distances now complete")

        self.number_of_columns(df)

        return df, df_summary


    def find_missing_data_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Finds missing Weather_Timestamp rows and attempts to fill metadata from nearby (example 24 hours) rows
        with valid Weather_Timestamp using ID, City, or County matching.

        Adds:
        - Duration_Start_to_Timestamp (HH:MM:SS)
        - Matching_Info (City, County, or TX fallback)
        - Matching_ID (ID of matched row)

        Returns:
            pd.DataFrame: Only the matched/missing rows with added metadata
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataPreparation] Starting search for missing 'Weather' data")    

        TIME_WINDOW_IN_HOURS = 24

        # Ensure Weather_Timestamp and Start_Time are in datetime format
        df['Weather_Timestamp'] = pd.to_datetime(df['Weather_Timestamp'], errors='coerce')
        df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')

        # Split into missing and valid
        df_missing = df[df['Weather_Timestamp'].isna()].copy()
        df_valid = df[df['Weather_Timestamp'].notna()].copy()

        if df_missing.empty:
            print("[Info] No missing Weather_Timestamp values found.")
            return df_missing

        # Pre-sort valid records
        df_valid = df_valid.sort_values('Weather_Timestamp')

        # Prepare columns to be added
        df_missing['Duration_Start_to_Timestamp'] = None
        df_missing['Matching_Info'] = None
        df_missing['Matching_ID'] = None

        #Example tqdm (not used)
        #for idx, row in tqdm(df_missing.iterrows(), total=len(df_missing), desc="Processing missing rows"):

        for idx, row in df_missing.iterrows():
            start_time_row = pd.to_datetime(row['Start_Time'], errors='coerce')
            if pd.isna(start_time_row):
                continue

            # Check Time_Delta
            time_window_start = start_time_row - timedelta(hours=TIME_WINDOW_IN_HOURS)
            time_window_end = start_time_row + timedelta(hours=TIME_WINDOW_IN_HOURS)

            df_window = df_valid[
                (df_valid['Weather_Timestamp'] >= time_window_start) &
                (df_valid['Weather_Timestamp'] <= time_window_end)
            ]

            if df_window.empty:
                continue

            df_window = df_window.copy()
            df_window['Time_Diff'] = (df_window['Weather_Timestamp'] - start_time_row).abs()

            best_match = None
            match_level = None

            # Try by City
            df_city = df_window[df_window['City'] == row['City']]
            if not df_city.empty:
                best_match = df_city.loc[df_city['Time_Diff'].idxmin()]
                match_level = 'City'
            else:
                # Try by County
                df_county = df_window[df_window['County'] == row['County']]
                if not df_county.empty:
                    best_match = df_county.loc[df_county['Time_Diff'].idxmin()]
                    match_level = 'County'
                else:
                    best_match = df_window.loc[df_window['Time_Diff'].idxmin()]
                    match_level = 'TX'

            if best_match is not None:
                # Format time difference
                td = timedelta(seconds=int(best_match['Time_Diff'].total_seconds()))
                df_missing.at[idx, 'Duration_Start_to_Timestamp'] = str(td)
                df_missing.at[idx, 'Matching_Info'] = match_level
                df_missing.at[idx, 'Matching_ID'] = best_match['ID']

        self.number_of_columns(df_missing)


        return df_missing
    

    def find_missing_data_period_of_day(self, df: pd.DataFrame) -> pd.DataFrame:
        from datetime import datetime, timedelta

        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataPreparation] Starting search for missing 'Period_of_Day' data")

        PRIMARY_WINDOW_MIN = 2
        TARGET_COLUMN = 'Sunrise_Sunset'
        MATCH_COLUMNS = ['City', 'County']

        df_missing = df[df[TARGET_COLUMN].isna()].copy()
        df_valid = df[df[TARGET_COLUMN].notna()].copy()

        if df_missing.empty:
            print(f"[Info] No missing '{TARGET_COLUMN}' values found.")
            return df_missing

        # === Preprocessing ===
        df_valid['Start_Timestamp'] = df_valid['Start_Time'].values.astype('datetime64[s]').astype('int64')
        df_missing['Start_Timestamp'] = df_missing['Start_Time'].values.astype('datetime64[s]').astype('int64')
        window_sec = PRIMARY_WINDOW_MIN * 60

        # === Build result columns
        df_missing['Duration_Start_to_Timestamp'] = None
        df_missing['Matching_ID'] = None
        df_missing['Matching_Info'] = None

        # === Group valid rows by City + County for efficient lookup
        valid_groups = df_valid.groupby(MATCH_COLUMNS)

        for idx, row in df_missing.iterrows():
            timestamp = row['Start_Timestamp']
            window_start = timestamp - window_sec
            window_end = timestamp + window_sec

            subset = None
            match_level = None

            # Try matching within City/County group
            for level in MATCH_COLUMNS:
                group_key = row[level]
                if level == 'City':
                    key = (group_key, slice(None))  # ('City', anything)
                else:  # 'County'
                    key = (slice(None), group_key)  # (anything, 'County')

                try:
                    df_group = valid_groups.get_group(key)
                except KeyError:
                    continue

                nearby = df_group[
                    (df_group['Start_Timestamp'] >= window_start) &
                    (df_group['Start_Timestamp'] <= window_end)
                ]

                if not nearby.empty:
                    match_level = level
                    subset = nearby
                    break

            # Fallback: try full df_valid
            if subset is None:
                subset = df_valid[
                    (df_valid['Start_Timestamp'] >= window_start) &
                    (df_valid['Start_Timestamp'] <= window_end)
                ]
                match_level = 'Fallback'

            if not subset.empty:
                # Find row with smallest time difference
                subset = subset.copy()  # Ensure it's a true copy, not a view
                subset.loc[:, 'Time_Diff'] = (subset['Start_Timestamp'] - timestamp).abs()
                best_match = subset.loc[subset['Time_Diff'].idxmin()]
                start_time_valid = best_match['Start_Time']
                row_time_valid = row['Start_Time']
                if pd.notna(start_time_valid) and pd.notna(row_time_valid):
                    delta = abs(start_time_valid - row_time_valid)
                    formatted = str(delta).split('.')[0]
                    df_missing.at[idx, 'Duration_Start_to_Timestamp'] = formatted
                    df_missing.at[idx, 'Matching_ID'] = best_match['ID']
                    df_missing.at[idx, 'Matching_Info'] = match_level

                formatted = str(delta).split('.')[0]  # Format HH:MM:SS
                df_missing.at[idx, 'Duration_Start_to_Timestamp'] = formatted
                df_missing.at[idx, 'Matching_ID'] = best_match['ID']
                df_missing.at[idx, 'Matching_Info'] = match_level

        # [Second loop could be rewritten similarly — let me know if you want it optimized too]

        self.number_of_columns(df_missing)

        return df_missing


    def update_modification_notes(self, existing: str, source: str) -> str:
        """
        Appends a new source tag to an existing 'Modification_Notes' string.
        """
        if pd.notna(existing) and existing:
            return f"{existing}; {source}"
        return source


    def find_missing_data_period_of_day_from_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns 'Day', 'Night', or similar label to rows where 'Sunrise_Sunset' is missing,
        using a sunrise/sunset lookup table. Also updates 'Modification_Notes' if filled.

        Adds:
            - Day_Period_By_Table: Only for rows with missing 'Sunrise_Sunset'
            - Updates 'Modification_Notes' with "From table" if filled.

        Returns:
            pd.DataFrame: Updated DataFrame.
        """

        def classify_day_period_with_info(row):
            start_time = row['Start_Time']
            if pd.isna(start_time):
                return 'Night', False  # Fallback for missing time

            month = start_time.strftime('%B')
            day_key = '15' if start_time.day > 7 else '1'

            times = DAY_TIMES.get(month, {}).get(day_key)
            if not times:
                return 'Night', False

            try:
                sunrise = datetime.strptime(times[0], "%I:%M %p").time()
                sunset = datetime.strptime(times[1], "%I:%M %p").time()
            except Exception:
                return 'Night', False

            current_time = start_time.time()
            buffer = timedelta(minutes=30)
            ref_date = datetime(2000, 1, 1)

            t = datetime.combine(ref_date, current_time)
            sr = datetime.combine(ref_date, sunrise)
            ss = datetime.combine(ref_date, sunset)

            if abs(t - sr) <= buffer:
                return 'Sunrise', True
            elif abs(t - ss) <= buffer:
                return 'Sunset', True
            elif sr < t < ss:
                return 'Day', True
            else:
                return 'Night', True

        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataPreparation] Starting sunrise/sunset table lookup")

        df = self.ensure_column_exists(df, 'Modification_Notes')

        results = []
        for i, row in df.iterrows():
            if pd.isna(row['Sunrise_Sunset']):
                label, used_table = classify_day_period_with_info(row)
                results.append(label)
                if used_table:
                    updated_info = self.update_modification_notes(row['Modification_Notes'], "From table")
                    df.at[i, 'Modification_Notes'] = updated_info
            else:
                results.append(None)  # Optional: keep this line for column alignment

        df['Day_Period_By_Table'] = results  # Will be None for all prefilled rows

        self.number_of_columns(df)

        return df


    def update_modification_notes(self, existing: str, source: str) -> str:
        """
        Appends a new source tag to an existing 'Modification_Notes' string.
        Handles cases where 'existing' is NaN, None, or not a string.
        """
        if not isinstance(existing, str) or pd.isna(existing):
            return source
        return f"{existing}; {source}"


    def retrieve_best_matches(self, df: pd.DataFrame, df_ids: pd.DataFrame, data_category: str) -> pd.DataFrame:
            """
            Vectorized: Fills missing fields in df based on best-matched IDs in df_ids for the given data_category.

            Args:
                df (pd.DataFrame): Full DataFrame containing all data and the ID-based matches.
                df_ids (pd.DataFrame): Subset of df with rows needing updates, and matching ID columns.
                data_category (str): 'Weather' or 'Day/Night'.

            Returns:
                pd.DataFrame: The full df with updated fields based on matching info.
            """
            from datetime import datetime

            print(f"[{datetime.now().strftime('%H:%M:%S')}][DataPreparation] Starting data retrieval for category: {data_category}")

            # Validate category
            valid_categories = {
                'Weather': {
                    'columns': WEATHER_COLUMNS,
                    'id_column': 'Matching_ID'
                },
                'Day/Night': {
                    'columns': DAY_NIGHT_COLUMNS,
                    'id_column': 'Matching_ID'
                }
            }

            if data_category not in valid_categories:
                raise ValueError(f"[ERROR] Invalid data_category '{data_category}'. Must be 'Weather' or 'Day/Night'.")

            config = valid_categories[data_category]
            cols_to_copy = config['columns']
            match_id_col = config['id_column']

            # === Build lookup DataFrame ===
            lookup_df = df[['ID'] + cols_to_copy].set_index('ID')

            # === Perform join on df_ids using the match ID ===
            df_ids_copy = df_ids.copy()
            matched_data = df_ids_copy[[match_id_col]].join(lookup_df, on=match_id_col)

            # === Update original df (in place) ===
            for col in cols_to_copy:
                df.loc[df_ids_copy.index, col] = matched_data[col]

            df = self.ensure_column_exists(df, 'Modification_Notes')

            # Loop through matching rows and apply update using the helper
            for idx in df_ids_copy.index:
                existing_note = df.at[idx, "Modification_Notes"]
                new_note = f"{data_category}+{df_ids_copy.at[idx, match_id_col]}"
                df.at[idx, "Modification_Notes"] = self.update_modification_notes(existing_note, new_note)

            print(f"[{datetime.now().strftime('%H:%M:%S')}][DataPreparation] Retrieval complete for category: {data_category}")

            self.number_of_columns(df)

            return df


    def summarize_period_of_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns 'Period_of_Day' based on combinations of:
        - 'Sunrise_Sunset'
        - 'Civil_Twilight'
        - 'Nautical_Twilight'

        If a row cannot be classified, and 'Day_Period_By_Table' is available,
        use it as a fallback for 'Period_of_Day'.
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataPreparation] Generating summarized 'Period_of_Day'")

        # Normalize twilight fields
        for col in ['Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight']:
            df[col] = df[col].astype(str)

        # Rule-based mapping
        conditions = {
            ('Night', 'Day', 'Day'): 'Day',
            ('Day', 'Day', 'Day'): 'Day',
            ('Day', 'Day', 'Night'): 'Twilight (AM)',
            ('Night', 'Night', 'Day'): 'Twilight (PM)',
            ('Night', 'Night', 'Night'): 'Night',
            ('Day', 'Night', 'Day'): 'Not possible',
            ('Day', 'Night', 'Night'): 'Sunrise',
            ('Night', 'Day', 'Day'): 'Sunset',
            ('Night', 'Day', 'Night'): 'Not possible'
        }

        def classify_period(row):
            key = (row['Sunrise_Sunset'], row['Civil_Twilight'], row['Nautical_Twilight'])
            return conditions.get(key, 'Unknown')

        df['Period_of_Day'] = df.apply(classify_period, axis=1)

        # Fallback: if 'Period_of_Day' is 'Unknown' and 'Day_Period_By_Table' is not null, use it
        fallback_mask = (df['Period_of_Day'] == 'Unknown') & (df['Day_Period_By_Table'].notna())
        df.loc[fallback_mask, 'Period_of_Day'] = df.loc[fallback_mask, 'Day_Period_By_Table']

        self.number_of_columns(df)

        return df


    def complete_period_of_day_from_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills empty or None values in 'Period_of_Day' column using values from 'Day_Period_By_Table'.
        Also updates 'Modification_Notes' with 'Period_of_day from table' for affected rows.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The updated DataFrame with filled values and modification notes.
        """
        # Ensure required columns exist
        df = self.ensure_column_exists(df, 'Period_of_Day')
        df = self.ensure_column_exists(df, 'Modification_Notes')

        # Identify rows where 'Period_of_Day' is empty or None
        mask_missing = df['Period_of_Day'].isna() | (df['Period_of_Day'] == '')

        # Fill 'Period_of_Day' with 'Day_Period_By_Table' values for those rows
        df.loc[mask_missing, 'Period_of_Day'] = df.loc[mask_missing, 'Day_Period_By_Table']

        # Update modification notes
        for idx in df[mask_missing].index:
            existing_note = df.at[idx, 'Modification_Notes']
            df.at[idx, 'Modification_Notes'] = self.update_modification_notes(existing_note, 'Period_of_Day from table')

        return df


    def change_weather_condition(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Updates the 'Weather_Condition' column based on the logic:
        - If Rainy == True and Precipitation(in) == 0,  then use insert_using_re() to add ' in the Vicinity'
        - If Rainy == False and Precipitation(in) > 0, then append '(rain at {Airport_Code})'

        Also updates 'Modification_Notes' using helper methods.
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}][DataPreparation] Updating 'Weather_Condition'")

        # Ensure required columns exist
        df = self.ensure_column_exists(df, 'Modification_Notes')
        df = self.ensure_column_exists(df, 'Weather_Condition')

        # Case 1: Rainy == True and Precipitation == 0
        mask_vicinity = (df['Rainy'] == True) & (df['Precipitation(in)'] == 0)
        df.loc[mask_vicinity, 'Weather_Condition'] = df.loc[mask_vicinity, 'Weather_Condition'].apply(insert_using_re)

        for idx in df[mask_vicinity].index:
            existing_note = df.at[idx, 'Modification_Notes']
            df.at[idx, 'Modification_Notes'] = self.update_modification_notes(existing_note, 'Added "vicinity" to WC')

        # Case 2: Rainy == False and Precipitation > 0
        mask_airport_rain = (df['Rainy'] == False) & (df['Precipitation(in)'] > 0)

        def append_airport_comment(text: str, code: str) -> str:
            if isinstance(text, str):
                return f"{text} (rain at {code})"
            return text

        for idx in df[mask_airport_rain].index:
            wc = df.at[idx, 'Weather_Condition']
            airport = df.at[idx, 'Airport_Code']
            df.at[idx, 'Weather_Condition'] = append_airport_comment(wc, airport)

            existing_note = df.at[idx, 'Modification_Notes']
            df.at[idx, 'Modification_Notes'] = self.update_modification_notes(existing_note, 'Added "rain at airport" to WC')

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
        return df










