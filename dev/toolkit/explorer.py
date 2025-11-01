import math
from collections import Counter

import pandas as pd
from plot_manager import PlotManager

from constants import MAX_UNIQUE_DISPLAY


class Explorer:
    @staticmethod
    def initial_exploration(df: pd.DataFrame, file_manager, plots_base: str) -> None:
        """
        Performs the initial exploration of a file, which includes:
        summary, uniqueness, statistics, and plotting.

        Args:
            df (pd.DataFrame): The input DataFrame.
            file_manager (FileManager): Handles file saving.
            plots_base (str): Directory path to save plots.
        """
        print("Initial Exploration...")
        df_datatypes, df_uniques, df_stats = Explorer.summarize_file(df)
        file_manager.write_report_csv(df_datatypes, "datatype_report_initial.csv")
        file_manager.write_report_csv(df_uniques, "uniques_report_initial.csv")
        file_manager.write_report_csv(df_stats, "stats_report_initial.csv")
        PlotManager.plot_datatype_summary_stacked_bar(df_datatypes, plots_base)

    @staticmethod
    def final(df: pd.DataFrame, file_manager, plots_base: str) -> None:
        """
        Performs the final exploration of a file: summary and plotting.

        Args:
            df (pd.DataFrame): The input DataFrame.
            file_manager (FileManager): Handles file saving.
            plots_base (str): Directory path to save plots.
        """
        print("Exploring...")
        df_datatypes, _df_uniques, _df_stats = Explorer.summarize_file(df)
        file_manager.write_report_csv(df_datatypes, "datatype_report_initial.csv")
        PlotManager.plot_datatype_summary_stacked_bar(df_datatypes, plots_base)

    @staticmethod
    def summarize_file(df: pd.DataFrame):
        datatype_summary = []
        uniques_summary = []
        stats_summary = []

        # Process each column
        for col in df.columns:
            col_data = df[col]
            type_counter = Counter()
            empty_counter = Counter()
            other_counter = Counter()

            # Analyze each value in column
            for val in col_data:
                if pd.isna(val):
                    other_counter["none"] += 1
                    continue
                elif isinstance(val, bool):
                    type_counter["bool"] += 1
                    if val is True:
                        other_counter["true"] += 1
                    elif val is False:
                        other_counter["false"] += 1
                elif isinstance(val, tuple):
                    type_counter["tuple"] += 1
                    if not val:
                        empty_counter["empty_tuple"] += 1
                elif isinstance(val, list):
                    type_counter["list"] += 1
                    if not val:
                        empty_counter["empty_list"] += 1
                elif isinstance(val, int):
                    type_counter["int"] += 1
                    if val == 0:
                        empty_counter["integer"] += 1
                elif isinstance(val, float):
                    type_counter["float"] += 1
                    if math.isclose(val, 0.0):
                        empty_counter["float"] += 1
                elif isinstance(val, str):
                    type_counter["str"] += 1
                    if val.strip() == "":
                        empty_counter["empty_string"] += 1
                    else:
                        # Check if string can be converted to int/float
                        try:
                            int(val)
                            other_counter["convertible_integer"] += 1
                        except:
                            try:
                                float(val)
                                other_counter["convertible_float"] += 1
                            except:
                                pass
                else:
                    # Catch any other data types
                    type_counter[str(type(val))] += 1

            # Sort types by frequency
            sorted_types = type_counter.most_common()

            # Compute fill percentage and store in datatype summary
            for dtype, count in sorted_types:
                fill_percentage = round(
                    (count - empty_counter.total()) / (count + other_counter["none"]) * 100, 2
                )
                datatype_summary.append(
                    (
                        col,
                        dtype,
                        count,
                        empty_counter.total(),
                        other_counter["none"],
                        other_counter["true"],
                        other_counter["false"],
                        other_counter["convertible_integer"],
                        other_counter["convertible_float"],
                        fill_percentage,
                    )
                )

            # UNIQUE VALUE SUMMARY
            col_valid = col_data.dropna()
            uniques = col_valid.value_counts()

            if len(uniques) > MAX_UNIQUE_DISPLAY:
                uniques_summary.append((col, "Too many occurrences", 999))
            else:
                for value, count in uniques.items():
                    uniques_summary.append((col, str(value), count))

            # NUMERIC STATISTICS (only for numeric columns)
            if sorted_types and sorted_types[0][0] in ["int", "float"]:
                numeric_col = pd.to_numeric(col_valid, errors="coerce").dropna()
                stats_summary.append(
                    (
                        col,
                        round(numeric_col.max(), 2),
                        round(numeric_col.min(), 2),
                        round(numeric_col.mean(), 2),
                        round(numeric_col.median(), 2),
                    )
                )

        # Convert results to DataFrames
        datatype_df = pd.DataFrame(
            datatype_summary,
            columns=[
                "Header",
                "DataType",
                "Count",
                "Count_Empty",
                "Count_None",
                "Count_True",
                "Count_False",
                "Convertible_to_Int",
                "Convertible_to_Float",
                "Fill%",
            ],
        )

        uniques_df = pd.DataFrame(uniques_summary, columns=["Header", "Unique Occurrence", "Count"])

        stats_df = pd.DataFrame(stats_summary, columns=["Header", "Max", "Min", "Mean", "Median"])

        return datatype_df, uniques_df, stats_df
