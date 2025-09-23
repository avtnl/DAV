"""
This script processes US accident data for Texas.
It provides a modular pipeline for data exploration, transformation, reporting, and visualization.

Main capabilities:
- Data Exploration (profiling, summaries, quality checks)
- Data Preparation (feature engineering, datetime processing, duration calculations)
- Data Transformation (adding columns, applying business rules, enriching fields)
- File Management (reading, writing, organizing outputs and reports)
- Plot Management (automated visualization generation)
"""

from constants import INPUT_BASE, OUTPUT_BASE, PLOTS_BASE, REPORTS_BASE
from file_manager import FileManager
from explorer import Explorer
from data_preparation import DataPreparation
from data_editor import DataEditor
from plot_manager import PlotManager

from datetime import datetime
import time as time_module

# Start time (track total script run time)
start_time_run = time_module.time()

# Instantiate FileManager
file_manager = FileManager(INPUT_BASE, OUTPUT_BASE, REPORTS_BASE)
data_preparation = DataPreparation()
editor = DataEditor(file_manager, data_preparation)  # When instantiating the editor, we tell thje editor which instance of DataPreparation to use

# Scripts
scripts=['script_1']
if 'script_1' in scripts:
    print("[DataPreparation] Starting script_1 ...")
    df = file_manager.read_csv("US_Accidents_TX_subset.csv")  # For script_1
    print("Reading input CSV...")
    Explorer.initial_exploration(df, file_manager, PLOTS_BASE)
    df, df_summary = editor.script_1(df)
    file_manager.write_report_csv(df_summary, "distance_report.csv")
    Explorer.final(df, file_manager, PLOTS_BASE)
    PlotManager.plot_cities_basic(df)
if 'script_2' in scripts:
    # Note: Define df_structure in Script_2!!!
    print("[DataPreparation] Starting script_2 ...")
    df = file_manager.read_csv("US_Accidents_TX_subset_amended_17Jun2025_0945.csv")  # For script_2 | Please change to correct file_name
    print("Reading input CSV...")
    Explorer.final(df, file_manager, PLOTS_BASE)
    df = editor.script_2(df)
    df_structures = file_manager.read_csv("Structures_Extracted_Details_17Jun2025_0947.csv") # TX-subset
    editor.verify_id_alignment(df, df_structures)
    df = editor.add_columns(df, df_structures)
    Explorer.final(df, file_manager, PLOTS_BASE)
if 'script_3' in scripts:
    print("[DataPreparation] Starting script_3 ...")
    df = file_manager.read_csv("US_Accidents_TX_subset_amended_17Jun2025_0949.csv")  # For script_3 | Please change to correct file_name
    print("Reading input CSV...")
    Explorer.initial_exploration(df, file_manager, PLOTS_BASE)
    df = editor.script_3(df)
    Explorer.final(df, file_manager, PLOTS_BASE)
if 'script_4' in scripts:
    print("[DataPreparation] Starting script_4 ...")
    df = file_manager.read_csv("US_Accidents_TX_subset_amended_17Jun2025_0950.csv")  # For script_4 | Please change to correct file_name
    print("Reading input CSV...")
    Explorer.final(df, file_manager, PLOTS_BASE)
    df = editor.script_4b(df)
    Explorer.final(df, file_manager, PLOTS_BASE)
if 'script_5' in scripts:
    df = file_manager.read_csv("US_Accidents_TX_subset_amended_17Jun2025_1037.csv")  # For script_5 | Please change to correct file_name
    print("[DataPreparation] Starting script_5 ...")
    print("Reading input CSV...")
    Explorer.final(df, file_manager, PLOTS_BASE)
    df = editor.script_5(df)
    Explorer.final(df, file_manager, PLOTS_BASE)

# # Generate Plots
# print("Generating Plots...")
# PlotManager.plot_tasks(df)

# Save amended csv
print("Saving amended CSV...")
amended_filename = file_manager.write_csv(df, "US_Accidents_TX_subset_amended")

# End time (track total script run time)
end_time_run = time_module.time()
elapsed_seconds = end_time_run - start_time_run
print("End time:", datetime.now().strftime("%H:%M:%S"))

# Convert elapsed seconds to HH:MM:SS
hours = int(elapsed_seconds // 3600)
minutes = int((elapsed_seconds % 3600) // 60)
seconds = int(elapsed_seconds % 60)
print(f"Total run time: {hours:02}:{minutes:02}:{seconds:02}")

