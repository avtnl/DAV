import pandas as pd
from loguru import logger
from file_manager import FileManager
from data_editor import DataEditor
from data_preparation import DataPreparation
from plot_manager import PlotManager
import tomllib
from pathlib import Path

def main():
    # Instantiate classes
    file_manager = FileManager()
    data_editor = DataEditor()
    data_preparation = DataPreparation()
    plot_manager = PlotManager()
  
    # Get CSV file(s), processed directory, group mapping, and Parquet files
    datafiles, processed, group_map, parq_files = file_manager.read_csv()
  
    # Check if any files were returned
    if datafiles is None or not datafiles or group_map is None:
        logger.error("No valid CSV files or group mapping returned")
        return
  
    # Load config for preprocess flag and image directory
    configfile = Path("config.toml").resolve()
    with configfile.open("rb") as f:
        config = tomllib.load(f)
    image_dir = Path(config["image"])
  
    # Process files
    dataframes = {}
    if config["preprocess"]:
        # Case A: Preprocess = True
        for datafile in datafiles:
            if not datafile.exists():
                logger.warning(
                    f"{datafile} does not exist. Maybe check timestamp!"
                )
                continue
            try:
                df = data_editor.convert_timestamp(datafile)
                df = data_editor.clean_author(df)
                df["has_emoji"] = df["message"].apply(data_editor.has_emoji)
                # Assign whatsapp_group based on parq_files
                for key, parq_name in parq_files.items():
                    if parq_name.replace(".parq", ".csv") == datafile.name:
                        df["whatsapp_group"] = group_map[key]
                        break
                else:
                    df["whatsapp_group"] = "unknown"
                logger.info(f"Processed DataFrame from {datafile}:\n{df.head()}")
                
                # Save individual processed DataFrame
                csv_file = file_manager.save_csv(df, processed)
                parq_file = file_manager.save_parq(df, processed)
                logger.info(f"Saved files: CSV={csv_file}, Parquet={parq_file}")
                
                # Store DataFrame with filename-based key
                dataframes[datafile.stem] = df
            except Exception as e:
                logger.error(f"Failed to process {datafile}: {e}")
                continue
    else:
        # Case B: Preprocess = False
        for key, group in group_map.items():
            datafile = processed / config[key]
            datafile = datafile.with_suffix(".csv")  # Convert .parq to .csv
            logger.debug(f"Attempting to load {datafile}")
            if not datafile.exists():
                logger.warning(f"{datafile} does not exist. Run preprocess.py or check timestamp!")
                continue
            try:
                df = data_editor.convert_timestamp(datafile)
                df = data_editor.clean_author(df)
                df["has_emoji"] = df["message"].apply(data_editor.has_emoji)
                df["whatsapp_group"] = group
                dataframes[key] = df
                logger.info(f"Loaded {datafile} with {len(df)} rows")
            except Exception as e:
                logger.exception(f"Failed to load {datafile}: {e}")
                continue
  
    # Check if any DataFrames were loaded
    if not dataframes:
        logger.error("No valid data files were loaded. Exiting.")
        return
  
    # Concatenate and filter DataFrames
    df = data_editor.concatenate_df(dataframes)
    if df is None:
        return
    df = data_editor.filter_group_names(df)
    if df is None:
        return

    # Clean messages for deleted media patterns
    df = data_editor.clean_for_deleted_media_patterns(df)
    if df is None:
        logger.error("Failed to clean messages for all groups.")
        return

    # Save combined DataFrame
    csv_file, parq_file = file_manager.save_combined_files(df, processed)
    if csv_file is None or parq_file is None:
        return
    df = data_editor.filter_group_names(df)
    if df is None:
        return
  
    # Assign STEPs to run
    Script = [4]

    # Initialize variables needed for STEPs 1 and 4
    if 1 in Script or 4 in Script:
        df, group_authors, non_anthony_group, anthony_group, sorted_groups = data_preparation.build_visual_categories(df)
        if df is None or group_authors is None or sorted_groups is None:
            logger.error("Failed to initialize required variables for STEPs 1 or 4.")
            return

    # STEP 1: Prepare data for visualization (categories)
    if 1 in Script:
        # Create bar chart (categories)
        fig = plot_manager.build_visual_categories(group_authors, non_anthony_group, anthony_group, sorted_groups)
        if fig is None:
            return
        
        # Save bar chart (categories)
        png_file = file_manager.save_png(fig, image_dir, filename="yearly_bar_chart_combined")
        if png_file is None:
            return
  
    # STEP 2: Time-based visualization for 'dac' group
    if 2 in Script:
        # Filter DataFrame for whatsapp_group='dac'
        df_dac = df[df['whatsapp_group'] == 'dac'].copy()
        if df_dac.empty:
            logger.error("No data found for WhatsApp group 'dac'. Skipping time-based visualization.")
        else:
            # Prepare data for time-based visualization
            df_dac, p, average_all = data_preparation.build_visual_time(df_dac)
            if df_dac is None or p is None or average_all is None:
                logger.error("Failed to prepare data for time-based visualization.")
            else:
                # Create time-based plot
                fig_time = plot_manager.build_visual_time(p, average_all)
                if fig_time is None:
                    logger.error("Failed to create time-based plot.")
                else:
                    # Save time-based plot
                    png_file_time = file_manager.save_png(fig_time, image_dir, filename="golf_decode_by_wa_heartbeat")
                    if png_file_time is None:
                        logger.error("Failed to save time-based plot.")
  
    # STEP 3: Distribution visualization for 'maap' group
    if 3 in Script:
        # Filter DataFrame for whatsapp_group='maap'
        df_maap = df[df['whatsapp_group'] == 'maap'].copy()
        if df_maap.empty:
            logger.error("No data found for WhatsApp group 'maap'. Skipping distribution visualization.")
        else:
            # Clean messages for deleted media patterns
            df_maap = data_editor.clean_for_deleted_media_patterns(df_maap)
            if df_maap is None:
                logger.error("Failed to clean messages for distribution visualization.")
            else:
                # Prepare data for distribution visualization
                df_maap, emoji_counts_df = data_preparation.build_visual_distribution(df_maap)
                if df_maap is None or emoji_counts_df is None:
                    logger.error("Failed to prepare data for distribution visualization.")
                else:
                    # Log number of unique emojis
                    logger.info(f"Number of unique emojis in emoji_counts_df: {len(emoji_counts_df)}")
                    
                    # Create distribution plot
                    fig_dist = plot_manager.build_visual_distribution(emoji_counts_df)
                    if fig_dist is None:
                        logger.error("Failed to create distribution plot.")
                    else:
                        # Save distribution plot
                        png_file_dist = file_manager.save_png(fig_dist, image_dir, filename="emoji_counts_once")
                        if png_file_dist is None:
                            logger.error("Failed to save distribution plot.")

    # STEP 4: Relationships visualization per group
    if 4 in Script:
        for group in sorted_groups:
            df_group = df[df['whatsapp_group'] == group].copy()
            if df_group.empty:
                logger.error(f"No data found for WhatsApp group '{group}'. Skipping relationships visualization.")
                continue
            
            table2 = data_preparation.build_visual_relationships(df_group, group_authors[group])
            if table2 is not None:
                fig_rel = plot_manager.build_visual_relationships(table2, group)
                if fig_rel is not None:
                    png_file_rel = file_manager.save_png(fig_rel, image_dir, filename=f"relationships_{group}")
                    if png_file_rel is None:
                        logger.error("Failed to save relationships plot.")

if __name__ == "__main__":
    main()