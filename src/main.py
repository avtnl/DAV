import pandas as pd
import numpy as np
from loguru import logger
from file_manager import FileManager
from data_editor import DataEditor
from data_preparation import DataPreparation
from plot_manager import PlotManager
import tomllib
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

def main():
    # Instantiate classes
    file_manager = FileManager()
    data_editor = DataEditor()
    data_preparation = DataPreparation(data_editor=data_editor)  # Pass DataEditor instance
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
    Script = [1]

    # Define tables directory
    tables_dir = Path("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Initialize variables needed for STEPs 1, 4, 5, 6, 7 and 8
    if 1 in Script or 4 in Script or 5 in Script or 6 in Script or 7 in Script or 8 in Script:
        df, group_authors, non_anthony_group, anthony_group, sorted_groups = data_preparation.build_visual_categories(df)
        if df is None or group_authors is None or sorted_groups is None:
            logger.error("Failed to initialize required variables for STEPs 1, 4, 5, 6, 7 or 8.")
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
                # Prepare data for distribution visualization (for build_visual_distribution)
                df_maap, emoji_counts_df = data_preparation.build_visual_distribution(df_maap)
                if df_maap is None or emoji_counts_df is None:
                    logger.error("Failed to prepare data for distribution visualization.")
                else:
                    # Log number of unique emojis
                    logger.info(f"Number of unique emojis in emoji_counts_df: {len(emoji_counts_df)}")
                    
                    # Create distribution plot (original bar chart)
                    fig_dist = plot_manager.build_visual_distribution(emoji_counts_df)
                    if fig_dist is None:
                        logger.error("Failed to create distribution bar chart.")
                    else:
                        # Save distribution bar chart
                        png_file_dist = file_manager.save_png(fig_dist, image_dir, filename="emoji_counts_once")
                        if png_file_dist is None:
                            logger.error("Failed to save distribution bar chart.")
                    
    # STEP 4: Relationships visualization as arc for daily participation in 'maap' group
    if 4 in Script:
        group = 'maap'
        df_group = df[df['whatsapp_group'] == group].copy()
        if df_group.empty:
            logger.error(f"No data found for WhatsApp group '{group}'. Skipping relationships_3 visualization.")
        else:
            combined_df = data_preparation.build_visual_relationships_arc(df_group, group_authors[group])
            if combined_df is not None and not combined_df.empty:
                file_manager.save_table(combined_df, tables_dir, f"participation_{group}")
                fig_net = plot_manager.build_visual_relationships_arc(combined_df, group)
                if fig_net is not None:
                    png_file_net = file_manager.save_png(fig_net, image_dir, filename=f"network_interactions_{group}")
                    if png_file_net is None:
                        logger.error("Failed to save network diagram.")
            else:
                logger.error("Failed to create combined participation table.")

    # STEP 5: Bubble Plot for Relationships, including both emoji usage categories
    if 5 in Script:
        groups = ['maap', 'golfmaten', 'dac', 'tillies']
        df_groups = df[df['whatsapp_group'].isin(groups)].copy()
        if df_groups.empty:
            logger.error(f"No data found for WhatsApp groups {groups}. Skipping bubble plot visualization.")
        else:
            try:
                # Prepare data for bubble plot including both has_emoji categories
                bubble_df = data_preparation.build_visual_relationships_bubble(df_groups, groups)
                if bubble_df is None or bubble_df.empty:
                    logger.error("Failed to prepare bubble plot data.")
                else:
                    # Create single bubble plot with both emoji categories
                    fig_bubble = plot_manager.build_visual_relationships_bubble_2(bubble_df)
                    if fig_bubble is None:
                        logger.error("Failed to create bubble plot.")
                    else:
                        # Save the single bubble plot
                        png_file_bubble = file_manager.save_png(fig_bubble, image_dir, filename="bubble_plot_words_vs_punct")
                        if png_file_bubble is None:
                            logger.error("Failed to save bubble plot.")
                        else:
                            logger.info(f"Saved bubble plot: {png_file_bubble}")
            except Exception as e:
                logger.exception(f"Error in STEP 5 - Relationship Visualizations: {e}")       

    # STEP 6: Normalize messages for golf-relatedness analysis
    if 6 in Script:
        try:
            # Ensure cleaned messages exist
            if 'message_cleaned' not in df.columns:
                logger.error("Column 'message_cleaned' not found. Ensure clean_for_deleted_media_patterns() is called first.")
                return
            # Apply emoji handling and text normalization
            df['message_with_emojis'] = df['message_cleaned'].apply(data_editor.handle_emojis)
            df['message_normalized'] = df['message_with_emojis'].apply(data_editor.normalize_text)
            logger.info(f"Normalized messages: {df[['message_cleaned', 'message_with_emojis', 'message_normalized']].head(10).to_string()}")
            # Save processed DataFrame
            csv_file = file_manager.save_csv(df, processed)
            parq_file = file_manager.save_parq(df, processed)
            if csv_file is None or parq_file is None:
                logger.error("Failed to save normalized DataFrame.")
                return
            logger.info(f"Saved normalized DataFrame: CSV={csv_file}, Parquet={parq_file}")

            golf_keywords = ['tee', 'birdie', 'bogey', 'bunker', 'caddie', 'chip', 'divot', 'draw', 'driver', 'eagle', 'fade', 'fairway', 'green', 'handicap', 'hcp', 'hole', 'hook', 'hybrid', 'lie', 'par', 'putter', 'qualifying', 'rough', 'slice', 'stroke', 'stablefore', 'stbf', 'stb', 'wedge', 'irons', 'masters', 'ryder', 'coupe', 'troffee', 'wedstrijd', 'majors', 'flag_in_hole', 'person_golfing', 'man_golfing', 'woman_golfing', 'trophy', 'white_circle']  # Your custom list
            
            # # Aggregate normalized messages per group
            # grouped = df.groupby('whatsapp_group')['message_normalized'].apply(' '.join).reset_index()
            # vectorizer = CountVectorizer(vocabulary=golf_keywords)
            # Aggregate normalized messages by group and author
            grouped = df.groupby(['whatsapp_group', 'author'])['message_normalized'].apply(' '.join).reset_index()
            # Create a label column for group/author combinations
            grouped['group_author'] = grouped['whatsapp_group'] + '/' + grouped['author']
            vectorizer = CountVectorizer(vocabulary=golf_keywords)
            X = vectorizer.fit_transform(grouped['message_normalized'])
            total_words = grouped['message_normalized'].apply(lambda x: len(x.split())).replace(0, 1)  # Avoid division by zero
            X_normalized = X.toarray() / np.array(total_words)[:, np.newaxis]

            # # Convert to DataFrame for saving/logging
            # feature_df = pd.DataFrame(X_normalized, columns=golf_keywords, index=grouped['whatsapp_group'])
            # feature_df = feature_df.reset_index().rename(columns={'index': 'whatsapp_group'})  # Make groups a column if needed

            # Convert to DataFrame for saving/logging
            feature_df = pd.DataFrame(X_normalized, columns=golf_keywords, index=grouped['group_author'])
            feature_df = feature_df.reset_index().rename(columns={'index': 'group_author'})

            # Save using existing save_table
            tables_dir = Path("tables")  # Already defined earlier in main()
            saved_table = file_manager.save_table(feature_df, tables_dir, prefix="golf_features_normalized")
            if saved_table:
                logger.info(f"Saved golf feature matrix: {saved_table}")
            else:
                logger.error("Failed to save golf feature matrix.")

            # Optional: Log a preview instead/alongside
            logger.debug(f"Golf feature matrix preview:\n{feature_df.head().to_string()}")

            # Call the new plot function
            fig_multi = plot_manager.build_visual_multi_dimensional_2(feature_df, method='pca')
            if fig_multi is None:
                logger.error("Failed to create multi-dimensional visualization_2.")
            else:
                # Save the plot
                png_file_multi = file_manager.save_png(fig_multi, image_dir, filename="golf_relatedness_pca")
                if png_file_multi is None:
                    logger.error("Failed to save multi-dimensional visualization.")
                else:
                    logger.info(f"Saved multi-dimensional visualization: {png_file_multi}")
        except Exception as e:
            logger.exception(f"Error in STEP 6 - Message Normalization: {e}")
            return

    # # STEP 7: Interaction and Network Dynamics (1 plot, including whatsapp_group "tillies")
    # if 7 in Script:
    #     feature_df = data_preparation.build_interaction_features(df, group_authors)
    #     if feature_df is not None:
    #         # Use the new specialized plot function
    #         fig_interact = plot_manager.build_visual_interactions(feature_df, method='pca')
    #         if fig_interact is not None:
    #             png_file_interact = file_manager.save_png(fig_interact, image_dir, filename="interaction_dynamics_pca")
    #             if png_file_interact is None:
    #                 logger.error("Failed to save interaction dynamics plot.")

    # # STEP 7: Interaction and Network Dynamics (1 plot, excluding whatsapp_group "tillies")
    # if 7 in Script:
    #     # Filter out the 'tillies' group
    #     df_filtered = df[df['whatsapp_group'] != 'tillies'].copy()
    #     if df_filtered.empty:
    #         logger.error("No data remains after filtering out 'tillies' group. Skipping STEP 7.")
    #     else:
    #         # Update group_authors to exclude 'tillies'
    #         group_authors_filtered = {group: authors for group, authors in group_authors.items() if group != 'tillies'}
    #         if not group_authors_filtered:
    #             logger.error("No groups remain after filtering out 'tillies'. Skipping STEP 7.")
    #         else:
    #             feature_df = data_preparation.build_interaction_features(df_filtered, group_authors_filtered)
    #             if feature_df is not None:
    #                 # Use the new specialized plot function
    #                 fig_interact = plot_manager.build_visual_interactions(feature_df, method='pca')
    #                 if fig_interact is not None:
    #                     png_file_interact = file_manager.save_png(fig_interact, image_dir, filename="interaction_dynamics_pca")
    #                     if png_file_interact is None:
    #                         logger.error("Failed to save interaction dynamics plot.")

    # STEP 7: Interaction and Network Dynamics
    if 7 in Script:
        df_filtered = df[df['whatsapp_group'] != 'tillies'].copy()
        if df_filtered.empty:
            logger.error("No data remains after filtering out 'tillies' group. Skipping STEP 7.")
        else:
            group_authors_filtered = {group: authors for group, authors in group_authors.items() if group != 'tillies'}
            if not group_authors_filtered:
                logger.error("No groups remain after filtering out 'tillies'. Skipping STEP 7.")
            else:
                feature_df = data_preparation.build_interaction_features(df_filtered, group_authors_filtered)
                if feature_df is not None:
                    fig_interact, fig_groups = plot_manager.build_visual_interactions_2(feature_df, method='pca')
                    if fig_interact is not None:
                        png_file_interact = file_manager.save_png(fig_interact, image_dir, filename="interaction_dynamics_pca")
                        if png_file_interact is None:
                            logger.error("Failed to save interaction dynamics plot.")
                        else:
                            logger.info(f"Saved interaction dynamics plot: {png_file_interact}")
                    if fig_groups is not None:
                        png_file_groups = file_manager.save_png(fig_groups, image_dir, filename="interaction_dynamics_groups_pca")
                        if png_file_groups is None:
                            logger.error("Failed to save group-based interaction dynamics plot.")
                        else:
                            logger.info(f"Saved group-based interaction dynamics plot: {png_file_groups}")

    # STEP 8: Model focussing on sequence handling for daily participation in 'maap' group
    if 8 in Script: 
        group = 'maap'
        df_group = df[df['whatsapp_group'] == group].copy()
        if df_group.empty:
            logger.error(f"No data found for WhatsApp group '{group}'. Skipping sequence analysis.")
        else:
            gender_map = {
                'Anthony van Tilburg': 'M',
                'Phons Berkemeijer': 'M',
                'Anja Berkemeijer': 'F',
                'Madeleine': 'F'
            }
            # Check for missing authors
            missing_authors = [a for a in group_authors[group] if a not in gender_map]
            if missing_authors:
                logger.warning(f"Authors missing from gender_map: {missing_authors}")
            
            sequence_handler = data_preparation.SequenceHandler(gender_map=gender_map)
            sequence_df = sequence_handler.build_sequence_scores(df_group, group_authors[group])
            if not sequence_df.empty:
                file_manager.save_table(sequence_df, tables_dir, f"sequence_scores_{group}")
                detected_couples = sequence_handler.detect_married_couples(sequence_df)
                logger.info(f"Detected couples: {detected_couples}")
                # Set married couples and recompute with alternation score
                sequence_handler.married_couples = [(m, f) for m, f in detected_couples.items()]
                sequence_df_with_married = sequence_handler.build_sequence_scores(
                    df_group, group_authors[group], include_married_alternation=True
                )
                if not sequence_df_with_married.empty:
                    file_manager.save_table(sequence_df_with_married, tables_dir, f"sequence_scores_with_married_{group}")
                    # Create and save visualization
                    fig = plot_manager.build_visual_model(sequence_df_with_married, group)
                    if fig is not None:
                        file_manager.save_png(fig, image_dir, f"model_{group}")
                    else:
                        logger.error(f"Failed to generate model plot for {group}.")
                else:
                    logger.error(f"Failed to generate sequence DataFrame with married alternation for {group}.")
              
if __name__ == "__main__":
    main()