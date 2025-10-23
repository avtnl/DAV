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
    data_preparation = DataPreparation(data_editor=data_editor)
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
        for datafile in datafiles:
            if not datafile.exists():
                logger.warning(f"{datafile} does not exist. Maybe check timestamp!")
                continue
            try:
                df = data_editor.convert_timestamp(datafile)
                df = data_editor.clean_author(df)
                df["has_emoji"] = df["message"].apply(data_editor.has_emoji)
                for key, parq_name in parq_files.items():
                    if parq_name.replace(".parq", ".csv") == datafile.name:
                        df["whatsapp_group"] = group_map[key]
                        break
                else:
                    df["whatsapp_group"] = "unknown"
                logger.info(f"Processed DataFrame from {datafile}:\n{df.head()}")
                
                csv_file = file_manager.save_csv(df, processed)
                parq_file = file_manager.save_parq(df, processed)
                logger.info(f"Saved files: CSV={csv_file}, Parquet={parq_file}")
                
                dataframes[datafile.stem] = df
            except Exception as e:
                logger.error(f"Failed to process {datafile}: {e}")
                continue
    else:
        for key, group in group_map.items():
            datafile = processed / config[key]
            datafile = datafile.with_suffix(".csv")
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
  
    if not dataframes:
        logger.error("No valid data files were loaded. Exiting.")
        return
  
    df = data_editor.concatenate_df(dataframes)
    if df is None:
        return
    df = data_editor.filter_group_names(df)
    if df is None:
        return

    df = data_editor.clean_for_deleted_media_patterns(df)
    if df is None:
        logger.error("Failed to clean messages for all groups.")
        return

    csv_file, parq_file = file_manager.save_combined_files(df, processed)
    if csv_file is None or parq_file is None:
        return
    df = data_editor.filter_group_names(df)
    if df is None:
        return
  
    # Assign STEPs to run
    Script = [10, 11]

    tables_dir = Path("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    if 1 in Script or 4 in Script or 5 in Script or 6 in Script or 7 in Script or 8 in Script:
        df, group_authors, non_anthony_group, anthony_group, sorted_groups = data_preparation.build_visual_categories(df)
        if df is None or group_authors is None or sorted_groups is None:
            logger.error("Failed to initialize required variables for STEPs 1, 4, 5, 6, 7 or 8.")
            return

    if 1 in Script:
        fig = plot_manager.build_visual_categories(group_authors, non_anthony_group, anthony_group, sorted_groups)
        if fig is None:
            return
        png_file = file_manager.save_png(fig, image_dir, filename="yearly_bar_chart_combined")
        if png_file is None:
            return

    if 2 in Script:
        df_dac = df[df['whatsapp_group'] == 'dac'].copy()
        if df_dac.empty:
            logger.error("No data found for WhatsApp group 'dac'. Skipping time-based visualization.")
        else:
            df_dac, p, average_all = data_preparation.build_visual_time(df_dac)
            if df_dac is None or p is None or average_all is None:
                logger.error("Failed to prepare data for time-based visualization.")
            else:
                fig_time = plot_manager.build_visual_time(p, average_all)
                if fig_time is None:
                    logger.error("Failed to create time-based plot.")
                else:
                    png_file_time = file_manager.save_png(fig_time, image_dir, filename="golf_decode_by_wa_heartbeat")
                    if png_file_time is None:
                        logger.error("Failed to save time-based plot.")

    if 3 in Script:
        df_maap = df[df['whatsapp_group'] == 'maap'].copy()
        if df_maap.empty:
            logger.error("No data found for WhatsApp group 'maap'. Skipping distribution visualization.")
        else:
            df_maap = data_editor.clean_for_deleted_media_patterns(df_maap)
            if df_maap is None:
                logger.error("Failed to clean messages for distribution visualization.")
            else:
                df_maap, emoji_counts_df = data_preparation.build_visual_distribution(df_maap)
                if df_maap is None or emoji_counts_df is None:
                    logger.error("Failed to prepare data for distribution visualization.")
                else:
                    logger.info(f"Number of unique emojis in emoji_counts_df: {len(emoji_counts_df)}")
                    fig_dist = plot_manager.build_visual_distribution(emoji_counts_df)
                    if fig_dist is None:
                        logger.error("Failed to create distribution bar chart.")
                    else:
                        png_file_dist = file_manager.save_png(fig_dist, image_dir, filename="emoji_counts_once")
                        if png_file_dist is None:
                            logger.error("Failed to save distribution bar chart.")

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

    if 5 in Script:
        groups = ['maap', 'golfmaten', 'dac', 'tillies']
        df_groups = df[df['whatsapp_group'].isin(groups)].copy()
        if df_groups.empty:
            logger.error(f"No data found for WhatsApp groups {groups}. Skipping bubble plot visualization.")
        else:
            try:
                bubble_df = data_preparation.build_visual_relationships_bubble(df_groups, groups)
                if bubble_df is None or bubble_df.empty:
                    logger.error("Failed to prepare bubble plot data.")
                else:
                    fig_bubble = plot_manager.build_visual_relationships_bubble_2(bubble_df)
                    if fig_bubble is None:
                        logger.error("Failed to create bubble plot.")
                    else:
                        png_file_bubble = file_manager.save_png(fig_bubble, image_dir, filename="bubble_plot_words_vs_punct")
                        if png_file_bubble is None:
                            logger.error("Failed to save bubble plot.")
                        else:
                            logger.info(f"Saved bubble plot: {png_file_bubble}")
            except Exception as e:
                logger.exception(f"Error in STEP 5 - Relationship Visualizations: {e}")

    if 6 in Script:
        try:
            if 'message_cleaned' not in df.columns:
                logger.error("Column 'message_cleaned' not found. Ensure clean_for_deleted_media_patterns() is called first.")
                return
            df['message_with_emojis'] = df['message_cleaned'].apply(data_editor.handle_emojis)
            df['message_normalized'] = df['message_with_emojis'].apply(data_editor.normalize_text)
            logger.info(f"Normalized messages: {df[['message_cleaned', 'message_with_emojis', 'message_normalized']].head(10).to_string()}")
            csv_file = file_manager.save_csv(df, processed)
            parq_file = file_manager.save_parq(df, processed)
            if csv_file is None or parq_file is None:
                logger.error("Failed to save normalized DataFrame.")
                return
            logger.info(f"Saved normalized DataFrame: CSV={csv_file}, Parquet={parq_file}")

            golf_keywords = ['tee', 'birdie', 'bogey', 'bunker', 'caddie', 'chip', 'divot', 'draw', 'driver', 'eagle', 'fade', 'fairway', 'green', 'handicap', 'hcp', 'hole', 'hook', 'hybrid', 'lie', 'par', 'putter', 'qualifying', 'rough', 'slice', 'stroke', 'stablefore', 'stbf', 'stb', 'wedge', 'irons', 'masters', 'ryder', 'coupe', 'troffee', 'wedstrijd', 'majors', 'flag_in_hole', 'person_golfing', 'man_golfing', 'woman_golfing', 'trophy', 'white_circle']
            grouped = df.groupby(['whatsapp_group', 'author'])['message_normalized'].apply(' '.join).reset_index()
            grouped['group_author'] = grouped['whatsapp_group'] + '/' + grouped['author']
            vectorizer = CountVectorizer(vocabulary=golf_keywords)
            X = vectorizer.fit_transform(grouped['message_normalized'])
            total_words = grouped['message_normalized'].apply(lambda x: len(x.split())).replace(0, 1)
            X_normalized = X.toarray() / np.array(total_words)[:, np.newaxis]
            feature_df = pd.DataFrame(X_normalized, columns=golf_keywords, index=grouped['group_author'])
            feature_df = feature_df.reset_index().rename(columns={'index': 'group_author'})
            saved_table = file_manager.save_table(feature_df, tables_dir, prefix="golf_features_normalized")
            if saved_table:
                logger.info(f"Saved golf feature matrix: {saved_table}")
            else:
                logger.error("Failed to save golf feature matrix.")
            logger.debug(f"Golf feature matrix preview:\n{feature_df.head().to_string()}")
            fig_multi = plot_manager.build_visual_multi_dimensional_2(feature_df, method='pca')
            if fig_multi is None:
                logger.error("Failed to create multi-dimensional visualization_2.")
            else:
                png_file_multi = file_manager.save_png(fig_multi, image_dir, filename="golf_relatedness_pca")
                if png_file_multi is None:
                    logger.error("Failed to save multi-dimensional visualization.")
                else:
                    logger.info(f"Saved multi-dimensional visualization: {png_file_multi}")
        except Exception as e:
            logger.exception(f"Error in STEP 6 - Message Normalization: {e}")
            return

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
                if feature_df is None:
                    logger.error("Failed to build interaction features.")
                else:
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
            missing_authors = [a for a in group_authors[group] if a not in gender_map]
            if missing_authors:
                logger.warning(f"Authors missing from gender_map: {missing_authors}")
            sequence_handler = data_preparation.SequenceHandler(gender_map=gender_map)
            sequence_df = sequence_handler.build_sequence_scores(df_group, group_authors[group])
            if not sequence_df.empty:
                file_manager.save_table(sequence_df, tables_dir, f"sequence_scores_{group}")
                detected_couples = sequence_handler.detect_married_couples(sequence_df)
                logger.info(f"Detected couples: {detected_couples}")
                sequence_handler.married_couples = [(m, f) for m, f in detected_couples.items()]
                sequence_df_with_married = sequence_handler.build_sequence_scores(
                    df_group, group_authors[group], include_married_alternation=True
                )
                if not sequence_df_with_married.empty:
                    file_manager.save_table(sequence_df_with_married, tables_dir, f"sequence_scores_with_married_{group}")
                    fig = plot_manager.build_visual_model(sequence_df_with_married, group)
                    if fig is None:
                        logger.error(f"Failed to generate model plot for {group}.")
                    else:
                        file_manager.save_png(fig, image_dir, f"model_{group}")
                else:
                    logger.error(f"Failed to generate sequence DataFrame with married alternation for {group}.")

    if 10 in Script:
        try:
            if 'message_cleaned' not in df.columns:
                logger.error("Column 'message_cleaned' not found. Ensure clean_for_deleted_media_patterns() is called first.")
                return
            # Add new columns before organizing
            df['year'] = data_editor.get_year(df)
            df['month'] = data_editor.get_month(df)
            df['week'] = data_editor.get_week(df)
            df['active_years'] = data_editor.active_years(df)
            df['early_leaver'] = data_editor.early_leaver(df)
            logger.info(f"Added columns: year, month, week, active_years, early_leaver")
            logger.debug(f"DataFrame with new columns:\n{df[['whatsapp_group', 'author', 'year', 'month', 'week', 'active_years', 'early_leaver']].head().to_string()}")
            
            df_organized = data_editor.organize_df(df)
            if df_organized is None:
                logger.error("Failed to organize DataFrame in STEP 10.")
                return
            
            # Convert boolean columns to strings after organizing
            df_organized = data_editor.convert_booleans(df_organized)
            if df_organized is None:
                logger.error("Failed to convert boolean columns in STEP 10.")
                return
            
            # Save DataFrame with redundant columns
            csv_file_with_redundancy = file_manager.save_csv(df_organized, processed, prefix="organized_data_with_redundancy")
            parq_file_with_redundancy = file_manager.save_parq(df_organized, processed, prefix="organized_data_with_redundancy")
            if csv_file_with_redundancy is None or parq_file_with_redundancy is None:
                logger.error("Failed to save organized DataFrame with redundancy.")
                return
            logger.info(f"Saved organized DataFrame with redundancy: CSV={csv_file_with_redundancy}, Parquet={parq_file_with_redundancy}")
            
            # Delete redundant attributes and save no redundancy version
            df_no_redundancy = data_editor.delete_redundant_attributes(df_organized)
            if df_no_redundancy is None:
                logger.error("Failed to delete redundant attributes in STEP 10.")
                return
            
            csv_file_no_redundancy = file_manager.save_csv(df_no_redundancy, processed, prefix="organized_data_no_redundancy")
            parq_file_no_redundancy = file_manager.save_parq(df_no_redundancy, processed, prefix="organized_data_no_redundancy")
            if csv_file_no_redundancy is None or parq_file_no_redundancy is None:
                logger.error("Failed to save organized DataFrame no redundancy.")
                return
            logger.info(f"Saved organized DataFrame no redundancy: CSV={csv_file_no_redundancy}, Parquet={parq_file_no_redundancy}")
            logger.debug(f"Organized DataFrame preview:\n{df_no_redundancy.head().to_string()}")
            # Save active years table
            active_years = df.groupby(['whatsapp_group', 'author'])['year'].agg(['min', 'max']).reset_index()
            active_years['active_years'] = active_years.apply(lambda x: f"{x['min']}-{x['max']}", axis=1)
            saved_table = file_manager.save_table(active_years, tables_dir, prefix="active_years")
            if saved_table:
                logger.info(f"Saved active years table: {saved_table}")
            else:
                logger.error("Failed to save active years table.")
        except Exception as e:
            logger.exception(f"Error in STEP 10 - DataFrame Organization: {e}")
            return

    if 11 in Script:
        try:
            data_feed = 'non_redundant'  # 'non_redundant' or 'redundant'
            plot_feed = 'both'  # 'both' or 'per_group', 'global'
            groupby_period = 'month'  # 'week' or 'month', 'year'
            delete_specific_attributes = False  # or True
            
            # Validate data_feed and plot_feed combinations
            if data_feed == 'non_redundant':
                if plot_feed not in ['per_group', 'global', 'both']:
                    logger.error("Invalid plot_feed for non_redundant data_feed. Must be 'per_group', 'global', or 'both'.")
                    return
            elif data_feed == 'redundant':
                if plot_feed != 'per_group':
                    logger.error("Invalid plot_feed for redundant data_feed. Can only be 'per_group'.")
                    return
            else:
                logger.error("Invalid data_feed. Must be 'non_redundant' or 'redundant'.")
                return
            
            # Validate groupby_period
            if groupby_period not in ['week', 'month', 'year']:
                logger.error("Invalid groupby_period. Must be 'week', 'month', or 'year'.")
                return
            
            # Validate delete_specific_attributes
            if delete_specific_attributes and (data_feed != 'non_redundant' or groupby_period != 'month'):
                logger.error("delete_specific_attributes can only be True when data_feed='non_redundant' and groupby_period='month'.")
                return
            
            # Find the latest appropriate organized_data csv based on data_feed
            prefix = f"organized_data_{'no_redundancy' if data_feed == 'non_redundant' else 'with_redundancy'}"
            datafile = file_manager.find_latest_file(processed, prefix=prefix, suffix=".csv")
            if datafile is None:
                logger.error(f"No {prefix}.csv found in processed directory.")
                return
            logger.info(f"Loading latest {prefix} data from {datafile}")
            df = pd.read_csv(datafile, parse_dates=['timestamp'])
            # Ensure week and month columns are present
            df['month'] = data_editor.get_month(df)
            df['week'] = data_editor.get_week(df)
            logger.debug(f"DataFrame after adding month and week columns:\n{df[['whatsapp_group', 'author', 'year', 'month', 'week']].head().to_string()}")
            # Exclude 'tillies' group
            df = df[df['whatsapp_group'] != 'tillies'].copy()
            if df.empty:
                logger.error("No data remains after filtering out 'tillies' group in STEP 11.")
                return
            logger.info(f"Filtered DataFrame excluding 'tillies': {len(df)} rows")
            
            # Apply delete_specific_attributes if conditions are met
            if delete_specific_attributes:
                df = data_editor.delete_specific_attributes(df)
                if df is None:
                    logger.error("Failed to delete specific attributes.")
                    return
                # Save the modified DataFrame
                csv_file_specific = file_manager.save_csv(df, processed, prefix="organized_data_specific")
                parq_file_specific = file_manager.save_parq(df, processed, prefix="organized_data_specific")
                if csv_file_specific is None or parq_file_specific is None:
                    logger.error("Failed to save organized DataFrame with specific attributes deleted.")
                    return
                logger.info(f"Saved organized DataFrame with specific attributes deleted: CSV={csv_file_specific}, Parquet={parq_file_specific}")
            
            # Build feature DataFrame for non-message content
            feature_df = data_preparation.build_visual_not_message_content(df, groupby_period=groupby_period)
            if feature_df is None:
                logger.error("Failed to build non-message content feature DataFrame.")
                return
            
            # Create and save visualizations for PCA and t-SNE based on plot_feed
            figs = plot_manager.build_visual_not_message_content(feature_df, draw_ellipse=True, alpha_per_group=0.6, alpha_global=0.6, plot_type=plot_feed, groupby_period=groupby_period)
            if figs is None:
                logger.error("Failed to create non-message content visualizations.")
                return
            # Save all figures
            for fig_info in figs:
                fig = fig_info['fig']
                filename = fig_info['filename']
                png_file = file_manager.save_png(fig, image_dir, filename=filename)
                if png_file is None:
                    logger.error(f"Failed to save visualization {filename}.")
                else:
                    logger.info(f"Saved visualization: {png_file}")
            
            # Plot month correlations if groupby_period is 'month'
            if groupby_period == 'month':
                correlations = data_preparation.compute_month_correlations(feature_df)
                if correlations is None:
                    logger.error("Failed to compute month correlations.")
                    return
                fig_correlations = plot_manager.plot_month_correlations(correlations)
                if fig_correlations is None:
                    logger.error("Failed to create month correlations plot.")
                    return
                png_file_correlations = file_manager.save_png(fig_correlations, image_dir, filename="month_correlations")
                if png_file_correlations is None:
                    logger.error("Failed to save month correlations plot.")
                    return
                logger.info(f"Saved month correlations plot: {png_file_correlations}")
        except Exception as e:
            logger.exception(f"Error in STEP 11 - Non-Message Content Visualization: {e}")
            return
            
            # Apply delete_specific_attributes if conditions are met
            if delete_specific_attributes:
                df = data_editor.delete_specific_attributes(df)
                if df is None:
                    logger.error("Failed to delete specific attributes.")
                    return
                # Save the modified DataFrame
                csv_file_specific = file_manager.save_csv(df, processed, prefix="organized_data_specific")
                parq_file_specific = file_manager.save_parq(df, processed, prefix="organized_data_specific")
                if csv_file_specific is None or parq_file_specific is None:
                    logger.error("Failed to save organized DataFrame with specific attributes deleted.")
                    return
                logger.info(f"Saved organized DataFrame with specific attributes deleted: CSV={csv_file_specific}, Parquet={parq_file_specific}")
            
            # Build feature DataFrame for non-message content
            feature_df = data_preparation.build_visual_not_message_content(df, groupby_period=groupby_period)
            if feature_df is None:
                logger.error("Failed to build non-message content feature DataFrame.")
                return
            # Create and save visualizations for PCA and t-SNE based on plot_feed
            figs = plot_manager.build_visual_not_message_content(feature_df, draw_ellipse=True, alpha_per_group=0.6, alpha_global=0.6, plot_type=plot_feed, groupby_period=groupby_period)
            if figs is None:
                logger.error("Failed to create non-message content visualizations.")
                return
            # Save all figures
            for fig_info in figs:
                fig = fig_info['fig']
                filename = fig_info['filename']
                png_file = file_manager.save_png(fig, image_dir, filename=filename)
                if png_file is None:
                    logger.error(f"Failed to save visualization {filename}.")
                else:
                    logger.info(f"Saved visualization: {png_file}")
        except Exception as e:
            logger.exception(f"Error in STEP 11 - Non-Message Content Visualization: {e}")
            return

    if 12 in Script:
        try:
            # Find the latest organized_data.csv
            datafile = file_manager.find_latest_file(processed, prefix="organized_data", suffix=".csv")
            if datafile is None:
                logger.error("No organized_data.csv found in processed directory.")
                return
            logger.info(f"Loading latest organized data from {datafile}")
            df = pd.read_csv(datafile, parse_dates=['timestamp'])
            # Ensure month columns are present
            df['month'] = data_editor.get_month(df)
            logger.debug(f"DataFrame after adding month column:\n{df[['whatsapp_group', 'author', 'year', 'month']].head().to_string()}")
            # Exclude 'tillies' group
            df = df[df['whatsapp_group'] != 'tillies'].copy()
            if df.empty:
                logger.error("No data remains after filtering out 'tillies' group in STEP 12.")
                return
            logger.info(f"Filtered DataFrame excluding 'tillies': {len(df)} rows")
            # Build feature DataFrame for non-message content
            feature_df = data_preparation.build_visual_not_message_content(df)
            if feature_df is None:
                logger.error("Failed to build non-message content feature DataFrame.")
                return
            # Compute correlations with month
            correlations = data_preparation.compute_month_correlations(feature_df)
            if correlations is None:
                logger.error("Failed to compute correlations with month.")
                return
            # Plot correlations
            fig = plot_manager.plot_month_correlations(correlations)
            if fig is None:
                logger.error("Failed to create month correlations plot.")
                return
            png_file = file_manager.save_png(fig, image_dir, filename="month_correlations")
            if png_file is None:
                logger.error("Failed to save month correlations plot.")
            else:
                logger.info(f"Saved month correlations plot: {png_file}")
        except Exception as e:
            logger.exception(f"Error in STEP 12 - Month Correlations Analysis: {e}")
            return

if __name__ == "__main__":
    main()