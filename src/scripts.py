import pandas as pd
import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer
from plot_manager import PlotSettings, CategoriesPlotSettings, TimePlotSettings, DistributionPlotSettings, NoMessageContentSettings as PlotManagerNoMessageContentSettings, DimReductionSettings, ArcPlotSettings, BubbleNewPlotSettings
from pydantic import BaseModel
from typing import Optional, Dict, List, Tuple
from pathlib import Path  # Added to resolve Path type hint

class BaseScript:
    """Base class for scripts with common functionality."""
    def __init__(self, file_manager, data_editor=None, data_preparation=None, plot_manager=None, settings: BaseModel = None):
        self.file_manager = file_manager
        self.data_editor = data_editor
        self.data_preparation = data_preparation
        self.plot_manager = plot_manager
        self.settings = settings or PlotSettings()

    def save_figure(self, fig, image_dir: Path, filename: str) -> Optional[Path]:
        """Save a figure to the specified directory."""
        png_file = self.file_manager.save_png(fig, image_dir, filename=filename)
        if png_file is None:
            logger.error(f"Failed to save {filename}.")
        else:
            logger.info(f"Saved {filename}: {png_file}")
        return png_file

    def save_table(self, df: pd.DataFrame, tables_dir: Path, prefix: str) -> Optional[Path]:
        """Save a DataFrame as a table."""
        saved_table = self.file_manager.save_table(df, tables_dir, prefix=prefix)
        if saved_table:
            logger.info(f"Saved table: {saved_table}")
        else:
            logger.error(f"Failed to save table with prefix {prefix}.")
        return saved_table

    def log_error(self, message: str):
        """Log an error and return None."""
        logger.error(message)
        return None

    def run(self):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement run()")

class Script1(BaseScript):
    """Script for: Build yearly bar chart."""
    def __init__(self, file_manager, plot_manager, image_dir, group_authors, non_anthony_group, anthony_group, sorted_groups, settings: Optional[CategoriesPlotSettings] = None):
        super().__init__(file_manager, plot_manager=plot_manager, settings=settings or CategoriesPlotSettings())
        self.image_dir = image_dir
        self.group_authors = group_authors
        self.non_anthony_group = non_anthony_group
        self.anthony_group = anthony_group
        self.sorted_groups = sorted_groups

    def run(self):
        fig = self.plot_manager.build_visual_categories(self.group_authors, self.non_anthony_group, self.anthony_group, self.sorted_groups, self.settings)
        if fig is None:
            return self.log_error("Failed to create yearly bar chart.")
        return self.save_figure(fig, self.image_dir, "yearly_bar_chart_combined")

class Script2(BaseScript):
    """Script for: Build time-based visualization."""
    def __init__(self, file_manager, data_preparation, plot_manager, image_dir, df, settings: Optional[TimePlotSettings] = None):
        super().__init__(file_manager, data_preparation=data_preparation, plot_manager=plot_manager, settings=settings or TimePlotSettings())
        self.image_dir = image_dir
        self.df = df

    def run(self):
        df_dac = self.df[self.df['whatsapp_group'] == 'dac'].copy()
        if df_dac.empty:
            return self.log_error("No data found for WhatsApp group 'dac'. Skipping time-based visualization.")
        df_dac, p, average_all = self.data_preparation.build_visual_time(df_dac)
        if df_dac is None or p is None or average_all is None:
            return self.log_error("Failed to prepare data for time-based visualization.")
        fig_time = self.plot_manager.build_visual_time(p, average_all, self.settings)
        if fig_time is None:
            return self.log_error("Failed to create time-based plot.")
        return self.save_figure(fig_time, self.image_dir, "golf_decode_by_wa_heartbeat")

class Script3(BaseScript):
    """Script for: Build emoji distribution visualization."""
    def __init__(self, file_manager, data_editor, data_preparation, plot_manager, image_dir, df, settings: Optional[DistributionPlotSettings] = None):
        super().__init__(file_manager, data_editor=data_editor, data_preparation=data_preparation, plot_manager=plot_manager, settings=settings or DistributionPlotSettings())
        self.image_dir = image_dir
        self.df = df

    def run(self):
        df_maap = self.df[self.df['whatsapp_group'] == 'maap'].copy()
        if df_maap.empty:
            return self.log_error("No data found for WhatsApp group 'maap'. Skipping distribution visualization.")
        df_maap = self.data_editor.clean_for_deleted_media_patterns(df_maap)
        if df_maap is None:
            return self.log_error("Failed to clean messages for distribution visualization.")
        df_maap, emoji_counts_df = self.data_preparation.build_visual_distribution(df_maap)
        if df_maap is None or emoji_counts_df is None:
            return self.log_error("Failed to prepare data for distribution visualization.")
        logger.info(f"Number of unique emojis in emoji_counts_df: {len(emoji_counts_df)}")
        fig_dist = self.plot_manager.build_visual_distribution(emoji_counts_df, self.settings)
        if fig_dist is None:
            return self.log_error("Failed to create distribution bar chart.")
        return self.save_figure(fig_dist, self.image_dir, "emoji_counts_once")

class Script4(BaseScript):
    """Script for: Build relationships arc diagram."""
    def __init__(self, file_manager, data_preparation, plot_manager,
                 image_dir, tables_dir, group_authors,
                 settings: Optional[ArcPlotSettings] = None):
        super().__init__(file_manager,
                         data_preparation=data_preparation,
                         plot_manager=plot_manager,
                         settings=settings or ArcPlotSettings())
        self.image_dir = image_dir
        self.tables_dir = tables_dir
        self.group_authors = group_authors

    def build_participation_table(self, df_group: pd.DataFrame, group: str) -> pd.DataFrame:
        """
        Build and save the participation table for the arc diagram.

        Args:
            df_group (pandas.DataFrame): Filtered DataFrame for the specified group (e.g., 'maap').
            group (str): WhatsApp group name (e.g., 'maap').

        Returns:
            pandas.DataFrame: Participation table DataFrame, or None if creation fails.
        """
        participation_df = self.data_preparation.build_visual_relationships_arc(
            df_group, self.group_authors.get(group, [])
        )
        if participation_df is None or participation_df.empty:
            self.log_error("Failed to create participation table.")
            return None
        self.save_table(participation_df,
                        self.tables_dir,
                        f"participation_{group}")
        logger.info(f"Saved participation table for group '{group}'")
        return participation_df

    def run(self):
        try:
            group = 'maap'
            df_group = self.data_preparation.df[
                self.data_preparation.df['whatsapp_group'] == group
            ].copy()
            if df_group.empty:
                return self.log_error(
                    f"No data found for WhatsApp group '{group}'. "
                    "Skipping relationships arc diagram."
                )

            # Build and save participation table
            participation_df = self.build_participation_table(df_group, group)
            if participation_df is None:
                return self.log_error("Failed to create participation table in run.")

            # Build the network plot
            fig = self.plot_manager.build_visual_relationships_arc(
                participation_df, group, self.settings
            )
            if fig is None:
                return self.log_error("Plot manager returned no figure for arc diagram.")
            return self.save_figure(fig,
                                    self.image_dir,
                                    f"network_interactions_{group}")
        except Exception as e:
            logger.exception(f"Error in Script 4 - Relationships Arc Diagram: {e}")
            return None

class Script5(BaseScript):
    """Script for: Build bubble plot visualization."""
    def __init__(self, file_manager, data_preparation, plot_manager, image_dir, df, settings: Optional[BubbleNewPlotSettings] = None):
        super().__init__(file_manager, data_preparation=data_preparation, plot_manager=plot_manager, settings=settings or BubbleNewPlotSettings())
        self.image_dir = image_dir
        self.df = df

    def run(self):
        groups = ['maap', 'golfmaten', 'dac', 'tillies']
        df_groups = self.df[self.df['whatsapp_group'].isin(groups)].copy()
        if df_groups.empty:
            return self.log_error(f"No data found for WhatsApp groups {groups}. Skipping bubble plot visualization.")
        try:
            bubble_df = self.data_preparation.build_visual_relationships_bubble(df_groups, groups)
            if bubble_df is None or bubble_df.empty:
                return self.log_error("Failed to prepare bubble plot data.")
            fig_bubble = self.plot_manager.build_visual_relationships_bubble(bubble_df, self.settings)
            if fig_bubble is None:
                return self.log_error("Failed to create bubble plot.")
            return self.save_figure(fig_bubble, self.image_dir, "bubble_plot_words_vs_punct")
        except Exception as e:
            logger.exception(f"Error in Script 5 - Relationship Visualizations: {e}")
            return None

class Script7(BaseScript):
    """Script for Step 7: Interaction dynamics visualization."""
    def __init__(self, file_manager, data_preparation, plot_manager, image_dir, group_authors, settings: Optional[PlotSettings] = None):
        super().__init__(file_manager, data_preparation=data_preparation, plot_manager=plot_manager, settings=settings or PlotSettings())
        self.image_dir = image_dir
        self.group_authors = group_authors

    def run(self):
        df_filtered = self.data_preparation.df[self.data_preparation.df['whatsapp_group'] != 'tillies'].copy()
        if df_filtered.empty:
            return self.log_error("No data remains after filtering out 'tillies' group. Skipping STEP 7.")
        group_authors_filtered = {group: authors for group, authors in self.group_authors.items() if group != 'tillies'}
        if not group_authors_filtered:
            return self.log_error("No groups remain after filtering out 'tillies'. Skipping STEP 7.")
        feature_df = self.data_preparation.build_interaction_features(df_filtered, group_authors_filtered)
        if feature_df is None:
            return self.log_error("Failed to build interaction features.")
        fig_interact, fig_groups = self.plot_manager.build_visual_interactions_2(
            feature_df, 
            method='pca', 
            settings=DimReductionSettings(), 
            nmc_settings=PlotManagerNoMessageContentSettings()
        )
        if fig_interact is not None:
            self.save_figure(fig_interact, self.image_dir, "interaction_dynamics_pca")
        if fig_groups is not None:
            return self.save_figure(fig_groups, self.image_dir, "interaction_dynamics_groups_pca")
        return None

class Script10(BaseScript):
    """Script for: Organize DataFrame with additional features."""
    def __init__(self, file_manager, data_editor, data_preparation, processed, tables_dir, settings: Optional[PlotSettings] = None):
        super().__init__(file_manager, data_editor=data_editor, data_preparation=data_preparation, settings=settings or PlotSettings())
        self.processed = processed
        self.tables_dir = tables_dir

    def run(self):
        try:
            # Assume the DataFrame is available from data_preparation.df or needs to be loaded
            if self.data_preparation.df is None or self.data_preparation.df.empty:
                logger.error("No valid DataFrame available in DataPreparation.")
                return None
            
            df = self.data_preparation.df.copy()  # Work with a copy to avoid modifying the original
            # Use DataEditor methods with the DataFrame
            df = self.data_editor.clean_author(df)  # Example method; adjust based on your needs
            df['year'] = self.data_editor.get_year(df)  # Add year column
            
            # Save the organized DataFrame
            csv_file = self.file_manager.save_csv(df, self.processed, prefix="organized_data_script_10")
            parq_file = self.file_manager.save_parq(df, self.processed, prefix="organized_data_script_10")
            if csv_file is None or parq_file is None:
                return self.log_error("Failed to save organized DataFrame.")
            logger.info(f"Saved organized DataFrame: CSV={csv_file}, Parquet={parq_file}")
            
            return csv_file  # Or return None if no further action is needed
        except Exception as e:
            logger.exception(f"Error in Script 10 - DataFrame Organization: {e}")
            return None

class Script11(BaseScript):
    """Script for: Non-message content visualization."""
    def __init__(self, file_manager, data_editor, data_preparation, plot_manager, processed, image_dir, settings: Optional[PlotSettings] = None):
        super().__init__(file_manager, data_editor=data_editor, data_preparation=data_preparation, plot_manager=plot_manager, settings=settings or PlotSettings())
        self.processed = processed
        self.image_dir = image_dir

    def check_input(self, data_feed: str, plot_feed: str, groupby_period: str, delete_specific_attributes: bool, plot_type: str) -> bool:
        """
        Validate input parameters for consistency.

        Args:
            data_feed (str): Data type ('non_redundant' or 'redundant').
            plot_feed (str): Plot type ('per_group', 'global', or 'both').
            groupby_period (str): Grouping period ('week', 'month', or 'year').
            delete_specific_attributes (bool): Whether to delete specific attributes.
            plot_type (str): Dimensionality reduction method ('both', 'pca', or 'tsne').

        Returns:
            bool: True if inputs are valid, False otherwise.
        """
        if data_feed not in ['non_redundant', 'redundant']:
            self.log_error(f"Invalid data_feed: {data_feed}. Must be 'non_redundant' or 'redundant'.")
            return False
        if data_feed == 'non_redundant':
            if plot_feed not in ['per_group', 'global', 'both']:
                self.log_error(f"Invalid plot_feed for non_redundant data_feed: {plot_feed}. Must be 'per_group', 'global', or 'both'.")
                return False
        elif data_feed == 'redundant':
            if plot_feed != 'per_group':
                self.log_error(f"Invalid plot_feed for redundant data_feed: {plot_feed}. Must be 'per_group'.")
                return False
        if groupby_period not in ['week', 'month', 'year']:
            self.log_error(f"Invalid groupby_period: {groupby_period}. Must be 'week', 'month', or 'year'.")
            return False
        if delete_specific_attributes and (data_feed != 'non_redundant' or groupby_period != 'month'):
            self.log_error("delete_specific_attributes can only be True when data_feed='non_redundant' and groupby_period='month'.")
            return False
        if plot_type not in ['both', 'pca', 'tsne']:
            self.log_error(f"Invalid plot_type: {plot_type}. Must be 'both', 'pca', or 'tsne'.")
            return False
        logger.info("Input parameters validated successfully.")
        return True

    def load_and_preprocess_data(self, data_feed: str) -> pd.DataFrame:
        """
        Load and preprocess data for non-message content visualization.

        Args:
            data_feed (str): Data type ('non_redundant' or 'redundant').

        Returns:
            pandas.DataFrame: Processed DataFrame, or None if loading or preprocessing fails.
        """
        try:
            prefix = f"organized_data_{'no_redundancy' if data_feed == 'non_redundant' else 'with_redundancy'}"
            datafile = self.file_manager.find_latest_file(self.processed, prefix=prefix, suffix=".csv")
            if datafile is None:
                self.log_error(f"No {prefix}.csv found.")
                return None
            logger.info(f"Loading latest {prefix} data from {datafile}")
            df = pd.read_csv(datafile, parse_dates=['timestamp'])
            df['month'] = self.data_editor.get_month(df)
            df['week'] = self.data_editor.get_week(df)
            logger.debug(f"DataFrame after adding month and week columns:\n{df[['whatsapp_group', 'author', 'year', 'month', 'week']].head().to_string()}")
            df = df[df['whatsapp_group'] != 'tillies'].copy()
            if df.empty:
                self.log_error("No data remains after filtering out 'tillies' group.")
                return None
            logger.info(f"Filtered DataFrame excluding 'tillies': {len(df)} rows")
            return df
        except Exception as e:
            self.log_error(f"Failed to load and preprocess data: {e}")
            return None

    def delete_and_save_attributes(self, df: pd.DataFrame, delete_specific_attributes: bool) -> pd.DataFrame:
        """
        Delete specific attributes and save the DataFrame if required.

        Args:
            df (pandas.DataFrame): Input DataFrame.
            delete_specific_attributes (bool): Whether to delete specific attributes.

        Returns:
            pandas.DataFrame: Modified DataFrame, or None if deletion or saving fails.
        """
        try:
            if delete_specific_attributes:
                df = self.data_editor.delete_specific_attributes(df)
                if df is None:
                    self.log_error("Failed to delete specific attributes.")
                    return None
                csv_file_specific = self.file_manager.save_csv(df, self.processed, prefix="organized_data_specific")
                parq_file_specific = self.file_manager.save_parq(df, self.processed, prefix="organized_data_specific")
                if csv_file_specific is None or parq_file_specific is None:
                    self.log_error("Failed to save organized DataFrame.")
                    return None
                logger.info(f"Saved organized DataFrame: CSV={csv_file_specific}, Parquet={parq_file_specific}")
            return df
        except Exception as e:
            self.log_error(f"Failed to delete attributes or save DataFrame: {e}")
            return None

    def run(self):
        try:
            data_feed = 'non_redundant'  # 'non_redundant' or 'redundant'
            plot_feed = 'global'  # 'both', 'per_group', or 'global'
            groupby_period = 'month'  # 'week', 'month', or 'year'
            delete_specific_attributes = False  # or True
            plot_type = 'tsne'  # 'both', 'pca', or 'tsne'

            # Validate inputs
            if not self.check_input(data_feed, plot_feed, groupby_period, delete_specific_attributes, plot_type):
                return None

            # Load and preprocess data
            df = self.load_and_preprocess_data(data_feed)
            if df is None:
                return None

            # Delete attributes and save if required
            df = self.delete_and_save_attributes(df, delete_specific_attributes)
            if df is None:
                return None

            # Build feature DataFrame
            feature_df = self.data_preparation.build_visual_no_message_content(df, groupby_period=groupby_period)
            if feature_df is None:
                return self.log_error("Failed to build non-message content feature DataFrame.")

            # Create visualizations
            figs = self.plot_manager.build_visual_no_message_content(
                feature_df,
                plot_type=plot_type,
                dr_settings=DimReductionSettings(),
                nmc_settings=PlotManagerNoMessageContentSettings(plot_type=plot_feed)
            )
            if figs is None:
                return self.log_error("Failed to create non-message content visualizations.")
            for fig_info in figs:
                self.save_figure(fig_info['fig'], self.image_dir, fig_info['filename'])

            # Create correlations plot if groupby_period is 'month'
            if groupby_period == 'month':
                correlations = self.data_preparation.compute_month_correlations(feature_df)
                if correlations is None:
                    return self.log_error("Failed to compute month correlations.")
                fig_correlations = self.plot_manager.plot_month_correlations(correlations)
                if fig_correlations is None:
                    return self.log_error("Failed to create month correlations plot.")
                return self.save_figure(fig_correlations, self.image_dir, "month_correlations")
            return None
        except Exception as e:
            logger.exception(f"Error in Script 11 - Non-Message Content Visualization: {e}")
            return None