import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer
from plot_manager import PlotSettings, CategoriesPlotSettings, TimePlotSettings, DistributionPlotSettings, NoMessageContentSettings as PlotManagerNoMessageContentSettings, DimReductionSettings, ArcPlotSettings, BubbleNewPlotSettings
from pydantic import BaseModel
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import copy
from constants import Columns, Groups, DataFeed, PlotFeed, GroupByPeriod, DeleteAttributes, PlotType
from datetime import datetime
import pytz
import tomllib

class BaseScript:
    """Base class for scripts with common functionality."""
    def __init__(self, file_manager, data_editor=None, data_preparation=None, plot_manager=None, settings: BaseModel = None, df=None, **kwargs):
        self.file_manager = file_manager
        self.data_editor = data_editor
        self.data_preparation = data_preparation
        self.plot_manager = plot_manager
        self.settings = settings or PlotSettings()
        self.df = df

    def save_figure(self, fig, image_dir: Path, filename: str) -> Optional[Path]:
        png_file = self.file_manager.save_png(fig, image_dir, filename=filename)
        if png_file is None:
            logger.error(f"Failed to save {filename}.")
        else:
            logger.info(f"Saved {filename}: {png_file}")
        return png_file

    def log_error(self, message: str):
        logger.error(message)
        return None

    def run(self):
        raise NotImplementedError("Subclasses must implement run()")

class Script0(BaseScript):
    """Script for: Reads raw WhatsApp files, preprocesses and processes these files."""
    
    def __init__(self, file_manager, data_editor, data_preparation, processed_dir, config, image_dir, scripts=None):
        super().__init__(file_manager, data_editor=data_editor, data_preparation=data_preparation)
        self.processed_dir = processed_dir
        self.config = config
        self.image_dir = image_dir
        self.dataframes = {}
        self.scripts = scripts or []

    def _setup_logging(self):
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now(tz=pytz.timezone('Europe/Amsterdam')).strftime("%Y%m%d-%H%M%S")
        log_file = log_dir / f"logfile-{timestamp}.log"
        logger.remove()
        logger.add(log_file, rotation=None, retention=None, level="DEBUG")
        logger.info(f"Logging configured: {log_file}")

    def _load_config(self):
        configfile = Path("config.toml").resolve()
        try:
            with configfile.open("rb") as f:
                config = tomllib.load(f)
            required_keys = ["image", "processed", "preprocess"]
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                logger.error(f"Missing keys in config.toml: {missing_keys}")
                raise ValueError(f"Invalid config.toml: missing keys {missing_keys}")
            self.config = config
            self.image_dir = Path(config["image"])
            self.processed_dir = Path(config["processed"])
            logger.info(f"Loaded config: {config}")
            logger.info(f"Config settings: image_dir={self.image_dir}, processed_dir={self.processed_dir}, preprocess={config['preprocess']}")
        except FileNotFoundError:
            logger.error("config.toml not found")
            self.config = None
            self.image_dir = None
            self.processed_dir = None
            raise
        except Exception as e:
            logger.error(f"Failed to load config.toml: {e}")
            self.config = None
            self.image_dir = None
            self.processed_dir = None
            raise

    def _add_columns(self, df):
        try:
            df['year'] = self.data_editor.get_year(df)
            df['month'] = self.data_editor.get_month(df)
            df['week'] = self.data_editor.get_week(df)
            df['day_of_week'] = self.data_editor.get_day_of_week(df)
            df['active_years'] = self.data_editor.active_years(df)
            df['early_leaver'] = self.data_editor.early_leaver(df)
            df = self.data_editor.mark_more_early_leavers(df)
            if df is None:
                raise ValueError("Failed to mark early leavers.")
            logger.debug(f"Added columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            logger.exception(f"Failed to add columns: {e}")
            return None

    def _compute_category_data(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Compute category data (authors per group) for scripts requiring group_authors.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'whatsapp_group' and 'author' columns.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping groups to sorted lists of unique authors, or None if failed.
        """
        try:
            group_authors = self.data_preparation.compute_group_authors(df)
            if group_authors is None:
                logger.error("Failed to compute group_authors in _compute_category_data")
                return None
            logger.info("Computed group_authors for scripts: {}".format(group_authors))
            return group_authors
        except Exception as e:
            logger.exception(f"Error in _compute_category_data: {e}")
            return None

    def run(self):
        try:
            self._setup_logging()
            self._load_config()
            logger.debug(f"Running Script0 with scripts: {self.scripts}")

            # Get CSV file(s), processed directory, group mapping, and Parquet files
            datafiles, processed, group_map, parq_files = self.file_manager.read_csv()
            logger.debug(f"Input files: {datafiles}")
            logger.debug(f"Group map: {group_map}")
            logger.debug(f"Parquet files: {parq_files}")

            # Check if any datafiles were returned
            if not datafiles or datafiles is None:
                return self.log_error("No valid data files were loaded. Exiting.")

            # Process files based on preprocess flag or Parquet availability
            if self.config["preprocess"] or not parq_files:
                logger.debug("Processing CSV files (preprocess=True or no valid Parquet files)")
                for datafile in datafiles:
                    if not datafile.exists():
                        return self.log_error(f"Input file does not exist: {datafile}")
                    df = self.data_editor.convert_timestamp(datafile)
                    if df is None:
                        return self.log_error("Failed to parse timestamp for {datafile}")
                    self.dataframes[datafile.stem] = df
                    if not self.dataframes:
                        return self.log_error("No DataFrames were created. Exiting.")
            else:
                # Load Parquet files
                logger.debug(f"Loading Parquet files: {parq_files}")
                for parq_file in parq_files:
                    try:
                        logger.debug(f"Loading {parq_file}")
                        df = pd.read_parquet(parq_file)
                        logger.info(f"Loaded {parq_file}: shape={df.shape}, rows={len(df)}")
                        self.dataframes[parq_file.stem] = df
                    except Exception as e:
                        logger.exception(f"Failed to load {parq_file}: {e}")
                        return None

            # Combine all DataFrames
            all_dfs = []
            for stem, df in self.dataframes.items():
                try:
                    # Map stem to group name using config.toml current_* values
                    group_name = None
                    for key, filename in self.config.items():
                        if key in ['current_1', 'current_2a', 'current_2b', 'current_3', 'current_4']:
                            if Path(filename).stem == stem:
                                group_name = group_map.get(key, None)
                                break
                    logger.debug(f"Mapping stem '{stem}' to group_name '{group_name}'")
                    if group_name is None:
                        logger.warning(f"No group mapping for {stem}. Skipping.")
                        continue
                    df["whatsapp_group"] = group_name
                    all_dfs.append(df)
                except Exception as e:
                    logger.exception(f"Error processing {stem}: {e}")
                    continue

            if not all_dfs:
                return self.log_error("No valid DataFrames to concatenate. Exiting.")

            df = pd.concat(all_dfs, ignore_index=True)
            logger.debug(f"Combined DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")

            # Filter group names
            df = self.data_editor.filter_group_names(df)
            if df is None:
                return self.log_error("Failed to filter group names.")

            # Add columns (year, month, week, day_of_week, active_years, early_leaver)
            df = self._add_columns(df)
            if df is None:
                return self.log_error("Failed to add columns.")

            # Clean messages for deleted media patterns (adds 'message_cleaned')
            df = self.data_editor.clean_for_deleted_media_patterns(df)
            if df is None:
                return self.log_error("Failed to clean messages for all groups.")

            # Mark additional early leavers
            df = self.data_editor.mark_more_early_leavers(df)
            if df is None:
                return self.log_error("Failed to mark early leavers.")

            # Add list_of_all_emojis column (now after mark_more_early_leavers)
            try:
                df['list_of_all_emojis'] = df['message_cleaned'].apply(self.data_editor.list_of_all_emojis)
                logger.debug("Added 'list_of_all_emojis' column")
                non_empty_emojis = df['list_of_all_emojis'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum()
                logger.debug(f"Number of messages with emojis: {non_empty_emojis}")
                logger.debug(f"Sample list_of_all_emojis: {df['list_of_all_emojis'].head().tolist()}")
            except Exception as e:
                logger.exception(f"Failed to add list_of_all_emojis column: {e}")
                return None

            # Save combined files
            logger.debug(f"DataFrame before saving: shape={df.shape}, columns={df.columns.tolist()}")
            csv_file, parq_file = self.file_manager.save_combined_files(df, self.processed_dir)
            if csv_file is None or parq_file is None:
                return self.log_error("Failed to save combined files.")
            logger.info(f"Saved combined files: CSV={csv_file}, Parquet={parq_file}")

            # Compute category data if needed for scripts
            if any(script in self.scripts for script in [1, 4, 5, 7]):
                group_authors = self._compute_category_data(df)
                if group_authors is None:
                    return self.log_error("Failed to compute group_authors for Scripts 1, 4, 5, or 7.")
            else:
                group_authors = None

            return {"df": df, "image_dir": self.image_dir, "processed_dir": self.processed_dir, "group_authors": group_authors}
        except Exception as e:
            logger.exception(f"Error in Script0: {e}")
            return None

class Script1(BaseScript):
    """Script for building visual categories and plotting bar chart."""
    def __init__(self, file_manager, data_preparation, plot_manager, image_dir, df):
        super().__init__(file_manager, data_preparation=data_preparation, plot_manager=plot_manager, df=df)
        self.image_dir = image_dir

    def run(self):
        try:
            logger.debug(f"Script1: Starting build_visual_categories with df shape: {self.df.shape}, columns: {self.df.columns.tolist()}")
            df, group_authors, non_anthony_group, anthony_group, sorted_groups = self.data_preparation.build_visual_categories(self.df)
            if df is None or group_authors is None or non_anthony_group is None or anthony_group is None or sorted_groups is None:
                return self.log_error("Failed to compute visual categories.")
            logger.debug(f"Script1: group_authors: {group_authors}")
            logger.debug(f"Script1: non_anthony_group:\n{non_anthony_group.to_string()}")
            logger.debug(f"Script1: anthony_group:\n{anthony_group.to_string()}")
            logger.debug(f"Script1: sorted_groups: {sorted_groups}")
            logger.debug(f"Script1: Calling build_visual_categories with settings: {CategoriesPlotSettings()}")
            fig = self.plot_manager.build_visual_categories(
                group_authors, non_anthony_group, anthony_group, sorted_groups,
                settings=CategoriesPlotSettings()
            )
            if fig is None:
                return self.log_error("Failed to create categories bar chart.")
            logger.debug(f"Script1: Figure created, saving to {self.image_dir}")
            saved_file = self.save_figure(fig, self.image_dir, "categories_bar_chart")
            if saved_file is None:
                return self.log_error("Failed to save categories bar chart.")
            logger.info(f"Script1: Saved plot: {saved_file}")
            return saved_file
        except Exception as e:
            logger.exception(f"Error in Script1 - Categories Bar Chart: {e}")
            return None

class Script2(BaseScript):
    """Script for creating time-based visualizations for the 'dac' group."""
    
    def __init__(self, file_manager, data_preparation, plot_manager, image_dir, df):
        super().__init__(file_manager, data_preparation=data_preparation, plot_manager=plot_manager, df=df)
        self.image_dir = image_dir
    
    def run(self):
        try:
            # Prepare time-based data for 'dac' group
            weekly_avg = self.data_preparation.compute_time_aggregates(self.df, group='dac')
            if weekly_avg is None:
                return self.log_error("Failed to prepare time-based data for 'dac'.")
            
            # Create weekly plot with green color
            settings = TimePlotSettings(
                title="Golf Season, Decoded by WhatsApp Heartbeat",
                line_color='green',  # Override for 'dac'
                rest_label="Rest",
                prep_label="Preparation",
                play_label="Play",
                vline_weeks=[14, 18, 35]  # 1st April, 1st May, 1st Sep
            )
            fig = self.plot_manager.build_visual_time(self.df, weekly_avg, settings)
            if fig is None:
                return self.log_error("Failed to create time plot for 'dac'.")
            
            # Save plot
            return self.save_figure(fig, self.image_dir, "time_line_chart_week_dac")
        except Exception as e:
            logger.exception(f"Error in Script 2 - Time Visualization: {e}")
            return None

class Script3(BaseScript):
    def __init__(self, file_manager, data_editor, data_preparation, plot_manager, image_dir, df):
        super().__init__(file_manager, data_editor=data_editor, data_preparation=data_preparation, plot_manager=plot_manager, df=df)
        self.image_dir = image_dir

    def run(self):
        try:
            df = self.df.copy()
            if df.empty:
                return self.log_error("No valid DataFrame for emoji distribution visualization")
            
            # Build visual distribution DataFrame (test with 'tillies')
            df_emojis = self.data_preparation.build_visual_distribution(df, group='tillies')
            if df_emojis is None:
                return self.log_error("Failed to build visual distribution DataFrame")
            logger.debug(f"df_emojis shape: {df_emojis.shape}, head:\n{df_emojis.head().to_string()}")
            
            # Create emoji distribution plot
            fig = self.plot_manager.build_visual_distribution(df_emojis)
            if fig is None:
                return self.log_error("Failed to create emoji distribution plot")
            
            # Save the figure
            return self.save_figure(fig, self.image_dir, "emoji_distribution_tillies")
        except Exception as e:
            logger.exception(f"Error in Script 3 - Emoji Distribution Visualization: {e}")
            return None

class Script3(BaseScript):
    """Script for: Build emoji distribution visualization."""
    def __init__(self, file_manager, data_editor, data_preparation, plot_manager, image_dir, df, settings: Optional[DistributionPlotSettings] = None):
        super().__init__(file_manager, data_editor=data_editor, data_preparation=data_preparation, plot_manager=plot_manager, settings=settings or DistributionPlotSettings())
        self.image_dir = image_dir
        self.df = df

    def run(self):
        try:
            df = self.df.copy()
            if df.empty:
                return self.log_error("No valid DataFrame for emoji distribution visualization")
            
            # Build visual distribution DataFrame (filtered to 'maap')
            df_emojis = self.data_preparation.build_visual_distribution(df, group='maap')
            if df_emojis is None:
                return self.log_error("Failed to build visual distribution DataFrame")
            
            # Create emoji distribution plot
            fig = self.plot_manager.build_visual_distribution(df_emojis)
            if fig is None:
                return self.log_error("Failed to create emoji distribution plot")
            
            # Save the figure
            return self.save_figure(fig, self.image_dir, "emoji_distribution_maap")
        except Exception as e:
            logger.exception(f"Error in Script 3 - Emoji Distribution Visualization: {e}")
            return None

class Script4(BaseScript):
    """Script for: Build relationships arc diagram."""
    def __init__(self, file_manager, data_preparation, plot_manager, image_dir, tables_dir, group_authors, original_df=None, settings: Optional[ArcPlotSettings] = None):
        super().__init__(file_manager, data_preparation=data_preparation, plot_manager=plot_manager, settings=settings or ArcPlotSettings(), df=original_df)
        self.image_dir = image_dir
        self.tables_dir = tables_dir
        self.group_authors = group_authors

    def build_participation_table(self, df_group: pd.DataFrame, group: str) -> pd.DataFrame:
        """
        Build and save the participation table for the arc diagram.

        This method constructs a participation table (e.g., message counts per author) from a
        filtered DataFrame and saves it to a table file. The table is specific to the provided
        WhatsApp group.

        Args:
            df_group (pandas.DataFrame): Filtered DataFrame for the specified group, where the
                'whatsapp_group' column matches the group value (e.g., filtered by Groups.MAAP.value).
            group (str): WhatsApp group name corresponding to an Enum value (e.g., Groups.MAAP.value).

        Returns:
            pd.DataFrame: Participation table DataFrame, or None if creation fails (e.g., empty
                input DataFrame or processing error).

        Raises:
            Exception: If data processing or file saving fails, logged via loguru and None is returned.

        Note:
            The table is saved to the 'tables_dir' using file_manager.save_table. Ensure this
            directory is accessible.
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
        """Execute the relationships arc diagram visualization."""
        try:
            group = Groups.MAAP.value
            # Use self.df (original_df) instead of data_preparation.df
            df_group = self.df[self.df[Columns.WHATSAPP_GROUP.value] == group].copy() if self.df is not None else \
                       self.data_preparation.df[self.data_preparation.df[Columns.WHATSAPP_GROUP.value] == group].copy()
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
    def __init__(self, file_manager, data_preparation, plot_manager, image_dir, df):
        super().__init__(file_manager, data_preparation=data_preparation, plot_manager=plot_manager, df=df)
        self.image_dir = image_dir

    def run(self):
        try:
            feature_df = self.data_preparation.build_interaction_features(self.df, groupby_period='year')
            if feature_df is None:
                return self.log_error("Failed to build interaction features.")
            figs = self.plot_manager.build_bubble_plot_new(
                feature_df, settings=BubbleNewPlotSettings()
            )
            if figs is None:
                return self.log_error("Failed to create bubble plots.")
            for fig_info in figs:
                self.save_figure(fig_info['fig'], self.image_dir, fig_info['filename'])
            return None
        except Exception as e:
            logger.exception(f"Error in Script5 - Bubble Plots: {e}")
            return None

class Script7(BaseScript):
    def __init__(self, file_manager, data_preparation, plot_manager, image_dir, group_authors, df):
        super().__init__(file_manager, data_preparation=data_preparation, plot_manager=plot_manager, df=df)
        self.image_dir = image_dir
        self.group_authors = group_authors

    def run(self):
        try:
            feature_df = self.data_preparation.build_interaction_features(self.df, groupby_period='year')
            if feature_df is None:
                return self.log_error("Failed to build interaction features.")
            figs = self.plot_manager.build_visual_interactions_2(feature_df, settings=DimReductionSettings(), nmc_settings=PlotManagerNoMessageContentSettings())
            if figs is None:
                return self.log_error("Failed to create interaction visualizations.")
            for i, fig in enumerate(figs):
                self.save_figure(fig, self.image_dir, f"interactions_{i+1}")
            return None
        except Exception as e:
            logger.exception(f"Error in Script7 - Interaction Visualizations: {e}")
            return None

class Script10(BaseScript):
    def __init__(self, file_manager, data_editor, data_preparation, processed_dir, image_dir):
        super().__init__(file_manager, data_editor=data_editor, data_preparation=data_preparation)
        self.processed = processed_dir
        self.image_dir = image_dir

    def check_input(self, data_feed: str, plot_feed: str, groupby_period: str, delete_specific_attributes: bool, plot_type: str) -> bool:
        if data_feed not in ['non_redundant', 'redundant']:
            self.log_error(f"Invalid data_feed: {data_feed}. Must be 'non_redundant' or 'redundant'.")
            return False
        if plot_feed not in ['both', 'per_group', 'global']:
            self.log_error(f"Invalid plot_feed: {plot_feed}. Must be 'both', 'per_group', or 'global'.")
            return False
        if data_feed == 'redundant' and plot_feed != 'per_group':
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

    def load_and_preprocess_data(self) -> pd.DataFrame:
        try:
            datafile = self.file_manager.find_latest_file(self.processed, prefix="whatsapp_all", suffix=".csv")
            if datafile is None:
                self.log_error("No whatsapp_all.csv found.")
                return None
            logger.info(f"Loading latest data from {datafile}")
            column_mapping = {}
            df = pd.read_csv(datafile, parse_dates=['timestamp'], date_parser=lambda x: pd.to_datetime(x, errors='coerce'))
            logger.debug(f"Timestamp column dtype: {df['timestamp'].dtype}")
            logger.debug(f"Timestamp sample: {df['timestamp'].head().to_list()}")
            if df['timestamp'].isna().any():
                logger.warning(f"Found {df['timestamp'].isna().sum()} NaT values in timestamp column")
            df = df.rename(columns=column_mapping)
            required_cols = [Columns.WHATSAPP_GROUP.value, Columns.YEAR.value, Columns.MONTH.value, Columns.WEEK.value, Columns.DAY_OF_WEEK.value]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.log_error(f"Missing columns in DataFrame: {missing_cols}")
                return None
            logger.debug(f"DataFrame after checking columns:\n{df[required_cols].head().to_string()}")
            df = df[df[Columns.WHATSAPP_GROUP.value] != Groups.TILLIES.value].copy()
            if df.empty:
                self.log_error(f"No data remains after filtering out '{Groups.TILLIES.value}' group.")
                return None
            logger.info(f"Filtered DataFrame excluding '{Groups.TILLIES.value}': {len(df)} rows")
            return df
        except Exception as e:
            logger.exception(f"Failed to load and preprocess data: {e}")
            return None

    def prepare_feature_dataframes(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            df_organized_no_redundancy = self.data_editor.organize_dataframe(df)
            if df_organized_no_redundancy is None:
                self.log_error("Failed to organize DataFrame (no redundancy).")
                return None, None
            
            logger.debug(f"Saving no_redundancy DataFrame: shape={df_organized_no_redundancy.shape}")
            csv_file_no_red = self.file_manager.save_csv(df_organized_no_redundancy, self.processed, prefix="organized_data_no_redundancy")
            parq_file_no_red = self.file_manager.save_parq(df_organized_no_redundancy, self.processed, prefix="organized_data_no_redundancy")
            if csv_file_no_red is None or parq_file_no_red is None:
                self.log_error("Failed to save organized DataFrame (no redundancy).")
                return None, None
            logger.info(f"Saved organized DataFrame (no redundancy): CSV={csv_file_no_red}, Parquet={parq_file_no_red}")
            
            df_organized_with_redundancy = df_organized_no_redundancy.copy()
            for group in df_organized_with_redundancy['whatsapp_group'].unique():
                group_df = df_organized_with_redundancy[df_organized_with_redundancy['whatsapp_group'] == group]
                group_df['whatsapp_group'] = 'all'
                df_organized_with_redundancy = pd.concat([df_organized_with_redundancy, group_df], ignore_index=True)
            logger.debug(f"Saving with_redundancy DataFrame: shape={df_organized_with_redundancy.shape}")
            csv_file_with_red = self.file_manager.save_csv(df_organized_with_redundancy, self.processed, prefix="organized_data_with_redundancy")
            parq_file_with_red = self.file_manager.save_parq(df_organized_with_redundancy, self.processed, prefix="organized_data_with_redundancy")
            if csv_file_no_red is None or parq_file_with_red is None:
                self.log_error("Failed to save organized DataFrame (with redundancy).")
                return None, None
            logger.info(f"Saved organized DataFrame (with redundancy): CSV={csv_file_with_red}, Parquet={parq_file_with_red}")
            
            return df_organized_no_redundancy, df_organized_with_redundancy
        except Exception as e:
            logger.exception(f"Failed to prepare feature DataFrames: {e}")
            return None, None

    def delete_and_save_attributes(self, df: pd.DataFrame, delete_specific_attributes: bool) -> pd.DataFrame:
        try:
            if delete_specific_attributes:
                df = self.data_editor.delete_specific_attributes(df)
                if df is None:
                    self.log_error("Failed to delete specific attributes.")
                    return None
                logger.debug(f"Saving specific DataFrame: shape={df.shape}")
                csv_file_specific = self.file_manager.save_csv(df, self.processed, prefix="organized_data_specific")
                parq_file_specific = self.file_manager.save_parq(df, self.processed, prefix="organized_data_specific")
                if csv_file_specific is None or parq_file_specific is None:
                    self.log_error("Failed to save organized DataFrame (specific).")
                    return None
                logger.info(f"Saved organized DataFrame: CSV={csv_file_specific}, Parquet={parq_file_specific}")
            return df
        except Exception as e:
            self.log_error(f"Failed to delete attributes or save DataFrame: {e}")
            return None

    def run(self):
        try:
            data_feed = DataFeed.NON_REDUNDANT.value
            plot_feed = PlotFeed.GLOBAL.value
            groupby_period = GroupByPeriod.MONTH.value
            delete_specific_attributes = DeleteAttributes.FALSE.value
            plot_type = PlotType.TSNE.value

            if not self.check_input(data_feed, plot_feed, groupby_period, delete_specific_attributes, plot_type):
                return None

            df = self.load_and_preprocess_data()
            if df is None:
                return None

            df_no_red, df_with_red = self.prepare_feature_dataframes(df)
            if df_no_red is None or df_with_red is None:
                return self.log_error("Failed to prepare feature DataFrames.")

            df = df_no_red if data_feed == 'non_redundant' else df_with_red
            df = self.delete_and_save_attributes(df, delete_specific_attributes)
            if df is None:
                return None

            feature_df = self.data_preparation.build_visual_no_message_content(df, groupby_period=groupby_period)
            if feature_df is None:
                return self.log_error("Failed to build non-message content feature DataFrame.")

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
            logger.exception(f"Error in Script 10 - Non-Message Content Visualization: {e}")
            return None

class Script11(BaseScript):
    def __init__(self, file_manager, data_editor, data_preparation, plot_manager, processed_dir, image_dir):
        super().__init__(file_manager, data_editor=data_editor, data_preparation=data_preparation, plot_manager=plot_manager)
        self.processed = processed_dir
        self.image_dir = image_dir

    def run(self):
        try:
            data_feed = DataFeed.NON_REDUNDANT.value
            plot_feed = PlotFeed.GLOBAL.value
            groupby_period = GroupByPeriod.MONTH.value
            delete_specific_attributes = DeleteAttributes.FALSE.value
            plot_type = PlotType.TSNE.value

            if not self.check_input(data_feed, plot_feed, groupby_period, delete_specific_attributes, plot_type):
                return None

            df = self.load_and_preprocess_data(data_feed)
            if df is None:
                return None

            df = self.delete_and_save_attributes(df, delete_specific_attributes)
            if df is None:
                return None

            feature_df = self.data_preparation.build_visual_no_message_content(df, groupby_period=groupby_period)
            if feature_df is None:
                return self.log_error("Failed to build non-message content feature DataFrame.")

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