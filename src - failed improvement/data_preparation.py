import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
from loguru import logger
import warnings
import matplotlib.font_manager as fm
import networkx as nx
import itertools
import emoji
import re
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple

class ThreadingSettings(BaseModel):
    """Settings for threading features."""
    married_couples: List[Tuple[str, str]] = []
    min_messages: int = 3
    max_lag: int = 5
    time_diff_threshold: float = 1.0  # hours

class InteractionSettings(BaseModel):
    """Settings for interaction features."""
    threading_min_length: int = 2

class NoMessageContentSettings(BaseModel):
    """Settings for non-message content features."""
    numerical_columns: List[str] = [
        'message_length', 'number_of_emojis'
    ]
    binary_columns: List[str] = [
        'has_link', 'was_deleted', 'has_emoji'
    ]
    sum_columns: List[str] = [
        'pictures_deleted', 'videos_deleted', 'audios_deleted', 'gifs_deleted',
        'stickers_deleted', 'documents_deleted', 'videonotes_deleted'
    ]
    optional_columns: List[str] = []

class BaseHandler:
    """Base class for data handlers with common behavior."""
    def __init__(self, settings: BaseModel):
        self.settings = settings

    def _log_debug(self, message: str):
        logger.debug(message)

    def _handle_empty_df(self, df: pd.DataFrame, context: str) -> pd.DataFrame:
        if df.empty:
            logger.error(f"Empty DataFrame in {context}.")
            return pd.DataFrame()
        return df

class DataPreparation(BaseHandler):
    def __init__(self, data_editor=None, threading_settings: ThreadingSettings = ThreadingSettings(),
                 int_settings: InteractionSettings = InteractionSettings(),
                 nmc_settings: NoMessageContentSettings = NoMessageContentSettings()):
        super().__init__(settings=nmc_settings)
        self.data_editor = data_editor
        self.threading_settings = threading_settings
        self.int_settings = int_settings
        self.nmc_settings = nmc_settings
        self.df = None

    def compute_group_authors(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Compute authors per WhatsApp group.

        Args:
            df (pandas.DataFrame): Input DataFrame with 'whatsapp_group' and 'author' columns.

        Returns:
            dict: Dictionary mapping each group to a sorted list of unique authors, or None if computation fails.
        """
        df = self._handle_empty_df(df, "group authors computation")
        if df.empty:
            return None

        try:
            group_authors = df.groupby("whatsapp_group")["author"].unique().to_dict()
            group_authors = {group: sorted(auths.tolist()) for group, auths in group_authors.items()}
            logger.info("Authors per WhatsApp group:")
            for group, auths in group_authors.items():
                logger.info(f"{group}: {auths}")
            return group_authors
        except Exception as e:
            logger.exception(f"Failed to compute group authors: {e}")
            return None

    def build_visual_categories(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]], pd.DataFrame, pd.DataFrame, List[str]]:
        df = self._handle_empty_df(df, "visual categories preparation")
        if df.empty:
            return df, None, None, None, None

        try:
            self.df = df.copy()
            logger.debug(f"Input DataFrame shape: {self.df.shape}, columns: {self.df.columns.tolist()}")

            # Log active years per author per group
            active_years = self.df.groupby(['whatsapp_group', 'author'])['year'].agg(['min', 'max']).reset_index()
            active_years['active_years'] = active_years.apply(lambda x: f"{x['min']}-{x['max']}", axis=1)
            logger.info("Active years per author per group:")
            logger.info(active_years[['whatsapp_group', 'author', 'active_years']].to_string())

            # Log early leavers
            early_leavers = self.df[self.df['early_leaver'] == True][['whatsapp_group', 'author']].drop_duplicates()
            logger.info("Authors who left early (max year < 2025 in July 2015 - July 2025):")
            logger.info(early_leavers.to_string() if not early_leavers.empty else "No authors left early.")

            # Get authors per group
            group_authors = self.df.groupby("whatsapp_group")["author"].unique().to_dict()
            group_authors = {group: sorted(auths.tolist()) for group, auths in group_authors.items()}
            logger.info("Authors per WhatsApp group:")
            for group, auths in group_authors.items():
                logger.info(f"{group}: {auths}")

            # Filter data for July 2015 - July 2025
            #filter_df = self.df[(self.df['timestamp'] >= '2015-07-01') & (self.df['timestamp'] <= '2025-07-31')]
            filter_df = self.df[(self.df['timestamp'] >= pd.to_datetime('2015-07-01', utc=True)) & (self.df['timestamp'] <= pd.to_datetime('2025-07-31', utc=True))]
            logger.debug(f"Filtered DataFrame (2015-07-01 to 2025-07-31): shape={filter_df.shape}")
            if filter_df.empty:
                logger.warning("No data remains after filtering for July 2015 - July 2025")
                return self.df, None, None, None, None

            # Calculate total messages per group for sorting
            group_total = filter_df.groupby('whatsapp_group').size().reset_index(name='total_messages')
            sorted_groups = group_total.sort_values('total_messages', ascending=False)['whatsapp_group'].tolist()
            logger.info(f"Sorted groups by total messages: {sorted_groups}")

            # Calculate average messages per non-Anthony author per group
            non_anthony = filter_df[filter_df['author'] != "Anthony van Tilburg"]
            non_anthony_counts = non_anthony.groupby(['whatsapp_group', 'author']).size().reset_index(name='messages')
            non_anthony_group = non_anthony_counts.groupby('whatsapp_group')['messages'].mean().reset_index(name='non_anthony_avg')
            non_anthony_authors_count = non_anthony_counts.groupby('whatsapp_group')['author'].nunique().reset_index(name='num_authors')
            non_anthony_group = non_anthony_group.merge(non_anthony_authors_count, on='whatsapp_group', how='left').fillna({'num_authors': 0, 'non_anthony_avg': 0})
            non_anthony_group = non_anthony_group.set_index('whatsapp_group').reindex(sorted_groups).reset_index().fillna({'non_anthony_avg': 0, 'num_authors': 0})
            logger.info(f"Non-Anthony average messages and author counts per group:\n{non_anthony_group.to_string()}")

            # Anthony messages per group
            anthony = filter_df[filter_df['author'] == "Anthony van Tilburg"]
            anthony_group = anthony.groupby('whatsapp_group').size().reset_index(name='anthony_messages')
            anthony_group = anthony_group.set_index('whatsapp_group').reindex(sorted_groups).reset_index().fillna({'anthony_messages': 0})
            logger.info(f"Anthony messages per group:\n{anthony_group.to_string()}")

            return self.df, group_authors, non_anthony_group, anthony_group, sorted_groups
        except Exception as e:
            logger.exception(f"Failed to build visual categories: {e}")
            return self.df, None, None, None, None

    def build_visual_time(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare time-based data for visualization for the 'dac' group only.

        Args:
            df (pd.DataFrame): Input DataFrame with 'timestamp', 'whatsapp_group', 'year', 'week'.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                - Original DataFrame (with 'year', 'week' columns).
                - weekly_avg: DataFrame with ['week', 'avg_messages'] (average messages per week for 'dac').
                - yearly_counts: DataFrame with ['year', 'total_messages'] (total messages per year for 'dac').
        """
        if df.empty:
            logger.error("Empty DataFrame provided for visual time preparation.")
            return df, None, None
        
        try:
            df = df.copy()
            # Filter for 'dac' group only
            dac_df = df[df['whatsapp_group'] == 'dac']
            if dac_df.empty:
                logger.warning("No data for 'dac' group in visual time preparation.")
                return df, None, None
            
            # Weekly data: group by year and week, then average across years for each week
            weekly_counts = dac_df.groupby(['year', 'week']).size().reset_index(name='message_count')
            weekly_avg = weekly_counts.groupby('week')['message_count'].mean().reset_index(name='avg_messages')
            
            # Yearly data: total messages per year
            yearly_counts = dac_df.groupby('year').size().reset_index(name='total_messages')
            
            logger.info(f"Weekly average messages for 'dac':\n{weekly_avg.to_string()}")
            logger.info(f"Yearly total messages for 'dac':\n{yearly_counts.to_string()}")
            return df, weekly_avg, yearly_counts
        except Exception as e:
            logger.exception(f"Failed to build visual time data: {e}")
            return df, None, None

    def compute_time_aggregates(self, df: pd.DataFrame, group: str = 'dac') -> pd.DataFrame:
        """
        Compute weekly average message counts (across years) for the specified group.
        
        Args:
            df (pandas.DataFrame): Input DataFrame with 'whatsapp_group', 'week', 'year', 'message'.
            group (str): WhatsApp group to filter (default: 'dac').
        
        Returns:
            pd.DataFrame: weekly_avg (week, avg_messages).
        """
        if df.empty:
            logger.error(f"No valid DataFrame provided for time aggregates in group '{group}'")
            return None
        try:
            df_filtered = df[df['whatsapp_group'] == group].copy()
            if df_filtered.empty:
                logger.warning(f"No data for group '{group}' in time aggregates.")
                return None
            
            # Weekly data: group by year and week, then average across years
            weekly_counts = df_filtered.groupby(['year', 'week']).size().reset_index(name='message_count')
            weekly_avg = weekly_counts.groupby('week')['message_count'].mean().reset_index(name='avg_messages')
            
            logger.info(f"Computed time aggregates for group '{group}': weekly_avg shape={weekly_avg.shape}")
            logger.debug(f"weekly_avg preview:\n{weekly_avg.head().to_string()}")
            return weekly_avg
        except Exception as e:
            logger.exception(f"Failed to compute time aggregates for group '{group}': {e}")
            return None

    def build_visual_distribution(self, df: pd.DataFrame, group: str = 'maap') -> pd.DataFrame:
        """
        Build a DataFrame with emoji distribution data for the specified group, using list_of_all_emojis.
        
        Args:
            df (pandas.DataFrame): Input DataFrame with 'whatsapp_group' and 'list_of_all_emojis' columns.
            group (str): WhatsApp group to filter (default: 'maap').
        
        Returns:
            pd.DataFrame: DataFrame with columns ['emoji', 'count', 'count_once', 'cumulative_percentage'].
        """
        if df.empty:
            logger.error("No valid DataFrame provided for distribution preparation")
            return None
        try:
            # Filter to specified group
            df_filtered = df[df['whatsapp_group'] == group].copy()
            if df_filtered.empty:
                logger.warning(f"No data for group '{group}' in distribution preparation.")
                return None
            
            # Check for required column
            if 'list_of_all_emojis' not in df_filtered.columns:
                logger.error("'list_of_all_emojis' column missing. Ensure it's added in Script0.")
                return None
            
            # Log sample for debugging
            logger.debug(f"Sample messages for '{group}': {df_filtered['message_cleaned'].head().tolist()}")
            logger.debug(f"Sample list_of_all_emojis for '{group}': {df_filtered['list_of_all_emojis'].head().tolist()}")
            
            # Count total emoji occurrences
            all_emojis = []
            for emojis_list in df_filtered['list_of_all_emojis']:
                if isinstance(emojis_list, list):
                    all_emojis.extend(emojis_list)
            
            if not all_emojis:
                logger.warning(f"No emojis found in group '{group}' messages.")
                return None
            
            emoji_counts = Counter(all_emojis)
            df_emojis = pd.DataFrame.from_dict(emoji_counts, orient='index', columns=['count']).reset_index()
            df_emojis.columns = ['emoji', 'count']
            
            # Count messages containing each emoji (count_once)
            unique_emojis = df_emojis['emoji'].unique()
            count_once_list = []
            for em in unique_emojis:
                count_once = df_filtered['list_of_all_emojis'].apply(
                    lambda l: em in l if isinstance(l, list) else False
                ).sum()
                count_once_list.append(count_once)
            df_emojis['count_once'] = count_once_list
            
            # Sort by count_once and compute cumulative percentage
            df_emojis = df_emojis.sort_values('count_once', ascending=False)
            total_count_once = df_emojis['count_once'].sum()
            if total_count_once == 0:
                logger.warning(f"All count_once values are zero for group '{group}'. Check emoji detection.")
                df_emojis['cumulative_percentage'] = 0.0
            else:
                df_emojis['cumulative_percentage'] = df_emojis['count_once'].cumsum() / total_count_once * 100
            
            logger.info(f"Built emoji distribution DataFrame for group '{group}' with {len(df_emojis)} unique emojis")
            logger.debug(f"Emoji distribution preview:\n{df_emojis.head().to_string()}")
            return df_emojis
        except Exception as e:
            logger.exception(f"Failed to build visual distribution for group '{group}': {e}")
            return None

    def build_visual_relationships_arc(self, df_group: pd.DataFrame, authors: List[str]) -> pd.DataFrame:
        """
        Analyze daily participation in a WhatsApp group and combine results into a single table.
        
        Args:
            df_group (pandas.DataFrame): Filtered DataFrame for the group with 'timestamp', 'author', 'message_cleaned', 'early_leaver' columns.
            authors (list): List of unique authors in the group.
        
        Returns:
            pd.DataFrame or None: Combined DataFrame with columns 'type', 'author', 'num_days', 'total_messages', '#participants', and author-specific columns.
        """
        if df_group.empty:
            logger.error("No valid DataFrame provided for relationships arc preparation")
            return None
        try:
            # Ensure timestamp is datetime and extract date
            df_group["timestamp"] = pd.to_datetime(df_group["timestamp"])
            df_group["date"] = df_group["timestamp"].dt.date
            # Compute message_length once
            df_group['message_length'] = df_group['message_cleaned'].apply(lambda x: len(str(x)) if isinstance(x, str) else 0)
            # Get sorted unique authors
            authors = sorted(set(df_group['author']))
            # Calculate overall totals
            total_messages = len(df_group)
            message_counts = df_group.groupby('author').size()
            message_percentages = (message_counts / total_messages * 100).round(0).astype(int)
            total_length = df_group['message_length'].sum()
            length_counts = df_group.groupby('author')['message_length'].sum()
            length_percentages = (length_counts / total_length * 100).round(0).astype(int)
            avg_message_length = (length_counts / message_counts).round(0).astype(int).fillna(0)
            # Daily message counts per author
            daily_counts = df_group.groupby(["date", "author"]).size().unstack(fill_value=0).reindex(columns=authors, fill_value=0)
            # Total messages per day
            daily_total = daily_counts.sum(axis=1)
            # Number of participants per day using the helper method
            num_unique_series = self.data_editor.number_of_unique_participants_that_day(df_group)
            daily_participants = num_unique_series.groupby(df_group['date']).first()
            # Overall period
            min_date = df_group["date"].min()
            max_date = df_group["date"].max()
            total_days = (max_date - min_date).days + 1 if min_date and max_date else 0
            days_with_messages = len(daily_counts)
            days_no_messages = total_days - days_with_messages
            # Helper function to compute author values
            def _compute_author_values(active_df: pd.DataFrame, active_authors: List[str]) -> Dict[str, str]:
                active_message_counts = active_df[active_df['author'].isin(active_authors)].groupby('author').size()
                active_total_messages = active_message_counts.sum()
                message_pct = (active_message_counts / active_total_messages * 100).round(0).astype(int) if active_total_messages > 0 else pd.Series()
                active_length_counts = active_df[active_df['author'].isin(active_authors)].groupby('author')['message_length'].sum()
                active_total_length = active_length_counts.sum()
                length_pct = (active_length_counts / active_total_length * 100).round(0).astype(int) if active_total_length > 0 else pd.Series()
                active_avg_length = (active_length_counts / active_message_counts).round(0).astype(int).fillna(0)
                author_values = {author: 0 for author in authors}
                for p in active_authors:
                    msg_pct = message_pct.get(p, 0)
                    len_pct = length_pct.get(p, 0)
                    avg_len = active_avg_length.get(p, 0)
                    author_values[p] = f"{msg_pct}%/{len_pct}%({avg_len})"
                return author_values
            # Initialize list for combined table
            combined_data = []
            # Messages (%) row
            msg_row = {
                "type": "Messages (%)",
                "author": None,
                "num_days": 0,
                "total_messages": 0,
                "#participants": 0
            }
            for author in authors:
                msg_pct = message_percentages.get(author, 0)
                avg_len = avg_message_length.get(author, 0)
                msg_row[author] = f"{msg_pct}%/{avg_len}"
            combined_data.append(msg_row)
            # Message Length (%) row
            len_row = {
                "type": "Message Length (%)",
                "author": None,
                "num_days": 0,
                "total_messages": 0,
                "#participants": 0
            }
            for author in authors:
                len_pct = length_percentages.get(author, 0)
                avg_len = avg_message_length.get(author, 0)
                len_row[author] = f"{len_pct}%/{avg_len}"
            combined_data.append(len_row)
            # Period (overall)
            combined_data.append({
                "type": "Period",
                "author": None,
                "num_days": total_days,
                "total_messages": 0,
                "#participants": 0,
                **{author: 0 for author in authors}
            })
            combined_data.append({
                "type": "Period",
                "author": "None",
                "num_days": days_no_messages,
                "total_messages": 0,
                "#participants": 0,
                **{author: 0 for author in authors}
            })
            # Days with only 1 participant, details per author
            for author in authors:
                other_authors = [a for a in authors if a != author]
                mask = (daily_counts[author] > 0) & (daily_counts[other_authors] == 0).all(axis=1)
                num_days = mask.sum()
                total_msg = daily_total[mask].sum() if num_days > 0 else 0
                if num_days > 0:
                    active_dates = daily_counts[mask].index
                    active_df = df_group[df_group['date'].isin(active_dates)]
                    author_values = _compute_author_values(active_df, [author])
                else:
                    author_values = {a: 0 for a in authors}
                combined_data.append({
                    "type": "Single",
                    "author": author,
                    "num_days": num_days,
                    "total_messages": total_msg,
                    "#participants": 1,
                    **author_values
                })
            # Days with only 2 participants, details per combination
            for comb in itertools.combinations(authors, 2):
                pair_str = " & ".join(sorted(comb))
                other_authors = [a for a in authors if a not in comb]
                mask = (daily_counts[list(comb)] > 0).all(axis=1) & (daily_counts[other_authors] == 0).all(axis=1)
                num_days = mask.sum()
                total_msg = daily_total[mask].sum() if num_days > 0 else 0
                if num_days > 0:
                    active_dates = daily_counts[mask].index
                    active_df = df_group[df_group['date'].isin(active_dates)]
                    author_values = _compute_author_values(active_df, list(comb))
                else:
                    author_values = {a: 0 for a in authors}
                combined_data.append({
                    "type": "Pairs",
                    "author": pair_str,
                    "num_days": num_days,
                    "total_messages": total_msg,
                    "#participants": 2,
                    **author_values
                })
            # Days with N-1 participants, details per non-participant
            for non_part in authors:
                participants = [a for a in authors if a != non_part]
                mask = (daily_counts[participants] > 0).all(axis=1) & (daily_counts[non_part] == 0)
                num_days = mask.sum()
                total_msg = daily_total[mask].sum() if num_days > 0 else 0
                if num_days > 0:
                    active_dates = daily_counts[mask].index
                    active_df = df_group[df_group['date'].isin(active_dates)]
                    author_values = _compute_author_values(active_df, participants)
                else:
                    author_values = {a: 0 for a in authors}
                combined_data.append({
                    "type": "Non-participant",
                    "author": non_part,
                    "num_days": num_days,
                    "total_messages": total_msg,
                    "#participants": len(authors) - 1,
                    **author_values
                })
            # Days with all participants
            mask = (daily_counts > 0).all(axis=1)
            num_days = mask.sum()
            total_msg = daily_total[mask].sum() if num_days > 0 else 0
            if num_days > 0:
                active_dates = daily_counts[mask].index
                active_df = df_group[df_group['date'].isin(active_dates)]
                author_values = _compute_author_values(active_df, authors)
            else:
                author_values = {a: 0 for a in authors}
            combined_data.append({
                "type": "All",
                "author": "All",
                "num_days": num_days,
                "total_messages": total_msg,
                "#participants": len(authors),
                **author_values
            })
            # Create combined DataFrame
            combined_df = pd.DataFrame(combined_data)
            combined_df = combined_df.sort_values(by=["#participants", "num_days"], ascending=[True, False])
            logger.info(f"Combined participation table for group {df_group['whatsapp_group'].iloc[0]}:\n{combined_df.to_string(index=False)}")
            return combined_df
        except Exception as e:
            logger.exception(f"Failed to prepare relationships arc data: {e}")
            return None

    def build_interaction_features(self, df: pd.DataFrame, groupby_period: str = 'year') -> pd.DataFrame:
        if df.empty:
            logger.error("No valid DataFrame provided for interaction features.")
            return None
        try:
            if groupby_period not in ['week', 'month', 'year']:
                logger.error(f"Invalid groupby_period: {groupby_period}. Must be 'week', 'month', or 'year'.")
                return None
            required_columns = ['author', 'year', 'whatsapp_group', 'previous_author']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns: {[col for col in required_columns if col not in df.columns]}")
                return None
            feature_list = []
            all_authors = sorted(df['author'].unique())
            for (author, year), sub_df in df.groupby(['author', groupby_period]):
                features = {'author_year': f"{author}_{year}"}
                mention_counts = defaultdict(int)
                for _, row in sub_df.iterrows():
                    message = row.get('message_cleaned', '')
                    if isinstance(message, str):
                        for tgt in all_authors:
                            if tgt != author and tgt in message:
                                mention_counts[tgt] += 1
                total_mentions = sum(mention_counts.values())
                for tgt in all_authors:
                    features[f'mention_{tgt.replace(" ", "_")}'] = mention_counts[tgt] / total_mentions if total_mentions > 0 else 0.0
                year_df = df[df['year'] == year].copy()
                G = nx.Graph()
                for _, row in year_df[year_df['previous_author'].notna()].iterrows():
                    G.add_edge(row['author'], row['previous_author'])
                if len(G) > 0 and author in G:
                    features['degree_centrality'] = nx.degree_centrality(G)[author]
                    features['betweenness_centrality'] = nx.betweenness_centrality(G)[author]
                    features['closeness_centrality'] = nx.closeness_centrality(G)[author]
                else:
                    features['degree_centrality'] = 0.0
                    features['betweenness_centrality'] = 0.0
                    features['closeness_centrality'] = 0.0
                features['num_groups_participated'] = sub_df['whatsapp_group'].nunique()
                total_msgs = len(sub_df)
                features['avg_msgs_per_group'] = total_msgs / features['num_groups_participated'] if features['num_groups_participated'] > 0 else 0.0
                threading_features = self._compute_threading_features(sub_df, author, self.threading_settings)
                features.update(threading_features)
                feature_list.append(features)
            
            feature_df = pd.DataFrame(feature_list).set_index('author_year')
            logger.info(f"Built interaction feature matrix with shape {feature_df.shape}")
            logger.debug(f"Feature matrix columns: {feature_df.columns.tolist()}")
            logger.debug(f"Feature matrix preview:\n{feature_df.head().to_string()}")
            return feature_df
        except Exception as e:
            logger.exception(f"Failed to build interaction features: {e}")
            return None

    def _compute_threading_features(self, df: pd.DataFrame, author: str, threading_settings: ThreadingSettings) -> dict:
        """
        Compute threading features for a given author.
        Used by build_interaction_features.

        Args:
            df (pandas.DataFrame): Input DataFrame with message data.
            author (str): Author name to compute features for.
            threading_settings (ThreadingSettings): Settings for threading computation.

        Returns:
            dict: Dictionary of threading features.
        """
        features = {}
        try:
            min_messages = threading_settings.min_messages
            max_lag = threading_settings.max_lag
            time_diff_threshold = threading_settings.time_diff_threshold
            df_author = df[df['author'] == author].sort_values('timestamp')
            sequence_count = 0
            for i in range(len(df_author) - min_messages + 1):
                sequence = df_author.iloc[i:i+min_messages]
                if len(sequence) == min_messages:
                    time_diffs = (sequence['timestamp'].diff().dt.total_seconds() / 3600).dropna()
                    if all(time_diffs <= time_diff_threshold):
                        sequence_count += 1
                features['threading_sequences'] = sequence_count
            logger.debug(f"Computed threading features for {author}: {features}")
            return features
        except Exception as e:
            logger.warning(f"Failed to compute threading features for {author}: {e}")
            return features

    def build_visual_no_message_content(self, df: pd.DataFrame, groupby_period: str = 'week') -> pd.DataFrame:
        if df.empty:
            logger.error("No valid DataFrame provided for non-message content preparation")
            return None
        try:
            if groupby_period not in ['week', 'month', 'year']:
                logger.error("Invalid groupby_period.")
                return None
            if groupby_period not in df.columns:
                logger.error(f"Column '{groupby_period}' not found.")
                return None
            required_columns = ['author', 'year', 'whatsapp_group']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns.")
                return None
            available_numerical = [col for col in self.nmc_settings.numerical_columns if col in df.columns]
            available_binary = [col for col in self.nmc_settings.binary_columns if col in df.columns]
            available_sum = [col for col in self.nmc_settings.sum_columns if col in df.columns]
            available_optional = [col for col in self.nmc_settings.optional_columns if col in df.columns]
            if not (available_numerical or available_binary or available_sum or available_optional):
                logger.error("No feature columns available.")
                return None
            feature_list = []
            groupby_cols = ['author', groupby_period, 'year', 'whatsapp_group'] if groupby_period != 'year' else ['author', 'year', 'whatsapp_group']
            for group_values, sub_group in df.groupby(groupby_cols):
                total_messages = len(sub_group)
                if total_messages == 0:
                    continue
                if groupby_period == 'year':
                    author, year, group = group_values
                    period_value = year
                else:
                    author, period_value, year, group = group_values
                features = {
                    'author_period_year_group': f"{author}_{period_value:02d}_{year}_{group}",
                    'author': author,
                    groupby_period: period_value,
                    'year': year,
                    'whatsapp_group': group,
                    'total_messages': total_messages
                }
                for col in available_numerical:
                    features[f'mean_{col}'] = sub_group[col].mean()
                for col in available_binary:
                    features[f'proportion_{col}'] = (sub_group[col] == 1).sum() / total_messages
                for col in available_sum:
                    features[f'sum_{col}'] = sub_group[col].sum()
                for col in available_optional:
                    features[f'mean_{col}'] = sub_group[col].mean() if col in sub_group.columns else 0.0
                feature_list.append(features)
            feature_df = pd.DataFrame(feature_list).set_index('author_period_year_group')
            logger.info(f"Built non-message content feature matrix with shape {feature_df.shape}")
            logger.debug(f"Feature matrix columns: {feature_df.columns.tolist()}")
            logger.debug(f"Feature matrix preview:\n{feature_df.head().to_string()}")
            return feature_df
        except Exception as e:
            logger.exception(f"Failed to build non-message content features: {e}")
            return None

    def compute_month_correlations(self, feature_df: pd.DataFrame) -> pd.Series:
        if feature_df.empty:
            logger.error("No valid feature DataFrame for month correlations")
            return None
        try:
            if 'month' not in feature_df.columns:
                logger.error("Column 'month' not found.")
                return None
            numerical_cols = [col for col in feature_df.columns 
                             if col not in ['author', 'month', 'year', 'whatsapp_group'] 
                             and feature_df[col].dtype in ['int64', 'float64']]
            if not numerical_cols:
                logger.warning("No numerical columns for correlation.")
                return None
            correlations = feature_df[numerical_cols + ['month']].corr()['month'].drop('month')
            logger.info(f"Computed correlations with 'month':\n{correlations.to_string()}")
            return correlations
        except Exception as e:
            logger.exception(f"Failed to compute month correlations: {e}")
            return None