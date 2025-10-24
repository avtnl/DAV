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

class VisualSettings(BaseModel):
    """Settings for visual preparations."""
    golf_keywords: List[str] = [
        'tee', 'birdie', 'bogey', 'bunker', 'caddie', 'chip', 'divot', 'draw', 'driver', 'eagle',
        'fade', 'fairway', 'green', 'handicap', 'hcp', 'hole', 'hook', 'hybrid', 'lie', 'par',
        'putter', 'qualifying', 'rough', 'slice', 'stroke', 'stablefore', 'stbf', 'stb', 'wedge',
        'irons', 'masters', 'ryder', 'coupe', 'troffee', 'wedstrijd', 'majors', 'flag_in_hole',
        'person_golfing', 'man_golfing', 'woman_golfing', 'trophy', 'white_circle'
    ]

class SequenceSettings(BaseModel):
    """Settings for sequence handling."""
    gender_map: Dict[str, str] = {
        'Anthony van Tilburg': 'M',
        'Phons Berkemeijer': 'M',
        'Anja Berkemeijer': 'F',
        'Madeleine': 'F'
    }
    married_couples: List[Tuple[str, str]] = []
    min_messages: int = 3
    max_lag: int = 5
    time_diff_threshold: float = 1.0  # hours

class InteractionSettings(BaseModel):
    """Settings for interaction features."""
    threading_min_length: int = 2

class NonMessageContentSettings(BaseModel):
    """Settings for non-message content features."""
    numerical_columns: List[str] = [
        'length_chat', 'response_time', 'number_of_emojis', 'number_of_punctuations',
        'pct_emojis'
    ]
    binary_columns: List[str] = [
        'has_link', 'was_deleted', 'has_emoji', 'ends_with_emoji',
        'has_punctuation', 'ends_with_punctuation'
    ]
    sum_columns: List[str] = [
        'pictures_deleted', 'videos_deleted', 'audios_deleted', 'gifs_deleted',
        'stickers_deleted', 'documents_deleted', 'videonotes_deleted'
    ]
    optional_columns: List[str] = [
        'number_of_chats_that_day', 'day_pct_length_chat', 'day_pct_length_emojis',
        'day_pct_length_punctuations', 'number_of_unique_participants_that_day'
    ]

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

class DataPreparation:
    """A class for preparing WhatsApp message data for visualization."""

    def __init__(self, data_editor=None, seq_settings: SequenceSettings = SequenceSettings(), int_settings: InteractionSettings = InteractionSettings(), nmc_settings: NonMessageContentSettings = NonMessageContentSettings(), vis_settings: VisualSettings = VisualSettings()):
        self.data_editor = data_editor
        self.seq_settings = seq_settings
        self.int_settings = int_settings
        self.nmc_settings = nmc_settings
        self.vis_settings = vis_settings
        self.df = None  # Added to store the DataFrame for use in scripts

    def build_visual_categories(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]], pd.DataFrame, pd.DataFrame, List[str]]:
        """Prepare data categories for visualization, including group authors, non-Anthony averages, Anthony's counts, and sorted groups."""
        if df.empty:
            logger.error("Empty DataFrame provided for visual categories.")
            return df, None, None, None, None
        try:
            self.df = df  # Store the DataFrame
            # Group by whatsapp_group to get authors
            group_authors = df.groupby('whatsapp_group')['author'].apply(list).to_dict()
            logger.debug(f"Group authors: {group_authors}")

            # Calculate non-Anthony averages and Anthony's counts
            non_anthony_df = df[df['author'] != 'Anthony van Tilburg'].groupby('whatsapp_group').agg(
                non_anthony_avg=('message', 'count'),
                num_authors=('author', 'nunique')
            ).reset_index()
            anthony_df = df[df['author'] == 'Anthony van Tilburg'].groupby('whatsapp_group').agg(
                anthony_messages=('message', 'count')
            ).reset_index().fillna(0)

            # Merge to align groups
            merged_df = non_anthony_df.merge(anthony_df, on='whatsapp_group', how='left').fillna({'anthony_messages': 0})
            # Compute total messages and sort
            merged_df['total_messages'] = merged_df['non_anthony_avg'] + merged_df['anthony_messages']
            sorted_df = merged_df.sort_values(by='total_messages', ascending=False)
            non_anthony_group = sorted_df[['whatsapp_group', 'non_anthony_avg', 'num_authors', 'anthony_messages']]
            anthony_group = sorted_df[['whatsapp_group', 'anthony_messages', 'non_anthony_avg', 'num_authors']]
            sorted_groups = sorted_df['whatsapp_group'].tolist()

            logger.info(f"Sorted groups by total messages: {sorted_groups}")
            return df, group_authors, non_anthony_group, anthony_group, sorted_groups
        except Exception as e:
            logger.exception(f"Failed to build visual categories: {e}")
            return df, None, None, None, None

    def build_visual_time(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare time-based data for visualization (e.g., average messages per week for 'dac' group)."""
        if df.empty:
            logger.error("Empty DataFrame provided for visual time preparation.")
            return df, None, None
        try:
            self.df = df  # Store the DataFrame
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['year'] = df['timestamp'].dt.year
            df['isoweek'] = df['timestamp'].dt.isocalendar().week
            p = df.groupby(['year', 'isoweek']).size().reset_index(name='message_count')
            average_all = p.groupby('isoweek')['message_count'].mean().reset_index(name='avg_count_all')
            logger.info(f"Prepared time-based data with {len(p)} rows and average data with {len(average_all)} rows.")
            return df, p, average_all
        except Exception as e:
            logger.exception(f"Failed to build visual time data: {e}")
            return df, None, None

    def build_visual_distribution(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare emoji distribution data for visualization (e.g., for 'maap' group)."""
        if df.empty:
            logger.error("Empty DataFrame provided for visual distribution preparation.")
            return df, None
        try:
            self.df = df  # Store the DataFrame
            # Extract emojis from messages
            df['emojis'] = df['message'].apply(lambda x: ''.join(c for c in str(x) if c in emoji.EMOJI_DATA))
            emoji_counts = Counter([e for msg_emojis in df['emojis'] for e in msg_emojis])
            emoji_counts_df = pd.DataFrame({
                'emoji': list(emoji_counts.keys()),
                'count_once': list(emoji_counts.values()),
                'percent_once': [count / len(df) * 100 for count in emoji_counts.values()]
            })
            emoji_counts_df = emoji_counts_df.sort_values(by='count_once', ascending=False)
            logger.info(f"Prepared emoji distribution data with {len(emoji_counts_df)} unique emojis.")
            return df, emoji_counts_df
        except Exception as e:
            logger.exception(f"Failed to build visual distribution data: {e}")
            return df, None

    def build_visual_relationships_arc(self,
                                    df: pd.DataFrame,
                                    group_authors: List[str]) -> pd.DataFrame:
        """Prepare participation table **and** return the network figure."""
        if df.empty:
            logger.error("Empty DataFrame supplied to build_visual_relationships_arc.")
            return None

        try:
            # -------------------------------------------------
            # 1. Build the participation table (same as before)
            # -------------------------------------------------
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            # count transitions between every pair of consecutive authors
            transitions = Counter(zip(df['author'].shift(-1), df['author']))
            participation_df = pd.DataFrame({
                'Pairs': [f"{prev}-{next}" for prev, next in transitions.keys()],
                'Count': list(transitions.values())
            })
            logger.info(
                f"Prepared relationships arc data with {len(participation_df)} pairs."
            )

            # -------------------------------------------------
            # 2. **Create the network figure** (the plot manager does the drawing)
            # -------------------------------------------------
            # The PlotManager already knows how to turn the table into a figure.
            # We simply forward the call and return the figure together with the table.
            fig = self.plot_manager.build_visual_relationships_arc(
                participation_df,
                df['whatsapp_group'].iloc[0],   # group name
                PlotSettings()                  # default settings – can be overridden later
            )

            # Return **both** the table (for saving) and the figure (for saving the PNG)
            return participation_df, fig

        except Exception as e:
            logger.exception(f"Failed to build visual relationships arc data: {e}")
            return None

    def build_visual_relationships_bubble(self, df: pd.DataFrame, groups: List[str] = None) -> pd.DataFrame:
        """
        Prepare data for a bubble plot: average words vs average punctuations per message,
        with bubble size as number of messages, split by group and has_emoji.
       
        Args:
            df (pandas.DataFrame): The full DataFrame with WhatsApp data.
            groups (list, optional): List of 2 group names to include. Defaults to first 2 unique groups.
       
        Returns:
            pandas.DataFrame: Aggregated data with columns: whatsapp_group, author, has_emoji,
                            message_count, avg_words, avg_punct.
        """
        try:
            if df.empty:
                logger.error("Empty DataFrame provided for visual relationships bubble preparation.")
                return None
            self.df = df  # Store the DataFrame
            if groups is None:
                groups = df['whatsapp_group'].unique()[:2]
            df_filtered = df[df['whatsapp_group'].isin(groups)].copy()
            
            # Use data_editor methods for word and punctuation counting
            df_filtered['word_count'] = df_filtered['message_cleaned'].apply(self.data_editor.count_words)
            df_filtered['punct_count'] = df_filtered['message_cleaned'].apply(self.data_editor.count_punctuations)
            
            agg_df = df_filtered.groupby(['whatsapp_group', 'author', 'has_emoji']).agg(
                message_count=('message_cleaned', 'count'),
                avg_words=('word_count', 'mean'),
                avg_punct=('punct_count', 'mean')
            ).reset_index()
            
            logger.info(f"Prepared bubble plot data for groups {groups}:\n{agg_df.to_string(index=False)}")
            return agg_df
        except Exception as e:
            logger.exception(f"Failed to prepare bubble plot data: {e}")
            return None

    def build_interaction_features(self, df: pd.DataFrame, group_authors: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Build a feature matrix for interaction and network dynamics analysis.
        Features include normalized reply frequencies, mention frequencies, centrality measures,
        and cross-group participation. Computed per author-year, with separate rows for
        'Anthony van Tilburg' for each group per year and an overall row across all groups.
    
        Args:
            df (pandas.DataFrame): Full cleaned DataFrame with 'whatsapp_group' column.
            group_authors (dict): Dictionary of group names to lists of authors.
    
        Returns:
            pandas.DataFrame: Feature matrix with 'author_year' or 'author_year_group' index and 'whatsapp_group' column.
        """
        try:
            if df.empty:
                logger.error("Empty DataFrame provided for interaction features.")
                return None
            self.df = df  # Store the DataFrame
            all_authors = sorted(set(a for authors in group_authors.values() for a in authors))
            first_names = {a: a.split()[0] for a in all_authors}
            first_name_to_author = {v: k for k, v in first_names.items()}
            
            df = df.copy()
            df['year'] = self.data_editor.get_year(df)
            df = df.sort_values(['whatsapp_group', 'timestamp'])
            df['prev_author'] = df.groupby('whatsapp_group')['author'].shift(1)
            df['mentions'] = df['message'].apply(lambda x: self.data_editor.extract_mentions(x, first_name_to_author))
            feature_list = []
            
            # Group by author and year
            for (author, year), author_year_df in df.groupby(['author', 'year']):
                if author == 'Anthony van Tilburg':
                    # Create separate rows for each group and an overall row
                    for group in author_year_df['whatsapp_group'].unique():
                        sub_df = author_year_df[author_year_df['whatsapp_group'] == group]
                        if sub_df.empty:
                            continue
                        features = {
                            'author_year': f"{author}_{year}_{group}",
                            'whatsapp_group': group
                        }
                        reply_counts = sub_df[sub_df['prev_author'].notna()]['prev_author'].value_counts(normalize=True)
                        for tgt in all_authors:
                            features[f'reply_to_{tgt.replace(" ", "_")}'] = reply_counts.get(tgt, 0.0)
                        mention_flat = [m for mentions in sub_df['mentions'] for m in mentions]
                        mention_counts = Counter(mention_flat)
                        total_mentions = sum(mention_counts.values())
                        for tgt in all_authors:
                            features[f'mention_{tgt.replace(" ", "_")}'] = mention_counts[tgt] / total_mentions if total_mentions > 0 else 0.0
                        year_df = df[df['year'] == year].copy()
                        G = nx.Graph()
                        for _, row in year_df[year_df['prev_author'].notna()].iterrows():
                            G.add_edge(row['author'], row['prev_author'])
                        if len(G) > 0 and author in G:
                            features['degree_centrality'] = nx.degree_centrality(G)[author]
                            features['betweenness_centrality'] = nx.betweenness_centrality(G)[author]
                            features['closeness_centrality'] = nx.closeness_centrality(G)[author]
                        else:
                            features['degree_centrality'] = 0.0
                            features['betweenness_centrality'] = 0.0
                            features['closeness_centrality'] = 0.0
                        features['num_groups_participated'] = author_year_df['whatsapp_group'].nunique()
                        total_msgs = len(sub_df)
                        features['avg_msgs_per_group'] = total_msgs / features['num_groups_participated'] if features['num_groups_participated'] > 0 else 0.0
                        threading_features = self._compute_threading_features(sub_df, author)
                        features.update(threading_features)
                        feature_list.append(features)
                    
                    # Add overall row for Anthony
                    sub_df = author_year_df
                    features = {
                        'author_year': f"{author}_{year}",
                        'whatsapp_group': 'overall'
                    }
                    reply_counts = sub_df[sub_df['prev_author'].notna()]['prev_author'].value_counts(normalize=True)
                    for tgt in all_authors:
                        features[f'reply_to_{tgt.replace(" ", "_")}'] = reply_counts.get(tgt, 0.0)
                    mention_flat = [m for mentions in sub_df['mentions'] for m in mentions]
                    mention_counts = Counter(mention_flat)
                    total_mentions = sum(mention_counts.values())
                    for tgt in all_authors:
                        features[f'mention_{tgt.replace(" ", "_")}'] = mention_counts[tgt] / total_mentions if total_mentions > 0 else 0.0
                    year_df = df[df['year'] == year].copy()
                    G = nx.Graph()
                    for _, row in year_df[year_df['prev_author'].notna()].iterrows():
                        G.add_edge(row['author'], row['prev_author'])
                    if len(G) > 0 and author in G:
                        features['degree_centrality'] = nx.degree_centrality(G)[author]
                        features['betweenness_centrality'] = nx.betweenness_centrality(G)[author]
                        features['closeness_centrality'] = nx.closeness_centrality(G)[author]
                    else:
                        features['degree_centrality'] = 0.0
                        features['betweenness_centrality'] = 0.0
                        features['closeness_centrality'] = 0.0
                    features['num_groups_participated'] = author_year_df['whatsapp_group'].nunique()
                    total_msgs = len(sub_df)
                    features['avg_msgs_per_group'] = total_msgs / features['num_groups_participated'] if features['num_groups_participated'] > 0 else 0.0
                    threading_features = self._compute_threading_features(sub_df, author)
                    features.update(threading_features)
                    feature_list.append(features)
                else:
                    # For other authors, use primary group (most messages)
                    group_counts = author_year_df['whatsapp_group'].value_counts()
                    primary_group = group_counts.index[0] if not group_counts.empty else 'unknown'
                    sub_df = author_year_df[author_year_df['whatsapp_group'] == primary_group]
                    features = {
                        'author_year': f"{author}_{year}",
                        'whatsapp_group': primary_group
                    }
                    reply_counts = sub_df[sub_df['prev_author'].notna()]['prev_author'].value_counts(normalize=True)
                    for tgt in all_authors:
                        features[f'reply_to_{tgt.replace(" ", "_")}'] = reply_counts.get(tgt, 0.0)
                    mention_flat = [m for mentions in sub_df['mentions'] for m in mentions]
                    mention_counts = Counter(mention_flat)
                    total_mentions = sum(mention_counts.values())
                    for tgt in all_authors:
                        features[f'mention_{tgt.replace(" ", "_")}'] = mention_counts[tgt] / total_mentions if total_mentions > 0 else 0.0
                    year_df = df[df['year'] == year].copy()
                    G = nx.Graph()
                    for _, row in year_df[year_df['prev_author'].notna()].iterrows():
                        G.add_edge(row['author'], row['prev_author'])
                    if len(G) > 0 and author in G:
                        features['degree_centrality'] = nx.degree_centrality(G)[author]
                        features['betweenness_centrality'] = nx.betweenness_centrality(G)[author]
                        features['closeness_centrality'] = nx.closeness_centrality(G)[author]
                    else:
                        features['degree_centrality'] = 0.0
                        features['betweenness_centrality'] = 0.0
                        features['closeness_centrality'] = 0.0
                    features['num_groups_participated'] = author_year_df['whatsapp_group'].nunique()
                    total_msgs = len(sub_df)
                    features['avg_msgs_per_group'] = total_msgs / features['num_groups_participated'] if features['num_groups_participated'] > 0 else 0.0
                    threading_features = self._compute_threading_features(sub_df, author)
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

    def _compute_threading_features(self, sub_df: pd.DataFrame, author: str) -> Dict[str, float]:
        features = {
            'num_initiated': 0,
            'avg_depth_initiated': 0.0,
            'num_joined': 0,
            'avg_position': 0.0
        }
        try:
            if len(sub_df) < self.int_settings.threading_min_length:
                return features
            sub_df = sub_df.copy()
            sub_df['timestamp'] = pd.to_datetime(sub_df['timestamp'])
            sub_df = sub_df.reset_index(drop=True)
            group_dfs = []
            for group, group_df in sub_df.groupby('whatsapp_group'):
                if len(group_df) < self.int_settings.threading_min_length:
                    continue
                group_df = group_df.sort_values('timestamp')
                group_df['time_diff'] = group_df['timestamp'].diff().dt.total_seconds() / 3600.0
                group_df['thread_id'] = (group_df['time_diff'] > self.seq_settings.time_diff_threshold).cumsum()
                group_dfs.append(group_df)
            if not group_dfs:
                return features
            sub_df = pd.concat(group_dfs, ignore_index=True)
            thread_starts = sub_df[sub_df['thread_id'].diff() != 0]
            thread_starts = thread_starts[thread_starts['author'] == author]
            features['num_initiated'] = len(thread_starts)
            features['avg_depth_initiated'] = 1.0 if features['num_initiated'] > 0 else 0.0
            features['num_joined'] = len(sub_df) - features['num_initiated']
            features['avg_position'] = 0.5 if len(sub_df) > 0 else 0.0
            logger.debug(f"Threading features for {author}: {features}")
            return features
        except Exception as e:
            logger.warning(f"Failed to compute threading features for {author}: {e}")
            return features

    def build_visual_not_message_content(self, df: pd.DataFrame, groupby_period: str = 'week') -> pd.DataFrame:
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
                    features[f'proportion_{col}'] = (sub_group[col] == col).sum() / total_messages
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