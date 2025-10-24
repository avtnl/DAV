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

class DataPreparation(BaseHandler):
    """A class for preparing WhatsApp message data for visualization, inheriting from BaseHandler."""
    
    def __init__(self, data_editor=None, seq_settings: SequenceSettings = SequenceSettings(), 
                 int_settings: InteractionSettings = InteractionSettings(), 
                 nmc_settings: NonMessageContentSettings = NonMessageContentSettings(), 
                 vis_settings: VisualSettings = VisualSettings()):
        # Initialize BaseHandler with vis_settings
        super().__init__(settings=vis_settings)
        self.data_editor = data_editor
        self.seq_settings = seq_settings
        self.int_settings = int_settings
        self.nmc_settings = nmc_settings
        self.vis_settings = vis_settings
        self.df = None  # Store the DataFrame for use in scripts

    def build_visual_categories(self, df: pd.DataFrame, settings: VisualSettings = None) -> Tuple[pd.DataFrame, Dict[str, List[str]], pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Prepare DataFrame for visualization by adding year, active years, and early leaver columns,
        computing group authors, non-Anthony averages, Anthony's message counts, and sorted groups.

        Args:
            df (pandas.DataFrame): Input DataFrame with 'timestamp', 'author', and 'whatsapp_group' columns.
            settings (VisualSettings, optional): Visualization settings (defaults to self.vis_settings).

        Returns:
            tuple: (pandas.DataFrame, dict, pandas.DataFrame, pandas.DataFrame, list) -
                Modified DataFrame with added columns, group authors dict, non-Anthony average DataFrame,
                Anthony messages DataFrame, sorted groups list.
        """
        # Use BaseHandler's empty DataFrame check
        df = self._handle_empty_df(df, "visual categories preparation")
        if df.empty:
            return df, None, None, None, None

        # Use provided settings or default to self.vis_settings
        settings = settings or self.vis_settings

        try:
            self.df = df.copy()  # Store a copy of the DataFrame
            # Ensure timestamp is datetime
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

            # Add year, active_years, and early_leaver columns using data_editor
            self.df["year"] = self.data_editor.get_year(self.df)
            self.df["active_years"] = self.data_editor.active_years(self.df)
            self.df["early_leaver"] = self.data_editor.early_leaver(self.df)
            logger.debug(f"Added year, active_years, and early_leaver columns. DataFrame columns: {self.df.columns.tolist()}")

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
            group_authors = {group: sorted(auths.tolist()) for group, auths in group_authors.items()}  # Sort for consistency
            logger.info("Authors per WhatsApp group:")
            for group, auths in group_authors.items():
                logger.info(f"{group}: {auths}")

            # Filter data for July 2015 - July 2025
            filter_df = self.df[(self.df['timestamp'] >= '2015-07-01') & (self.df['timestamp'] <= '2025-07-31')]
            logger.info(f"Filtered DataFrame for July 2015 - July 2025: {len(filter_df)} rows")

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

    def build_visual_relationships_bubble(self, df_groups: pd.DataFrame, groups: List[str]) -> pd.DataFrame:
        """
        Prepare data for the new bubble plot with average words, average punctuation, and message count per author within each group.
        
        Args:
            df_groups (pandas.DataFrame): Filtered DataFrame with 'whatsapp_group', 'author', 'message_cleaned' columns.
            groups (List[str]): List of WhatsApp groups to process.
        
        Returns:
            pd.DataFrame or None: DataFrame with 'whatsapp_group', 'author', 'avg_words', 'avg_punct', 'message_count' columns.
        """
        if df_groups.empty or not any(df_groups['whatsapp_group'].isin(groups)):
            logger.error("No valid data provided for bubble plot preparation.")
            return None
        try:
            # Compute message length and punctuation count per message
            df_groups['message_length'] = df_groups['message_cleaned'].apply(lambda x: len(str(x).split()) if isinstance(x, str) else 0)
            df_groups['punctuation_count'] = df_groups['message_cleaned'].apply(lambda x: len(re.findall(r'[.!?]', str(x))) if isinstance(x, str) else 0)
            
            # Aggregate per whatsapp_group and author
            result_df = df_groups.groupby(['whatsapp_group', 'author']).agg({
                'message_length': 'mean',
                'punctuation_count': 'mean',
                'message_cleaned': 'count'
            }).rename(columns={
                'message_length': 'avg_words',
                'punctuation_count': 'avg_punct',
                'message_cleaned': 'message_count'
            }).reset_index()
            
            # Filter to requested groups
            result_df = result_df[result_df['whatsapp_group'].isin(groups)]
            logger.info(f"Prepared bubble plot data with shape {result_df.shape}: {result_df.columns.tolist()}")
            return result_df
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