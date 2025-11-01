# === Module Docstring ===
"""
WhatsApp Chat Analyzer – Data Preparation Module

Prepares data for visualization and analysis by adding derived columns,
computing aggregates, and building feature matrices for interactions,
threading, and non-message content. Integrates with ``DataEditor`` for
data transformations.

Key responsibilities:
    * Time-based aggregations and averages
    * Emoji and participation distributions
    * Network and interaction features using NetworkX
    * Threading and non-message content analysis
"""

# === Imports ===
import itertools
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import emoji
import networkx as nx
import pandas as pd
from loguru import logger

from src.constants import Columns
from src.data_editor import DataEditor

# === Feature Column Groups ===
NMC_NUMERICAL = [
    Columns.LENGTH_CHAT,
    Columns.RESPONSE_TIME,
    Columns.NUMBER_OF_EMOJIS,
    Columns.NUMBER_OF_PUNCTUATIONS,
    Columns.PCT_EMOJIS,
]

NMC_BINARY = [
    Columns.HAS_LINK,
    Columns.WAS_DELETED,
    Columns.HAS_EMOJI,
    Columns.ENDS_WITH_EMOJI,
    Columns.HAS_PUNCTUATION,
    Columns.ENDS_WITH_PUNCTUATION,
]

NMC_SUM = [
    Columns.PICTURES_DELETED,
    Columns.VIDEOS_DELETED,
    Columns.AUDIOS_DELETED,
    Columns.GIFS_DELETED,
    Columns.STICKERS_DELETED,
    Columns.DOCUMENTS_DELETED,
    Columns.VIDEONOTES_DELETED,
]

NMC_OPTIONAL = [
    Columns.NUMBER_OF_CHATS_THAT_DAY,
    Columns.X_DAY_PCT_LENGTH_CHAT,
    Columns.X_DAY_PCT_LENGTH_EMOJIS,
    Columns.X_DAY_PCT_LENGTH_PUNCTUATIONS,
    Columns.X_NUMBER_OF_UNIQUE_PARTICIPANTS_THAT_DAY,
]

# === Threading Constants ===
THREADING_MIN_MESSAGES = 3
THREADING_TIME_THRESHOLD = 1.0  # Hours


# === Base Handler Class ===
class BaseHandler:
    """Base class for data handlers with common behavior."""

    def _log_debug(self, message: str) -> None:
        logger.debug(message)

    def _handle_empty_df(self, df: pd.DataFrame, context: str) -> pd.DataFrame:
        if df.empty:
            logger.error(f"Empty DataFrame in {context}.")
            return pd.DataFrame()
        return df


# === Data Preparation Class ===
class DataPreparation(BaseHandler):
    """Prepares WhatsApp chat data for visualization and advanced analysis."""

    def __init__(self, data_editor: Optional[DataEditor] = None) -> None:
        self.data_editor = data_editor
        self.df: Optional[pd.DataFrame] = None

    # === Visualization Preparation Methods ===

    def build_visual_categories(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[Dict[str, List[str]]], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[List[str]]]:
        """
        Prepare DataFrame for visualization by adding year, active years, and early leaver columns,
        computing group authors, non-Anthony averages, Anthony's message counts, and sorted groups.

        Args:
            df: Input DataFrame with 'timestamp', 'author', and 'whatsapp_group' columns.

        Returns:
            Tuple of:
                - Modified DataFrame with added columns
                - Group authors dictionary
                - Non-Anthony average DataFrame
                - Anthony messages DataFrame
                - Sorted groups list

        Examples:
            >>> dp = DataPreparation()
            >>> mod_df, groups, non_ant, ant, sorted_g = dp.build_visual_categories(df)
        """
        df = self._handle_empty_df(df, "visual categories preparation")
        if df.empty:
            return df, None, None, None, None

        try:
            self.df = df.copy()
            self.df[Columns.TIMESTAMP] = pd.to_datetime(self.df[Columns.TIMESTAMP])

            self.df[Columns.YEAR] = self.data_editor.get_year(self.df)
            self.df[Columns.ACTIVE_YEARS] = self.data_editor.active_years(self.df)
            self.df[Columns.EARLY_LEAVER] = self.data_editor.early_leaver(self.df)
            logger.debug(
                f"Added year, active_years, and early_leaver columns. DataFrame columns: {self.df.columns.tolist()}"
            )

            active_years = (
                self.df.groupby([Columns.WHATSAPP_GROUP, Columns.AUTHOR])[Columns.YEAR]
                .agg(["min", "max"])
                .reset_index()
            )
            active_years[Columns.ACTIVE_YEARS] = active_years.apply(
                lambda x: f"{x['min']}-{x['max']}", axis=1
            )
            logger.info("Active years per author per group:")
            logger.info(active_years[[Columns.WHATSAPP_GROUP, Columns.AUTHOR, Columns.ACTIVE_YEARS]].to_string())

            early_leavers = self.df[self.df[Columns.EARLY_LEAVER]][
                [Columns.WHATSAPP_GROUP, Columns.AUTHOR]
            ].drop_duplicates()
            logger.info("Authors who left early (max year < 2025 in July 2015 - July 2025):")
            logger.info(
                early_leavers.to_string() if not early_leavers.empty else "No authors left early."
            )

            group_authors = self.df.groupby(Columns.WHATSAPP_GROUP)[Columns.AUTHOR].unique().to_dict()
            group_authors = {
                group: sorted(auths.tolist()) for group, auths in group_authors.items()
            }
            logger.info("Authors per WhatsApp group:")
            for group, auths in group_authors.items():
                logger.info(f"{group}: {auths}")

            filter_df = self.df[
                (self.df[Columns.TIMESTAMP] >= "2015-07-01") & (self.df[Columns.TIMESTAMP] <= "2025-07-31")
            ]
            logger.info(f"Filtered DataFrame for July 2015 - July 2025: {len(filter_df)} rows")

            group_total = (
                filter_df.groupby(Columns.WHATSAPP_GROUP).size().reset_index(name="total_messages")
            )
            sorted_groups = group_total.sort_values("total_messages", ascending=False)[
                Columns.WHATSAPP_GROUP
            ].tolist()
            logger.info(f"Sorted groups by total messages: {sorted_groups}")

            non_anthony = filter_df[filter_df[Columns.AUTHOR] != "Anthony van Tilburg"]
            non_anthony_counts = (
                non_anthony.groupby([Columns.WHATSAPP_GROUP, Columns.AUTHOR])
                .size()
                .reset_index(name="messages")
            )
            non_anthony_group = (
                non_anthony_counts.groupby(Columns.WHATSAPP_GROUP)["messages"]
                .mean()
                .reset_index(name="non_anthony_avg")
            )
            non_anthony_authors_count = (
                non_anthony_counts.groupby(Columns.WHATSAPP_GROUP)[Columns.AUTHOR]
                .nunique()
                .reset_index(name="num_authors")
            )
            non_anthony_group = non_anthony_group.merge(
                non_anthony_authors_count, on=Columns.WHATSAPP_GROUP, how="left"
            ).fillna({"num_authors": 0, "non_anthony_avg": 0})
            non_anthony_group = (
                non_anthony_group.set_index(Columns.WHATSAPP_GROUP)
                .reindex(sorted_groups)
                .reset_index()
                .fillna({"non_anthony_avg": 0, "num_authors": 0})
            )
            logger.info(
                f"Non-Anthony average messages and author counts per group:\n{non_anthony_group.to_string()}"
            )

            anthony = filter_df[filter_df[Columns.AUTHOR] == "Anthony van Tilburg"]
            anthony_group = (
                anthony.groupby(Columns.WHATSAPP_GROUP).size().reset_index(name="anthony_messages")
            )
            anthony_group = (
                anthony_group.set_index(Columns.WHATSAPP_GROUP)
                .reindex(sorted_groups)
                .reset_index()
                .fillna({"anthony_messages": 0})
            )
            logger.info(f"Anthony messages per group:\n{anthony_group.to_string()}")

            return self.df, group_authors, non_anthony_group, anthony_group, sorted_groups
        except Exception as e:
            logger.exception(f"Failed to build visual categories: {e}")
            return self.df, None, None, None, None

    def build_visual_time(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Prepare time-based data for visualization (e.g., average messages per week for 'dac' group).

        Args:
            df: Input DataFrame with 'timestamp' column.

        Returns:
            Tuple of:
                - Original DataFrame with added 'year' and 'isoweek' columns
                - Weekly message counts DataFrame
                - Average weekly counts DataFrame

        Examples:
            >>> dp = DataPreparation()
            >>> df_time, weekly, avg_weekly = dp.build_visual_time(df)
        """
        if df.empty:
            logger.error("Empty DataFrame provided for visual time preparation.")
            return df, None, None
        try:
            self.df = df
            df = df.copy()
            df[Columns.TIMESTAMP] = pd.to_datetime(df[Columns.TIMESTAMP])
            df[Columns.YEAR] = df[Columns.TIMESTAMP].dt.year
            df["isoweek"] = df[Columns.TIMESTAMP].dt.isocalendar().week
            p = df.groupby([Columns.YEAR, "isoweek"]).size().reset_index(name="message_count")
            average_all = (
                p.groupby("isoweek")["message_count"].mean().reset_index(name="avg_count_all")
            )
            logger.info(
                f"Prepared time-based data with {len(p)} rows and average data with {len(average_all)} rows."
            )
            return df, p, average_all
        except Exception as e:
            logger.exception(f"Failed to build visual time data: {e}")
            return df, None, None

    def build_visual_distribution(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Prepare emoji distribution data for visualization (e.g., for 'maap' group).

        Args:
            df: Input DataFrame with 'message' column.

        Returns:
            Tuple of:
                - DataFrame with added 'emojis' column
                - Emoji counts DataFrame

        Examples:
            >>> dp = DataPreparation()
            >>> df_emoji, emoji_df = dp.build_visual_distribution(df)
        """
        if df.empty:
            logger.error("Empty DataFrame provided for visual distribution preparation.")
            return df, None
        try:
            self.df = df
            df["emojis"] = df[Columns.MESSAGE].apply(
                lambda x: "".join(c for c in str(x) if c in emoji.EMOJI_DATA)
            )
            emoji_counts = Counter([e for msg_emojis in df["emojis"] for e in msg_emojis])
            emoji_counts_df = pd.DataFrame(
                {
                    "emoji": list(emoji_counts.keys()),
                    "count_once": list(emoji_counts.values()),
                    "percent_once": [count / len(df) * 100 for count in emoji_counts.values()],
                }
            )
            emoji_counts_df = emoji_counts_df.sort_values(by="count_once", ascending=False)
            logger.info(
                f"Prepared emoji distribution data with {len(emoji_counts_df)} unique emojis."
            )
            return df, emoji_counts_df
        except Exception as e:
            logger.exception(f"Failed to build visual distribution data: {e}")
            return df, None

    def build_visual_relationships_arc(
        self, df_group: pd.DataFrame, authors: List[str]
    ) -> Optional[pd.DataFrame]:
        """
        Analyze daily participation in a WhatsApp group and combine results into a single table.

        Args:
            df_group: Filtered DataFrame for the group with 'timestamp', 'author', 'message_cleaned', 'early_leaver' columns.
            authors: List of unique authors in the group.

        Returns:
            Combined DataFrame with columns 'type', 'author', 'num_days', 'total_messages', '#participants', and author-specific columns,
            or ``None`` on failure.

        Examples:
            >>> dp = DataPreparation()
            >>> participation_df = dp.build_visual_relationships_arc(df_maap, authors_maap)
        """
        if df_group.empty:
            logger.error("No valid DataFrame provided for relationships arc preparation")
            return None
        try:
            df_group[Columns.TIMESTAMP] = pd.to_datetime(df_group[Columns.TIMESTAMP])
            df_group["date"] = df_group[Columns.TIMESTAMP].dt.date
            df_group["message_length"] = df_group[Columns.MESSAGE_CLEANED].apply(
                lambda x: len(str(x)) if isinstance(x, str) else 0
            )
            authors = sorted(set(df_group[Columns.AUTHOR]))
            total_messages = len(df_group)
            message_counts = df_group.groupby(Columns.AUTHOR).size()
            message_percentages = (message_counts / total_messages * 100).round(0).astype(int)
            total_length = df_group["message_length"].sum()
            length_counts = df_group.groupby(Columns.AUTHOR)["message_length"].sum()
            length_percentages = (length_counts / total_length * 100).round(0).astype(int)
            avg_message_length = (length_counts / message_counts).round(0).astype(int).fillna(0)

            daily_counts = (
                df_group.groupby(["date", Columns.AUTHOR])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=authors, fill_value=0)
            )
            daily_total = daily_counts.sum(axis=1)
            num_unique_series = self.data_editor.number_of_unique_participants_that_day(df_group)
            num_unique_series = num_unique_series.groupby(df_group["date"]).first()

            min_date = df_group["date"].min()
            max_date = df_group["date"].max()
            total_days = (max_date - min_date).days + 1 if min_date and max_date else 0
            days_with_messages = len(daily_counts)
            days_no_messages = total_days - days_with_messages

            def _compute_author_values(
                active_df: pd.DataFrame, active_authors: List[str]
            ) -> Dict[str, str]:
                active_message_counts = (
                    active_df[active_df[Columns.AUTHOR].isin(active_authors)].groupby(Columns.AUTHOR).size()
                )
                active_total_messages = active_message_counts.sum()
                message_pct = (
                    (active_message_counts / active_total_messages * 100).round(0).astype(int)
                    if active_total_messages > 0
                    else pd.Series()
                )
                active_length_counts = (
                    active_df[active_df[Columns.AUTHOR].isin(active_authors)]
                    .groupby(Columns.AUTHOR)["message_length"]
                    .sum()
                )
                active_total_length = active_length_counts.sum()
                length_pct = (
                    (active_length_counts / active_total_length * 100).round(0).astype(int)
                    if active_total_length > 0
                    else pd.Series()
                )
                active_avg_length = (
                    (active_length_counts / active_message_counts).round(0).astype(int).fillna(0)
                )
                author_values = dict.fromkeys(authors, 0)
                for p in active_authors:
                    msg_pct = message_pct.get(p, 0)
                    len_pct = length_pct.get(p, 0)
                    avg_len = active_avg_length.get(p, 0)
                    author_values[p] = f"{msg_pct}%/{len_pct}%({avg_len})"
                return author_values

            combined_data: List[Dict[str, any]] = []

            msg_row = {
                "type": "Messages (%)",
                "author": None,
                "num_days": 0,
                "total_messages": 0,
                "#participants": 0,
            }
            for author in authors:
                msg_pct = message_percentages.get(author, 0)
                avg_len = avg_message_length.get(author, 0)
                msg_row[author] = f"{msg_pct}%/{avg_len}"
            combined_data.append(msg_row)

            len_row = {
                "type": "Message Length (%)",
                "author": None,
                "num_days": 0,
                "total_messages": 0,
                "#participants": 0,
            }
            for author in authors:
                len_pct = length_percentages.get(author, 0)
                avg_len = avg_message_length.get(author, 0)
                len_row[author] = f"{len_pct}%/{avg_len}"
            combined_data.append(len_row)

            combined_data.append(
                {
                    "type": "Period",
                    "author": None,
                    "num_days": total_days,
                    "total_messages": 0,
                    "#participants": 0,
                    **dict.fromkeys(authors, 0),
                }
            )
            combined_data.append(
                {
                    "type": "Period",
                    "author": "None",
                    "num_days": days_no_messages,
                    "total_messages": 0,
                    "#participants": 0,
                    **dict.fromkeys(authors, 0),
                }
            )

            for author in authors:
                other_authors = [a for a in authors if a != author]
                mask = (daily_counts[author] > 0) & (daily_counts[other_authors] == 0).all(axis=1)
                num_days = mask.sum()
                total_msg = daily_total[mask].sum() if num_days > 0 else 0
                if num_days > 0:
                    active_dates = daily_counts[mask].index
                    active_df = df_group[df_group["date"].isin(active_dates)]
                    author_values = _compute_author_values(active_df, [author])
                else:
                    author_values = dict.fromkeys(authors, 0)
                combined_data.append(
                    {
                        "type": "Single",
                        "author": author,
                        "num_days": num_days,
                        "total_messages": total_msg,
                        "#participants": 1,
                        **author_values,
                    }
                )

            for comb in itertools.combinations(authors, 2):
                pair_str = " & ".join(sorted(comb))
                other_authors = [a for a in authors if a not in comb]
                mask = (daily_counts[list(comb)] > 0).all(axis=1) & (
                    daily_counts[other_authors] == 0
                ).all(axis=1)
                num_days = mask.sum()
                total_msg = daily_total[mask].sum() if num_days > 0 else 0
                if num_days > 0:
                    active_dates = daily_counts[mask].index
                    active_df = df_group[df_group["date"].isin(active_dates)]
                    author_values = _compute_author_values(active_df, list(comb))
                else:
                    author_values = dict.fromkeys(authors, 0)
                combined_data.append(
                    {
                        "type": "Pairs",
                        "author": pair_str,
                        "num_days": num_days,
                        "total_messages": total_msg,
                        "#participants": 2,
                        **author_values,
                    }
                )

            for non_part in authors:
                participants = [a for a in authors if a != non_part]
                mask = (daily_counts[participants] > 0).all(axis=1) & (daily_counts[non_part] == 0)
                num_days = mask.sum()
                total_msg = daily_total[mask].sum() if num_days > 0 else 0
                if num_days > 0:
                    active_dates = daily_counts[mask].index
                    active_df = df_group[df_group["date"].isin(active_dates)]
                    author_values = _compute_author_values(active_df, participants)
                else:
                    author_values = dict.fromkeys(authors, 0)
                combined_data.append(
                    {
                        "type": "N-1",
                        "author": f"Without {non_part}",
                        "num_days": num_days,
                        "total_messages": total_msg,
                        "#participants": len(authors) - 1,
                        **author_values,
                    }
                )

            mask = (daily_counts > 0).all(axis=1)
            num_days = mask.sum()
            total_msg = daily_total[mask].sum() if num_days > 0 else 0
            if num_days > 0:
                active_dates = daily_counts[mask].index
                active_df = df_group[df_group["date"].isin(active_dates)]
                author_values = _compute_author_values(active_df, authors)
            else:
                author_values = dict.fromkeys(authors, 0)
            combined_data.append(
                {
                    "type": "All",
                    "author": "All",
                    "num_days": num_days,
                    "total_messages": total_msg,
                    "#participants": len(authors),
                    **author_values,
                }
            )

            combined_df = pd.DataFrame(combined_data)
            combined_df = combined_df.sort_values(
                by=["#participants", "num_days"], ascending=[True, False]
            )
            logger.info(
                f"Combined participation table for group {df_group[Columns.WHATSAPP_GROUP].iloc[0]}:\n{combined_df.to_string(index=False)}"
            )
            return combined_df
        except Exception as e:
            logger.exception(f"Failed to prepare relationships arc data: {e}")
            return None

    def build_visual_relationships_bubble(
        self, df_groups: pd.DataFrame, groups: List[str]
    ) -> Optional[pd.DataFrame]:
        """
        Prepare data for the new bubble plot with average words, average punctuation, and message count per author within each group.

        Args:
            df_groups: Filtered DataFrame with 'whatsapp_group', 'author', 'message_cleaned' columns.
            groups: List of WhatsApp groups to process.

        Returns:
            DataFrame with 'whatsapp_group', 'author', 'avg_words', 'avg_punct', 'message_count' columns,
            or ``None`` on failure.

        Examples:
            >>> dp = DataPreparation()
            >>> bubble_df = dp.build_visual_relationships_bubble(df_all, ["maap", "dac"])
        """
        if df_groups.empty or not any(df_groups[Columns.WHATSAPP_GROUP].isin(groups)):
            logger.error("No valid data provided for bubble plot preparation.")
            return None
        try:
            df_groups["message_length"] = df_groups[Columns.MESSAGE_CLEANED].apply(
                lambda x: len(str(x).split()) if isinstance(x, str) else 0
            )
            df_groups["punctuation_count"] = df_groups[Columns.MESSAGE_CLEANED].apply(
                lambda x: len(re.findall(r"[.!?]", str(x))) if isinstance(x, str) else 0
            )

            result_df = (
                df_groups.groupby([Columns.WHATSAPP_GROUP, Columns.AUTHOR])
                .agg(
                    {
                        "message_length": "mean",
                        "punctuation_count": "mean",
                        Columns.MESSAGE_CLEANED: "count",
                    }
                )
                .rename(
                    columns={
                        "message_length": "avg_words",
                        "punctuation_count": "avg_punct",
                        Columns.MESSAGE_CLEANED: "message_count",
                    }
                )
                .reset_index()
            )

            result_df = result_df[result_df[Columns.WHATSAPP_GROUP].isin(groups)]
            logger.info(
                f"Prepared bubble plot data with shape {result_df.shape}: {result_df.columns.tolist()}"
            )
            return result_df
        except Exception as e:
            logger.exception(f"Failed to prepare bubble plot data: {e}")
            return None

    # === Feature Building Methods ===

    def build_interaction_features(
        self, df: pd.DataFrame, group_authors: Dict[str, List[str]]
    ) -> Optional[pd.DataFrame]:
        """
        Build a feature matrix for interaction and network dynamics analysis.

        Features include normalized reply frequencies, mention frequencies, centrality measures,
        and cross-group participation. Computed per author-year, with separate rows for
        'Anthony van Tilburg' for each group per year and an overall row across all groups.

        Args:
            df: Full cleaned DataFrame with 'whatsapp_group' column.
            group_authors: Dictionary of group names to lists of authors.

        Returns:
            Feature matrix with 'author_year' or 'author_year_group' index and 'whatsapp_group' column,
            or ``None`` on failure.

        Examples:
            >>> dp = DataPreparation()
            >>> features = dp.build_interaction_features(df, group_authors)
        """
        try:
            if df.empty:
                logger.error("Empty DataFrame provided for interaction features.")
                return None
            self.df = df
            all_authors = sorted({a for authors in group_authors.values() for a in authors})
            first_names = {a: a.split()[0] for a in all_authors}
            first_name_to_author = {v: k for k, v in first_names.items()}

            df = df.copy()
            df[Columns.YEAR] = self.data_editor.get_year(df)
            df = df.sort_values([Columns.WHATSAPP_GROUP, Columns.TIMESTAMP])
            df[Columns.PREVIOUS_AUTHOR] = df.groupby(Columns.WHATSAPP_GROUP)[Columns.AUTHOR].shift(1)
            df["mentions"] = df[Columns.MESSAGE].apply(
                lambda x: self.data_editor.extract_mentions(x, first_name_to_author)
            )
            feature_list: List[Dict[str, any]] = []

            for (author, year), author_year_df in df.groupby([Columns.AUTHOR, Columns.YEAR]):
                if author == "Anthony van Tilburg":
                    for group in author_year_df[Columns.WHATSAPP_GROUP].unique():
                        sub_df = author_year_df[author_year_df[Columns.WHATSAPP_GROUP] == group]
                        if sub_df.empty:
                            continue
                        features = {
                            "author_year": f"{author}_{year}_{group}",
                            Columns.WHATSAPP_GROUP: group,
                        }
                        reply_counts = sub_df[sub_df[Columns.PREVIOUS_AUTHOR].notna()][
                            Columns.PREVIOUS_AUTHOR
                        ].value_counts(normalize=True)
                        for tgt in all_authors:
                            features[f"reply_to_{tgt.replace(' ', '_')}"] = reply_counts.get(tgt, 0.0)
                        mention_flat = [m for mentions in sub_df["mentions"] for m in mentions]
                        mention_counts = Counter(mention_flat)
                        total_mentions = sum(mention_counts.values())
                        for tgt in all_authors:
                            features[f"mention_{tgt.replace(' ', '_')}"] = (
                                mention_counts[tgt] / total_mentions if total_mentions > 0 else 0.0
                            )
                        year_df = df[df[Columns.YEAR] == year].copy()
                        G = nx.Graph()
                        for _, row in year_df[year_df[Columns.PREVIOUS_AUTHOR].notna()].iterrows():
                            G.add_edge(row[Columns.AUTHOR], row[Columns.PREVIOUS_AUTHOR])
                        if len(G) > 0 and author in G:
                            features["degree_centrality"] = nx.degree_centrality(G)[author]
                            features["betweenness_centrality"] = nx.betweenness_centrality(G)[author]
                            features["closeness_centrality"] = nx.closeness_centrality(G)[author]
                        else:
                            features["degree_centrality"] = 0.0
                            features["betweenness_centrality"] = 0.0
                            features["closeness_centrality"] = 0.0
                        features["num_groups_participated"] = author_year_df[Columns.WHATSAPP_GROUP].nunique()
                        total_msgs = len(sub_df)
                        features["avg_msgs_per_group"] = (
                            total_msgs / features["num_groups_participated"]
                            if features["num_groups_participated"] > 0
                            else 0.0
                        )
                        threading_features = self._compute_threading_features(sub_df, author)
                        features.update(threading_features)
                        feature_list.append(features)

                    sub_df = author_year_df
                    features = {"author_year": f"{author}_{year}", Columns.WHATSAPP_GROUP: "overall"}
                    reply_counts = sub_df[sub_df[Columns.PREVIOUS_AUTHOR].notna()][
                        Columns.PREVIOUS_AUTHOR
                    ].value_counts(normalize=True)
                    for tgt in all_authors:
                        features[f"reply_to_{tgt.replace(' ', '_')}"] = reply_counts.get(tgt, 0.0)
                    mention_flat = [m for mentions in sub_df["mentions"] for m in mentions]
                    mention_counts = Counter(mention_flat)
                    total_mentions = sum(mention_counts.values())
                    for tgt in all_authors:
                        features[f"mention_{tgt.replace(' ', '_')}"] = (
                            mention_counts[tgt] / total_mentions if total_mentions > 0 else 0.0
                        )
                    year_df = df[df[Columns.YEAR] == year].copy()
                    G = nx.Graph()
                    for _, row in year_df[year_df[Columns.PREVIOUS_AUTHOR].notna()].iterrows():
                        G.add_edge(row[Columns.AUTHOR], row[Columns.PREVIOUS_AUTHOR])
                    if len(G) > 0 and author in G:
                        features["degree_centrality"] = nx.degree_centrality(G)[author]
                        features["betweenness_centrality"] = nx.betweenness_centrality(G)[author]
                        features["closeness_centrality"] = nx.closeness_centrality(G)[author]
                    else:
                        features["degree_centrality"] = 0.0
                        features["betweenness_centrality"] = 0.0
                        features["closeness_centrality"] = 0.0
                    features["num_groups_participated"] = author_year_df[Columns.WHATSAPP_GROUP].nunique()
                    total_msgs = len(sub_df)
                    features["avg_msgs_per_group"] = (
                        total_msgs / features["num_groups_participated"]
                        if features["num_groups_participated"] > 0
                        else 0.0
                    )
                    threading_features = self._compute_threading_features(sub_df, author)
                    features.update(threading_features)
                    feature_list.append(features)
                else:
                    group_counts = author_year_df[Columns.WHATSAPP_GROUP].value_counts()
                    primary_group = group_counts.index[0] if not group_counts.empty else "unknown"
                    sub_df = author_year_df[author_year_df[Columns.WHATSAPP_GROUP] == primary_group]
                    features = {"author_year": f"{author}_{year}", Columns.WHATSAPP_GROUP: primary_group}
                    reply_counts = sub_df[sub_df[Columns.PREVIOUS_AUTHOR].notna()][
                        Columns.PREVIOUS_AUTHOR
                    ].value_counts(normalize=True)
                    for tgt in all_authors:
                        features[f"reply_to_{tgt.replace(' ', '_')}"] = reply_counts.get(tgt, 0.0)
                    mention_flat = [m for mentions in sub_df["mentions"] for m in mentions]
                    mention_counts = Counter(mention_flat)
                    total_mentions = sum(mention_counts.values())
                    for tgt in all_authors:
                        features[f"mention_{tgt.replace(' ', '_')}"] = (
                            mention_counts[tgt] / total_mentions if total_mentions > 0 else 0.0
                        )
                    year_df = df[df[Columns.YEAR] == year].copy()
                    G = nx.Graph()
                    for _, row in year_df[year_df[Columns.PREVIOUS_AUTHOR].notna()].iterrows():
                        G.add_edge(row[Columns.AUTHOR], row[Columns.PREVIOUS_AUTHOR])
                    if len(G) > 0 and author in G:
                        features["degree_centrality"] = nx.degree_centrality(G)[author]
                        features["betweenness_centrality"] = nx.betweenness_centrality(G)[author]
                        features["closeness_centrality"] = nx.closeness_centrality(G)[author]
                    else:
                        features["degree_centrality"] = 0.0
                        features["betweenness_centrality"] = 0.0
                        features["closeness_centrality"] = 0.0
                    features["num_groups_participated"] = author_year_df[Columns.WHATSAPP_GROUP].nunique()
                    total_msgs = len(sub_df)
                    features["avg_msgs_per_group"] = (
                        total_msgs / features["num_groups_participated"]
                        if features["num_groups_participated"] > 0
                        else 0.0
                    )
                    threading_features = self._compute_threading_features(sub_df, author)
                    features.update(threading_features)
                    feature_list.append(features)

            feature_df = pd.DataFrame(feature_list).set_index("author_year")
            logger.info(f"Built interaction feature matrix with shape {feature_df.shape}")
            logger.debug(f"Feature matrix columns: {feature_df.columns.tolist()}")
            logger.debug(f"Feature matrix preview:\n{feature_df.head().to_string()}")
            return feature_df
        except Exception as e:
            logger.exception(f"Failed to build interaction features: {e}")
            return None

    def _compute_threading_features(
        self, df: pd.DataFrame, author: str
    ) -> Dict[str, any]:
        """
        Compute threading features for a given author.

        Args:
            df: Input DataFrame with message data.
            author: Author name to compute features for.

        Returns:
            Dictionary of threading features.

        Examples:
            >>> dp = DataPreparation()
            >>> threading_feats = dp._compute_threading_features(df_author, "John Doe")
        """
        features: Dict[str, any] = {}
        try:
            min_messages = THREADING_MIN_MESSAGES
            time_diff_threshold = THREADING_TIME_THRESHOLD
            df_author = df[df[Columns.AUTHOR] == author].sort_values(Columns.TIMESTAMP)
            sequence_count = 0
            for i in range(len(df_author) - min_messages + 1):
                sequence = df_author.iloc[i : i + min_messages]
                if len(sequence) == min_messages:
                    time_diffs = (sequence[Columns.TIMESTAMP].diff().dt.total_seconds() / 3600).dropna()
                    if all(time_diffs <= time_diff_threshold):
                        sequence_count += 1
            features["threading_sequences"] = sequence_count
            logger.debug(f"Computed threading features for {author}: {features}")
            return features
        except Exception as e:
            logger.warning(f"Failed to compute threading features for {author}: {e}")
            return features

    def build_visual_no_message_content(
        self, df: pd.DataFrame, groupby_period: str = "week"
    ) -> Optional[pd.DataFrame]:
        """
        Build non-message content feature matrix for PCA/t-SNE analysis.

        Args:
            df: Input DataFrame with required columns.
            groupby_period: Time period for grouping ("week", "month", "year").

        Returns:
            Feature matrix DataFrame, or ``None`` on failure.

        Examples:
            >>> dp = DataPreparation()
            >>> nmc_df = dp.build_visual_no_message_content(df, "month")
        """
        if df.empty:
            logger.error("No valid DataFrame provided for non-message content preparation")
            return None
        try:
            if groupby_period not in {"week", "month", "year"}:
                logger.error("Invalid groupby_period.")
                return None
            period_col = Columns.WEEK if groupby_period == "week" else Columns.MONTH if groupby_period == "month" else Columns.YEAR
            if period_col not in df.columns:
                logger.error(f"Column '{period_col}' not found.")
                return None
            required_columns = [Columns.AUTHOR, Columns.YEAR, Columns.WHATSAPP_GROUP]
            if not all(col in df.columns for col in required_columns):
                logger.error("Missing required columns.")
                return None

            available_numerical = [col for col in NMC_NUMERICAL if col in df.columns]
            available_binary = [col for col in NMC_BINARY if col in df.columns]
            available_sum = [col for col in NMC_SUM if col in df.columns]
            available_optional = [col for col in NMC_OPTIONAL if col in df.columns]

            if not (available_numerical or available_binary or available_sum or available_optional):
                logger.error("No feature columns available.")
                return None

            feature_list: List[Dict[str, any]] = []
            groupby_cols = (
                [Columns.AUTHOR, period_col, Columns.YEAR, Columns.WHATSAPP_GROUP]
                if groupby_period != "year"
                else [Columns.AUTHOR, Columns.YEAR, Columns.WHATSAPP_GROUP]
            )
            for group_values, sub_group in df.groupby(groupby_cols):
                total_messages = len(sub_group)
                if total_messages == 0:
                    continue
                if groupby_period == "year":
                    author, year, group = group_values
                    period_value = year
                else:
                    author, period_value, year, group = group_values
                features = {
                    "author_period_year_group": f"{author}_{period_value:02d}_{year}_{group}",
                    Columns.AUTHOR: author,
                    groupby_period: period_value,
                    Columns.YEAR: year,
                    Columns.WHATSAPP_GROUP: group,
                    "total_messages": total_messages,
                }
                for col in available_numerical:
                    features[f"mean_{col}"] = sub_group[col].mean()
                for col in available_binary:
                    features[f"proportion_{col}"] = sub_group[col].sum() / total_messages
                for col in available_sum:
                    features[f"sum_{col}"] = sub_group[col].sum()
                for col in available_optional:
                    features[f"mean_{col}"] = (
                        sub_group[col].mean() if col in sub_group.columns else 0.0
                    )
                feature_list.append(features)
            feature_df = pd.DataFrame(feature_list).set_index("author_period_year_group")
            logger.info(f"Built non-message content feature matrix with shape {feature_df.shape}")
            logger.debug(f"Feature matrix columns: {feature_df.columns.tolist()}")
            logger.debug(f"Feature matrix preview:\n{feature_df.head().to_string()}")
            return feature_df
        except Exception as e:
            logger.exception(f"Failed to build non-message content features: {e}")
            return None

    # === Correlation Methods ===

    def compute_month_correlations(self, feature_df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Compute correlations between numerical features and month.

        Args:
            feature_df: Feature DataFrame with 'month' and numerical columns.

        Returns:
            Series of correlations with 'month', or ``None`` on failure.

        Examples:
            >>> dp = DataPreparation()
            >>> corrs = dp.compute_month_correlations(feature_df)
        """
        if feature_df.empty:
            logger.error("No valid feature DataFrame for month correlations")
            return None
        try:
            if Columns.MONTH not in feature_df.columns:
                logger.error("Column 'month' not found.")
                return None
            numerical_cols = [
                col
                for col in feature_df.columns
                if col not in {Columns.AUTHOR, Columns.MONTH, Columns.YEAR, Columns.WHATSAPP_GROUP}
                and feature_df[col].dtype in {"int64", "float64"}
            ]
            if not numerical_cols:
                logger.warning("No numerical columns for correlation.")
                return None
            correlations = feature_df[[*numerical_cols, Columns.MONTH]].corr()[Columns.MONTH].drop(Columns.MONTH)
            logger.info(f"Computed correlations with 'month':\n{correlations.to_string()}")
            return correlations
        except Exception as e:
            logger.exception(f"Failed to compute month correlations: {e}")
            return None


# NEW: Removed Pydantic settings; replaced with StrEnum-based constants (2025-10-31)