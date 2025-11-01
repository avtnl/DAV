# === Module Docstring ===
"""
WhatsApp Chat Analyzer - Data Preparation Module

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
from typing import TYPE_CHECKING

import emoji
import pandas as pd
from loguru import logger
from src.constants import Columns

if TYPE_CHECKING:
    from src.data_editor import DataEditor


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

    def __init__(self, data_editor: "DataEditor" | None = None) -> None:
        self.data_editor = data_editor
        self.df: pd.DataFrame | None = None

    # === Visualization Preparation Methods ===

    def build_visual_categories(
        self, df: pd.DataFrame
    ) -> tuple[
        pd.DataFrame,
        dict[str, list[str]] | None,
        pd.DataFrame | None,
        pd.DataFrame | None,
        list[str] | None,
    ]:
        df = self._handle_empty_df(df, "visual categories preparation")

        try:
            self.df = df.copy()
            if self.df.empty:
                return self.df, None, None, None, None

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
            logger.info(
                active_years[
                    [Columns.WHATSAPP_GROUP, Columns.AUTHOR, Columns.ACTIVE_YEARS]
                ].to_string()
            )

            early_leavers = self.df[self.df[Columns.EARLY_LEAVER]][
                [Columns.WHATSAPP_GROUP, Columns.AUTHOR]
            ].drop_duplicates()
            logger.info("Authors who left early (max year < 2025 in July 2015 - July 2025):")
            logger.info(
                early_leavers.to_string() if not early_leavers.empty else "No authors left early."
            )

            group_authors = (
                self.df.groupby(Columns.WHATSAPP_GROUP)[Columns.AUTHOR].unique().to_dict()
            )
            group_authors = {
                group: sorted(auths.tolist()) for group, auths in group_authors.items()
            }
            logger.info("Authors per WhatsApp group:")
            for group, auths in group_authors.items():
                logger.info(f"{group}: {auths}")

            filter_df = self.df[
                (self.df[Columns.TIMESTAMP] >= "2015-07-01")
                & (self.df[Columns.TIMESTAMP] <= "2025-07-31")
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
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
        try:
            if df.empty:
                logger.error("Empty DataFrame provided for visual time preparation.")
                return df, None, None

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

    def build_visual_distribution(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        try:
            if df.empty:
                logger.error("Empty DataFrame provided for visual distribution preparation.")
                return df, None

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
        self, df_group: pd.DataFrame, authors: list[str]
    ) -> pd.DataFrame | None:
        try:
            if df_group.empty:
                logger.error("No valid DataFrame provided for relationships arc preparation")
                return None

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
                active_df: pd.DataFrame, active_authors: list[str]
            ) -> dict[str, str]:
                active_message_counts = (
                    active_df[active_df[Columns.AUTHOR].isin(active_authors)]
                    .groupby(Columns.AUTHOR)
                    .size()
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

            combined_data: list[dict[str, any]] = []

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
        self, df_groups: pd.DataFrame, groups: list[str]
    ) -> pd.DataFrame | None:
        try:
            if df_groups.empty or not any(df_groups[Columns.WHATSAPP_GROUP].isin(groups)):
                logger.error("No valid data provided for bubble plot preparation.")
                return None

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

    # === Correlation Methods ===

    def compute_month_correlations(self, feature_df: pd.DataFrame) -> pd.Series | None:
        try:
            if feature_df.empty:
                logger.error("No valid feature DataFrame for month correlations")
                return None

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
            correlations = (
                feature_df[[*numerical_cols, Columns.MONTH]]
                .corr()[Columns.MONTH]
                .drop(Columns.MONTH)
            )
            logger.info(f"Computed correlations with 'month':\n{correlations.to_string()}")
            return correlations
        except Exception as e:
            logger.exception(f"Failed to compute month correlations: {e}")
            return None


# NEW: Removed Pydantic settings; replaced with StrEnum-based constants (2025-10-31)
