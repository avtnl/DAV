# === Module Docstring ===
"""
Data Preparation Module

Prepares validated data for all 5 visualizations:
1. Categories: Total messages by group/author (Script1)
2. Time: DAC weekly heartbeat (Script2)
3. Distribution: Emoji frequency (Script3)
4. Arc: Author interactions (Script4)
5. Bubble: Words vs punctuation (Script5)

**All Pydantic models are defined here** for clean data contracts.
"""

# === Imports ===
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Dict

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from .constants import Columns, Groups

if TYPE_CHECKING:
    from .data_editor import DataEditor


# === 1. Category Plot Data Contracts (Script1) ===
class AuthorMessages(BaseModel):
    author: str = Field(..., min_length=1)
    message_count: int = Field(..., ge=0)
    is_avt: bool = False


class GroupMessages(BaseModel):
    whatsapp_group: Literal["dac", "golfmaten", "maap", "tillies"]
    authors: list[AuthorMessages] = Field(..., min_items=1)
    group_avg: float = Field(..., ge=0.0)

    @property
    def total_messages(self) -> int:
        return sum(a.message_count for a in self.authors)


class CategoryPlotData(BaseModel):
    groups: list[GroupMessages]
    total_messages: int
    date_range: tuple[datetime, datetime]

    @property
    def group_order(self) -> list[str]:
        return [g.whatsapp_group for g in self.groups]


# === 2. Time Plot Data Contract (Script2) ===
class TimePlotData(BaseModel):
    weekly_avg: Dict[int, float]
    global_avg: float
    date_range: tuple[datetime, datetime]

    class Config:
        arbitrary_types_allowed = False


# === 3. Distribution Plot Data Contract (Script3) ===
class DistributionPlotData(BaseModel):
    """Validated container for the emoji-frequency DataFrame."""
    emoji_counts_df: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True


# === 4. Arc Plot Data Contract (Script4) ===
class ArcPlotData(BaseModel):
    """Validated container for the participation table used by the arc diagram."""
    participation_df: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True


# === 5. Bubble Plot Data Contract (Script5) – placeholder ===
class BubblePlotData(BaseModel):
    pass  # to be filled later


# === Base Handler ===
class BaseHandler:
    def _log_debug(self, message: str) -> None:
        logger.debug(message)

    def _handle_empty_df(self, df: pd.DataFrame, context: str) -> pd.DataFrame:
        if df.empty:
            logger.error(f"Empty DataFrame in {context}.")
            return pd.DataFrame()
        return df


# === Data Preparation Class ===
class DataPreparation(BaseHandler):
    def __init__(self, data_editor: DataEditor | None = None) -> None:
        self.data_editor = data_editor
        self.df: pd.DataFrame | None = None

    # === 1. Categories (Script1) ===
    def build_visual_categories(self, df: pd.DataFrame) -> CategoryPlotData | None:
        df = self._handle_empty_df(df, "build_visual_categories")
        if df is None or df.empty:
            return None

        try:
            mask = (
                (df[Columns.TIMESTAMP] >= "2015-07-01")
                & (df[Columns.TIMESTAMP] <= "2025-07-31")
            )
            df_filtered = df[mask].copy()
            if df_filtered.empty:
                logger.warning("No messages in date range 2015-07-01 to 2025-07-31")
                return None

            counts = (
                df_filtered.groupby([Columns.WHATSAPP_GROUP, Columns.AUTHOR], as_index=False)
                .size()
                .rename(columns={"size": Columns.MESSAGE_COUNT})
            )
            if counts.empty:
                logger.error("No message counts after grouping")
                return None

            group_order = sorted(counts[Columns.WHATSAPP_GROUP].unique())
            counts[Columns.WHATSAPP_GROUP] = pd.Categorical(
                counts[Columns.WHATSAPP_GROUP],
                categories=group_order,
                ordered=True,
            )
            counts = counts.sort_values(Columns.WHATSAPP_GROUP).reset_index(drop=True)
            counts["is_avt"] = counts[Columns.AUTHOR] == "AvT"

            groups_data = []
            total_messages = 0

            for group in group_order:
                grp = counts[counts[Columns.WHATSAPP_GROUP] == group]
                non_avt = grp[~grp["is_avt"]].sort_values(Columns.MESSAGE_COUNT, ascending=False)
                avt_row = grp[grp["is_avt"]]
                group_avg = non_avt[Columns.MESSAGE_COUNT].mean() if not non_avt.empty else 0.0
                authors = []

                for _, row in non_avt.iterrows():
                    msg_count = int(row[Columns.MESSAGE_COUNT])
                    authors.append(
                        AuthorMessages(
                            author=str(row[Columns.AUTHOR]),
                            message_count=msg_count,
                            is_avt=False,
                        )
                    )
                    total_messages += msg_count

                if not avt_row.empty:
                    row = avt_row.iloc[0]
                    msg_count = int(row[Columns.MESSAGE_COUNT])
                    authors.append(
                        AuthorMessages(
                            author=str(row[Columns.AUTHOR]),
                            message_count=msg_count,
                            is_avt=True,
                        )
                    )
                    total_messages += msg_count

                groups_data.append(
                    GroupMessages(
                        whatsapp_group=group,
                        authors=authors,
                        group_avg=float(group_avg),
                    )
                )

            result = CategoryPlotData(
                groups=groups_data,
                total_messages=total_messages,
                date_range=(datetime(2015, 7, 1), datetime(2025, 7, 31)),
            )

            logger.success(
                f"CategoryPlotData built: {len(result.groups)} groups, "
                f"{result.total_messages:,} messages"
            )
            return result

        except Exception as e:
            logger.exception(f"build_visual_categories failed: {e}")
            return None

    # === 2. Time (Script2) - Returns dict, not Series ===
    def build_visual_time(self, df: pd.DataFrame) -> TimePlotData | None:
        df = self._handle_empty_df(df, "build_visual_time")
        if df is None or df.empty:
            return None

        try:
            df_dac = df[df[Columns.WHATSAPP_GROUP] == Groups.DAC].copy()
            if df_dac.empty:
                logger.warning("No messages in DAC group for time plot")
                return None

            weekly_counts = df_dac.groupby(Columns.WEEK).size()
            p = weekly_counts.groupby(weekly_counts.index).mean()
            p = p.reindex(range(1, 53), fill_value=0.0)

            # Convert Series → dict
            weekly_avg_dict = p.to_dict()  # {1: 12.3, 2: 15.1, ...}

            average_all = float(p.mean())
            date_range = (
                df_dac[Columns.TIMESTAMP].min().date(),
                df_dac[Columns.TIMESTAMP].max().date(),
            )

            result = TimePlotData(
                weekly_avg=weekly_avg_dict,
                global_avg=average_all,
                date_range=date_range,
            )
            logger.success(f"TimePlotData built: {len(weekly_avg_dict)} weeks")
            return result

        except Exception as e:
            logger.exception(f"build_visual_time failed: {e}")
            return None

    # === 3. Distribution (Script3) ===
    def build_visual_distribution(self, df: pd.DataFrame) -> DistributionPlotData | None:
        df = self._handle_empty_df(df, "build_visual_distribution")
        if df.empty:
            return None

        try:
            import ast
            import emoji

            def parse_emoji_list(cell):
                if pd.isna(cell) or cell in {'[]', '', ' ', None}:
                    return []
                try:
                    return ast.literal_eval(cell)
                except (ValueError, SyntaxError):
                    # Fallback: split by space and filter valid emojis
                    return [e.strip() for e in str(cell).split() if e.strip() in emoji.EMOJI_DATA]

            # Parse and flatten
            emoji_lists = df[Columns.LIST_OF_ALL_EMOJIS.value].apply(parse_emoji_list)
            all_emojis = pd.Series([e for sublist in emoji_lists for e in sublist])

            if all_emojis.empty:
                logger.warning("No emojis found after parsing.")
                return None

            emoji_counts = Counter(all_emojis)
            emoji_counts_df = pd.DataFrame(
                {
                    "emoji": list(emoji_counts.keys()),
                    "count_once": list(emoji_counts.values()),
                }
            )
            emoji_counts_df["percent_once"] = (
                emoji_counts_df["count_once"] / len(df) * 100
            )
            emoji_counts_df = emoji_counts_df.sort_values(
                by="count_once", ascending=False
            ).reset_index(drop=True)

            result = DistributionPlotData(emoji_counts_df=emoji_counts_df)
            logger.success(
                f"DistributionPlotData built – {len(emoji_counts_df)} unique emojis."
            )
            return result

        except Exception as e:
            logger.exception(f"build_visual_distribution failed: {e}")
            return None

    # === 4. Arc (Script4) ===
    def build_visual_relationships_arc(self, df_group: pd.DataFrame) -> ArcPlotData | None:
        """
        Build the participation table for the MAAP arc diagram.

        Uses ``x_number_of_unique_participants_that_day`` to identify 2- or 3-person days.
        Authors are derived from the data (must be exactly 4).

        Args:
            df_group: DataFrame filtered to MAAP group.

        Returns:
            ArcPlotData or None.
        """
        df = self._handle_empty_df(df_group, "build_visual_relationships_arc")
        if df.empty:
            return None

        try:
            # 1. Extract authors from data
            authors = sorted(df[Columns.AUTHOR].unique())
            if len(authors) != 4:
                logger.error(f"MAAP group must have exactly 4 authors, found {len(authors)}")
                return None

            # 2. Add date column
            df = df.copy()
            df["date"] = df[Columns.TIMESTAMP].dt.date

            rows = []

            # 3. Group by date
            for day, day_df in df.groupby("date"):
                n_part = day_df[Columns.X_NUMBER_OF_UNIQUE_PARTICIPANTS_THAT_DAY].iloc[0]
                if n_part not in (2, 3):
                    continue

                total_messages = len(day_df)
                author_counts = day_df[Columns.AUTHOR].value_counts()
                pct = {a: (author_counts.get(a, 0) / total_messages) * 100 for a in authors}

                if n_part == 2:
                    active = sorted([a for a in authors if pct[a] > 0])
                    author_label = " & ".join(active)
                    row = {
                        "type": "Pairs",
                        "author": author_label,
                        "total_messages": total_messages,
                    }
                    for a in authors:
                        row[a] = f"{pct[a]:.0f}%" if a in active else 0

                else:  # n_part == 3
                    missing = [a for a in authors if pct[a] == 0][0]
                    author_label = f"Missing: {missing}"
                    row = {
                        "type": "Non-participant",
                        "author": author_label,
                        "total_messages": total_messages,
                    }
                    for a in authors:
                        row[a] = f"{pct[a]:.0f}%" if pct[a] > 0 else 0

                rows.append(row)

            if not rows:
                logger.error("No qualifying days found for arc diagram.")
                return None

            participation_df = pd.DataFrame(rows)
            col_order = ["type", "author", "total_messages"] + authors
            participation_df = participation_df[col_order]

            logger.success(f"Arc table built: {len(participation_df)} rows")
            return ArcPlotData(participation_df=participation_df)

        except Exception as e:
            logger.exception(f"build_visual_relationships_arc failed: {e}")
            return None

    # === 5. Bubble (Script5) – stub ===
    def build_visual_relationships_bubble(self, df_groups: pd.DataFrame, groups: list[str]) -> BubblePlotData | None:
        ...


# === CODING STANDARD ===
# - `# === Module Docstring ===` before """
# - Google-style docstrings
# - `# === Section Name ===` for all blocks
# - Inline: `# One space, sentence case`
# - Tags: `# TODO:`, `# NOTE:`, `# NEW: (YYYY-MM-DD)`, `# FIXME:`
# - Type hints in function signatures
# - Examples: with >>>
# - No long ----- lines
# - No mixed styles
# - Add markers #NEW at the end of the module

# NEW: DistributionPlotData + build_visual_distribution using LIST_OF_ALL_EMOJIS (2025-11-01)
# NEW: High-level placeholders for Arc & Bubble (2025-11-01)