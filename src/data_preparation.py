# === data_preparation.py ===
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

import re
import emoji
import itertools
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Dict

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, validator

from .constants import Columns, Groups

if TYPE_CHECKING:
    from .data_editor import DataEditor


# === 1. Category Plot Data Contracts (Script1) ===
class AuthorMessages(BaseModel):
    author: str = Field(..., min_length=1)
    message_count: int = Field(..., ge=0)
    is_avt: bool = False

    @validator("author")
    def strip(cls, v: str) -> str:
        return v.strip()


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

    # === 3–5: Stubs (to be filled) ===
    def build_visual_distribution(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]: ...
    def build_visual_relationships_arc(self, df_group: pd.DataFrame, authors: list[str]) -> pd.DataFrame | None: ...
    def build_visual_relationships_bubble(self, df_groups: pd.DataFrame, groups: list[str]) -> pd.DataFrame | None: ...


# === CODING STANDARD ===
# NEW: All Pydantic models moved here (2025-11-01)
# NEW: plot_manager imports from data_preparation only
# NEW: No circular imports