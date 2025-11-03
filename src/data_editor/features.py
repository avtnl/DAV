# === Module Docstring ===
"""
Feature engineering module within data_editor.

Computes behavioral and linguistic features:
- Time features
- Core style (length, response time, etc.)
- Emoji & punctuation
- Daily percentages
- Text analysis
- Active years & early leaver

All features are vectorized and use ``utilities`` functions.
"""

from __future__ import annotations

# === Imports ===
from typing import Dict, List, Any
import pandas as pd

from src.constants import Columns
from .utilities import (
    length_chat,
    count_words,
    avg_word_length,
    list_of_all_emojis,
    list_of_connected_emojis,
    count_punctuations,
    has_punctuation,
    list_of_all_punctuations,
    list_of_connected_punctuations,
    ends_with_emoji,
    emoji_ending_chat,
    ends_with_punctuation,
    punctuation_ending_chat,
    starts_with_emoji,
    emoji_starting_chat,
    has_question_mark,
    ends_with_question_mark,
    count_capitals,
    has_capitals,
    list_of_connected_capitals,
    starts_with_capital,
    capitalized_words_ratio,
    count_number_characters,
    has_number_characters,
    count_numbers,
)


# === Main Feature Engineer ===
class FeatureEngineer:
    """Compute all behavioral and linguistic features."""

    # === Initialization ===
    def __init__(self, initials_map: Dict[str, str]) -> None:
        """Initialize with author initials mapping.

        Args:
            initials_map: Mapping from full name to initials.
        """
        self.initials_map = initials_map

    # === Step 2: Time Features ===
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add year, month, week, and day of week.

        Args:
            df: DataFrame with ``TIMESTAMP``.

        Returns:
            pd.DataFrame: With added time columns.
        """
        df[Columns.YEAR] = df[Columns.TIMESTAMP].dt.year
        df[Columns.MONTH] = df[Columns.TIMESTAMP].dt.month
        df[Columns.WEEK] = df[Columns.TIMESTAMP].dt.isocalendar().week
        df[Columns.DAY_OF_WEEK] = df[Columns.TIMESTAMP].dt.dayofweek
        return df

    # === Step 3: Core Style Columns ===
    def add_core_style(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add length, previous/next author, response time, and daily chat count.

        Args:
            df: DataFrame with ``MESSAGE_CLEANED``, ``AUTHOR``, ``TIMESTAMP``.

        Returns:
            pd.DataFrame: With core style columns.
        """
        df[Columns.NUMBER_OF_CHATS_THAT_DAY] = (
            df.groupby(df[Columns.TIMESTAMP].dt.date)[Columns.TIMESTAMP]
            .transform("cumcount") + 1
        )
        df[Columns.LENGTH_CHAT] = df[Columns.MESSAGE_CLEANED].apply(length_chat)
        df[Columns.PREVIOUS_AUTHOR] = df[Columns.AUTHOR].shift(1).fillna("")
        df[Columns.NEXT_AUTHOR] = df[Columns.AUTHOR].shift(-1).fillna("")
        df[Columns.RESPONSE_TIME] = (
            df[Columns.TIMESTAMP] - df[Columns.TIMESTAMP].shift(1)
        ).dt.total_seconds().fillna(0)
        return df

    # === Step 4: Emoji & Punctuation Features ===
    def add_emoji_punct_features(self, df: pd.DataFrame, ignore_emojis: set, patterns: dict) -> pd.DataFrame:
        """Add all emoji and punctuation features.

        Args:
            df: DataFrame with ``MESSAGE_CLEANED``.
            ignore_emojis: Set of emoji codes to ignore.
            patterns: Dict with 'connected_emoji' and 'connected_punct' regexes.

        Returns:
            pd.DataFrame: With emoji/punct columns.
        """
        msg = df[Columns.MESSAGE_CLEANED]

        df[Columns.LIST_OF_ALL_EMOJIS] = msg.apply(list_of_all_emojis, ignore_emojis=ignore_emojis)
        df[Columns.NUMBER_OF_EMOJIS] = df[Columns.LIST_OF_ALL_EMOJIS].str.len()
        df[Columns.HAS_EMOJI] = df[Columns.NUMBER_OF_EMOJIS] > 0

        df[Columns.LIST_OF_CONNECTED_EMOJIS] = msg.apply(
            list_of_connected_emojis, pattern=patterns["connected_emoji"]
        )

        df[Columns.NUMBER_OF_PUNCTUATIONS] = msg.apply(
            count_punctuations, pattern=patterns["punctuation"]
        )
        df[Columns.HAS_PUNCTUATION] = msg.apply(has_punctuation, pattern=patterns["punctuation"])
        df[Columns.LIST_OF_ALL_PUNCTUATIONS] = msg.apply(
            list_of_all_punctuations, pattern=patterns["punctuation"]
        )
        df[Columns.LIST_OF_CONNECTED_PUNCTUATIONS] = msg.apply(
            list_of_connected_punctuations, pattern=patterns["connected_punct"]
        )

        df[Columns.ENDS_WITH_EMOJI] = msg.apply(ends_with_emoji, ignore_emojis=ignore_emojis)
        df[Columns.EMOJI_ENDING_CHAT] = msg.apply(emoji_ending_chat, ignore_emojis=ignore_emojis)
        df[Columns.ENDS_WITH_PUNCTUATION] = msg.apply(ends_with_punctuation, pattern=patterns["punctuation"])
        df[Columns.PUNCTUATION_ENDING_CHAT] = msg.apply(punctuation_ending_chat, pattern=patterns["punctuation"])

        df[Columns.PCT_EMOJIS] = (df[Columns.NUMBER_OF_EMOJIS] / df[Columns.LENGTH_CHAT].replace(0, 1)) * 100
        df[Columns.PCT_PUNCTUATIONS] = (df[Columns.NUMBER_OF_PUNCTUATIONS] / df[Columns.LENGTH_CHAT].replace(0, 1)) * 100

        return df

    # === Step 5: Daily Percentages ===
    def add_daily_percentages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add daily percentage features.

        Args:
            df: DataFrame with ``TIMESTAMP``, ``LENGTH_CHAT``, ``NUMBER_OF_EMOJIS``, etc.

        Returns:
            pd.DataFrame: With daily % columns.
        """
        date = df[Columns.TIMESTAMP].dt.date

        df[Columns.X_DAY_PCT_LENGTH_CHAT] = (
            df[Columns.LENGTH_CHAT] / df.groupby(date)[Columns.LENGTH_CHAT].transform("sum")
        ) * 100

        df[Columns.X_DAY_PCT_LENGTH_EMOJIS] = (
            df[Columns.NUMBER_OF_EMOJIS] / df.groupby(date)[Columns.NUMBER_OF_EMOJIS].transform("sum")
        ) * 100

        df[Columns.X_DAY_PCT_LENGTH_PUNCTUATIONS] = (
            df[Columns.NUMBER_OF_PUNCTUATIONS] / df.groupby(date)[Columns.NUMBER_OF_PUNCTUATIONS].transform("sum")
        ) * 100

        df[Columns.X_NUMBER_OF_UNIQUE_PARTICIPANTS_THAT_DAY] = df.groupby(date)[Columns.AUTHOR].transform("nunique")

        daily_total = df.groupby(date).size()
        daily_author = df.groupby([date, Columns.AUTHOR]).size()
        pct = (daily_author / daily_total.reindex(daily_author.index.get_level_values(0))) * 100
        df[Columns.X_DAY_PCT_MESSAGES_OF_AUTHOR] = pct.reindex(
            df.set_index([date, Columns.AUTHOR]).index
        ).values

        df[Columns.Y_SEQUENCE_AUTHORS_THAT_DAY] = df.groupby(date)[Columns.AUTHOR].transform(lambda g: g.tolist())
        df[Columns.Y_SEQUENCE_RESPONSE_TIMES_THAT_DAY] = df.groupby(date)[Columns.RESPONSE_TIME].transform(lambda g: g.tolist())

        return df

    # === Step 6: Text Features ===
    def add_text_features(self, df: pd.DataFrame, stopwords: set) -> pd.DataFrame:
        """Add word count, capitalization, numbers, and question marks.

        Args:
            df: DataFrame with ``MESSAGE_CLEANED``.
            stopwords: Set of Dutch stopwords.

        Returns:
            pd.DataFrame: With text feature columns.
        """
        msg = df[Columns.MESSAGE_CLEANED]

        df[Columns.NUMBER_OF_WORDS] = msg.apply(count_words, stopwords=stopwords)
        df[Columns.AVG_WORD_LENGTH] = msg.apply(avg_word_length, stopwords=stopwords)

        df[Columns.STARTS_WITH_EMOJI] = msg.apply(starts_with_emoji, ignore_emojis=set())
        df[Columns.EMOJI_STARTING_CHAT] = msg.apply(emoji_starting_chat, ignore_emojis=set())

        df[Columns.HAS_QUESTION_MARK] = msg.apply(has_question_mark)
        df[Columns.ENDS_WITH_QUESTION_MARK] = msg.apply(ends_with_question_mark)

        df[Columns.NUMBER_OF_CAPITALS] = msg.apply(count_capitals)
        df[Columns.HAS_CAPITALS] = msg.apply(has_capitals)
        df[Columns.LIST_OF_CONNECTED_CAPITALS] = msg.apply(list_of_connected_capitals)
        df[Columns.STARTS_WITH_CAPITAL] = msg.apply(starts_with_capital)
        df[Columns.CAPITALIZED_WORDS_RATIO] = msg.apply(capitalized_words_ratio)

        df[Columns.NUMBER_OF_NUMBER_CHARACTERS] = msg.apply(count_number_characters)
        df[Columns.HAS_NUMBER_CHARACTERS] = msg.apply(has_number_characters)
        df[Columns.NUMBER_OF_NUMBERS] = msg.apply(count_numbers)

        # Attachment & media summary
        media_cols = [
            Columns.PICTURES_DELETED, Columns.VIDEOS_DELETED, Columns.AUDIOS_DELETED,
            Columns.GIFS_DELETED, Columns.STICKERS_DELETED, Columns.DOCUMENTS_DELETED,
            Columns.VIDEONOTES_DELETED
        ]
        df[Columns.HAS_ATTACHMENT] = df[media_cols].sum(axis=1) > 0
        df[Columns.NUMBER_OF_PICTURES_VIDEOS] = df[Columns.PICTURES_DELETED] + df[Columns.VIDEOS_DELETED]

        return df

    # === Step 7: Active Years & Early Leaver ===
    def add_active_years_and_leaver(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add active years and early leaver flag, then drop leavers.

        Args:
            df: DataFrame with ``TIMESTAMP`` and ``AUTHOR``.

        Returns:
            pd.DataFrame: Filtered DataFrame with active years.
        """
        if Columns.TIMESTAMP not in df.columns or Columns.AUTHOR not in df.columns:
            df[Columns.ACTIVE_YEARS] = 0
            df[Columns.EARLY_LEAVER] = False
            return df

        year_series = df[Columns.TIMESTAMP].dt.year
        active_years = df.groupby(Columns.AUTHOR)[year_series.name].nunique()
        df[Columns.ACTIVE_YEARS] = df[Columns.AUTHOR].map(active_years).fillna(0)

        last_year = year_series.max()
        active_last_year = df[df[Columns.TIMESTAMP].dt.year == last_year][Columns.AUTHOR].unique()
        df[Columns.EARLY_LEAVER] = ~df[Columns.AUTHOR].isin(active_last_year)

        df = df[~df[Columns.EARLY_LEAVER]].reset_index(drop=True)
        return df

    # === Orchestrator: Add All Features ===
    def add_all_features(
        self,
        df: pd.DataFrame,
        ignore_emojis: set,
        patterns: dict,
        stopwords: set,
    ) -> pd.DataFrame:
        """Run all feature engineering steps.

        Args:
            df: Cleaned DataFrame from ``MessageCleaner``.
            ignore_emojis: Set of emoji codes to ignore.
            patterns: Dict with pre-compiled regexes.
            stopwords: Set of Dutch stopwords.

        Returns:
            pd.DataFrame: Fully enriched DataFrame.
        """
        df = self.add_time_features(df)
        df = self.add_core_style(df)
        df = self.add_emoji_punct_features(df, ignore_emojis, patterns)
        df = self.add_daily_percentages(df)
        df = self.add_text_features(df, stopwords)
        df = self.add_active_years_and_leaver(df)
        return df


# === CODING STANDARD (APPLY TO ALL CODE) ===
# - `# === Module Docstring ===` before """
# - Google-style docstrings
# - `# === Section Name ===` for all blocks
# - Inline: `# One space, sentence case`
# - Tags: `# TODO:`, `# NOTE:`, `# NEW: (YYYY-MM-DD)`, `# FIXME:`
# - Type hints in function signatures
# - Examples: with >>>
# - No long ----- lines
# - No mixed styles
# - Add markers #NEW at the end of the module capturing the latest changes.

# NEW: Created features.py with FeatureEngineer class (2025-11-03)
# NEW: All features grouped by pipeline step (2025-11-03)
# NEW: add_all_features orchestrates full enrichment (2025-11-03)
# NEW: Strict 1-blank-line rule enforced (2025-11-03)
# NEW: Full Google-style docstrings (2025-11-03)