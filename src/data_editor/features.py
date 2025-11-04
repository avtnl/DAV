# === Module Docstring ===
"""
Feature engineering module within data_editor

Adds behavioral, linguistic, and temporal features to the cleaned DataFrame.

Key features:
- Daily percentages (chat length, emojis, punctuation)
- Response time, previous/next author
- Emoji & punctuation chains
- Capitalization, numbers, links
- Author activity stats
"""

from __future__ import annotations

# === Imports ===
import re
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from loguru import logger
import emoji

from src.constants import Columns


# === Feature Engineer ===
class FeatureEngineer:
    """Add all engineered features to the DataFrame."""

    def __init__(self) -> None:
        """Initialize with no state."""
        pass

    # === Main Entry Point ===
    def add_all_features(
        self,
        df: pd.DataFrame,
        ignore_emojis: Set[str],
        patterns: Dict[str, re.Pattern],
        stopwords: Set[str],
    ) -> pd.DataFrame:
        """
        Add all features in sequence.

        Args:
            df: Cleaned DataFrame with core columns.
            ignore_emojis: Emojis to skip (e.g., skin tones).
            patterns: Pre-compiled regex patterns.
            stopwords: Dutch stopwords for word analysis.

        Returns:
            DataFrame with all features added.
        """
        try:
            # === 1. Add LENGTH_CHAT first ===
            df[Columns.LENGTH_CHAT] = df[Columns.MESSAGE_CLEANED].str.len()

            # === 2. Then all other features ===
            df = self.add_basic_features(df)
            df = self.add_emoji_punct_features(df, ignore_emojis, patterns)
            df = self.add_capitalization_features(df)
            df = self.add_number_features(df)
            df = self.add_daily_percentages(df)
            df = self.add_response_features(df)
            df = self.add_author_stats(df)
            return df
        except Exception as e:
            logger.exception(f"Feature engineering failed: {e}")
            return df

    # === Basic Features ===
    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add year, month, week, day of week."""
        try:
            df[Columns.YEAR] = df[Columns.TIMESTAMP].dt.year
            df[Columns.MONTH] = df[Columns.TIMESTAMP].dt.month
            df[Columns.WEEK] = df[Columns.TIMESTAMP].dt.isocalendar().week
            df[Columns.DAY_OF_WEEK] = df[Columns.TIMESTAMP].dt.dayofweek
            return df
        except Exception as e:
            logger.exception(f"Failed to add basic features: {e}")
            return df

    # === Emoji & Punctuation Features ===
    def add_emoji_punct_features(
        self, df: pd.DataFrame, ignore_emojis: Set[str], patterns: Dict[str, re.Pattern]
    ) -> pd.DataFrame:
        """Add emoji and punctuation chain features."""
        try:
            msg = df[Columns.MESSAGE_CLEANED]

            # Emoji counts
            df[Columns.NUMBER_OF_EMOJIS] = msg.apply(
                lambda x: sum(1 for c in x if c in emoji.EMOJI_DATA and c not in ignore_emojis)
            )
            df[Columns.HAS_EMOJI] = df[Columns.NUMBER_OF_EMOJIS] > 0
            df[Columns.LIST_OF_ALL_EMOJIS] = msg.apply(
                lambda x: "".join(c for c in x if c in emoji.EMOJI_DATA and c not in ignore_emojis)
            )
            df[Columns.LIST_OF_CONNECTED_EMOJIS] = msg.apply(
                lambda x: "".join(re.findall(patterns["connected_emoji"], x))
            )
            df[Columns.STARTS_WITH_EMOJI] = msg.str.startswith(tuple(emoji.EMOJI_DATA.keys()))
            df[Columns.ENDS_WITH_EMOJI] = msg.str.endswith(tuple(emoji.EMOJI_DATA.keys()))
            df[Columns.PCT_EMOJIS] = (df[Columns.NUMBER_OF_EMOJIS] / df[Columns.LENGTH_CHAT].replace(0, 1)) * 100

            # Punctuation counts
            df[Columns.NUMBER_OF_PUNCTUATIONS] = msg.apply(
                lambda x: len(re.findall(patterns["punctuation"], x))
            )
            df[Columns.HAS_PUNCTUATION] = df[Columns.NUMBER_OF_PUNCTUATIONS] > 0
            df[Columns.LIST_OF_ALL_PUNCTUATIONS] = msg.apply(
                lambda x: "".join(re.findall(patterns["punctuation"], x))
            )
            df[Columns.LIST_OF_CONNECTED_PUNCTUATIONS] = msg.apply(
                lambda x: "".join(re.findall(patterns["connected_punct"], x))
            )
            df[Columns.HAS_QUESTION_MARK] = msg.str.contains(r"\?", regex=True)
            df[Columns.ENDS_WITH_QUESTION_MARK] = msg.str.endswith("?")
            df[Columns.PCT_PUNCTUATIONS] = (df[Columns.NUMBER_OF_PUNCTUATIONS] / df[Columns.LENGTH_CHAT].replace(0, 1)) * 100

            return df
        except Exception as e:
            logger.exception(f"Failed to add emoji/punct features: {e}")
            return df

    # === Capitalization Features ===
    def add_capitalization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add capitalization and word-level features."""
        try:
            msg = df[Columns.MESSAGE_CLEANED]

            df[Columns.NUMBER_OF_CAPITALS] = msg.apply(lambda x: sum(1 for c in x if c.isupper()))
            df[Columns.HAS_CAPITALS] = df[Columns.NUMBER_OF_CAPITALS] > 0
            df[Columns.LIST_OF_CONNECTED_CAPITALS] = msg.apply(lambda x: "".join(re.findall(r"[A-Z]{2,}", x)))
            df[Columns.STARTS_WITH_CAPITAL] = msg.str.startswith(tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            df[Columns.CAPITALIZED_WORDS_RATIO] = msg.apply(
                lambda x: sum(1 for w in x.split() if w and w[0].isupper()) / len(x.split()) if x.split() else 0
            )

            return df
        except Exception as e:
            logger.exception(f"Failed to add capitalization features: {e}")
            return df

    # === Number Features ===
    def add_number_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add number character and digit features."""
        try:
            msg = df[Columns.MESSAGE_CLEANED]

            df[Columns.NUMBER_OF_NUMBER_CHARACTERS] = msg.apply(lambda x: sum(1 for c in x if c.isdigit()))
            df[Columns.HAS_NUMBER_CHARACTERS] = df[Columns.NUMBER_OF_NUMBER_CHARACTERS] > 0
            df[Columns.NUMBER_OF_NUMBERS] = msg.apply(lambda x: len(re.findall(r"\b\d+\b", x)))

            return df
        except Exception as e:
            logger.exception(f"Failed to add number features: {e}")
            return df

    # === Daily Percentages ===
    def add_daily_percentages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add daily percentage of chat length by author."""
        try:
            # Daily total length (all authors)
            daily_total = df.groupby(df[Columns.TIMESTAMP].dt.date)[Columns.LENGTH_CHAT].transform('sum')

            # Percentage of daily chat length
            df[Columns.X_DAY_PCT_LENGTH_CHAT] = (df[Columns.LENGTH_CHAT] / daily_total.replace(0, 1)) * 100

            return df
        except Exception as e:
            logger.exception(f"Failed to add daily percentages: {e}")
            return df

    # === Response Time & Author Sequence ===
    def add_response_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add response time and previous/next author."""
        try:
            df = df.sort_values(Columns.TIMESTAMP)

            df[Columns.PREVIOUS_AUTHOR] = df[Columns.AUTHOR].shift(1)
            df[Columns.NEXT_AUTHOR] = df[Columns.AUTHOR].shift(-1)

            # Response time (seconds)
            df[Columns.RESPONSE_TIME] = df[Columns.TIMESTAMP].diff().dt.total_seconds().fillna(0)

            return df
        except Exception as e:
            logger.exception(f"Failed to add response features: {e}")
            return df

    # === Author Activity Stats ===
    def add_author_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add active years, early leaver, message count."""
        try:
            author_year = df.groupby(Columns.AUTHOR)[Columns.YEAR]
            df[Columns.ACTIVE_YEARS] = df[Columns.AUTHOR].map(author_year.nunique())
            df[Columns.EARLY_LEAVER] = df[Columns.AUTHOR].map(author_year.min()) <= df[Columns.YEAR].min() + 1
            df[Columns.NUMBER_OF_CHATS_THAT_DAY] = df.groupby([df[Columns.TIMESTAMP].dt.date, Columns.AUTHOR]).cumcount() + 1

            return df
        except Exception as e:
            logger.exception(f"Failed to add author stats: {e}")
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
# NEW: Added all feature methods with error handling (2025-11-03)
# NEW: (2025-11-04) – Fixed add_daily_percentages using transform('sum') to avoid index alignment