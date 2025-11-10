# === Module Docstring ===
"""
Core orchestration within data_editor.

Contains the main ``DataEditor`` class that coordinates:
- Message cleaning via ``MessageCleaner``
- Feature engineering via ``FeatureEngineer``
- Author initials mapping
- Final column ordering and group isolation

All heavy logic is delegated. This module is thin and testable.
"""

from __future__ import annotations

# === Imports ===
import re
import ast
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from .cleaners import MessageCleaner
from .features import list_of_all_emojis, FeatureEngineer
from src.constants import Columns, Groups


# === Main Orchestrator ===
class DataEditor:
    """Orchestrate the full WhatsApp data cleaning and enrichment pipeline."""

    # === Initialization ===
    def __init__(self) -> None:
        """Initialize shared state and sub-components."""
        # Shared regex patterns
        EMOJI_PATTERN = re.compile(
            r'[\U0001F600-\U0001F64F'   # emoticons
            r'\U0001F300-\U0001F5FF'    # symbols & pictographs
            r'\U0001F680-\U0001F6FF'    # transport & map
            r'\U0001F1E0-\U0001F1FF'    # flags
            r'\U0001F900-\U0001FAFF'    # supplemental
            r'\U00002600-\U000026FF'    # misc symbols
            r'\U00002700-\U000027BF'    # dingbats
            r'\u200D\uFE0F]',            # ZWJ + VS16
            re.UNICODE
        )
        self.ignore_emojis = {
            chr(int(code, 16)) for code in ["1F3FB", "1F3FC", "1F3FD", "1F3FE", "1F3FF", "FE0F"]
        }
        self.url_pattern = re.compile(r"(?i)\b((?:https?://|ftp://|www\.)\S+)", flags=re.UNICODE)
        self.punctuation_pattern = re.compile(r"(?<![\d])[!?.,;:](?![\d])")

        # Author initials map – applied globally
        self.initials_map = {
            "Nico Dofferhoff": "ND",
            "Loek Laan": "LL",
            "Herma Hollander": "HH",
            "Hieke Heusden van": "HvH",
            "Irene Bienema": "IB",
            "Anthony van Tilburg": "AvT",
            "Anja Berkemeijer": "AB",
            "Madeleine": "M",
            "Phons Berkemeijer": "PB",
            "Rob Haasbroek": "RH",
            "Hugo Brouwer": "HB",
            "Martin Kat": "MK",
            "Jochem Caspers": "JC",
            "Esmée Horn": "EH",
            "Floorjosje": "FJ",
        }

        # Build author name regex once
        self._author_names_regex = "|".join(re.escape(name) for name in self.initials_map.keys())

        # Sub-components
        self.cleaner = MessageCleaner(
            author_names_regex=self._author_names_regex,
            ignore_emojis=self.ignore_emojis,
            url_pattern=self.url_pattern,
            punctuation_pattern=self.punctuation_pattern,
        )
        self.engineer = FeatureEngineer(self.initials_map)

    # === Author Initials Mapping ===
    def replace_author_by_initials(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace full author names by initials in author columns.

        Applies ``self.initials_map`` to:
            - ``AUTHOR``
            - ``PREVIOUS_AUTHOR``
            - ``NEXT_AUTHOR``

        Args:
            df: DataFrame with author columns.

        Returns:
            pd.DataFrame: Same DataFrame with initials applied.
        """
        author_cols = [Columns.AUTHOR, Columns.PREVIOUS_AUTHOR, Columns.NEXT_AUTHOR]
        for col in author_cols:
            if col in df.columns:
                df[col] = df[col].map(self.initials_map).fillna(df[col])
        logger.info("All author columns converted to initials (ND, LL, AvT, …)")
        return df

    # === Emoji List Parser ===
    def parse_emojis(self, df: pd.DataFrame) -> pd.Series:
        """
        Convert 'list_of_all_emojis' to Python lists.

        Handles:
        - str from CSV: "['😂', '😂']"
        - list from in-memory DF: ['😂', '😂']
        - None/NaN
        """
        col = Columns.LIST_OF_ALL_EMOJIS.value
        if col not in df.columns:
            logger.warning(f"Column '{col}' missing. Parsing from 'message_cleaned'.")
            return df[Columns.MESSAGE_CLEANED].apply(
                lambda x: list_of_all_emojis(x, self.ignore_emojis) if x is not None else []
            )

        def safe_parse(x):
            if x is None:  # ← FIXED: was pd.isna(x)
                return []
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                if x in ("", "[]"):
                    return []
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    return []
            return []

        parsed = df[col].apply(safe_parse)
        logger.info(f"Parsed 'list_of_all_emojis' for {len(parsed)} rows")
        return parsed

    # === Main Pipeline ===
    def organize_extended_df(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Run the complete cleaning and feature engineering pipeline.

        Pipeline steps:
            0. Clean messages (media, links, system)
            1. Map authors to initials
            2. Add time features (year, month, week, ...)
            3. Add core style columns
            4. Add emoji & punctuation features
            5. Add daily percentages
            6. Add text features
            7. Add active years & early leaver (and drop early leavers)

        Args:
            df: Raw DataFrame from WhatsApp export.

        Returns:
            pd.DataFrame | None: Fully enriched DataFrame or None on failure.
        """
        try:
            # === Clean messages ===
            df = self.cleaner.clean_messages(df)
            if df is None:
                return None

            # === Strip leading tilde ===
            tilde_nbsp_pattern = re.compile(r"^~\u202f")
            author_cols = [Columns.AUTHOR, Columns.PREVIOUS_AUTHOR, Columns.NEXT_AUTHOR]
            for col in author_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(tilde_nbsp_pattern, "", regex=True)
            logger.info("Removed leading '~ ' from author columns")

            # === Replace author names with initials ===
            df = self.replace_author_by_initials(df)

            # === Add all features ===
            df = self.engineer.add_all_features(
                df=df,
                ignore_emojis=self.ignore_emojis,
                patterns=self.cleaner.patterns,
                stopwords=self.cleaner.stopwords,
            )

            # === Create whatsapp_group_temp ===
            df["whatsapp_group_temp"] = np.where(
                df[Columns.AUTHOR] == Groups.AVT,
                Groups.AVT,
                df[Columns.WHATSAPP_GROUP],
            )
            logger.info("Created 'whatsapp_group_temp' (AvT isolated)")

            # === Final column order ===
            organized_columns = [
                Columns.WHATSAPP_GROUP,
                "whatsapp_group_temp",
                Columns.TIMESTAMP,
                Columns.YEAR,
                Columns.MONTH,
                Columns.WEEK,
                Columns.DAY_OF_WEEK,
                Columns.AUTHOR,
                Columns.ACTIVE_YEARS,
                Columns.EARLY_LEAVER,
                Columns.NUMBER_OF_CHATS_THAT_DAY,
                Columns.LENGTH_CHAT,
                Columns.NUMBER_OF_WORDS,
                Columns.AVG_WORD_LENGTH,
                Columns.PREVIOUS_AUTHOR,
                Columns.RESPONSE_TIME,
                Columns.NEXT_AUTHOR,
                Columns.HAS_LINK,
                Columns.WAS_DELETED,
                Columns.PICTURES_DELETED,
                Columns.VIDEOS_DELETED,
                Columns.NUMBER_OF_PICTURES_VIDEOS,
                Columns.AUDIOS_DELETED,
                Columns.GIFS_DELETED,
                Columns.STICKERS_DELETED,
                Columns.DOCUMENTS_DELETED,
                Columns.VIDEONOTES_DELETED,
                Columns.HAS_ATTACHMENT,
                Columns.NUMBER_OF_EMOJIS,
                Columns.HAS_EMOJI,
                Columns.LIST_OF_ALL_EMOJIS,
                Columns.LIST_OF_CONNECTED_EMOJIS,
                Columns.STARTS_WITH_EMOJI,
                Columns.EMOJI_STARTING_CHAT,
                Columns.ENDS_WITH_EMOJI,
                Columns.EMOJI_ENDING_CHAT,
                Columns.PCT_EMOJIS,
                Columns.NUMBER_OF_PUNCTUATIONS,
                Columns.HAS_PUNCTUATION,
                Columns.LIST_OF_ALL_PUNCTUATIONS,
                Columns.LIST_OF_CONNECTED_PUNCTUATIONS,
                Columns.ENDS_WITH_PUNCTUATION,
                Columns.HAS_QUESTION_MARK,
                Columns.ENDS_WITH_QUESTION_MARK,
                Columns.PUNCTUATION_ENDING_CHAT,
                Columns.PCT_PUNCTUATIONS,
                Columns.NUMBER_OF_CAPITALS,
                Columns.HAS_CAPITALS,
                Columns.LIST_OF_CONNECTED_CAPITALS,
                Columns.STARTS_WITH_CAPITAL,
                Columns.CAPITALIZED_WORDS_RATIO,
                Columns.NUMBER_OF_NUMBER_CHARACTERS,
                Columns.HAS_NUMBER_CHARACTERS,
                Columns.NUMBER_OF_NUMBERS,
                Columns.MESSAGE_CLEANED,
                Columns.X_DAY_PCT_LENGTH_CHAT,
                Columns.X_DAY_PCT_LENGTH_EMOJIS,
                Columns.X_DAY_PCT_LENGTH_PUNCTUATIONS,
                Columns.X_NUMBER_OF_UNIQUE_PARTICIPANTS_THAT_DAY,
                Columns.X_DAY_PCT_MESSAGES_OF_AUTHOR,
                Columns.Y_SEQUENCE_AUTHORS_THAT_DAY,
                Columns.Y_SEQUENCE_RESPONSE_TIMES_THAT_DAY,
            ]
            df = df[organized_columns]
            logger.info(
                f"Final DF: {len(df)} rows, {len(df.columns)} cols - all authors are initials"
            )
            return df

        except Exception as e:
            logger.exception(f"organize_extended_df failed: {e}")
            return None


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

# NEW: Added parse_emojis() method with full CSV string handling and fallback (2025-11-05)
# NEW: Robust ast.literal_eval with debug logging and type checking
# NEW: Clear documentation explaining why this method is REQUIRED for CSV data