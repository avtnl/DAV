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
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from .cleaners import MessageCleaner
from .features import FeatureEngineer
from src.constants import Columns, Groups

if TYPE_CHECKING:
    from .utilities import Utililities  # Type hint only


# === Main Orchestrator ===
class DataEditor:
    """Orchestrate the full WhatsApp data cleaning and enrichment pipeline."""

    # === Initialization ===
    def __init__(self) -> None:
        """Initialize shared state and sub-components."""
        # Shared regex patterns
        self.emoji_pattern = re.compile(
            r"["
            r"\U0001f600-\U0001f64f"  # emoticons
            r"\U0001f300-\U0001f5ff"  # symbols & pictographs
            r"\U0001f680-\U0001f6ff"  # transport & map symbols
            r"\U0001f1e0-\U0001f1ff"  # flags (iOS)
            r"\U00002702-\U000027b0"  # Dingbats
            r"\U000024c2-\U0001f251"
            r"\U0001f900-\U0001f9ff"  # supplemental symbols & pictographs
            r"]+",
            flags=re.UNICODE,
        )
        self.ignore_emojis = {
            chr(int(code, 16)) for code in ["1F3FB", "1F3FC", "1F3FD", "1F3FE", "1F3FF"]
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
        self.engineer = FeatureEngineer(initials_map=self.initials_map)

    # === Step 1: Author Initials Mapping ===
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

    # === Main Pipeline ===
    def organize_extended_df(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Run the complete cleaning and feature engineering pipeline.

        Pipeline steps:
            0. Clean messages (media, links, system)
            1. Map authors to initials
            2. Add time features (year, month, week, day)
            3. Add core style (length, response time, etc.)
            4. Add emoji & punctuation features
            5. Add daily percentage features
            6. Add text features (words, capitals, numbers)
            7. Add active years & early leaver (drop leavers)
            8. Create ``whatsapp_group_temp`` (AvT isolated)
            9. Reorder columns

        Args:
            df: Raw WhatsApp export with at least ``TIMESTAMP``, ``AUTHOR``, ``MESSAGE``.

        Returns:
            pd.DataFrame | None: Final feature-complete DataFrame or None on error.

        Examples:
            >>> editor = DataEditor()
            >>> final_df = editor.organize_extended_df(raw_df)
        """
        try:
            if df is None or df.empty:
                logger.error("No valid DataFrame provided for organizing")
                return None

            # 0. Clean messages
            df = self.cleaner.clean_messages(df)
            if df is None:
                return None

            # 1. Apply initials
            df = self.replace_author_by_initials(df)

            # 2–6. Feature engineering
            df = self.engineer.add_all_features(df)

            # 7. Drop early leavers (done in engineer)
            # Note: ``EARLY_LEAVER`` column is added and filtered

            # 8. Create whatsapp_group_temp
            df["whatsapp_group_temp"] = np.where(
                df[Columns.AUTHOR] == Groups.AVT,
                Groups.AVT,
                df[Columns.WHATSAPP_GROUP],
            )
            logger.info("Created 'whatsapp_group_temp' (AvT isolated)")

            # 9. Final column order
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

# NEW: Full blank line standardization (1 line only) (2025-11-03)
# NEW: Enhanced organize_extended_df docstring with full step list (2025-11-03)
# NEW: Added note about early leaver drop (2025-11-03)