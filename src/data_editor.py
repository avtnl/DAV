# === Module Docstring ===
"""
WhatsApp Chat Analyzer – Data Editor

Cleans and enriches raw WhatsApp message data into a feature-complete DataFrame.
Applies text processing, emoji/punctuation analysis, response time, daily stats,
and author initials mapping.

All column names use :class:`constants.Columns` for consistency.
"""

# === Imports ===
import re
import string
from typing import List

import emoji
import nltk
import numpy as np
import pandas as pd
from loguru import logger
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from .constants import Columns

# Download required NLTK data
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Global: Punctuation to remove during word analysis (exclude @ and &)
BROAD_PUNCTUATION = "".join(set(string.punctuation) - {"@", "&"})


# === Main Class ===
class DataEditor:
    """Clean and enrich WhatsApp message data with behavioral and linguistic features."""

    # === Initialization ===
    def __init__(self) -> None:
        """Initialize regex patterns, emoji sets, stopwords, and author initials map."""
        self.emoji_pattern = re.compile(
            r"["  # noqa: E501
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
        self.stopwords = set(stopwords.words("dutch"))
        self.punctuation_pattern = re.compile(r"(?<![\d])[!?.,;:](?![\d])")
        self.connected_emoji_pattern = re.compile(
            r"([" + "".join(self.emoji_pattern.pattern[1:-2]) + r"]{2,})", flags=re.UNICODE
        )
        self.connected_punctuation_pattern = re.compile(r"([!?.,;:]{2,})")

        # Initials map – applied to all author columns
        self.initials_map = {
            "Nico Dofferhoff": "ND",
            "Loek van der Laan": "LL",
            "Herma Hollander": "HH",
            "Hieke van Heusden": "HvH",
            "Irene Bienema": "IB",
            "Anthony van Tilburg": "AvT",
            "Anja Berkemeijer": "AB",
            "Madeleine": "M",
            "Phons Berkemeijer": "PB",
            "Rob Haasbroek": "RH",
            "Hugo Brouwer": "HB",
            "Martin Kat": "MK",
        }

    # === Step 0: Message Cleaning ===
    def clean_for_deleted_media_patterns(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Remove media placeholders, links, and system messages.

        Args:
            df (pd.DataFrame): Raw input with message column.

        Returns:
            pd.DataFrame | None: Cleaned DataFrame or None on error.
        """
        if df is None or df.empty:
            logger.error("No valid DataFrame provided for cleaning deleted media patterns")
            return None
        try:
            # Initialize feature columns
            df[Columns.HAS_EMOJI] = False
            df[Columns.NUMBER_OF_EMOJIS] = 0
            df[Columns.HAS_LINK] = False
            df[Columns.WAS_DELETED] = False
            df["number_of_changes_to_group"] = 0  # Internal only
            for col in [
                Columns.PICTURES_DELETED, Columns.VIDEOS_DELETED, Columns.AUDIOS_DELETED,
                Columns.GIFS_DELETED, Columns.STICKERS_DELETED, Columns.DOCUMENTS_DELETED,
                Columns.VIDEONOTES_DELETED
            ]:
                df[col] = 0
            df[Columns.MESSAGE_CLEANED] = df[Columns.MESSAGE]

            non_media_patterns = [
                (r"Dit bericht is verwijderd\.", "message deleted", re.IGNORECASE),
                (
                    r"(?:Anthony van Tilburg|Anja Berkemeijer|Phons Berkemeijer|Madeleine) heeft de groepsafbeelding gewijzigd",
                    "grouppicture",
                    re.IGNORECASE,
                ),
            ]
            media_patterns = [
                (r"afbeelding\s*weggelaten", "picture deleted", re.IGNORECASE),
                (r"video\s*weggelaten", "video deleted", re.IGNORECASE),
                (r"audio\s*weggelaten", "audio deleted", re.IGNORECASE),
                (r"GIF\s*weggelaten", "GIF deleted", re.IGNORECASE),
                (r"sticker\s*weggelaten", "sticker deleted", re.IGNORECASE),
                (r"document\s*weggelaten", "document deleted", re.IGNORECASE),
                (r"videonotitie\s*weggelaten", "video note deleted", re.IGNORECASE),
            ]
            link_removal_pattern = r'(\s*https?://[^\s<>"{}|\\^`\[\]]+[\.,;:!?]?\s*)'
            fallback_pattern = r"\s*[\u200e\u200f]*\[\d{2}-\d{2}-\d{4},\s*\d{2}:\d{2}:\d{2}\]\s*(?:Anthony van Tilburg|Anja Berkemeijer|Phons Berkemeijer|Madeleine)[\s\u200e\u200f]*:.*"

            def clean_message(row):
                message = row[Columns.MESSAGE]
                row[Columns.HAS_EMOJI] = self.has_emoji(message)
                row[Columns.NUMBER_OF_EMOJIS] = self.count_emojis(message)
                row[Columns.HAS_LINK] = self.has_link(message)
                row[Columns.WAS_DELETED] = self.was_deleted(message)
                row["number_of_changes_to_group"] = self.changes_to_grouppicture(message)

                if self.has_link(message):
                    message = re.sub(link_removal_pattern, " ", message, flags=re.IGNORECASE)
                    message = re.sub(r"\s+", " ", message).strip()

                for pattern, _, flags in non_media_patterns:
                    message = re.sub(pattern, "", message, flags=flags).strip()

                for pattern, change, flags in media_patterns:
                    matches = re.findall(pattern, message, flags=flags)
                    count = len(matches)
                    if count > 0:
                        if change == "picture deleted":
                            row[Columns.PICTURES_DELETED] += count
                        elif change == "video deleted":
                            row[Columns.VIDEOS_DELETED] += count
                        elif change == "audio deleted":
                            row[Columns.AUDIOS_DELETED] += count
                        elif change == "GIF deleted":
                            row[Columns.GIFS_DELETED] += count
                        elif change == "sticker deleted":
                            row[Columns.STICKERS_DELETED] += count
                        elif change == "document deleted":
                            row[Columns.DOCUMENTS_DELETED] += count
                        elif change == "video note deleted":
                            row[Columns.VIDEONOTES_DELETED] += count
                        message = re.sub(pattern, "", message, flags=flags).strip()

                total_media_deleted = sum(
                    row[col]
                    for col in [
                        Columns.PICTURES_DELETED, Columns.VIDEOS_DELETED, Columns.AUDIOS_DELETED,
                        Columns.GIFS_DELETED, Columns.STICKERS_DELETED, Columns.DOCUMENTS_DELETED,
                        Columns.VIDEONOTES_DELETED
                    ]
                )
                if total_media_deleted > 0:
                    ta_pattern = r"[\s\u200e\u200f]*\[\d{2}-\d{2}-\d{4},\s*\d{2}:\d{2}:\d{2}\]\s*[^:]*:[\s\u200e\u200f]*$"
                    message = re.sub(ta_pattern, "", message).strip()

                if re.search(fallback_pattern, message, flags=re.IGNORECASE):
                    message = re.sub(fallback_pattern, "", message, flags=re.IGNORECASE).strip()

                if not message or message.strip() == "":
                    message = "completely removed"
                row[Columns.MESSAGE_CLEANED] = message
                return row

            df = df.apply(clean_message, axis=1)
            logger.info(
                f"Cleaned messages: {df[[Columns.MESSAGE, Columns.MESSAGE_CLEANED]].head(10).to_string()}"
            )
            return df
        except Exception as e:
            logger.exception(f"Failed to clean messages: {e}")
            return None

    # === Step 1: Author Initials Mapping ===
    def replace_author_by_initials(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace full names with initials in all author columns.

        Args:
            df (pd.DataFrame): Input with author, previous_author, next_author.

        Returns:
            pd.DataFrame: Authors replaced with initials.
        """
        author_cols = [Columns.AUTHOR, Columns.PREVIOUS_AUTHOR, Columns.NEXT_AUTHOR]
        for col in author_cols:
            if col in df.columns:
                df[col] = df[col].map(self.initials_map).fillna(df[col])
        logger.info("All author columns converted to initials (ND, LL, AvT, …)")
        return df

    # === Step 2: Time Features (computed inline in organize_extended_df) ===
    # df[Columns.YEAR] = df[Columns.TIMESTAMP].dt.year
    # df[Columns.MONTH] = df[Columns.TIMESTAMP].dt.month
    # df[Columns.WEEK] = df[Columns.TIMESTAMP].dt.isocalendar().week
    # df[Columns.DAY_OF_WEEK] = df[Columns.TIMESTAMP].dt.dayofweek

    # === Step 3: Core Style Columns ===
    def number_of_chats_that_day(self, df: pd.DataFrame) -> pd.Series:
        """Cumulative message count per day.

        Args:
            df (pd.DataFrame): Input with timestamp.

        Returns:
            pd.Series: Message number that day (1-based).
        """
        return df.groupby(df[Columns.TIMESTAMP].dt.date)[Columns.TIMESTAMP].transform("cumcount") + 1

    def length_chat(self, text) -> int:
        """Length of cleaned message.

        Args:
            text: Message string or NaN.

        Returns:
            int: Character count.
        """
        return len(str(text)) if pd.notna(text) else 0

    def previous_author(self, df: pd.DataFrame) -> pd.Series:
        """Author of previous message.

        Args:
            df (pd.DataFrame): Input with author column.

        Returns:
            pd.Series: Previous author (empty string if first).
        """
        return df[Columns.AUTHOR].shift(1).fillna("")

    def response_time(self, df: pd.DataFrame) -> pd.Series:
        """Time in seconds since previous message.

        Args:
            df (pd.DataFrame): Input with timestamp.

        Returns:
            pd.Series: Response time in seconds (0 if first).
        """
        prev = df[Columns.TIMESTAMP].shift(1)
        return (df[Columns.TIMESTAMP] - prev).dt.total_seconds().fillna(0)

    def next_author(self, df: pd.DataFrame) -> pd.Series:
        """Author of next message.

        Args:
            df (pd.DataFrame): Input with author column.

        Returns:
            pd.Series: Next author (empty string if last).
        """
        return df[Columns.AUTHOR].shift(-1).fillna("")

    # === Step 4: Emoji / Punctuation ===
    def list_of_all_emojis(self, text: str) -> List[str]:
        """List all individual emojis in message.

        Args:
            text (str): Input message.

        Returns:
            List[str]: List of emoji characters.
        """
        if not isinstance(text, str):
            return []
        return [c for c in text if c in emoji.EMOJI_DATA and c not in self.ignore_emojis]

    def list_of_connected_emojis(self, text: str) -> List[str]:
        """Find sequences of 2+ connected emojis.

        Args:
            text (str): Input message.

        Returns:
            List[str]: List of emoji sequences.
        """
        if not isinstance(text, str):
            return []
        return re.findall(self.connected_emoji_pattern, text)

    def count_punctuations(self, text: str) -> int:
        """Count standalone punctuation marks.

        Args:
            text (str): Input message.

        Returns:
            int: Number of punctuation marks.
        """
        if not isinstance(text, str):
            return 0
        return len(re.findall(self.punctuation_pattern, text))

    def has_punctuation(self, text: str) -> bool:
        """Check for any punctuation.

        Args:
            text (str): Input message.

        Returns:
            bool: True if punctuation exists.
        """
        return self.count_punctuations(text) > 0

    def list_of_all_punctuations(self, text: str) -> List[str]:
        """List all standalone punctuation.

        Args:
            text (str): Input message.

        Returns:
            List[str]: List of punctuation characters.
        """
        if not isinstance(text, str):
            return []
        return re.findall(self.punctuation_pattern, text)

    def list_of_connected_punctuations(self, text: str) -> List[str]:
        """Find sequences of 2+ punctuation.

        Args:
            text (str): Input message.

        Returns:
            List[str]: List of punctuation sequences.
        """
        if not isinstance(text, str):
            return []
        return re.findall(self.connected_punctuation_pattern, text)

    def ends_with_emoji(self, text: str) -> bool:
        """Check if message ends with emoji.

        Args:
            text (str): Input message.

        Returns:
            bool: True if ends with emoji.
        """
        if not isinstance(text, str) or not text.strip():
            return False
        return text.strip()[-1] in emoji.EMOJI_DATA and text.strip()[-1] not in self.ignore_emojis

    def emoji_ending_chat(self, text: str) -> List[str]:
        """Extract trailing emoji sequence.

        Args:
            text (str): Input message.

        Returns:
            List[str]: Reversed and corrected ending emojis.
        """
        if not isinstance(text, str):
            return []
        match = re.search(self.emoji_pattern, text.strip()[::-1])
        return list(match.group(0)[::-1]) if match else []

    def ends_with_punctuation(self, text: str) -> bool:
        """Check if message ends with punctuation.

        Args:
            text (str): Input message.

        Returns:
            bool: True if ends with !?.,;:
        """
        if not isinstance(text, str) or not text.strip():
            return False
        return text.strip()[-1] in "!?.,;:"

    def punctuation_ending_chat(self, df: pd.DataFrame) -> pd.Series:
        """Apply ends_with_punctuation to cleaned message.

        Args:
            df (pd.DataFrame): Input with message_cleaned.

        Returns:
            pd.Series: Boolean per row.
        """
        return df[Columns.MESSAGE_CLEANED].apply(self.ends_with_punctuation)

    def calc_pct_emojis(self, df: pd.DataFrame) -> pd.Series:
        """Percentage of message length that is emojis.

        Args:
            df (pd.DataFrame): Input with number_of_emojis and length_chat.

        Returns:
            pd.Series: Emoji percentage (0–1).
        """
        return (df[Columns.NUMBER_OF_EMOJIS] / df[Columns.LENGTH_CHAT].replace(0, np.nan)).fillna(0)

    def calc_pct_punctuations(self, df: pd.DataFrame) -> pd.Series:
        """Percentage of message length that is punctuation.

        Args:
            df (pd.DataFrame): Input with number_of_punctuations and length_chat.

        Returns:
            pd.Series: Punctuation percentage (0–1).
        """
        return (df[Columns.NUMBER_OF_PUNCTUATIONS] / df[Columns.LENGTH_CHAT].replace(0, np.nan)).fillna(0)

    # === Step 5: Daily Percentages ===
    def calc_day_pct_length_chat(self, df: pd.DataFrame) -> pd.Series:
        """Daily percentage of total chat length.

        Args:
            df (pd.DataFrame): Input with length_chat and timestamp.

        Returns:
            pd.Series: Daily contribution.
        """
        daily = df.groupby(df[Columns.TIMESTAMP].dt.date)[Columns.LENGTH_CHAT].transform("sum")
        return (df[Columns.LENGTH_CHAT] / daily.replace(0, np.nan)).fillna(0)

    def calc_day_pct_length_emojis(self, df: pd.DataFrame) -> pd.Series:
        """Daily percentage of total emojis.

        Args:
            df (pd.DataFrame): Input with number_of_emojis and timestamp.

        Returns:
            pd.Series: Daily emoji share.
        """
        daily = df.groupby(df[Columns.TIMESTAMP].dt.date)[Columns.NUMBER_OF_EMOJIS].transform("sum")
        return (df[Columns.NUMBER_OF_EMOJIS] / daily.replace(0, np.nan)).fillna(0)

    def calc_day_pct_length_punctuations(self, df: pd.DataFrame) -> pd.Series:
        """Daily percentage of total punctuations.

        Args:
            df (pd.DataFrame): Input with number_of_punctuations and timestamp.

        Returns:
            pd.Series: Daily punctuation share.
        """
        daily = df.groupby(df[Columns.TIMESTAMP].dt.date)[Columns.NUMBER_OF_PUNCTUATIONS].transform("sum")
        return (df[Columns.NUMBER_OF_PUNCTUATIONS] / daily.replace(0, np.nan)).fillna(0)

    def number_of_unique_participants_that_day(self, df: pd.DataFrame) -> pd.Series:
        """Number of unique authors per day.

        Args:
            df (pd.DataFrame): Input with author and timestamp.

        Returns:
            pd.Series: Unique count per day.
        """
        return df.groupby(df[Columns.TIMESTAMP].dt.date)[Columns.AUTHOR].transform("nunique")

    def calc_day_pct_authors(self, df: pd.DataFrame) -> pd.Series:
        """Percentage of daily messages by author.

        Args:
            df (pd.DataFrame): Input with timestamp.

        Returns:
            pd.Series: Author’s share of daily messages.
        """
        daily = df.groupby(df[Columns.TIMESTAMP].dt.date).size()
        return (
            df.groupby(df[Columns.TIMESTAMP].dt.date).cumcount().add(1)
            / daily.reindex(df[Columns.TIMESTAMP].dt.date).values
        )

    def find_sequence_authors(self, df: pd.DataFrame) -> pd.Series:
        """List of authors in order per day.

        Args:
            df (pd.DataFrame): Input with author and timestamp.

        Returns:
            pd.Series: List of authors per day.
        """
        return (
            df.groupby(df[Columns.TIMESTAMP].dt.date)[Columns.AUTHOR]
            .apply(list)
            .reindex(df[Columns.TIMESTAMP].dt.date)
            .values
        )

    def find_sequence_response_times(self, df: pd.DataFrame) -> pd.Series:
        """List of response times per day.

        Args:
            df (pd.DataFrame): Input with response_time and timestamp.

        Returns:
            pd.Series: List of response times per day.
        """
        return (
            df.groupby(df[Columns.TIMESTAMP].dt.date)[Columns.RESPONSE_TIME]
            .apply(list)
            .reindex(df[Columns.TIMESTAMP].dt.date)
            .values
        )

    # === Step 6: Text Features ===
    def has_emoji(self, text: str) -> bool:
        """Check if message contains any valid emoji (excluding skin tones).

        Args:
            text (str): Input message.

        Returns:
            bool: True if contains emoji.
        """
        if not isinstance(text, str):
            return False
        return any(char in emoji.EMOJI_DATA and char not in self.ignore_emojis for char in text)

    def count_emojis(self, text: str) -> int:
        """Count number of valid emojis in message.

        Args:
            text (str): Input message.

        Returns:
            int: Count of emojis.
        """
        if not isinstance(text, str):
            return 0
        return len([c for c in text if c in emoji.EMOJI_DATA and c not in self.ignore_emojis])

    def has_link(self, text: str) -> bool:
        """Detect URLs in message.

        Args:
            text (str): Input message.

        Returns:
            bool: True if URL found.
        """
        if not isinstance(text, str):
            return False
        return bool(self.url_pattern.search(text))

    def was_deleted(self, message: str) -> bool:
        """Detect deleted message in Dutch.

        Args:
            message (str): Input message.

        Returns:
            bool: True if message was deleted.
        """
        if not isinstance(message, str):
            return False
        return bool(re.search(r"Dit bericht is verwijderd\.", message, flags=re.IGNORECASE))

    def changes_to_grouppicture(self, message: str) -> int:
        """Count group picture change notifications.

        Args:
            message (str): Input message.

        Returns:
            int: Number of group picture changes.
        """
        if not isinstance(message, str):
            return 0
        return len(re.findall(r"heeft de groepsafbeelding gewijzigd", message, flags=re.IGNORECASE))

    def _clean_for_word_analysis(self, text: str) -> str:
        """Remove emojis, numbers, and punctuation for word analysis.

        Args:
            text (str): Input message.

        Returns:
            str: Cleaned text for tokenization.
        """
        if not isinstance(text, str):
            return ""
        text = re.sub(self.emoji_pattern, "", text)
        text = re.sub(
            r"""
            (?:[\$€£¥]\s*)?      # currency + space
            \d+(?:[.,]\d+)*       # digits + fraction
            (?:%\s?)?             # optional %
            [.,=]*                # trailing
        """,
            " ",
            text,
            flags=re.VERBOSE,
        )
        text = re.sub(f"[{re.escape(BROAD_PUNCTUATION)}]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def count_words(self, text: str) -> int:
        """Count words containing letters, @, or &.

        Args:
            text (str): Input message.

        Returns:
            int: Number of valid words.
        """
        cleaned = self._clean_for_word_analysis(text)
        if not cleaned:
            return 0
        tokens = word_tokenize(cleaned)
        return len([t for t in tokens if any(c.isalpha() or c in {"@", "&"} for c in t)])

    def avg_word_length(self, text: str) -> float:
        """Compute average length of valid words.

        Args:
            text (str): Input message.

        Returns:
            float: Average word length (0.0 if no words).
        """
        cleaned = self._clean_for_word_analysis(text)
        if not cleaned:
            return 0.0
        tokens = word_tokenize(cleaned)
        words = [t for t in tokens if any(c.isalpha() or c in {"@", "&"} for c in t)]
        return sum(len(w) for w in words) / len(words) if words else 0.0

    def starts_with_emoji(self, text: str) -> bool:
        """Check if message starts with emoji.

        Args:
            text (str): Input message.

        Returns:
            bool: True if starts with emoji.
        """
        if not isinstance(text, str) or not text.strip():
            return False
        return bool(re.match(self.emoji_pattern, text.strip()))

    def emoji_starting_chat(self, text: str) -> List[str]:
        """Extract sequence of emojis at start of message.

        Args:
            text (str): Input message.

        Returns:
            List[str]: List of starting emojis.
        """
        if not isinstance(text, str):
            return []
        match = re.match(self.emoji_pattern, text.strip())
        return list(match.group(0)) if match else []

    def has_question_mark(self, text: str) -> bool:
        """Check for question mark in message.

        Args:
            text (str): Input message.

        Returns:
            bool: True if contains '?'.
        """
        return isinstance(text, str) and "?" in text

    def ends_with_question_mark(self, text: str) -> bool:
        """Check if message ends with question mark.

        Args:
            text (str): Input message.

        Returns:
            bool: True if ends with '?'.
        """
        return isinstance(text, str) and text.strip() and text.strip()[-1] == "?"

    def count_capitals(self, text: str) -> int:
        """Count uppercase letters.

        Args:
            text (str): Input message.

        Returns:
            int: Number of capital letters.
        """
        return sum(1 for c in text if c.isupper()) if isinstance(text, str) else 0

    def has_capitals(self, text: str) -> bool:
        """Check for any uppercase letters.

        Args:
            text (str): Input message.

        Returns:
            bool: True if contains capitals.
        """
        return isinstance(text, str) and any(c.isupper() for c in text)

    def list_of_connected_capitals(self, text: str) -> List[str]:
        """Find sequences of 2+ uppercase letters.

        Args:
            text (str): Input message.

        Returns:
            List[str]: List of capital sequences.
        """
        return re.findall(r"[A-Z]{2,}", text) if isinstance(text, str) else []

    def starts_with_capital(self, text: str) -> bool:
        """Check if message starts with capital, @, or &.

        Args:
            text (str): Input message.

        Returns:
            bool: True if starts with capital or special.
        """
        if not isinstance(text, str) or not text.strip():
            return False
        first = text.strip()[0]
        return first.isupper() or first in {"@", "&"}

    def capitalized_words_ratio(self, text: str) -> float:
        """Ratio of capitalized words to total words.

        Args:
            text (str): Input message.

        Returns:
            float: Capitalized word ratio.
        """
        if not isinstance(text, str):
            return 0.0
        words = self._clean_for_word_analysis(text).split()
        if not words:
            return 0.0
        caps = [w for w in words if w[0].isupper()]
        return len(caps) / len(words)

    def count_number_characters(self, text: str) -> int:
        """Count digit characters.

        Args:
            text (str): Input message.

        Returns:
            int: Number of digits.
        """
        return sum(1 for c in text if c.isdigit()) if isinstance(text, str) else 0

    def has_number_characters(self, text: str) -> bool:
        """Check for any digits.

        Args:
            text (str): Input message.

        Returns:
            bool: True if contains digits.
        """
        return isinstance(text, str) and any(c.isdigit() for c in text)

    def count_numbers(self, text: str) -> int:
        """Count numeric expressions (e.g., 123, €50, 12.5%).

        Args:
            text (str): Input message.

        Returns:
            int: Number of numeric tokens.
        """
        if not isinstance(text, str):
            return 0
        pattern = re.compile(
            r"""
            (?:[\$€£¥]\s*)?       # currency + space
            \d+(?:[.,]\d+)*       # digits
            (?:%\s?)?             # optional %
            [.,=]*                # trailing
        """,
            re.VERBOSE,
        )
        return len([m for m in pattern.finditer(text) if re.search(r"\d", m.group(0))])

    def has_attachment(self, row: pd.Series) -> bool:
        """Check if Rowe has any media attachment.

        Args:
            row (pd.Series): DataFrame row.

        Returns:
            bool: True if any media deleted.
        """
        cols = [
            Columns.PICTURES_DELETED,
            Columns.VIDEOS_DELETED,
            Columns.AUDIOS_DELETED,
            Columns.GIFS_DELETED,
            Columns.STICKERS_DELETED,
            Columns.DOCUMENTS_DELETED,
            Columns.VIDEONOTES_DELETED,
        ]
        return any(row.get(col, 0) > 0 for col in cols)

    def number_of_pictures_videos(self, row: pd.Series) -> int:
        """Count pictures + videos in row.

        Args:
            row (pd.Series): DataFrame row.

        Returns:
            int: Total pictures and videos.
        """
        return row.get(Columns.PICTURES_DELETED, 0) + row.get(Columns.VIDEOS_DELETED, 0)

    # === Step 7: Active Years & Early Leaver ===
    def active_years(self, df: pd.DataFrame) -> pd.Series:
        """Count unique years each author was active.

        Args:
            df (pd.DataFrame): Input with timestamp and author.

        Returns:
            pd.Series: Number of active years per row.
        """
        if Columns.TIMESTAMP not in df.columns or Columns.AUTHOR not in df.columns:
            return pd.Series([0] * len(df), index=df.index)
        year_series = df[Columns.TIMESTAMP].dt.year
        return (
            df.groupby(Columns.AUTHOR)[year_series.name]
            .nunique()
            .reindex(df[Columns.AUTHOR], fill_value=0)
            .values
        )

    def early_leaver(self, df: pd.DataFrame) -> pd.Series:
        """Identify authors not active in the last year.

        Args:
            df (pd.DataFrame): Input with timestamp and author.

        Returns:
            pd.Series: True if author left early.
        """
        if Columns.TIMESTAMP not in df.columns or Columns.AUTHOR not in df.columns:
            return pd.Series([False] * len(df), index=df.index)
        last_year = df[Columns.TIMESTAMP].dt.year.max()
        active_last = df[df[Columns.TIMESTAMP].dt.year == last_year][Columns.AUTHOR].unique()
        return ~df[Columns.AUTHOR].isin(active_last)

    # === Step 8: Create whatsapp_group_temp ===
    # df["whatsapp_group_temp"] = np.where(df[Columns.AUTHOR] == "AvT", "AvT", df[Columns.WHATSAPP_GROUP])

    # === Step 9: Final Column Order ===
    # See organized_columns list in organize_extended_df()

    # === Main Pipeline ===
    def organize_extended_df(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Orchestrate full cleaning and feature engineering pipeline.

        Args:
            df (pd.DataFrame): Raw WhatsApp export.

        Returns:
            pd.DataFrame | None: Final organized DataFrame or None on error.

        Examples:
            >>> editor = DataEditor()
            >>> final_df = editor.organize_extended_df(raw_df)
        """
        if df is None or df.empty:
            logger.error("No valid DataFrame provided for organizing")
            return None
        try:
            # 0. Clean messages
            df = self.clean_for_deleted_media_patterns(df)
            if df is None:
                return None

            # 1. Apply initials
            df = self.replace_author_by_initials(df)

            # 2. Time features
            df[Columns.YEAR] = df[Columns.TIMESTAMP].dt.year
            df[Columns.MONTH] = df[Columns.TIMESTAMP].dt.month
            df[Columns.WEEK] = df[Columns.TIMESTAMP].dt.isocalendar().week
            df[Columns.DAY_OF_WEEK] = df[Columns.TIMESTAMP].dt.dayofweek

            # 3. Core style columns
            df[Columns.NUMBER_OF_CHATS_THAT_DAY] = self.number_of_chats_that_day(df)
            df[Columns.LENGTH_CHAT] = df[Columns.MESSAGE_CLEANED].apply(self.length_chat)
            df[Columns.PREVIOUS_AUTHOR] = self.previous_author(df)
            df[Columns.NEXT_AUTHOR] = self.next_author(df)
            df[Columns.RESPONSE_TIME] = self.response_time(df)

            # 4. Emoji / Punctuation
            df[Columns.LIST_OF_ALL_EMOJIS] = df[Columns.MESSAGE_CLEANED].apply(self.list_of_all_emojis)
            df[Columns.LIST_OF_CONNECTED_EMOJIS] = df[Columns.MESSAGE_CLEANED].apply(self.list_of_connected_emojis)
            df[Columns.NUMBER_OF_PUNCTUATIONS] = df[Columns.MESSAGE_CLEANED].apply(self.count_punctuations)
            df[Columns.HAS_PUNCTUATION] = df[Columns.MESSAGE_CLEANED].apply(self.has_punctuation)
            df[Columns.LIST_OF_ALL_PUNCTUATIONS] = df[Columns.MESSAGE_CLEANED].apply(self.list_of_all_punctuations)
            df[Columns.LIST_OF_CONNECTED_PUNCTUATIONS] = df[Columns.MESSAGE_CLEANED].apply(self.list_of_connected_punctuations)
            df[Columns.ENDS_WITH_EMOJI] = df[Columns.MESSAGE_CLEANED].apply(self.ends_with_emoji)
            df[Columns.EMOJI_ENDING_CHAT] = df[Columns.MESSAGE_CLEANED].apply(self.emoji_ending_chat)
            df[Columns.ENDS_WITH_PUNCTUATION] = df[Columns.MESSAGE_CLEANED].apply(self.ends_with_punctuation)
            df[Columns.PUNCTUATION_ENDING_CHAT] = self.punctuation_ending_chat(df)
            df[Columns.PCT_EMOJIS] = self.calc_pct_emojis(df)
            df[Columns.PCT_PUNCTUATIONS] = self.calc_pct_punctuations(df)

            # 5. Daily percentages
            df[Columns.X_DAY_PCT_LENGTH_CHAT] = self.calc_day_pct_length_chat(df)
            df[Columns.X_DAY_PCT_LENGTH_EMOJIS] = self.calc_day_pct_length_emojis(df)
            df[Columns.X_DAY_PCT_LENGTH_PUNCTUATIONS] = self.calc_day_pct_length_punctuations(df)
            df[Columns.X_NUMBER_OF_UNIQUE_PARTICIPANTS_THAT_DAY] = self.number_of_unique_participants_that_day(df)
            df[Columns.X_DAY_PCT_MESSAGES_OF_AUTHOR] = self.calc_day_pct_authors(df)
            df[Columns.Y_SEQUENCE_AUTHORS_THAT_DAY] = self.find_sequence_authors(df)
            df[Columns.Y_SEQUENCE_RESPONSE_TIMES_THAT_DAY] = self.find_sequence_response_times(df)

            # 6. Text features
            df[Columns.NUMBER_OF_WORDS] = df[Columns.MESSAGE_CLEANED].apply(self.count_words)
            df[Columns.AVG_WORD_LENGTH] = df[Columns.MESSAGE_CLEANED].apply(self.avg_word_length)
            df[Columns.STARTS_WITH_EMOJI] = df[Columns.MESSAGE_CLEANED].apply(self.starts_with_emoji)
            df[Columns.EMOJI_STARTING_CHAT] = df[Columns.MESSAGE_CLEANED].apply(self.emoji_starting_chat)
            df[Columns.HAS_QUESTION_MARK] = df[Columns.MESSAGE_CLEANED].apply(self.has_question_mark)
            df[Columns.ENDS_WITH_QUESTION_MARK] = df[Columns.MESSAGE_CLEANED].apply(self.ends_with_question_mark)
            df[Columns.NUMBER_OF_CAPITALS] = df[Columns.MESSAGE_CLEANED].apply(self.count_capitals)
            df[Columns.HAS_CAPITALS] = df[Columns.MESSAGE_CLEANED].apply(self.has_capitals)
            df[Columns.LIST_OF_CONNECTED_CAPITALS] = df[Columns.MESSAGE_CLEANED].apply(self.list_of_connected_capitals)
            df[Columns.STARTS_WITH_CAPITAL] = df[Columns.MESSAGE_CLEANED].apply(self.starts_with_capital)
            df[Columns.CAPITALIZED_WORDS_RATIO] = df[Columns.MESSAGE_CLEANED].apply(self.capitalized_words_ratio)
            df[Columns.NUMBER_OF_NUMBER_CHARACTERS] = df[Columns.MESSAGE_CLEANED].apply(self.count_number_characters)
            df[Columns.HAS_NUMBER_CHARACTERS] = df[Columns.MESSAGE_CLEANED].apply(self.has_number_characters)
            df[Columns.NUMBER_OF_NUMBERS] = df[Columns.MESSAGE_CLEANED].apply(self.count_numbers)
            df[Columns.HAS_ATTACHMENT] = df.apply(self.has_attachment, axis=1)
            df[Columns.NUMBER_OF_PICTURES_VIDEOS] = df.apply(self.number_of_pictures_videos, axis=1)

            # 7. Active years & early leaver
            df[Columns.ACTIVE_YEARS] = self.active_years(df)
            df[Columns.EARLY_LEAVER] = self.early_leaver(df)
            df = df[~df[Columns.EARLY_LEAVER]].reset_index(drop=True)

            # 8. Create whatsapp_group_temp
            df["whatsapp_group_temp"] = np.where(df[Columns.AUTHOR] == "AvT", "AvT", df[Columns.WHATSAPP_GROUP])
            logger.info("Created 'whatsapp_group_temp' (AvT isolated)")

            # 9. Final column order
            organized_columns = [
                Columns.WHATSAPP_GROUP, "whatsapp_group_temp", Columns.TIMESTAMP,
                Columns.YEAR, Columns.MONTH, Columns.WEEK, Columns.DAY_OF_WEEK,
                Columns.AUTHOR, Columns.ACTIVE_YEARS, Columns.EARLY_LEAVER,
                Columns.NUMBER_OF_CHATS_THAT_DAY, Columns.LENGTH_CHAT, Columns.NUMBER_OF_WORDS,
                Columns.AVG_WORD_LENGTH, Columns.PREVIOUS_AUTHOR, Columns.RESPONSE_TIME,
                Columns.NEXT_AUTHOR, Columns.HAS_LINK, Columns.WAS_DELETED,
                Columns.PICTURES_DELETED, Columns.VIDEOS_DELETED, Columns.NUMBER_OF_PICTURES_VIDEOS,
                Columns.AUDIOS_DELETED, Columns.GIFS_DELETED, Columns.STICKERS_DELETED,
                Columns.DOCUMENTS_DELETED, Columns.VIDEONOTES_DELETED, Columns.HAS_ATTACHMENT,
                Columns.NUMBER_OF_EMOJIS, Columns.HAS_EMOJI, Columns.LIST_OF_ALL_EMOJIS,
                Columns.LIST_OF_CONNECTED_EMOJIS, Columns.STARTS_WITH_EMOJI, Columns.EMOJI_STARTING_CHAT,
                Columns.ENDS_WITH_EMOJI, Columns.EMOJI_ENDING_CHAT, Columns.PCT_EMOJIS,
                Columns.NUMBER_OF_PUNCTUATIONS, Columns.HAS_PUNCTUATION, Columns.LIST_OF_ALL_PUNCTUATIONS,
                Columns.LIST_OF_CONNECTED_PUNCTUATIONS, Columns.ENDS_WITH_PUNCTUATION,
                Columns.HAS_QUESTION_MARK, Columns.ENDS_WITH_QUESTION_MARK, Columns.PUNCTUATION_ENDING_CHAT,
                Columns.PCT_PUNCTUATIONS, Columns.NUMBER_OF_CAPITALS, Columns.HAS_CAPITALS,
                Columns.LIST_OF_CONNECTED_CAPITALS, Columns.STARTS_WITH_CAPITAL,
                Columns.CAPITALIZED_WORDS_RATIO, Columns.NUMBER_OF_NUMBER_CHARACTERS,
                Columns.HAS_NUMBER_CHARACTERS, Columns.NUMBER_OF_NUMBERS, Columns.MESSAGE_CLEANED,
                Columns.X_DAY_PCT_LENGTH_CHAT, Columns.X_DAY_PCT_LENGTH_EMOJIS,
                Columns.X_DAY_PCT_LENGTH_PUNCTUATIONS, Columns.X_NUMBER_OF_UNIQUE_PARTICIPANTS_THAT_DAY,
                Columns.X_DAY_PCT_MESSAGES_OF_AUTHOR, Columns.Y_SEQUENCE_AUTHORS_THAT_DAY,
                Columns.Y_SEQUENCE_RESPONSE_TIMES_THAT_DAY,
            ]
            df = df[organized_columns]
            logger.info(
                f"Final DF: {len(df)} rows, {len(df.columns)} cols – all authors are initials"
            )
            return df

        except Exception as e:
            logger.exception(f"organize_extended_df failed: {e}")
            return None


# NEW: Reorganized by pipeline order + Step headers (2025-10-31)