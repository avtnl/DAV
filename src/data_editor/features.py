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

All features are vectorized and fully self-contained for consistency.
"""

from __future__ import annotations

# === Imports ===
from typing import Dict, List, Set, Any
import pandas as pd
import emoji
import re
import string

from src.constants import Columns


# === Global Constants ===
BROAD_PUNCTUATION = "".join(set(string.punctuation) - {"@", "&"})


# === Shared Emoji Pattern ===
EMOJI_PATTERN = re.compile(
    r'[\U0001F600-\U0001F64F'
    r'\U0001F300-\U0001F5FF'
    r'\U0001F680-\U0001F6FF'
    r'\U0001F1E0-\U0001F1FF'
    r'\U0001F900-\U0001FAFF'
    r'\U00002600-\U000026FF'
    r'\U00002700-\U000027BF'
    r'\u200D\uFE0F'
    r']+', re.UNICODE
)


# === Reusable Sub-Functions ===
def tokenize_text(text: str) -> List[str]:
    """Extract non-whitespace tokens (handles punctuation-attached content)."""
    return re.findall(r'\S+', str(text))


def extract_emoji_groups(text: str) -> int:
    """Count emoji sequences in text (each sequence = 1 word)."""
    return len(EMOJI_PATTERN.findall(text))


def extract_text_words(token: str) -> List[str]:
    """Extract all words from token after removing emojis."""
    remaining = EMOJI_PATTERN.sub('', token)
    words = re.findall(r'\w+', remaining.lower())
    return words


# === Core Utilities ===
def length_chat(text: str | None) -> int:
    """Return character length of cleaned message."""
    return len(str(text)) if pd.notna(text) else 0


def list_of_all_emojis(text: str, ignore_emojis: Set[str]) -> List[str]:
    """
    Extract individual base emojis using `emoji.emoji_list()`.

    - Returns list of dicts: [{'match_start': 0, 'match_end': 3, 'emoji': '❤️'}]
    - Filters out sequences with ignored chars (skin tones, U+FE0F)
    - Strips U+FE0F
    - Returns list of clean base emojis
    """
    if not isinstance(text, str):
        return []

    try:
        matches = emoji.emoji_list(text)
    except:
        return []

    cleaned = []
    for item in matches:
        seq = item["emoji"]
        # Skip if contains any ignored char
        if any(c in ignore_emojis for c in seq):
            continue
        # Remove U+FE0F and other ignored
        cleaned_seq = ''.join(c for c in seq if c not in ignore_emojis)
        if cleaned_seq:
            cleaned.append(cleaned_seq)

    return cleaned


def list_of_connected_emojis(text: str, pattern: re.Pattern) -> List[str]:
    """Find sequences of 2+ connected emojis."""
    if not isinstance(text, str):
        return []
    return pattern.findall(text)


def ends_with_emoji(text: str, ignore_emojis: Set[str]) -> bool:
    """Check if message ends with emoji (after punctuation)."""
    if not isinstance(text, str) or not text:
        return False
    match = re.search(r'\S+$', text)
    return bool(match and EMOJI_PATTERN.search(match.group(0)))


def emoji_ending_chat(text: str, ignore_emojis: Set[str]) -> str:
    """Return the last emoji."""
    if not isinstance(text, str) or not text:
        return ""
    matches = EMOJI_PATTERN.findall(text)
    return matches[-1] if matches else ""


def count_punctuations(text: str, pattern: re.Pattern) -> int:
    """Count standalone punctuation."""
    if not isinstance(text, str):
        return 0
    return len(pattern.findall(text))


def has_punctuation(text: str, pattern: re.Pattern) -> bool:
    return count_punctuations(text, pattern) > 0


def list_of_all_punctuations(text: str, pattern: re.Pattern) -> List[str]:
    if not isinstance(text, str):
        return []
    return pattern.findall(text)


def list_of_connected_punctuations(text: str, pattern: re.Pattern) -> List[str]:
    if not isinstance(text, str):
        return []
    return pattern.findall(text)


def ends_with_punctuation(text: str, pattern: re.Pattern) -> bool:
    if not isinstance(text, str) or not text:
        return False
    return bool(pattern.search(text.rstrip()))


def punctuation_ending_chat(text: str, pattern: re.Pattern) -> str:
    if not isinstance(text, str) or not text:
        return ""
    matches = pattern.findall(text)
    return matches[-1] if matches else ""


# === Text Analysis ===
def count_words(text: str) -> int:
    """Count all words + emoji groups."""
    if not text or pd.isna(text):
        return 0
    text = str(text)
    emoji_groups = extract_emoji_groups(text)
    tokens = tokenize_text(text)
    text_word_count = 0
    for token in tokens:
        if EMOJI_PATTERN.search(token):
            text_word_count += len(extract_text_words(token))
        else:
            clean = re.sub(r'[^\w]', '', token.lower())
            if clean:
                text_word_count += 1
    return text_word_count + emoji_groups


def avg_word_length(text: str) -> float:
    """Average word length — counts all words."""
    if not text or pd.isna(text):
        return 0.0
    text = str(text)
    emoji_groups = extract_emoji_groups(text)
    tokens = tokenize_text(text)
    total_length = 0.0
    counted_words = 0
    for token in tokens:
        if EMOJI_PATTERN.search(token):
            words = extract_text_words(token)
            for w in words:
                total_length += len(w)
                counted_words += 1
            token_emojis = len(EMOJI_PATTERN.findall(token))
            total_length += 4 * token_emojis
            counted_words += token_emojis
        else:
            clean = re.sub(r'[^\w]', '', token.lower())
            if clean:
                total_length += len(clean)
                counted_words += 1
    if counted_words == 0 and emoji_groups > 0:
        return 4.0
    return total_length / counted_words if counted_words > 0 else 0.0


def starts_with_emoji(text: str) -> bool:
    """Check if first non-whitespace token contains an emoji."""
    if not text or pd.isna(text):
        return False
    match = re.search(r'\S+', str(text))
    return bool(match and EMOJI_PATTERN.search(match.group(0)))


def emoji_starting_chat(text: str) -> bool:
    """Check if message starts with emoji after removing leading punctuation."""
    if not text or pd.isna(text):
        return False
    text = str(text)
    stripped = re.sub(r'^[^\w\s]+', '', text).lstrip()
    return bool(stripped and EMOJI_PATTERN.match(stripped))


# === Capitalization & Numbers ===
def count_capitals(text: str) -> int:
    return sum(1 for c in text if c.isupper()) if isinstance(text, str) else 0


def has_capitals(text: str) -> bool:
    return isinstance(text, str) and any(c.isupper() for c in text)


def list_of_connected_capitals(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"\b[A-Z]{2,}\b", text)


def starts_with_capital(text: str) -> bool:
    if not isinstance(text, str) or not text:
        return False
    return text[0].isupper()


def capitalized_words_ratio(text: str) -> float:
    if not isinstance(text, str):
        return 0.0
    try:
        import nltk
        words = nltk.word_tokenize(text)
    except:
        words = text.split()
    caps = [w for w in words if w and w[0].isupper()]
    return len(caps) / len(words) if words else 0.0


def count_number_characters(text: str) -> int:
    return sum(1 for c in text if c.isdigit()) if isinstance(text, str) else 0


def has_number_characters(text: str) -> bool:
    return isinstance(text, str) and any(c.isdigit() for c in text)


def count_numbers(text: str) -> int:
    if not isinstance(text, str):
        return 0
    pattern = re.compile(r"(?:[\$€£¥]\s*)?\d+(?:[.,]\d+)*(?:%\s?)?[.,=]*", re.VERBOSE)
    return len([m for m in pattern.finditer(text) if re.search(r"\d", m.group(0))])


def has_question_mark(text: str) -> bool:
    return isinstance(text, str) and "?" in text


def ends_with_question_mark(text: str) -> bool:
    return isinstance(text, str) and text and text[-1] == "?"


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
        # df[Columns.NUMBER_OF_EMOJIS] = df[Columns.LIST_OF_ALL_EMOJIS].str.len()
        df[Columns.NUMBER_OF_EMOJIS] = msg.apply(
        lambda x: len(list_of_all_emojis(x, ignore_emojis)) if isinstance(x, str) else 0
        )
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
        pct = daily_author.div(daily_total, level=0) * 100
        df[Columns.X_DAY_PCT_MESSAGES_OF_AUTHOR] = pct.reindex(
            df.set_index([date, Columns.AUTHOR]).index
        ).values

        df[Columns.Y_SEQUENCE_AUTHORS_THAT_DAY] = df.groupby(date)[Columns.AUTHOR].transform(lambda g: g.tolist())
        df[Columns.Y_SEQUENCE_RESPONSE_TIMES_THAT_DAY] = df.groupby(date)[Columns.RESPONSE_TIME].transform(lambda g: g.tolist())

        return df

    # === Step 6: Text Features ===
    def add_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add word count, capitalization, numbers, and question marks.

        Args:
            df: DataFrame with ``MESSAGE_CLEANED``.

        Returns:
            pd.DataFrame: With text feature columns.
        """
        msg = df[Columns.MESSAGE_CLEANED]

        df[Columns.NUMBER_OF_WORDS] = msg.apply(count_words)
        df[Columns.AVG_WORD_LENGTH] = msg.apply(avg_word_length)

        df[Columns.STARTS_WITH_EMOJI] = msg.apply(starts_with_emoji)
        df[Columns.EMOJI_STARTING_CHAT] = msg.apply(emoji_starting_chat)

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
        """Add active years and early leaver flag, then drop early leavers."""
        if Columns.TIMESTAMP not in df.columns or Columns.AUTHOR not in df.columns:
            df[Columns.ACTIVE_YEARS] = 0
            df[Columns.EARLY_LEAVER] = False
            return df

        year_stats = df.groupby(Columns.AUTHOR)[Columns.YEAR].agg(['min', 'max'])
        active_years = (year_stats['max'] - year_stats['min'] + 1).astype(int)
        df[Columns.ACTIVE_YEARS] = df[Columns.AUTHOR].map(active_years).fillna(0).astype(int)

        last_year = df[Columns.YEAR].max()
        active_last_year = df[df[Columns.YEAR] == last_year][Columns.AUTHOR].unique()
        df[Columns.EARLY_LEAVER] = ~df[Columns.AUTHOR].isin(active_last_year)

        df = df[~df[Columns.EARLY_LEAVER]].reset_index(drop=True)
        return df


    # === Orchestrator: Add All Features ===
    def add_all_features(
        self,
        df: pd.DataFrame,
        ignore_emojis: set,
        patterns: dict,
        stopwords: set = None,
    ) -> pd.DataFrame:
        """Run all feature engineering steps.

        Args:
            df: Cleaned DataFrame from ``MessageCleaner``.
            ignore_emojis: Set of emoji codes to ignore.
            patterns: Dict with pre-compiled regexes.
            stopwords: Set of Dutch stopwords (unused).

        Returns:
            pd.DataFrame: Fully enriched DataFrame.
        """
        df = self.add_time_features(df)
        df = self.add_core_style(df)
        df = self.add_emoji_punct_features(df, ignore_emojis, patterns)
        df = self.add_daily_percentages(df)
        df = self.add_text_features(df)
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

# NEW: Removed all stopword filtering from count_words and avg_word_length (2025-11-06)
# NEW: Stopwords param now unused and optional (2025-11-06)
# NEW: Counts all words including fillers like "ik", "het", "the", "and" (2025-11-06)
# NEW: No NLTK dependency in core functions (2025-11-06)
# NEW: Full consistency with no zeros for short messages (2025-11-06)