# === Module Docstring ===
"""
Constants Module

Defines standardized column names, group identifiers, and configuration enums
using StrEnum for direct string access and type safety.

All string enums can be used directly in DataFrame operations:
    df[Columns.TIMESTAMP]
    author = Groups.MAAP
"""

# === Imports ===
from enum import Enum, StrEnum
from typing import Any, Dict, Tuple
from pathlib import Path

# === Column Definitions ===
class Columns(StrEnum):
    """Standardized column names for DataFrame consistency.

    Use directly as strings: ``Columns.MESSAGE`` → ``"message"``.
    """

    TIMESTAMP = "timestamp"
    AUTHOR = "author"
    WHATSAPP_GROUP = "whatsapp_group"
    WHATSAPP_GROUP_TEMP = "whatsapp_group_temp"
    MESSAGE = "message"
    YEAR = "year"
    MONTH = "month"
    ISOWEEK = "isoweek"
    WEEK = "week"
    DAY_OF_WEEK = "day_of_week"
    ACTIVE_YEARS = "active_years"
    EARLY_LEAVER = "early_leaver"
    NUMBER_OF_CHATS_THAT_DAY = "number_of_chats_that_day"
    LENGTH_CHAT = "length_chat"
    NUMBER_OF_WORDS = "number_of_words"
    AVG_WORD_LENGTH = "avg_word_length"
    PREVIOUS_AUTHOR = "previous_author"
    RESPONSE_TIME = "response_time"
    NEXT_AUTHOR = "next_author"
    HAS_LINK = "has_link"
    WAS_DELETED = "was_deleted"
    PICTURES_DELETED = "pictures_deleted"
    VIDEOS_DELETED = "videos_deleted"
    NUMBER_OF_PICTURES_VIDEOS = "number_of_pictures_videos"
    AUDIOS_DELETED = "audios_deleted"
    GIFS_DELETED = "gifs_deleted"
    STICKERS_DELETED = "stickers_deleted"
    DOCUMENTS_DELETED = "documents_deleted"
    VIDEONOTES_DELETED = "videonotes_deleted"
    HAS_ATTACHMENT = "has_attachment"
    NUMBER_OF_EMOJIS = "number_of_emojis"
    HAS_EMOJI = "has_emoji"
    LIST_OF_ALL_EMOJIS = "list_of_all_emojis"
    LIST_OF_CONNECTED_EMOJIS = "list_of_connected_emojis"
    STARTS_WITH_EMOJI = "starts_with_emoji"
    EMOJI_STARTING_CHAT = "emoji_starting_chat"
    ENDS_WITH_EMOJI = "ends_with_emoji"
    EMOJI_ENDING_CHAT = "emoji_ending_chat"
    PCT_EMOJIS = "pct_emojis"
    NUMBER_OF_PUNCTUATIONS = "number_of_punctuations"
    HAS_PUNCTUATION = "has_punctuation"
    LIST_OF_ALL_PUNCTUATIONS = "list_of_all_punctuations"
    LIST_OF_CONNECTED_PUNCTUATIONS = "list_of_connected_punctuations"
    ENDS_WITH_PUNCTUATION = "ends_with_punctuation"
    HAS_QUESTION_MARK = "has_question_mark"
    ENDS_WITH_QUESTION_MARK = "ends_with_question_mark"
    PUNCTUATION_ENDING_CHAT = "punctuation_ending_chat"
    PCT_PUNCTUATIONS = "pct_punctuations"
    NUMBER_OF_CAPITALS = "number_of_capitals"
    HAS_CAPITALS = "has_capitals"
    LIST_OF_CONNECTED_CAPITALS = "list_of_connected_capitals"
    STARTS_WITH_CAPITAL = "starts_with_capital"
    CAPITALIZED_WORDS_RATIO = "capitalized_words_ratio"
    NUMBER_OF_NUMBER_CHARACTERS = "number_of_number_characters"
    HAS_NUMBER_CHARACTERS = "has_number_characters"
    NUMBER_OF_NUMBERS = "number_of_numbers"
    MESSAGE_CLEANED = "message_cleaned"
    X_DAY_PCT_LENGTH_CHAT = "x_day_pct_length_chat"
    X_DAY_PCT_LENGTH_EMOJIS = "x_day_pct_length_emojis"
    X_DAY_PCT_LENGTH_PUNCTUATIONS = "x_day_pct_length_punctuations"
    X_NUMBER_OF_UNIQUE_PARTICIPANTS_THAT_DAY = "x_number_of_unique_participants_that_day"
    X_DAY_PCT_MESSAGES_OF_AUTHOR = "x_day_pct_messages_of_author"
    Y_SEQUENCE_AUTHORS_THAT_DAY = "y_sequence_authors_that_day"
    Y_SEQUENCE_RESPONSE_TIMES_THAT_DAY = "y_sequence_response_times_that_day"
    AVG_WORDS = "avg_words"
    AVG_PUNCT = "avg_punctuations"
    MESSAGE_COUNT = "message_count"
    NON_ANTHONY_AVG = "non_anthony_avg"
    ANTHONY_MESSAGES = "anthony_messages"
    NUM_AUTHORS = "num_authors"

    @property
    def human(self) -> str:
        return {
            self.WHATSAPP_GROUP: "WhatsApp Group",
            self.AUTHOR: "Author",
            self.YEAR: "Year",
            self.WEEK: "Week",
            self.AVG_WORDS: "Average Words per Message",
            self.AVG_PUNCT: "Average Punctuation per Message",
            self.MESSAGE_COUNT: "Total Messages",
            self.NON_ANTHONY_AVG: "Non-Anthony Average Messages",
            self.ANTHONY_MESSAGES: "Anthony's Messages",
            self.NUM_AUTHORS: "Number of Authors",
        }.get(self, self.value.replace("_", " ").title())


# === Group Identifiers ===
class Groups(StrEnum):
    """Standardized WhatsApp group names."""

    MAAP = "maap"
    GOLFMATEN = "golfmaten"
    DAC = "dac"
    TILLIES = "tillies"
    AVT = "AvT"
    UNKNOWN = "unknown"


# === File-name prefixes, suffixes and patterns ===
class FilePrefixes(StrEnum):
    """Prefixes used when generating timestamped output files."""
    ORGANIZED = "organized_data"
    WHATSAPP = "whatsapp"
    WHATSAPP_ALL = "whatsapp_all"
    WHATSAPP_ALL_ENRICHED = "whatsapp_all_enriched"
    TABLE = "table"


class FileExtensions(StrEnum):
    """Common file extensions."""
    CSV = ".csv"
    PARQUET = ".parq"


class FilePatterns(StrEnum):
    """Glob patterns for locating processed files."""
    CLEANED_CSV = "whatsapp-*-cleaned.csv"


# === Config keys & static paths ===
class ConfigKeys(StrEnum):
    """Keys read from ``config.toml``."""
    PROCESSED = "processed"
    RAW = "raw"
    PREPROCESS = "preprocess"


CONFIG_FILE: Path = Path("config.toml")
TEMP_CHAT_FILE: Path = Path("_chat.txt")


# === Pre-processor CLI arguments ===
class PreprocessorArgs(StrEnum):
    """Arguments passed to ``preprocessor.main()``."""
    DEVICE = "--device"
    IOS = "ios"


# === Mapping raw → current → group ===
# NOTE: Keep the raw-key names exactly as they appear in config.toml.
RAW_FILE_MAPPING: Dict[str, Tuple[str, str]] = {
    "raw_1": ("current_1", Groups.MAAP),
    "raw_2a": ("current_2a", Groups.GOLFMATEN),
    "raw_2b": ("current_2b", Groups.GOLFMATEN),
    "raw_3": ("current_3", Groups.DAC),
    "raw_4": ("current_4", Groups.TILLIES),
}

# Helper to get only the group part (used in enrich_all_groups)
GROUP_MAP_FROM_CLEANED: Dict[str, str] = {
    "maap": Groups.MAAP,
    "golf": Groups.GOLFMATEN,
    "dac": Groups.DAC,
    "voorganger-golf": Groups.GOLFMATEN,
    "til": Groups.TILLIES,
}

# === NEW: Default image filename ===
class ImageFilenames(StrEnum):
    """Default base names for saved figures."""
    YEARLY_BAR_CHART = "yearly_bar_chart_combined"


# === Interaction Types for Arc Diagram ===
class InteractionType(StrEnum):
    """Types of author interactions in the arc diagram.

    Used in participation table (Script4) to distinguish:
    - Direct 2-person conversations
    - 3-person days with one non-participant
    """
    PAIRS = "Pairs"
    NON_PARTICIPANT = "Non-participant"


# === Data Feed Types ===
class DataFeed(StrEnum):
    """Types of data feeds for analysis."""

    NON_REDUNDANT = "non_redundant"
    REDUNDANT = "redundant"


# === Plot Configuration ===
class PlotFeed(StrEnum):
    """Plot output modes."""

    BOTH = "both"
    PER_GROUP = "per_group"
    GLOBAL = "global"


class GroupByPeriod(StrEnum):
    """Time periods for grouping."""

    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class PlotType(StrEnum):
    """Dimensionality reduction methods."""

    BOTH = "both"
    PCA = "pca"
    TSNE = "tsne"


# === Boolean Flags ===
class DeleteAttributes(Enum):
    """Boolean flags for deletion attributes."""

    FALSE = False
    TRUE = True


# === Embedding Models ===
class EmbeddingModel(int, Enum):
    """Mapping for Script6 embedding models."""
    STYLE = 1
    MINILM = 2
    MPNET = 3


# === Script6 Configuration Keys ===
class Script6ConfigKeys(StrEnum):
    """
    Configuration keys for Script6 passed via SCRIPT_6_DETAILS.

    Used in main.py, pipeline.py, script6.py, and data_preparation.py.
    """
    PLOT_TYPE = "plot_type"
    BY_GROUP = "by_group"
    DRAW_ELLIPSES = "draw_ellipses"
    USE_EMBEDDINGS = "use_embeddings"
    HYBRID_FEATURES = "hybrid_features"
    EMBEDDING_MODEL = "embedding_model"


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
# - Add markers #NEW at the end of the module

# NEW: Added missing columns and footer (2025-11-01)
# NEW: Added EmbeddingModel, Script6 config keys (2025-11-03)
# NEW: (2025-11-03) – All hard-coded literals moved to constants

