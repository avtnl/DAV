# === constants.py ===
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
            self.TIMESTAMP: "Timestamp",
            self.AUTHOR: "Author",
            self.WHATSAPP_GROUP: "WhatsApp Group",
            self.WHATSAPP_GROUP_TEMP: "WhatsApp Group Temp",
            self.MESSAGE: "Message",
            self.YEAR: "Year",
            self.MONTH: "Month",
            self.WEEK: "Week",
            self.DAY_OF_WEEK: "Day of Week",
            self.ACTIVE_YEARS: "Active Years",
            self.EARLY_LEAVER: "Early Leaver",
            self.NUMBER_OF_CHATS_THAT_DAY: "Chats That Day",
            self.LENGTH_CHAT: "Chat Length",
            self.NUMBER_OF_WORDS: "Word Count",
            self.AVG_WORD_LENGTH: "Avg Word Length",
            self.PREVIOUS_AUTHOR: "Previous Author",
            self.RESPONSE_TIME: "Response Time",
            self.NEXT_AUTHOR: "Next Author",
            self.HAS_LINK: "Has Link",
            self.WAS_DELETED: "Was Deleted",
            self.PICTURES_DELETED: "Pictures Deleted",
            self.VIDEOS_DELETED: "Videos Deleted",
            self.NUMBER_OF_PICTURES_VIDEOS: "Pictures/Videos Count",
            self.AUDIOS_DELETED: "Audios Deleted",
            self.GIFS_DELETED: "GIFs Deleted",
            self.STICKERS_DELETED: "Stickers Deleted",
            self.DOCUMENTS_DELETED: "Documents Deleted",
            self.VIDEONOTES_DELETED: "Videonotes Deleted",
            self.HAS_ATTACHMENT: "Has Attachment",
            self.NUMBER_OF_EMOJIS: "Emoji Count",
            self.HAS_EMOJI: "Has Emoji",
            self.LIST_OF_ALL_EMOJIS: "All Emojis",
            self.LIST_OF_CONNECTED_EMOJIS: "Connected Emojis",
            self.STARTS_WITH_EMOJI: "Starts with Emoji",
            self.EMOJI_STARTING_CHAT: "Emoji Starting Chat",
            self.ENDS_WITH_EMOJI: "Ends with Emoji",
            self.EMOJI_ENDING_CHAT: "Emoji Ending Chat",
            self.PCT_EMOJIS: "% Emojis",
            self.NUMBER_OF_PUNCTUATIONS: "Punctuation Count",
            self.HAS_PUNCTUATION: "Has Punctuation",
            self.LIST_OF_ALL_PUNCTUATIONS: "All Punctuations",
            self.LIST_OF_CONNECTED_PUNCTUATIONS: "Connected Punctuations",
            self.ENDS_WITH_PUNCTUATION: "Ends with Punctuation",
            self.HAS_QUESTION_MARK: "Has Question Mark",
            self.ENDS_WITH_QUESTION_MARK: "Ends with Question Mark",
            self.PUNCTUATION_ENDING_CHAT: "Punctuation Ending Chat",
            self.PCT_PUNCTUATIONS: "% Punctuations",
            self.NUMBER_OF_CAPITALS: "Capitals Count",
            self.HAS_CAPITALS: "Has Capitals",
            self.LIST_OF_CONNECTED_CAPITALS: "Connected Capitals",
            self.STARTS_WITH_CAPITAL: "Starts with Capital",
            self.CAPITALIZED_WORDS_RATIO: "Capitalized Words Ratio",
            self.NUMBER_OF_NUMBER_CHARACTERS: "Number Characters Count",
            self.HAS_NUMBER_CHARACTERS: "Has Number Characters",
            self.NUMBER_OF_NUMBERS: "Numbers Count",
            self.MESSAGE_CLEANED: "Cleaned Message",
            self.X_DAY_PCT_LENGTH_CHAT: "Day % Chat Length",
            self.X_DAY_PCT_LENGTH_EMOJIS: "Day % Emoji Length",
            self.X_DAY_PCT_LENGTH_PUNCTUATIONS: "Day % Punctuation Length",
            self.X_NUMBER_OF_UNIQUE_PARTICIPANTS_THAT_DAY: "Unique Participants That Day",
            self.X_DAY_PCT_MESSAGES_OF_AUTHOR: "Day % Author Messages",
            self.Y_SEQUENCE_AUTHORS_THAT_DAY: "Sequence Authors That Day",
            self.Y_SEQUENCE_RESPONSE_TIMES_THAT_DAY: "Sequence Response Times That Day",
            self.AVG_WORDS: "Avg Words",
            self.AVG_PUNCT: "Avg Punctuations",
            self.MESSAGE_COUNT: "Message Count",
            self.NON_ANTHONY_AVG: "Non-Anthony Avg",
            self.ANTHONY_MESSAGES: "Anthony Messages",
            self.NUM_AUTHORS: "Num Authors",
        }.get(self, self.value.replace("_", " ").title())

# === Group Definitions ===
class Groups(StrEnum):
    """WhatsApp group identifiers.

    Use directly as strings: ``Groups.MAAP`` → ``"maap"``.
    """

    MAAP = "maap"
    GOLFMATEN = "golfmaten"
    DAC = "dac"
    TILLIES = "tillies"
    NAJAARSCOMPETITIE = "najaarscompetitie"
    AVT = "AvT"
    UNKNOWN = "unknown"

# === File Prefixes ===
class FilePrefixes(StrEnum):
    """Prefixes used when generating timestamped output files."""

    ORGANIZED = "organized_data"
    WHATSAPP = "whatsapp"
    WHATSAPP_ALL_ENRICHED = "whatsapp_all_enriched"
    TABLE = "table"

# === File Extensions ===
class FileExtensions(StrEnum):
    """Common file extensions."""

    CSV = ".csv"
    PARQUET = ".parq"

# === File Patterns ===
class FilePatterns(StrEnum):
    """Glob patterns for locating processed files."""

    CLEANED_CSV = "whatsapp-*-cleaned.csv"

# === Config keys & static paths ===
class ConfigKeys(StrEnum):
    """Keys read from ``config.toml``."""

    PROCESSED = "processed"
    RAW = "raw"
    PREPROCESS = "preprocess"
    REUSE_WHATSAPP_ALL = "reuse_whatsapp_all"

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
    "raw_5": ("current_5", Groups.NAJAARSCOMPETITIE),
}

# Helper to get only the group part (used in enrich_all_groups)
GROUP_MAP_FROM_CLEANED: Dict[str, str] = {
    "maap": Groups.MAAP,
    "golf": Groups.GOLFMATEN,
    "dac": Groups.DAC,
    "voorganger-golf": Groups.GOLFMATEN,
    "til": Groups.TILLIES,
    "najaarscompetitie": Groups.NAJAARSCOMPETITIE,
}

# === Default image filename ===
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
    """Mapping for Script5 embedding models."""

    STYLE = 1
    MINILM = 2
    MPNET = 3

# === Script Run Mode Options ===
class RunMode(StrEnum):
    """Run modes for scripts like Script3 (individual groups, combined, or both)."""
    INDIVIDUAL = "individual"
    COMBINED = "combined"
    BOTH = "both"

# === Script5 Configuration Keys ===
class Script5ConfigKeys(StrEnum):
    """
    Configuration keys for Script5 passed via SCRIPT_5_DETAILS.

    Used in main.py, pipeline.py, script5.py, and data_preparation.py.
    """

    PLOT_TYPE = "plot_type"
    BY_GROUP = "by_group"
    ELLIPSE_MODE = "ellipse_mode"
    CONFIDENCE_LEVEL = "confidence_level"
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
# NEW: Added EmbeddingModel, Script5 config keys (2025-11-03)
# NEW: (2025-11-03) – All hard-coded literals moved to constants
# NEW: (2025-11-04) – Added REUSE_WHATSAPP_ALL to ConfigKeys; Renamed WHATSAPP_ALL to WHATSAPP_ALL_ENRICHED in FilePrefixes