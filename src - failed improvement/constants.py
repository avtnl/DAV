from enum import Enum

class Columns(Enum):
    """Enum for standardized DataFrame column names."""
    TIMESTAMP = 'timestamp'
    AUTHOR = 'author'
    WHATSAPP_GROUP = 'whatsapp_group'
    MESSAGE = 'message'
    YEAR = 'year'
    MONTH = 'month'
    WEEK = 'week'
    DAY_OF_WEEK = 'day_of_week'
    ACTIVE_YEARS = 'active_years'
    EARLY_LEAVER = 'early_leaver'
    NUMBER_OF_CHATS_THAT_DAY = 'number_of_chats_that_day'
    LENGTH_CHAT = 'length_chat'
    PREVIOUS_AUTHOR = 'previous_author'
    RESPONSE_TIME = 'response_time'
    NEXT_AUTHOR = 'next_author'
    HAS_LINK = 'has_link'
    WAS_DELETED = 'was_deleted'
    PICTURES_DELETED = 'pictures_deleted'
    VIDEOS_DELETED = 'videos_deleted'
    AUDIOS_DELETED = 'audios_deleted'
    GIFS_DELETED = 'gifs_deleted'
    STICKERS_DELETED = 'stickers_deleted'
    DOCUMENTS_DELETED = 'documents_deleted'
    VIDEONOTES_DELETED = 'videonotes_deleted'
    NUMBER_OF_EMOJIS = 'number_of_emojis'
    HAS_EMOJI = 'has_emoji'
    LIST_OF_ALL_EMOJIS = 'list_of_all_emojis'
    LIST_OF_CONNECTED_EMOJIS = 'list_of_connected_emojis'
    ENDS_WITH_EMOJI = 'ends_with_emoji'
    EMOJI_ENDING_CHAT = 'emoji_ending_chat'
    PCT_EMOJIS = 'pct_emojis'
    NUMBER_OF_PUNCTUATIONS = 'number_of_punctuations'
    HAS_PUNCTUATION = 'has_punctuation'
    LIST_OF_ALL_PUNCTUATIONS = 'list_of_all_punctuations'
    LIST_OF_CONNECTED_PUNCTUATIONS = 'list_of_connected_punctuations'
    ENDS_WITH_PUNCTUATION = 'ends_with_punctuation'
    PUNCTUATION_ENDING_CHAT = 'punctuation_ending_chat'
    PCT_PUNCTUATIONS = 'pct_punctuations'
    DAY_PCT_LENGTH_CHAT = 'day_pct_length_chat'
    DAY_PCT_LENGTH_EMOJIS = 'day_pct_length_emojis'
    DAY_PCT_LENGTH_PUNCTUATIONS = 'day_pct_length_punctuations'
    NUMBER_OF_UNIQUE_PARTICIPANTS_THAT_DAY = 'number_of_unique_participants_that_day'
    DAY_PCT_AUTHORS = 'day_pct_authors'
    SEQUENCE_AUTHORS = 'sequence_authors'
    SEQUENCE_RESPONSE_TIMES = 'sequence_response_times'
    # Add any additional columns here if discovered during testing

class Groups(Enum):
    """Enum for standardized WhatsApp group names."""
    MAAP = 'maap'
    GOLFMATEN = 'golfmaten'
    DAC = 'dac'
    TILLIES = 'tillies'
    UNKNOWN = 'unknown'

class DataFeed(Enum):
    """Enum for dataset variant types."""
    NON_REDUNDANT = 'non_redundant'
    REDUNDANT = 'redundant'

class PlotFeed(Enum):
    """Enum for plot visualization modes."""
    BOTH = 'both'
    PER_GROUP = 'per_group'
    GLOBAL = 'global'

class GroupByPeriod(Enum):
    """Enum for time period grouping options."""
    WEEK = 'week'
    MONTH = 'month'
    YEAR = 'year'

class DeleteAttributes(Enum):
    """Enum for delete specific attributes option."""
    FALSE = False
    TRUE = True

class PlotType(Enum):
    """Enum for dimensionality reduction plot types."""
    BOTH = 'both'
    PCA = 'pca'
    TSNE = 'tsne'    