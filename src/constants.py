# constants.py
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
    NUMBER_OF_WORDS = 'number_of_words'
    AVG_WORD_LENGTH = 'avg_word_length'
    PREVIOUS_AUTHOR = 'previous_author'
    RESPONSE_TIME = 'response_time'
    NEXT_AUTHOR = 'next_author'
    HAS_LINK = 'has_link'
    WAS_DELETED = 'was_deleted'
    PICTURES_DELETED = 'pictures_deleted'
    VIDEOS_DELETED = 'videos_deleted'
    NUMBER_OF_PICTURES_VIDEOS = 'number_of_pictures_videos'
    AUDIOS_DELETED = 'audios_deleted'
    GIFS_DELETED = 'gifs_deleted'
    STICKERS_DELETED = 'stickers_deleted'
    DOCUMENTS_DELETED = 'documents_deleted'
    VIDEONOTES_DELETED = 'videonotes_deleted'
    HAS_ATTACHMENT = 'has_attachment'
    NUMBER_OF_EMOJIS = 'number_of_emojis'
    HAS_EMOJI = 'has_emoji'
    LIST_OF_ALL_EMOJIS = 'list_of_all_emojis'
    LIST_OF_CONNECTED_EMOJIS = 'list_of_connected_emojis'
    STARTS_WITH_EMOJI = 'starts_with_emoji'
    EMOJI_STARTING_CHAT = 'emoji_starting_chat'
    ENDS_WITH_EMOJI = 'ends_with_emoji'
    EMOJI_ENDING_CHAT = 'emoji_ending_chat'
    PCT_EMOJIS = 'pct_emojis'
    NUMBER_OF_PUNCTUATIONS = 'number_of_punctuations'
    HAS_PUNCTUATION = 'has_punctuation'
    LIST_OF_ALL_PUNCTUATIONS = 'list_of_all_punctuations'
    LIST_OF_CONNECTED_PUNCTUATIONS = 'list_of_connected_punctuations'
    ENDS_WITH_PUNCTUATION = 'ends_with_punctuation'
    HAS_QUESTION_MARK = 'has_question_mark'
    ENDS_WITH_QUESTION_MARK = 'ends_with_question_mark'
    PUNCTUATION_ENDING_CHAT = 'punctuation_ending_chat'
    PCT_PUNCTUATIONS = 'pct_punctuations'
    NUMBER_OF_CAPITALS = 'number_of_capitals'
    HAS_CAPITALS = 'has_capitals'
    LIST_OF_CONNECTED_CAPITALS = 'list_of_connected_capitals'
    STARTS_WITH_CAPITAL = 'starts_with_capital'
    CAPITALIZED_WORDS_RATIO = 'capitalized_words_ratio'
    NUMBER_OF_NUMBER_CHARACTERS = 'number_of_number_characters'
    HAS_NUMBER_CHARACTERS = 'has_number_characters'
    NUMBER_OF_NUMBERS = 'number_of_numbers'
    MESSAGE_CLEANED = 'message_cleaned'
    X_DAY_PCT_LENGTH_CHAT = 'x_day_pct_length_chat'
    X_DAY_PCT_LENGTH_EMOJIS = 'x_day_pct_length_emojis'
    X_DAY_PCT_LENGTH_PUNCTUATIONS = 'x_day_pct_length_punctuations'
    X_NUMBER_OF_UNIQUE_PARTICIPANTS_THAT_DAY = 'x_number_of_unique_participants_that_day'
    X_DAY_PCT_MESSAGES_OF_AUTHOR = 'x_day_pct_messages_of_author'
    Y_SEQUENCE_AUTHORS_THAT_DAY = 'y_sequence_authors_that_day'
    Y_SEQUENCE_RESPONSE_TIMES_THAT_DAY = 'y_sequence_response_times_that_day'

class Groups(Enum):
    """Enum for standardized WhatsApp group names."""
    MAAP = 'maap'
    GOLFMATEN = 'golfmaten'
    DAC = 'dac'
    TILLIES = 'tillies'
    UNKNOWN = 'unknown'

class DataFeed(Enum):
    NON_REDUNDANT = 'non_redundant'
    REDUNDANT = 'redundant'

class PlotFeed(Enum):
    BOTH = 'both'
    PER_GROUP = 'per_group'
    GLOBAL = 'global'

class GroupByPeriod(Enum):
    WEEK = 'week'
    MONTH = 'month'
    YEAR = 'year'

class DeleteAttributes(Enum):
    FALSE = False
    TRUE = True

class PlotType(Enum):
    BOTH = 'both'
    PCA = 'pca'
    TSNE = 'tsne'