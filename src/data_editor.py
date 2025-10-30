# data_editor.py
import pandas as pd
import numpy as np
import re
from loguru import logger
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download required NLTK data
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Global: Punctuation to remove during word analysis (exclude @ and &)
BROAD_PUNCTUATION = ''.join(set(string.punctuation) - {'@', '&'})


class DataEditor:
    """A class for cleaning and processing WhatsApp message data."""

    def __init__(self):
        """Initialize DataEditor with regex patterns, emoji sets, and stopwords."""
        self.emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f1e0-\U0001f1ff"  # flags (iOS)
            "\U00002702-\U000027b0"  # Dingbats
            "\U000024c2-\U0001f251"
            "\U0001f900-\U0001f9ff"  # supplemental symbols & pictographs
            "]+",
            flags=re.UNICODE,
        )
        self.ignore_emojis = {chr(int(code, 16)) for code in ['1F3FB', '1F3FC', '1F3FD', '1F3FE', '1F3FF']}
        self.url_pattern = re.compile(r"(?i)\b((?:https?://|ftp://|www\.)\S+)", flags=re.UNICODE)
        self.stopwords = set(stopwords.words('dutch'))
        self.punctuation_pattern = re.compile(r'(?<![\d])[!?.,;:](?![\d])')
        self.connected_emoji_pattern = re.compile(
            r'([' + ''.join(self.emoji_pattern.pattern[1:-2]) + r']{2,})', flags=re.UNICODE)
        self.connected_punctuation_pattern = re.compile(r'([!?.,;:]{2,})')

        # INITIALS MAP – applied to ALL author columns
        self.initials_map = {
            'Nico Dofferhoff': 'ND',
            'Loek van der Laan': 'LL',
            'Herma Hollander': 'HH',
            'Hieke van Heusden': 'HvH',
            'Irene Bienema': 'IB',
            'Anthony van Tilburg': 'AvT',
            'Anja Berkemeijer': 'AB',
            'Madeleine': 'M',
            'Phons Berkemeijer': 'PB',
            'Rob Haasbroek': 'RH',
            'Hugo Brouwer': 'HB',
            'Martin Kat': 'MK'
        }

    # -----------------------------------------------------------------
    # CORE CLEANING & LOADING
    # -----------------------------------------------------------------
    def convert_timestamp(self, datafile):
        df = pd.read_csv(datafile, parse_dates=["timestamp"])
        logger.info(f"DataFrame head:\n{df.head()}")
        return df

    def clean_author(self, df):
        clean_tilde = r"^~\u202f"
        df["author"] = df["author"].apply(lambda x: re.sub(clean_tilde, "", str(x)))
        return df

    # -----------------------------------------------------------------
    # TEXT ANALYSIS HELPERS – ALL DEFINED BEFORE organize_extended_df
    # -----------------------------------------------------------------
    def has_emoji(self, text):
        if not isinstance(text, str): return False
        return any(char in emoji.EMOJI_DATA and char not in self.ignore_emojis for char in text)

    def count_emojis(self, text):
        if not isinstance(text, str): return 0
        return len([c for c in text if c in emoji.EMOJI_DATA and c not in self.ignore_emojis])

    def has_link(self, text):
        if not isinstance(text, str): return False
        return bool(self.url_pattern.search(text))

    def was_deleted(self, message):
        if not isinstance(message, str): return False
        return bool(re.search(r'Dit bericht is verwijderd\.', message, flags=re.IGNORECASE))

    def changes_to_grouppicture(self, message):
        if not isinstance(message, str): return 0
        return len(re.findall(r'heeft de groepsafbeelding gewijzigd', message, flags=re.IGNORECASE))

    def _clean_for_word_analysis(self, text):
        if not isinstance(text, str): return ""
        text = re.sub(self.emoji_pattern, '', text)
        text = re.sub(r'''
            (?:[\$€£¥]\s*)?      # currency + space
            \d+(?:[.,]\d+)*       # digits + fraction
            (?:%\s?)?             # optional % 
            [.,=]*                # trailing
        ''', ' ', text, flags=re.VERBOSE)
        text = re.sub(f'[{re.escape(BROAD_PUNCTUATION)}]+', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def count_words(self, text):
        cleaned = self._clean_for_word_analysis(text)
        if not cleaned: return 0
        tokens = word_tokenize(cleaned)
        return len([t for t in tokens if any(c.isalpha() or c in {'@', '&'} for c in t)])

    def avg_word_length(self, text):
        cleaned = self._clean_for_word_analysis(text)
        if not cleaned: return 0.0
        tokens = word_tokenize(cleaned)
        words = [t for t in tokens if any(c.isalpha() or c in {'@', '&'} for c in t)]
        return sum(len(w) for w in words) / len(words) if words else 0.0

    def starts_with_emoji(self, text):
        if not isinstance(text, str) or not text.strip(): return False
        return bool(re.match(self.emoji_pattern, text.strip()))

    def emoji_starting_chat(self, text):
        if not isinstance(text, str): return []
        match = re.match(self.emoji_pattern, text.strip())
        return list(match.group(0)) if match else []

    def has_question_mark(self, text):
        return isinstance(text, str) and '?' in text

    def ends_with_question_mark(self, text):
        return isinstance(text, str) and text.strip() and text.strip()[-1] == '?'

    def count_capitals(self, text):
        return sum(1 for c in text if c.isupper()) if isinstance(text, str) else 0

    def has_capitals(self, text):
        return isinstance(text, str) and any(c.isupper() for c in text)

    def list_of_connected_capitals(self, text):
        return re.findall(r'[A-Z]{2,}', text) if isinstance(text, str) else []

    def starts_with_capital(self, text):
        if not isinstance(text, str) or not text.strip(): return False
        first = text.strip()[0]
        return first.isupper() or first in {'@', '&'}

    def capitalized_words_ratio(self, text):
        if not isinstance(text, str): return 0.0
        words = self._clean_for_word_analysis(text).split()
        if not words: return 0.0
        caps = [w for w in words if w[0].isupper()]
        return len(caps) / len(words)

    def count_number_characters(self, text):
        return sum(1 for c in text if c.isdigit()) if isinstance(text, str) else 0

    def has_number_characters(self, text):
        return isinstance(text, str) and any(c.isdigit() for c in text)

    def count_numbers(self, text):
        if not isinstance(text, str): return 0
        pattern = re.compile(r'''
            (?:[\$€£¥]\s*)?       # currency + space
            \d+(?:[.,]\d+)*       # digits
            (?:%\s?)?             # optional %
            [.,=]*                # trailing
        ''', re.VERBOSE)
        return len([m for m in pattern.finditer(text) if re.search(r'\d', m.group(0))])

    def has_attachment(self, row):
        cols = ['pictures_deleted', 'videos_deleted', 'audios_deleted', 'gifs_deleted',
                'stickers_deleted', 'documents_deleted', 'videonotes_deleted']
        return any(row.get(col, 0) > 0 for col in cols)

    def number_of_pictures_videos(self, row):
        return row.get('pictures_deleted', 0) + row.get('videos_deleted', 0)

    # -----------------------------------------------------------------
    # MISSING CORE COLUMNS – NOW DEFINED BEFORE organize_extended_df
    # -----------------------------------------------------------------
    def number_of_chats_that_day(self, df):
        return df.groupby(df['timestamp'].dt.date)['timestamp'].transform('cumcount') + 1

    def length_chat(self, text):
        return len(str(text)) if pd.notna(text) else 0

    def previous_author(self, df):
        return df['author'].shift(1).fillna('')

    def response_time(self, df):
        prev = df['timestamp'].shift(1)
        return (df['timestamp'] - prev).dt.total_seconds().fillna(0)

    def next_author(self, df):
        return df['author'].shift(-1).fillna('')

    def list_of_all_emojis(self, text):
        if not isinstance(text, str): return []
        return [c for c in text if c in emoji.EMOJI_DATA and c not in self.ignore_emojis]

    def list_of_connected_emojis(self, text):
        if not isinstance(text, str): return []
        return re.findall(self.connected_emoji_pattern, text)

    def count_punctuations(self, text):
        if not isinstance(text, str): return 0
        return len(re.findall(self.punctuation_pattern, text))

    def has_punctuation(self, text):
        return self.count_punctuations(text) > 0

    def list_of_all_punctuations(self, text):
        if not isinstance(text, str): return []
        return re.findall(self.punctuation_pattern, text)

    def list_of_connected_punctuations(self, text):
        if not isinstance(text, str): return []
        return re.findall(self.connected_punctuation_pattern, text)

    def ends_with_emoji(self, text):
        if not isinstance(text, str) or not text.strip(): return False
        return text.strip()[-1] in emoji.EMOJI_DATA and text.strip()[-1] not in self.ignore_emojis

    def emoji_ending_chat(self, text):
        if not isinstance(text, str): return []
        match = re.search(self.emoji_pattern, text.strip()[::-1])
        return list(match.group(0)[::-1]) if match else []

    def ends_with_punctuation(self, text):
        if not isinstance(text, str) or not text.strip(): return False
        return text.strip()[-1] in '!?.,;:'

    def punctuation_ending_chat(self, df):
        return df['message_cleaned'].apply(self.ends_with_punctuation)

    def calc_pct_emojis(self, df):
        return (df['number_of_emojis'] / df['length_chat'].replace(0, np.nan)).fillna(0)

    def calc_pct_punctuations(self, df):
        return (df['number_of_punctuations'] / df['length_chat'].replace(0, np.nan)).fillna(0)

    def calc_day_pct_length_chat(self, df):
        daily = df.groupby(df['timestamp'].dt.date)['length_chat'].transform('sum')
        return (df['length_chat'] / daily.replace(0, np.nan)).fillna(0)

    def calc_day_pct_length_emojis(self, df):
        daily = df.groupby(df['timestamp'].dt.date)['number_of_emojis'].transform('sum')
        return (df['number_of_emojis'] / daily.replace(0, np.nan)).fillna(0)

    def calc_day_pct_length_punctuations(self, df):
        daily = df.groupby(df['timestamp'].dt.date)['number_of_punctuations'].transform('sum')
        return (df['number_of_punctuations'] / daily.replace(0, np.nan)).fillna(0)

    def number_of_unique_participants_that_day(self, df):
        return df.groupby(df['timestamp'].dt.date)['author'].transform('nunique')

    def calc_day_pct_authors(self, df):
        daily = df.groupby(df['timestamp'].dt.date).size()
        return df.groupby(df['timestamp'].dt.date).cumcount().add(1) / daily.reindex(df['timestamp'].dt.date).values

    def find_sequence_authors(self, df):
        return df.groupby(df['timestamp'].dt.date)['author'].apply(list).reindex(df['timestamp'].dt.date).values

    def find_sequence_response_times(self, df):
        return df.groupby(df['timestamp'].dt.date)['response_time'].apply(list).reindex(df['timestamp'].dt.date).values

    # -----------------------------------------------------------------
    # INITIALS REPLACEMENT – APPLIED TO ALL AUTHOR COLUMNS
    # -----------------------------------------------------------------
    def replace_author_by_initials(self, df: pd.DataFrame) -> pd.DataFrame:
        author_cols = ['author', 'previous_author', 'next_author']
        for col in author_cols:
            if col in df.columns:
                df[col] = df[col].map(self.initials_map).fillna(df[col])
        logger.info("All author columns converted to initials (ND, LL, AvT, …)")
        return df

    # -----------------------------------------------------------------
    # CLEAN MESSAGES (unchanged – kept for context)
    # -----------------------------------------------------------------
    def clean_for_deleted_media_patterns(self, df):
        if df is None or df.empty:
            logger.error("No valid DataFrame provided for cleaning deleted media patterns")
            return None
        try:
            df["has_emoji"] = False
            df["number_of_emojis"] = 0
            df["has_link"] = False
            df["was_deleted"] = False
            df["number_of_changes_to_group"] = 0
            df["pictures_deleted"] = 0
            df["videos_deleted"] = 0
            df["audios_deleted"] = 0
            df["gifs_deleted"] = 0
            df["stickers_deleted"] = 0
            df["documents_deleted"] = 0
            df["videonotes_deleted"] = 0
            df["message_cleaned"] = df["message"]

            non_media_patterns = [
                (r'Dit bericht is verwijderd\.', "message deleted", re.IGNORECASE),
                (r'(?:Anthony van Tilburg|Anja Berkemeijer|Phons Berkemeijer|Madeleine) heeft de groepsafbeelding gewijzigd', "grouppicture", re.IGNORECASE)
            ]
            media_patterns = [
                (r'afbeelding\s*weggelaten', "picture deleted", re.IGNORECASE),
                (r'video\s*weggelaten', "video deleted", re.IGNORECASE),
                (r'audio\s*weggelaten', "audio deleted", re.IGNORECASE),
                (r'GIF\s*weggelaten', "GIF deleted", re.IGNORECASE),
                (r'sticker\s*weggelaten', "sticker deleted", re.IGNORECASE),
                (r'document\s*weggelaten', "document deleted", re.IGNORECASE),
                (r'videonotitie\s*weggelaten', "video note deleted", re.IGNORECASE)
            ]
            link_removal_pattern = r'(\s*https?://[^\s<>"{}|\\^`\[\]]+[\.,;:!?]?\s*)'
            fallback_pattern = r'\s*[\u200e\u200f]*\[\d{2}-\d{2}-\d{4},\s*\d{2}:\d{2}:\d{2}\]\s*(?:Anthony van Tilburg|Anja Berkemeijer|Phons Berkemeijer|Madeleine)[\s\u200e\u200f]*:.*'

            def clean_message(row):
                message = row["message"]
                row["has_emoji"] = self.has_emoji(message)
                row["number_of_emojis"] = self.count_emojis(message)
                row["has_link"] = self.has_link(message)
                row["was_deleted"] = self.was_deleted(message)
                row["number_of_changes_to_group"] = self.changes_to_grouppicture(message)

                if self.has_link(message):
                    message = re.sub(link_removal_pattern, ' ', message, flags=re.IGNORECASE)
                    message = re.sub(r'\s+', ' ', message).strip()

                for pattern, _, flags in non_media_patterns:
                    message = re.sub(pattern, '', message, flags=flags).strip()

                for pattern, change, flags in media_patterns:
                    matches = re.findall(pattern, message, flags=flags)
                    count = len(matches)
                    if count > 0:
                        if change == "picture deleted": row["pictures_deleted"] += count
                        elif change == "video deleted": row["videos_deleted"] += count
                        elif change == "audio deleted": row["audios_deleted"] += count
                        elif change == "GIF deleted": row["gifs_deleted"] += count
                        elif change == "sticker deleted": row["stickers_deleted"] += count
                        elif change == "document deleted": row["documents_deleted"] += count
                        elif change == "video note deleted": row["videonotes_deleted"] += count
                        message = re.sub(pattern, '', message, flags=flags).strip()

                total_media_deleted = sum(row[col] for col in [
                    "pictures_deleted", "videos_deleted", "audios_deleted",
                    "gifs_deleted", "stickers_deleted", "documents_deleted", "videonotes_deleted"
                ])
                if total_media_deleted > 0:
                    ta_pattern = r'[\s\u200e\u200f]*\[\d{2}-\d{2}-\d{4},\s*\d{2}:\d{2}:\d{2}\]\s*[^:]*:[\s\u200e\u200f]*$'
                    message = re.sub(ta_pattern, '', message).strip()

                if re.search(fallback_pattern, message, flags=re.IGNORECASE):
                    message = re.sub(fallback_pattern, '', message, flags=re.IGNORECASE).strip()

                if not message or message.strip() == "": message = "completely removed"
                row["message_cleaned"] = message
                return row

            df = df.apply(clean_message, axis=1)
            logger.info(f"Cleaned messages: {df[['message', 'message_cleaned']].head(10).to_string()}")
            return df
        except Exception as e:
            logger.exception(f"Failed to clean messages: {e}")
            return None

    # -----------------------------------------------------------------
    # ACTIVE YEARS & EARLY LEAVER
    # -----------------------------------------------------------------
    def active_years(self, df: pd.DataFrame) -> pd.Series:
        if 'timestamp' not in df.columns or 'author' not in df.columns:
            return pd.Series([0] * len(df), index=df.index)
        year_series = df['timestamp'].dt.year
        return df.groupby('author')[year_series.name].nunique().reindex(df['author'], fill_value=0).values

    def early_leaver(self, df: pd.DataFrame) -> pd.Series:
        if 'timestamp' not in df.columns or 'author' not in df.columns:
            return pd.Series([False] * len(df), index=df.index)
        last_year = df['timestamp'].dt.year.max()
        active_last = df[df['timestamp'].dt.year == last_year]['author'].unique()
        return ~df['author'].isin(active_last)

    # -----------------------------------------------------------------
    # ONE FUNCTION TO RULE THEM ALL – NOW AFTER ALL HELPERS
    # -----------------------------------------------------------------
    def organize_extended_df(self, df):
        if df is None or df.empty:
            logger.error("No valid DataFrame provided for organizing")
            return None
        try:
            # 0. CLEAN MESSAGES
            df = self.clean_for_deleted_media_patterns(df)
            if df is None: return None

            # 1. APPLY INITIALS TO ALL AUTHOR COLUMNS (FIRST!)
            df = self.replace_author_by_initials(df)

            # 2. TIME FEATURES
            df['year'] = df['timestamp'].dt.year
            df['month'] = df['timestamp'].dt.month
            df['week'] = df['timestamp'].dt.isocalendar().week
            df['day_of_week'] = df['timestamp'].dt.dayofweek

            # 3. CORE STYLE COLUMNS
            df['number_of_chats_that_day'] = self.number_of_chats_that_day(df)
            df['length_chat'] = df['message_cleaned'].apply(self.length_chat)
            df['previous_author'] = self.previous_author(df)
            df['next_author'] = self.next_author(df)
            df['response_time'] = self.response_time(df)

            # 4. EMOJI / PUNCTUATION
            df['list_of_all_emojis'] = df['message_cleaned'].apply(self.list_of_all_emojis)
            df['list_of_connected_emojis'] = df['message_cleaned'].apply(self.list_of_connected_emojis)
            df['number_of_punctuations'] = df['message_cleaned'].apply(self.count_punctuations)
            df['has_punctuation'] = df['message_cleaned'].apply(self.has_punctuation)
            df['list_of_all_punctuations'] = df['message_cleaned'].apply(self.list_of_all_punctuations)
            df['list_of_connected_punctuations'] = df['message_cleaned'].apply(self.list_of_connected_punctuations)
            df['ends_with_emoji'] = df['message_cleaned'].apply(self.ends_with_emoji)
            df['emoji_ending_chat'] = df['message_cleaned'].apply(self.emoji_ending_chat)
            df['ends_with_punctuation'] = df['message_cleaned'].apply(self.ends_with_punctuation)
            df['punctuation_ending_chat'] = self.punctuation_ending_chat(df)
            df['pct_emojis'] = self.calc_pct_emojis(df)
            df['pct_punctuations'] = self.calc_pct_punctuations(df)

            # 5. DAILY PERCENTAGES
            df['x_day_pct_length_chat'] = self.calc_day_pct_length_chat(df)
            df['x_day_pct_length_emojis'] = self.calc_day_pct_length_emojis(df)
            df['x_day_pct_length_punctuations'] = self.calc_day_pct_length_punctuations(df)
            df['x_number_of_unique_participants_that_day'] = self.number_of_unique_participants_that_day(df)
            df['x_day_pct_messages_of_author'] = self.calc_day_pct_authors(df)
            df['y_sequence_authors_that_day'] = self.find_sequence_authors(df)
            df['y_sequence_response_times_that_day'] = self.find_sequence_response_times(df)

            # 6. TEXT FEATURES
            df['number_of_words'] = df['message_cleaned'].apply(self.count_words)
            df['avg_word_length'] = df['message_cleaned'].apply(self.avg_word_length)
            df['starts_with_emoji'] = df['message_cleaned'].apply(self.starts_with_emoji)
            df['emoji_starting_chat'] = df['message_cleaned'].apply(self.emoji_starting_chat)
            df['has_question_mark'] = df['message_cleaned'].apply(self.has_question_mark)
            df['ends_with_question_mark'] = df['message_cleaned'].apply(self.ends_with_question_mark)
            df['number_of_capitals'] = df['message_cleaned'].apply(self.count_capitals)
            df['has_capitals'] = df['message_cleaned'].apply(self.has_capitals)
            df['list_of_connected_capitals'] = df['message_cleaned'].apply(self.list_of_connected_capitals)
            df['starts_with_capital'] = df['message_cleaned'].apply(self.starts_with_capital)
            df['capitalized_words_ratio'] = df['message_cleaned'].apply(self.capitalized_words_ratio)
            df['number_of_number_characters'] = df['message_cleaned'].apply(self.count_number_characters)
            df['has_number_characters'] = df['message_cleaned'].apply(self.has_number_characters)
            df['number_of_numbers'] = df['message_cleaned'].apply(self.count_numbers)
            df['has_attachment'] = df.apply(self.has_attachment, axis=1)
            df['number_of_pictures_videos'] = df.apply(self.number_of_pictures_videos, axis=1)

            # 7. ACTIVE YEARS & EARLY LEAVER
            df['active_years'] = self.active_years(df)
            df['early_leaver'] = self.early_leaver(df)
            df = df[df['early_leaver'] == False].reset_index(drop=True)

            # 8. CREATE whatsapp_group_temp USING INITIALS 'AvT'
            df['whatsapp_group_temp'] = np.where(
                df['author'] == 'AvT',
                'AvT',
                df['whatsapp_group']
            )
            logger.info("Created 'whatsapp_group_temp' (AvT isolated)")

            # 9. FINAL COLUMN ORDER
            organized_columns = [
                'whatsapp_group', 'whatsapp_group_temp',
                'timestamp', 'year', 'month', 'week', 'day_of_week',
                'author', 'active_years', 'early_leaver',
                'number_of_chats_that_day', 'length_chat', 'number_of_words', 'avg_word_length',
                'previous_author', 'response_time', 'next_author',
                'has_link', 'was_deleted',
                'pictures_deleted', 'videos_deleted', 'number_of_pictures_videos',
                'audios_deleted', 'gifs_deleted', 'stickers_deleted',
                'documents_deleted', 'videonotes_deleted', 'has_attachment',
                'number_of_emojis', 'has_emoji', 'list_of_all_emojis', 'list_of_connected_emojis',
                'starts_with_emoji', 'emoji_starting_chat',
                'ends_with_emoji', 'emoji_ending_chat', 'pct_emojis',
                'number_of_punctuations', 'has_punctuation',
                'list_of_all_punctuations', 'list_of_connected_punctuations',
                'ends_with_punctuation', 'has_question_mark', 'ends_with_question_mark',
                'punctuation_ending_chat', 'pct_punctuations',
                'number_of_capitals', 'has_capitals', 'list_of_connected_capitals',
                'starts_with_capital', 'capitalized_words_ratio',
                'number_of_number_characters', 'has_number_characters', 'number_of_numbers',
                'message_cleaned',
                'x_day_pct_length_chat', 'x_day_pct_length_emojis', 'x_day_pct_length_punctuations',
                'x_number_of_unique_participants_that_day', 'x_day_pct_messages_of_author',
                'y_sequence_authors_that_day', 'y_sequence_response_times_that_day'
            ]
            df = df[organized_columns]
            logger.info(f"Final DF: {len(df)} rows, {len(df.columns)} cols – all authors are initials")
            return df

        except Exception as e:
            logger.exception(f"organize_extended_df failed: {e}")
            return None