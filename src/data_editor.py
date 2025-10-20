import pandas as pd
import re
from loguru import logger
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datetime import timedelta

# Download required NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')

class DataEditor:
    """A class for cleaning and processing WhatsApp message data, including timestamp conversion,
    author cleaning, emoji/URL detection, and additional message analysis."""

    def __init__(self):
        """Initialize DataEditor with regular expression patterns, emoji sets, and stopwords.

        Attributes:
            emoji_pattern (re.Pattern): Regex pattern to match sequences of emojis (one or more).
            ignore_emojis (set): Set of skin tone modifier emojis to exclude from emoji counts.
            url_pattern (re.Pattern): Regex pattern to detect URLs in messages.
            stopwords (set): Set of stopwords for text normalization.
            punctuation_pattern (re.Pattern): Regex pattern to match punctuation marks.
            connected_emoji_pattern (re.Pattern): Regex pattern to match two or more connected emojis.
            connected_punctuation_pattern (re.Pattern): Regex pattern to match two or more connected punctuations.
        """
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
        self.connected_emoji_pattern = re.compile(r'([' + ''.join(self.emoji_pattern.pattern[1:-2]) + r']{2,})', flags=re.UNICODE)
        self.connected_punctuation_pattern = re.compile(r'([!?.,;:]{2,})')
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

    def convert_timestamp(self, datafile):
        """
        Convert the 'timestamp' column from a string to a datetime object.
        Args:
            datafile (Path): Path to the CSV file to read.
        Returns:
            pandas.DataFrame: DataFrame with converted timestamp column.
        """
        df = pd.read_csv(datafile, parse_dates=["timestamp"])
        logger.info(f"DataFrame head:\n{df.head()}")
        return df

    def clean_author(self, df):
        """
        Clean author names by removing leading tilde characters.
        Args:
            df (pandas.DataFrame): DataFrame with an 'author' column.
        Returns:
            pandas.DataFrame: DataFrame with cleaned 'author' column.
        """
        clean_tilde = r"^~\u202f"
        df["author"] = df["author"].apply(lambda x: re.sub(clean_tilde, "", str(x)))
        return df
    
    def has_emoji(self, text):
        """
        Check if the input text contains any emojis.
        Args:
            text (str): Text to check for emojis.
        Returns:
            bool: True if the text contains emojis, False otherwise.
        """
        if not isinstance(text, str):
            return False
        return any(char in emoji.EMOJI_DATA and char not in self.ignore_emojis for char in text)

    def count_emojis(self, text):
        """
        Count the total number of emojis in the input text, excluding skin tone modifiers.
        Args:
            text (str): Text to count emojis in.
        Returns:
            int: Total number of emojis, excluding ignored ones.
        """
        if not isinstance(text, str):
            return 0
        emojis = [char for char in text if char in emoji.EMOJI_DATA and char not in self.ignore_emojis]
        return len(emojis)

    def has_link(self, text):
        """
        Check if the input text contains any URLs.
        Args:
            text (str): Text to check for URLs.
        Returns:
            bool: True if the text contains URLs, False otherwise.
        """
        if not isinstance(text, str):
            return False
        return bool(self.url_pattern.search(text))

    def was_deleted(self, message):
        """
        Check if the message was deleted.
        Args:
            message (str): Message to check.
        Returns:
            bool: True if the message was deleted, False otherwise.
        """
        if not isinstance(message, str):
            return False
        pattern = r'Dit bericht is verwijderd\.'
        return bool(re.search(pattern, message, flags=re.IGNORECASE))

    def changes_to_grouppicture(self, message):
        """
        Count the number of group picture changes in the message.
        Args:
            message (str): Message to check.
        Returns:
            int: Number of group picture changes.
        """
        if not isinstance(message, str):
            return 0
        pattern = r'heeft de groepsafbeelding gewijzigd'
        matches = re.findall(pattern, message, flags=re.IGNORECASE)
        return len(matches)

    def concatenate_df(self, dataframes):
        """
        Concatenate multiple DataFrames into a single DataFrame and log verification details.
        Args:
            dataframes (dict): Dictionary of DataFrames to concatenate.
        Returns:
            pandas.DataFrame or None: Concatenated DataFrame, or None if concatenation fails.
        """
        if not dataframes:
            logger.error("No DataFrames provided for concatenation")
            return None
        
        logger.debug("Concatenating DataFrames")
        try:
            df = pd.concat(dataframes.values(), ignore_index=True)
            logger.info(f"Concatenated DataFrame with {len(df)} rows and columns: {df.columns.tolist()}")
            logger.info(f"Unique WhatsApp groups: {df['whatsapp_group'].unique().tolist()}")
            logger.debug(f"DataFrame head:\n{df.head().to_string()}")
            logger.debug(f"DataFrame dtypes:\n{df.dtypes}")
            return df
        except Exception as e:
            logger.exception(f"Failed to concatenate DataFrames: {e}")
            return None

    def filter_group_names(self, df):
        """
        Filter out specific group names from the 'author' column and reset the index.
        Args:
            df (pandas.DataFrame): DataFrame to filter.
        Returns:
            pandas.DataFrame or None: Filtered DataFrame, or None if filtering fails.
        """
        if df is None or df.empty:
            logger.error("No valid DataFrame provided for filtering")
            return None
        
        try:
            logger.info("Author counts per WhatsApp group:")
            logger.info(df.groupby(["whatsapp_group", "author"]).size().to_string())
            rows_before = len(df)
            df = df.loc[df["author"] != "MAAP"]
            df = df.loc[df["author"] != "Golfmaten"]
            df = df.loc[df["author"] != "What's up with golf"]
            df = df.loc[df["author"] != "DAC cie"]
            df = df.loc[df["author"] != "Tillies & co"]
            df = df.reset_index(drop=True)
            rows_after = len(df)
            logger.info(f"DataFrame filtered: {rows_before} rows reduced to {rows_after} rows")
            return df
        except Exception as e:
            logger.exception(f"Failed to filter DataFrame: {e}")
            return None

    def clean_for_deleted_media_patterns(self, df):
        """
        Clean messages in the DataFrame by removing deleted messages, media patterns, and links,
        and add columns to track the count of each type of deleted media, group picture changes, and links.
        Args:
            df (pandas.DataFrame): Input DataFrame with 'message' column.
        Returns:
            pandas.DataFrame or None: Modified DataFrame with new columns, or None if processing fails.
        """
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
                    logger.debug(f"Removing link from message: {message}")
                    message = re.sub(link_removal_pattern, ' ', message, flags=re.IGNORECASE)
                    message = re.sub(r'\s+', ' ', message).strip()
                
                for pattern, change, flags in non_media_patterns:
                    message = re.sub(pattern, '', message, flags=flags).strip()
                
                for pattern, change, flags in media_patterns:
                    matches = re.findall(pattern, message, flags=flags)
                    count = len(matches)
                    if count > 0:
                        logger.debug(f"Matched media pattern '{pattern}' {count} times in message: {message}")
                        if change == "picture deleted":
                            row["pictures_deleted"] += count
                        elif change == "video deleted":
                            row["videos_deleted"] += count
                        elif change == "audio deleted":
                            row["audios_deleted"] += count
                        elif change == "GIF deleted":
                            row["gifs_deleted"] += count
                        elif change == "sticker deleted":
                            row["stickers_deleted"] += count
                        elif change == "document deleted":
                            row["documents_deleted"] += count
                        elif change == "video note deleted":
                            row["videonotes_deleted"] += count
                        message = re.sub(pattern, '', message, flags=flags).strip()
                
                total_media_deleted = (row["pictures_deleted"] + row["videos_deleted"] + row["audios_deleted"] +
                                    row["gifs_deleted"] + row["stickers_deleted"] + row["documents_deleted"] +
                                    row["videonotes_deleted"])
                if total_media_deleted > 0:
                    ta_pattern = r'[\s\u200e\u200f]*\[\d{2}-\d{2}-\d{4},\s*\d{2}:\d{2}:\d{2}\]\s*[^:]*:[\s\u200e\u200f]*$'
                    message = re.sub(ta_pattern, '', message).strip()
                
                if re.search(fallback_pattern, message, flags=re.IGNORECASE):
                    logger.debug(f"Matched fallback pattern '{fallback_pattern}' in message: {message}")
                    message = re.sub(fallback_pattern, '', message, flags=re.IGNORECASE).strip()
                
                if message is None or message == "" or message.strip() == "":
                    message = "completely removed"
                
                row["message_cleaned"] = message
                return row
            
            df = df.apply(clean_message, axis=1)
            
            if df["has_link"].sum() == 0:
                logger.info("No links found in the messages")
            else:
                logger.info(f"Found {df['has_link'].sum()} messages with links")
            
            logger.info(f"Cleaned messages: {df[['message', 'message_cleaned', 'has_emoji', 'number_of_emojis', 'has_link', 'was_deleted', 'number_of_changes_to_group', 'pictures_deleted', 'videos_deleted', 'audios_deleted', 'gifs_deleted', 'stickers_deleted', 'documents_deleted', 'videonotes_deleted']].head(10).to_string()}")
            return df
        except Exception as e:
            logger.exception(f"Failed to clean messages for deleted media patterns: {e}")
            return None

    def handle_emojis(self, text):
        """
        Convert emojis in the input text to their Unicode names, excluding skin tone modifiers.
        Args:
            text (str): Text containing emojis to convert.
        Returns:
            str: Text with emojis replaced by their Unicode names (e.g., ⛳ -> flag_in_hole).
        """
        if not isinstance(text, str):
            return text
        text = emoji.demojize(text, delimiters=("", ""))
        text = re.sub(r'_(light|medium_light|medium|medium_dark|dark)_skin_tone', '', text)
        return text

    def normalize_text(self, text):
        """
        Normalize text by lowercasing, removing punctuation, removing stopwords, and tokenizing.
        Args:
            text (str): Text to normalize.
        Returns:
            str: Normalized text as a space-separated string of tokens.
        """
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stopwords]
        return ' '.join(tokens)

    def extract_mentions(self, text, first_name_to_author):
        """
        Extract mentioned authors from the message text based on @mentions of first names.
        Args:
            text (str): Message text to analyze.
            first_name_to_author (dict): Mapping of first names to full author names.
        Returns:
            list: List of mentioned full author names.
        """
        if not isinstance(text, str):
            return []
        mentions = []
        for first_name, author in first_name_to_author.items():
            if re.search(r'@' + re.escape(first_name), text, re.IGNORECASE):
                mentions.append(author)
        return list(set(mentions))

    def number_of_chats_that_day(self, df):
        """
        Count the number of messages for each day based on the timestamp.
        Args:
            df (pandas.DataFrame): DataFrame with 'timestamp' column.
        Returns:
            pandas.Series: Series with the count of messages per day for each row.
        """
        if 'timestamp' not in df.columns:
            logger.error("Timestamp column not found in DataFrame")
            return pd.Series([0] * len(df), index=df.index)
        df['date'] = df['timestamp'].dt.date
        chat_counts = df.groupby('date').size()
        return df['date'].map(chat_counts).fillna(0).astype(int)

    def length_chat(self, message):
        """
        Count the number of characters in the cleaned message, counting each emoji as 1 character.
        Args:
            message (str): Cleaned message text.
        Returns:
            int: Number of characters, with emojis counted as 1 each.
        """
        if not isinstance(message, str):
            return 0
        emoji_count = self.count_emojis(message)
        no_emoji_text = re.sub(self.emoji_pattern, '', message)
        char_count = len(no_emoji_text)
        return char_count + emoji_count

    def previous_author(self, df):
        """
        Determine the previous author for each message. Empty if the message is the first of the day.
        Args:
            df (pandas.DataFrame): DataFrame with 'timestamp' and 'author' columns.
        Returns:
            pandas.Series: Series with the previous author for each row.
        """
        if 'timestamp' not in df.columns or 'author' not in df.columns:
            logger.error("Required columns (timestamp, author) not found in DataFrame")
            return pd.Series([''] * len(df), index=df.index)
        df = df.sort_values('timestamp')
        df['date'] = df['timestamp'].dt.date
        previous_authors = []
        for date, group in df.groupby('date'):
            authors = group['author'].values
            previous = ['']  # First message of the day has no previous author
            previous.extend(authors[:-1])
            previous_authors.extend(previous)
        return pd.Series(previous_authors, index=df.index)

    def next_author(self, df):
        """
        Determine the author of the next message that day. If the message is the last of the day, use the author of the first message the next day. If no message the next day, return empty string.

        Args:
            df (pandas.DataFrame): DataFrame with 'timestamp' and 'author' columns.

        Returns:
            pandas.Series: Series with the next author for each row.
        """
        if 'timestamp' not in df.columns or 'author' not in df.columns:
            logger.error("Required columns (timestamp, author) not found in DataFrame")
            return pd.Series([''] * len(df), index=df.index)
        df = df.sort_values('timestamp')
        df['date'] = df['timestamp'].dt.date
        next_authors = []
        for date, group in df.groupby('date'):
            authors = group['author'].values
            next_a = authors[1:]
            next_a = np.append(next_a, '')  # Last message of the day has empty next author temporarily
            next_authors.extend(next_a)
        # Handle last message of the day: use first message of next day if exists
        df['next_author'] = next_authors
        dates = df['date'].unique()
        dates.sort()
        for i in range(len(dates) - 1):
            current_date = dates[i]
            next_date = dates[i + 1]
            last_mask = (df['date'] == current_date) & (df['next_author'] == '')
            first_next = df[df['date'] == next_date]['author'].iloc[0] if not df[df['date'] == next_date].empty else ''
            df.loc[last_mask, 'next_author'] = first_next
        return df['next_author']

    def response_time(self, df):
        """
        Calculate the time difference in seconds since the previous message on the same day.
        Args:
            df (pandas.DataFrame): DataFrame with 'timestamp' column.
        Returns:
            pandas.Series: Series with response time in seconds (integer) for each row.
        """
        if 'timestamp' not in df.columns:
            logger.error("Timestamp column not found in DataFrame")
            return pd.Series([0] * len(df), index=df.index)
        df = df.sort_values('timestamp')
        df['date'] = df['timestamp'].dt.date
        response_times = []
        for date, group in df.groupby('date'):
            timestamps = group['timestamp']
            diffs = timestamps.diff().dt.total_seconds().fillna(0).astype(int)
            response_times.extend(diffs)
        return pd.Series(response_times, index=df.index)

    def list_of_all_emojis(self, message):
        """
        Extract all emojis in the message as a comma-separated list.
        Args:
            message (str): Cleaned message text.
        Returns:
            str: Comma-separated list of emoji Unicode names.
        """
        if not isinstance(message, str):
            return ''
        emojis = [char for char in message if char in emoji.EMOJI_DATA and char not in self.ignore_emojis]
        emoji_names = [emoji.demojize(char, delimiters=("", "")).replace('_', '') for char in emojis]
        return ','.join(emoji_names)

    def list_of_connected_emojis(self, message):
        """
        Extract sequences of two or more connected emojis as a comma-separated list.
        Args:
            message (str): Cleaned message text.
        Returns:
            str: Comma-separated list of connected emoji sequences (Unicode names).
        """
        if not isinstance(message, str):
            return ''
        matches = self.connected_emoji_pattern.findall(message)
        connected_emojis = []
        for match in matches:
            emoji_seq = [emoji.demojize(char, delimiters=("", "")).replace('_', '')
                        for char in match if char in emoji.EMOJI_DATA and char not in self.ignore_emojis]
            if len(emoji_seq) >= 2:
                connected_emojis.append(''.join(emoji_seq))
        return ','.join(connected_emojis)

    def count_punctuations(self, message):
        """
        Count the number of punctuation marks in the message, excluding decimals in numbers.
        Args:
            message (str): Cleaned message text.
        Returns:
            int: Number of punctuation marks.
        """
        if not isinstance(message, str):
            return 0
        message = re.sub(r'([!?.,;:])\1+', r'\1', message)
        punctuations = re.findall(r'(?<![\d])[!?.,;:](?![\d])', message)
        return len(punctuations)

    def has_punctuation(self, message):
        """
        Check if the message contains any punctuation.
        Args:
            message (str): Cleaned message text.
        Returns:
            bool: True if punctuation is present, False otherwise.
        """
        return self.count_punctuations(message) > 0

    def list_of_all_punctuations(self, message):
        """
        Extract all punctuation marks as a comma-separated list.
        Args:
            message (str): Cleaned message text.
        Returns:
            str: Comma-separated list of punctuation marks.
        """
        if not isinstance(message, str):
            return ''
        punctuations = self.punctuation_pattern.findall(message)
        return ','.join(punctuations)

    def list_of_connected_punctuations(self, message):
        """
        Extract sequences of two or more connected punctuation marks as a comma-separated list.
        Args:
            message (str): Cleaned message text.
        Returns:
            str: Comma-separated list of connected punctuation sequences.
        """
        if not isinstance(message, str):
            return ''
        matches = self.connected_punctuation_pattern.findall(message)
        return ','.join(matches)

    def ends_with_emoji(self, message):
        """
        Check if the message ends with an emoji.
        Args:
            message (str): Cleaned message text.
        Returns:
            bool: True if the last character is an emoji, False otherwise.
        """
        if not isinstance(message, str) or not message.strip():
            return False
        last_char = message.strip()[-1]
        return last_char in emoji.EMOJI_DATA and last_char not in self.ignore_emojis

    def emoji_ending_chat(self, message):
        """
        Return the Unicode name of the emoji that ends the message, if any.
        Args:
            message (str): Cleaned message text.
        Returns:
            str: Unicode name of the ending emoji, or empty string if none.
        """
        if not self.ends_with_emoji(message):
            return ''
        last_char = message.strip()[-1]
        return emoji.demojize(last_char, delimiters=("", "")).replace('_', '')

    def ends_with_punctuation(self, message):
        """
        Check if the message ends with a punctuation mark.
        Args:
            message (str): Cleaned message text.
        Returns:
            bool: True if the last character is a punctuation mark, False otherwise.
        """
        if not isinstance(message, str) or not message.strip():
            return False
        last_char = message.strip()[-1]
        return bool(self.punctuation_pattern.match(last_char))

    def punctuation_ending_chat(self, message):
        """
        Return the punctuation mark that ends the message, if any.
        Args:
            message (str): Cleaned message text.
        Returns:
            str: Punctuation mark that ends the message, or empty string if none.
        """
        if not self.ends_with_punctuation(message):
            return ''
        return message.strip()[-1]

    def get_year(self, df):
        """
        Extract the year from the timestamp column.
        Args:
            df (pandas.DataFrame): DataFrame with 'timestamp' column.
        Returns:
            pandas.Series: Series with the year for each row.
        """
        if 'timestamp' not in df.columns:
            logger.error("Timestamp column not found in DataFrame")
            return pd.Series([0] * len(df), index=df.index)
        return df['timestamp'].dt.year

    def get_month(self, df):
        """
        Extract the month from the timestamp column.
        Args:
            df (pandas.DataFrame): DataFrame with 'timestamp' column.
        Returns:
            pandas.Series: Series with the month (1-12) for each row.
        """
        if 'timestamp' not in df.columns:
            logger.error("Timestamp column not found in DataFrame")
            return pd.Series([0] * len(df), index=df.index)
        return df['timestamp'].dt.month

    def get_week(self, df):
        """
        Extract the ISO week number (1-52) from the timestamp column.
        Args:
            df (pandas.DataFrame): DataFrame with 'timestamp' column.
        Returns:
            pandas.Series: Series with the week number for each row.
        """
        if 'timestamp' not in df.columns:
            logger.error("Timestamp column not found in DataFrame")
            return pd.Series([0] * len(df), index=df.index)
        return df['timestamp'].dt.isocalendar().week

    def active_years(self, df):
        """
        Compute the active years (min and max) for each author in each group.
        Args:
            df (pandas.DataFrame): DataFrame with 'whatsapp_group', 'author', and 'year' columns.
        Returns:
            pandas.Series: Series with active years as 'min-max' string for each row.
        """
        if not all(col in df.columns for col in ['whatsapp_group', 'author', 'year']):
            logger.error("Required columns (whatsapp_group, author, year) not found in DataFrame")
            return pd.Series([''] * len(df), index=df.index)
        active_years = df.groupby(['whatsapp_group', 'author'])['year'].agg(['min', 'max']).reset_index()
        active_years['active_years'] = active_years.apply(lambda x: f"{x['min']}-{x['max']}", axis=1)
        active_years_dict = {(row['whatsapp_group'], row['author']): row['active_years'] 
                            for _, row in active_years.iterrows()}
        return df.apply(lambda x: active_years_dict.get((x['whatsapp_group'], x['author']), ''), axis=1)

    def early_leaver(self, df):
        """
        Flag authors who were not active in 2025 within the period 2015-07-01 to 2025-07-31.
        Args:
            df (pandas.DataFrame): DataFrame with 'whatsapp_group', 'author', 'year', and 'timestamp' columns.
        Returns:
            pandas.Series: Series with True for early leavers, False otherwise.
        """
        if not all(col in df.columns for col in ['whatsapp_group', 'author', 'year', 'timestamp']):
            logger.error("Required columns (whatsapp_group, author, year, timestamp) not found in DataFrame")
            return pd.Series([False] * len(df), index=df.index)
        filter_df = df[(df['timestamp'] >= '2015-07-01') & (df['timestamp'] <= '2025-07-31')]
        active_years_period = filter_df.groupby(['whatsapp_group', 'author'])['year'].agg(['max']).reset_index()
        early_leavers = set(active_years_period[active_years_period['max'] < 2025][['whatsapp_group', 'author']].itertuples(index=False, name=None))
        return df.apply(lambda x: (x['whatsapp_group'], x['author']) in early_leavers, axis=1)

    def convert_booleans(self, df):
        """
        Convert specified boolean columns to string values.
        Args:
            df (pandas.DataFrame): DataFrame with boolean columns 'has_link', 'was_deleted', 'has_emoji',
                                   'ends_with_emoji', 'has_punctuation', 'ends_with_punctuation'.
        Returns:
            pandas.DataFrame or None: Modified DataFrame with boolean columns converted to strings, or None if processing fails.
        """
        if df is None or df.empty:
            logger.error("No valid DataFrame provided for boolean conversion")
            return None
        try:
            boolean_mappings = {
                'has_link': {True: 'link', False: ''},
                'was_deleted': {True: 'deleted', False: ''},
                'has_emoji': {True: 'emoji(s)', False: ''},
                'ends_with_emoji': {True: 'ends_with_emoji', False: ''},
                'has_punctuation': {True: 'punctuation(s)', False: ''},
                'ends_with_punctuation': {True: 'ends_with_punctuation', False: ''}
            }
            for column, mapping in boolean_mappings.items():
                if column not in df.columns:
                    logger.error(f"Column {column} not found in DataFrame")
                    return None
                df[column] = df[column].map(mapping)
            logger.info(f"Converted boolean columns: {list(boolean_mappings.keys())}")
            logger.debug(f"DataFrame after boolean conversion:\n{df[list(boolean_mappings.keys())].head().to_string()}")
            return df
        except Exception as e:
            logger.exception(f"Failed to convert boolean columns: {e}")
            return None

    def calc_pct_emojis(self, df):
        """
        Calculate percentage of emojis as (number_of_emojis / length_chat) * 100, as integer.

        Args:
            df (pandas.DataFrame): DataFrame with 'number_of_emojis' and 'length_chat' columns.

        Returns:
            pandas.Series: Series with percentage of emojis for each row.
        """
        if 'number_of_emojis' not in df.columns or 'length_chat' not in df.columns:
            logger.error("Required columns (number_of_emojis, length_chat) not found in DataFrame")
            return pd.Series([0] * len(df), index=df.index)
        return ((df['number_of_emojis'] / df['length_chat']) * 100).fillna(0).astype(int)

    def calc_pct_punctuations(self, df):
        """
        Calculate percentage of punctuations as (number_of_punctuations / length_chat) * 100, as integer.

        Args:
            df (pandas.DataFrame): DataFrame with 'number_of_punctuations' and 'length_chat' columns.

        Returns:
            pandas.Series: Series with percentage of punctuations for each row.
        """
        if 'number_of_punctuations' not in df.columns or 'length_chat' not in df.columns:
            logger.error("Required columns (number_of_punctuations, length_chat) not found in DataFrame")
            return pd.Series([0] * len(df), index=df.index)
        return ((df['number_of_punctuations'] / df['length_chat']) * 100).fillna(0).astype(int)

    def calc_day_pct_length_chat(self, df):
        """
        Calculate percentage of length_chat as (length_chat / sum of length_chat for that day) * 100, as integer.

        Args:
            df (pandas.DataFrame): DataFrame with 'length_chat', 'timestamp' columns.

        Returns:
            pandas.Series: Series with day percentage of length_chat for each row.
        """
        if 'length_chat' not in df.columns or 'timestamp' not in df.columns:
            logger.error("Required columns (length_chat, timestamp) not found in DataFrame")
            return pd.Series([0] * len(df), index=df.index)
        df['date'] = df['timestamp'].dt.date
        daily_sums = df.groupby('date')['length_chat'].sum()
        pct = (df['length_chat'] / df['date'].map(daily_sums)) * 100
        return pct.fillna(0).astype(int)

    def calc_day_pct_length_emojis(self, df):
        """
        Calculate percentage of number_of_emojis as (number_of_emojis / sum of number_of_emojis for that day) * 100, as integer.

        Args:
            df (pandas.DataFrame): DataFrame with 'number_of_emojis', 'timestamp' columns.

        Returns:
            pandas.Series: Series with day percentage of number_of_emojis for each row.
        """
        if 'number_of_emojis' not in df.columns or 'timestamp' not in df.columns:
            logger.error("Required columns (number_of_emojis, timestamp) not found in DataFrame")
            return pd.Series([0] * len(df), index=df.index)
        df['date'] = df['timestamp'].dt.date
        daily_sums = df.groupby('date')['number_of_emojis'].sum()
        pct = (df['number_of_emojis'] / df['date'].map(daily_sums)) * 100
        return pct.fillna(0).astype(int)

    def calc_day_pct_length_punctuations(self, df):
        """
        Calculate percentage of number_of_punctuations as (number_of_punctuations / sum of number_of_punctuations for that day) * 100, as integer.

        Args:
            df (pandas.DataFrame): DataFrame with 'number_of_punctuations', 'timestamp' columns.

        Returns:
            pandas.Series: Series with day percentage of number_of_punctuations for each row.
        """
        if 'number_of_punctuations' not in df.columns or 'timestamp' not in df.columns:
            logger.error("Required columns (number_of_punctuations, timestamp) not found in DataFrame")
            return pd.Series([0] * len(df), index=df.index)
        df['date'] = df['timestamp'].dt.date
        daily_sums = df.groupby('date')['number_of_punctuations'].sum()
        pct = (df['number_of_punctuations'] / df['date'].map(daily_sums)) * 100
        return pct.fillna(0).astype(int)

    def number_of_unique_participants_that_day(self, df):
        """
        Count the number of unique participants that day (excluding early_leaver == True).

        Args:
            df (pandas.DataFrame): DataFrame with 'timestamp', 'author', 'early_leaver' columns.

        Returns:
            pandas.Series: Series with number of unique participants per day for each row.
        """
        if 'timestamp' not in df.columns or 'author' not in df.columns or 'early_leaver' not in df.columns:
            logger.error("Required columns (timestamp, author, early_leaver) not found in DataFrame")
            return pd.Series([0] * len(df), index=df.index)
        df_filtered = df[df['early_leaver'] == False]
        df_filtered['date'] = df_filtered['timestamp'].dt.date
        daily_unique = df_filtered.groupby('date')['author'].nunique()
        df['date'] = df['timestamp'].dt.date
        return df['date'].map(daily_unique).fillna(0).astype(int)

    def calc_day_pct_authors(self, df):
        """
        Calculate list of percentages = (message count per author that day / number_of_chats_that_day) * 100, sorted alphabetically by author (excluding early_leaver == True).

        Args:
            df (pandas.DataFrame): DataFrame with 'author', 'timestamp', 'early_leaver', 'number_of_chats_that_day' columns.

        Returns:
            pandas.Series: Series with list of percentages for each row.
        """
        if not all(col in df.columns for col in ['author', 'timestamp', 'early_leaver', 'number_of_chats_that_day']):
            logger.error("Required columns (author, timestamp, early_leaver, number_of_chats_that_day) not found in DataFrame")
            return pd.Series([[]] * len(df), index=df.index)
        df_filtered = df[df['early_leaver'] == False]
        df_filtered['date'] = df_filtered['timestamp'].dt.date
        daily_counts = df_filtered.groupby(['date', 'author']).size().reset_index(name='msg_count')
        daily_pcts = []
        for date, group in daily_counts.groupby('date'):
            group = group.sort_values('author')
            pcts = (group['msg_count'] / group['msg_count'].sum() * 100).astype(int).tolist()
            daily_pcts.append((date, pcts))
        pct_dict = dict(daily_pcts)
        df['date'] = df['timestamp'].dt.date
        return df['date'].map(pct_dict).fillna([])

    def find_sequence_authors(self, df):
        """
        Create list of initials of authors in order of sequence of messages that day.

        Args:
            df (pandas.DataFrame): DataFrame with 'timestamp', 'author' columns.

        Returns:
            pandas.Series: Series with list of initials per day for each row.
        """
        if 'timestamp' not in df.columns or 'author' not in df.columns:
            logger.error("Required columns (timestamp, author) not found in DataFrame")
            return pd.Series([[]] * len(df), index=df.index)
        df['date'] = df['timestamp'].dt.date
        daily_sequences = []
        for date, group in df.groupby('date'):
            group = group.sort_values('timestamp')
            initials = group['author'].map(self.initials_map).tolist()
            daily_sequences.append((date, initials))
        seq_dict = dict(daily_sequences)
        return df['date'].map(seq_dict).fillna([])

    def find_sequence_response_times(self, df):
        """
        Create list of response times in order of sequence of messages that day (first is 0).

        Args:
            df (pandas.DataFrame): DataFrame with 'timestamp' columns.

        Returns:
            pandas.Series: Series with list of response times per day for each row.
        """
        if 'timestamp' not in df.columns:
            logger.error("Timestamp column not found in DataFrame")
            return pd.Series([[]] * len(df), index=df.index)
        df['date'] = df['timestamp'].dt.date
        daily_times = []
        for date, group in df.groupby('date'):
            group = group.sort_values('timestamp')
            diffs = group['timestamp'].diff().dt.total_seconds().fillna(0).astype(int).tolist()
            daily_times.append((date, diffs))
        time_dict = dict(daily_times)
        return df['date'].map(time_dict).fillna([])

    def replace_author_by_initials(self, df):
        """
        Replace 'author', 'previous_author', 'next_author' with initials (after filtering early_leaver == True).

        Args:
            df (pandas.DataFrame): DataFrame with 'author', 'previous_author', 'next_author', 'early_leaver' columns.

        Returns:
            pandas.DataFrame or None: Modified DataFrame with replaced authors, or None if processing fails.
        """
        if df is None or df.empty:
            logger.error("No valid DataFrame provided for replacing authors")
            return None
        try:
            df_filtered = df[df['early_leaver'] == False].copy()
            for col in ['author', 'previous_author', 'next_author']:
                if col in df_filtered.columns:
                    df_filtered[col] = df_filtered[col].map(self.initials_map).fillna('')
            logger.info("Replaced authors with initials")
            logger.debug(f"DataFrame after replacement:\n{df_filtered[['author', 'previous_author', 'next_author']].head().to_string()}")
            return df_filtered
        except Exception as e:
            logger.exception(f"Failed to replace authors with initials: {e}")
            return None

    def delete_redundant_attributes(self, df):
        """
        Delete specified redundant columns: 'active_years', 'early_leaver', 'number_of_chats_that_day', 'day_pct_authors', 'sequence_authors', 'sequence_response_times'.

        Args:
            df (pandas.DataFrame): Input DataFrame.

        Returns:
            pandas.DataFrame or None: Modified DataFrame with deleted columns, or None if processing fails.
        """
        if df is None or df.empty:
            logger.error("No valid DataFrame provided for deleting attributes")
            return None
        try:
            redundant_cols = ['active_years', 'early_leaver', 'number_of_chats_that_day', 'day_pct_authors', 'sequence_authors', 'sequence_response_times']
            existing_cols = [col for col in redundant_cols if col in df.columns]
            if existing_cols:
                df = df.drop(columns=existing_cols)
                logger.info(f"Deleted redundant columns: {existing_cols}")
            else:
                logger.info("No redundant columns found to delete")
            logger.debug(f"DataFrame after deletion:\n{df.head().to_string()}")
            return df
        except Exception as e:
            logger.exception(f"Failed to delete redundant attributes: {e}")
            return None

    def organize_df(self, df):
        """
        Create a new DataFrame with the specified columns in the desired order, excluding rows where early_leaver is True.
        Args:
            df (pandas.DataFrame): Input DataFrame with required columns.
        Returns:
            pandas.DataFrame or None: Organized DataFrame, or None if processing fails.
        """
        if df is None or df.empty:
            logger.error("No valid DataFrame provided for organizing")
            return None
        try:
            # Ensure all required columns are present
            required_columns = [
                'timestamp', 'author', 'message_cleaned', 'has_emoji', 'whatsapp_group',
                'number_of_emojis', 'has_link', 'was_deleted', 'pictures_deleted',
                'videos_deleted', 'audios_deleted', 'gifs_deleted', 'stickers_deleted',
                'documents_deleted', 'videonotes_deleted', 'year', 'month', 'week', 'active_years', 'early_leaver'
            ]
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns: {[col for col in required_columns if col not in df.columns]}")
                return None

            # Filter out rows where early_leaver is True
            rows_before = len(df)
            df = df[df['early_leaver'] == False].reset_index(drop=True)
            rows_after = len(df)
            logger.info(f"Filtered out early leavers: {rows_before} rows reduced to {rows_after} rows")

            # Add new columns
            df['number_of_chats_that_day'] = self.number_of_chats_that_day(df)
            df['length_chat'] = df['message_cleaned'].apply(self.length_chat)
            df['previous_author'] = self.previous_author(df)
            df['response_time'] = self.response_time(df)
            df['next_author'] = self.next_author(df)
            df['list_of_all_emojis'] = df['message_cleaned'].apply(self.list_of_all_emojis)
            df['list_of_connected_emojis'] = df['message_cleaned'].apply(self.list_of_connected_emojis)
            df['number_of_punctuations'] = df['message_cleaned'].apply(self.count_punctuations)
            df['has_punctuation'] = df['message_cleaned'].apply(self.has_punctuation)
            df['list_of_all_punctuations'] = df['message_cleaned'].apply(self.list_of_all_punctuations)
            df['list_of_connected_punctuations'] = df['message_cleaned'].apply(self.list_of_connected_punctuations)
            df['ends_with_emoji'] = df['message_cleaned'].apply(self.ends_with_emoji)
            df['emoji_ending_chat'] = df['message_cleaned'].apply(self.emoji_ending_chat)
            df['ends_with_punctuation'] = df['message_cleaned'].apply(self.ends_with_punctuation)
            df['punctuation_ending_chat'] = df['message_cleaned'].apply(self.punctuation_ending_chat)
            df['pct_emojis'] = self.calc_pct_emojis(df)
            df['pct_punctuations'] = self.calc_pct_punctuations(df)
            df['day_pct_length_chat'] = self.calc_day_pct_length_chat(df)
            df['day_pct_length_emojis'] = self.calc_day_pct_length_emojis(df)
            df['day_pct_length_punctuations'] = self.calc_day_pct_length_punctuations(df)
            df['number_of_unique_participants_that_day'] = self.number_of_unique_participants_that_day(df)
            df['day_pct_authors'] = self.calc_day_pct_authors(df)
            df['sequence_authors'] = self.find_sequence_authors(df)
            df['sequence_response_times'] = self.find_sequence_response_times(df)
            df = self.replace_author_by_initials(df)

            # Select and order columns (excluding number_of_changes_to_group and convert_emoji)
            organized_columns = [
                'whatsapp_group', 'timestamp', 'year', 'month', 'week', 'author', 'active_years', 'early_leaver',
                'number_of_chats_that_day', 'length_chat', 'previous_author', 'response_time', 'next_author',
                'has_link', 'was_deleted', 'pictures_deleted', 'videos_deleted', 'audios_deleted',
                'gifs_deleted', 'stickers_deleted', 'documents_deleted', 'videonotes_deleted',
                'number_of_emojis', 'has_emoji', 'list_of_all_emojis', 'list_of_connected_emojis',
                'ends_with_emoji', 'emoji_ending_chat', 'pct_emojis', 'number_of_punctuations', 'has_punctuation',
                'list_of_all_punctuations', 'list_of_connected_punctuations', 'ends_with_punctuation',
                'punctuation_ending_chat', 'pct_punctuations', 'day_pct_length_chat', 'day_pct_length_emojis', 'day_pct_length_punctuations',
                'number_of_unique_participants_that_day', 'day_pct_authors', 'sequence_authors', 'sequence_response_times'
            ]
            df_organized = df[organized_columns]
            logger.info(f"Organized DataFrame with {len(df_organized)} rows and columns: {df_organized.columns.tolist()}")
            logger.debug(f"Organized DataFrame head:\n{df_organized.head().to_string()}")
            return df_organized
        except Exception as e:
            logger.exception(f"Failed to organize DataFrame: {e}")
            return None