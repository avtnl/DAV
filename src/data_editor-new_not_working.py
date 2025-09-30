import pandas as pd
import re
from loguru import logger
import emoji

class DataEditor:
    def __init__(self):
        self.emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f1e0-\U0001f1ff"  # flags (iOS)
            "\U00002702-\U000027b0"  # Dingbats
            "\U000024c2-\U0001f251"
            "\U0001f900-\U0001f9ff"  # supplemental symbols & pictographs
            "]",
            flags=re.UNICODE,
        )
        # Define emojis to ignore (skin tone modifiers)
        self.ignore_emojis = {chr(int(code, 16)) for code in ['1F3FB', '1F3FC', '1F3FD', '1F3FE', '1F3FF']}
        # Define URL pattern
        self.url_pattern = re.compile(r"(?i)\b((?:https?://|ftp://|www\.)\S+)", flags=re.UNICODE)

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
        Sometimes, author names have a tilde in front due to formatting issues.
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
        Check for the presence of emojis and count the total number of emojis in the input text,
        excluding specified skin tone modifiers.
        Args:
            text (str): Text to check and count emojis in.
        Returns:
            tuple: (bool, int) - (True if emojis are present, number of emojis excluding ignored ones).
        """
        if not isinstance(text, str):
            return False, 0
        emojis = [char for char in text if char in emoji.EMOJI_DATA and char not in self.ignore_emojis]
        return bool(emojis), len(emojis)

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
            
            # Verify the result
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
            # Debug: Check author counts per group
            logger.info("Author counts per WhatsApp group:")
            logger.info(df.groupby(["whatsapp_group", "author"]).size().to_string())
            
            # Filter out group names
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
        Clean messages in the DataFrame by removing deleted messages and media patterns,
        and add columns to track the count of each type of deleted media and group picture changes.
        Args:
            df (pandas.DataFrame): Input DataFrame with 'message' column.
        Returns:
            pandas.DataFrame or None: Modified DataFrame with new columns, or None if processing fails.
        """
        if df is None or df.empty:
            logger.error("No valid DataFrame provided for cleaning deleted media patterns")
            return None
        try:
            # Initialize new columns
            df["has_emoji"] = False
            df["number_of_emojis"] = 0
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

            # Define regex patterns for cleaning
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
            fallback_pattern = r'\s*[\u200e\u200f]*\[\d{2}-\d{2}-\d{4},\s*\d{2}:\d{2}:\d{2}\]\s*(?:Anthony van Tilburg|Anja Berkemeijer|Phons Berkemeijer|Madeleine)[\s\u200e\u200f]*:.*'

            # Clean messages and update columns
            def clean_message(row):
                message = row["message"]
                # Update deletion status
                row["was_deleted"] = self.was_deleted(message)
                # Update group picture changes
                row["number_of_changes_to_group"] = self.changes_to_grouppicture(message)

                # Apply non-media patterns
                for pattern, change, flags in non_media_patterns:
                    message = re.sub(pattern, '', message, flags=flags).strip()

                # Count media pattern occurrences
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
                        # Remove matched patterns
                        message = re.sub(pattern, '', message, flags=flags).strip()

                # Fallback: Remove any trailing timestamp and author/media info
                if re.search(fallback_pattern, message, flags=re.IGNORECASE):
                    logger.debug(f"Matched fallback pattern '{fallback_pattern}' in message: {message}")
                    message = re.sub(fallback_pattern, '', message, flags=re.IGNORECASE).strip()

                # If message is empty, contains only spaces, or is None, set to "completely removed"
                if message is None or message == "" or message.strip() == "":
                    message = "completely removed"

                row["message_cleaned"] = message
                return row

            df = df.apply(clean_message, axis=1)
            logger.info(f"Cleaned messages: {df[['message', 'message_cleaned', 'has_emoji', 'number_of_emojis', 'was_deleted', 'number_of_changes_to_group', 'pictures_deleted', 'videos_deleted', 'audios_deleted', 'gifs_deleted', 'stickers_deleted', 'documents_deleted', 'videonotes_deleted']].head(10).to_string()}")
            return df
        except Exception as e:
            logger.exception(f"Failed to clean messages for deleted media patterns: {e}")
            return None