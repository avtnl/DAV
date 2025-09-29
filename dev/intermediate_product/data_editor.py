import pandas as pd
import re
from loguru import logger

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
            "]+",
            flags=re.UNICODE,
        )

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
        Check if the input text contains any emojis and add that as a feature.

        Args:
            text (str): Text to check for emojis.

        Returns:
            bool: True if the text contains emojis, False otherwise.
        """
        return bool(self.emoji_pattern.search(str(text)))

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

    def clean_for_deleted_media_patterns(self, df, whatsapp_group="all"):
        """
        Clean messages in the DataFrame by removing deleted messages and media patterns,
        and add a 'changes' column to track modifications.

        Args:
            df (pandas.DataFrame): Input DataFrame with 'message' and 'whatsapp_group' columns.
            whatsapp_group (str): WhatsApp group to process, or "all" to process all groups (default: "all").

        Returns:
            pandas.DataFrame or None: Modified DataFrame with 'message_cleaned' and 'changes' columns,
                                     or None if processing fails.
        """
        if df is None or df.empty:
            logger.error("No valid DataFrame provided for cleaning deleted media patterns")
            return None

        try:
            # Filter by whatsapp_group if not "all"
            if whatsapp_group != "all":
                df = df[df["whatsapp_group"] == whatsapp_group].copy()
                if df.empty:
                    logger.error(f"No data found for WhatsApp group '{whatsapp_group}'")
                    return None

            # Add changes column
            df["changes"] = ""

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

            # Clean messages and update changes column
            def clean_message(row):
                message = row["message"]
                changes = []
                # Apply non-media patterns first
                for pattern, change, flags in non_media_patterns:
                    if re.search(pattern, message, flags=flags):
                        logger.debug(f"Matched non-media pattern '{pattern}' in message: {message}")
                        if change not in changes:
                            changes.append(change)
                        message = re.sub(pattern, '', message, flags=flags).strip()
                # Loop over media patterns until no more matches
                while True:
                    matched = False
                    for pattern, change, flags in media_patterns:
                        match = re.search(pattern, message, flags=flags)
                        if match:
                            logger.debug(f"Matched media pattern '{pattern}' in message: {message}")
                            if change not in changes:
                                changes.append(change)
                            # Find the preceding '['
                            start_idx = match.start()
                            bracket_idx = message.rfind('[', 0, start_idx)
                            if bracket_idx != -1:
                                # Check for space before '['
                                if bracket_idx > 0 and message[bracket_idx - 1] == ' ':
                                    remove_start = bracket_idx - 1  # Include space
                                else:
                                    remove_start = bracket_idx
                                # Remove from bracket_idx (or bracket_idx - 1) to end of match
                                message = message[:remove_start] + message[match.end():]
                            else:
                                # If no '[', just remove the matched media phrase
                                message = re.sub(pattern, '', message, flags=flags)
                            message = message.strip()
                            matched = True
                            break  # Restart loop to check for more media patterns
                    if not matched:
                        break  # Exit loop if no media patterns matched
                # Fallback: Remove any trailing timestamp and author/media info
                if re.search(fallback_pattern, message, flags=re.IGNORECASE):
                    logger.debug(f"Matched fallback pattern '{fallback_pattern}' in message: {message}")
                    if "generic deleted" not in changes:
                        changes.append("generic deleted")
                    message = re.sub(fallback_pattern, '', message, flags=re.IGNORECASE).strip()
                # If message is empty, contains only spaces, or is None, set to "completely removed"
                if message is None or message == "" or message.strip() == "":
                    message = "completely removed"
                # Update changes column
                row["changes"] = ", ".join(changes) if changes else row["changes"]
                row["message_cleaned"] = message
                return row

            df = df.apply(clean_message, axis=1)
            logger.info(f"Cleaned messages: {df[['message', 'message_cleaned', 'changes']].head(10).to_string()}")

            return df
        except Exception as e:
            logger.exception(f"Failed to clean messages for deleted media patterns: {e}")
            return None