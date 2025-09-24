import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from loguru import logger
import warnings
import tomllib
import re
import pytz
from datetime import datetime
import numpy as np
import emoji
import matplotlib.font_manager as fm
import string

# Configure logger to write to a file
logger.add("logs/app_{time}.log", rotation="1 MB", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
warnings.simplefilter(action="ignore", category=FutureWarning)

def main():
    # Set font to Segoe UI Emoji for emoji support
    try:
        plt.rcParams['font.family'] = 'Segoe UI Emoji'
    except:
        logger.warning("Segoe UI Emoji font not found. Falling back to default font. Some emojis may not render correctly.")
        plt.rcParams['font.family'] = 'DejaVu Sans'

    # Read configuration
    logger.debug("Loading configuration from config.toml")
    configfile = Path("config.toml").resolve()
    try:
        with configfile.open("rb") as f:
            config = tomllib.load(f)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.exception(f"Failed to load config.toml: {e}")
        exit(1)

    # Define processed directory
    processed = Path("data/processed")
    logger.debug(f"Processed directory set to: {processed}")

    # Load and validate data files
    datafile = processed / config["current"]
    if not datafile.exists():
        logger.warning(
            "Datafile does not exist. First run src/preprocess.py, and check the timestamp!"
        )
        exit(1)
    df = pd.read_parquet(datafile)
    logger.info(df)

    # Filter out group name and reset index
    df = df[df["author"] != "MAAP"].reset_index(drop=True)

    # Add Changes column
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

    # Clean messages and update Changes column
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

        # Update Changes column
        row["changes"] = ", ".join(changes) if changes else row["changes"]
        row["message_cleaned"] = message
        return row

    df = df.apply(clean_message, axis=1)
    logger.info(f"Cleaned messages: {df[['message', 'message_cleaned', 'changes']].head(10).to_string()}")

    # Save amended DataFrame to [filename]_filtered.csv
    filename_base = config["current"].rsplit('.', 1)[0]  # Remove extension
    output_csv = processed / f"{filename_base}_filtered.csv"
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved amended DataFrame to {output_csv}")

    # Filter messages with emojis
    emoji_msgs = df[df["has_emoji"] == True]

    # Initialize dictionaries for counts
    count_all = {}
    count_once = {}

    # Process each message
    for message in emoji_msgs["message"]:
        # Extract all emojis in the message
        emojis = [char for char in message if char in emoji.EMOJI_DATA]
        # Count all occurrences
        for e in emojis:
            count_all[e] = count_all.get(e, 0) + 1
        # Count unique emojis once per message
        unique_emojis = set(emojis)
        for e in unique_emojis:
            count_once[e] = count_once.get(e, 0) + 1

    # Create DataFrame for emoji counts
    emoji_counts_df = pd.DataFrame({
        "emoji": list(count_all.keys()),
        "count_all": list(count_all.values()),
        "count_once": [count_once.get(e, 0) for e in count_all.keys()]
    })

    # Log emoji counts
    logger.info("Emoji usage counts (before sorting):")
    logger.info(emoji_counts_df)

    # Calculate percentages
    total_all = emoji_counts_df["count_all"].sum()
    total_once = emoji_counts_df["count_once"].sum()
    emoji_counts_df["percent_all"] = (emoji_counts_df["count_all"] / total_all) * 100
    emoji_counts_df["percent_once"] = (emoji_counts_df["count_once"] / total_once) * 100

    # Sort by count_all for consistency
    emoji_counts_df = emoji_counts_df.sort_values(by="count_all", ascending=False)

    # Generate labels (aa-az, ba-bz, ..., za-zz)
    letters = string.ascii_lowercase
    labels = [f"{a}{b}" for a in letters for b in letters]
    if len(labels) < len(emoji_counts_df):
        logger.warning("Not enough labels for all emojis. Truncating to available labels.")
        emoji_counts_df = emoji_counts_df.head(len(labels))
    emoji_counts_df["label"] = labels[:len(emoji_counts_df)]

    # Create legend mapping for top 26 emojis (aa-az)
    legend_dict = dict(zip(emoji_counts_df["label"].head(26), emoji_counts_df["emoji"].head(26)))

    # Plotting count_all
    fig1, ax1 = plt.subplots(figsize=(len(emoji_counts_df) * 0.3, 6))
    ax1.bar(emoji_counts_df["label"], emoji_counts_df["percent_all"], color="#808080")
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Emoji Usage: All Occurrences")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='x', labelsize=8)
    ax1.tick_params(axis='y', labelsize=10)
    # Add legend for aa-az
    legend_labels = [f"{label}: {emoji}" for label, emoji in legend_dict.items()]
    ax1.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.tight_layout()
    output_path1 = Path("img/emoji_counts_all.png")
    plt.savefig(output_path1, dpi=300, bbox_inches="tight")
    logger.info(f"Saved count_all plot: {output_path1}")
    plt.show()

    # Plotting count_once
    fig2, ax2 = plt.subplots(figsize=(len(emoji_counts_df) * 0.3, 6))
    ax2.bar(emoji_counts_df["label"], emoji_counts_df["percent_once"], color="#ADD8E6")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Emoji Usage: Once per Message")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='x', labelsize=8)
    ax2.tick_params(axis='y', labelsize=10)
    # Add legend for aa-az
    ax2.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.tight_layout()
    output_path2 = Path("img/emoji_counts_once.png")
    plt.savefig(output_path2, dpi=300, bbox_inches="tight")
    logger.info(f"Saved count_once plot: {output_path2}")
    plt.show()

    # Save emoji counts to CSV
    output_csv = Path("data/emoji_counts_summary.csv")
    emoji_counts_df.to_csv(output_csv, index=False)
    logger.info(f"Saved emoji counts summary to {output_csv}")

if __name__ == "__main__":
    main()