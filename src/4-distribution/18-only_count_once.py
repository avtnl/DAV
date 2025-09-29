import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from loguru import logger
import warnings
import tomllib
import re
import numpy as np
import emoji
import matplotlib.font_manager as fm

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
                            remove_start = bracket_idx - 1 # Include space
                        else:
                            remove_start = bracket_idx
                        # Remove from bracket_idx (or bracket_idx - 1) to end of match
                        message = message[:remove_start] + message[match.end():]
                    else:
                        # If no '[', just remove the matched media phrase
                        message = re.sub(pattern, '', message, flags=flags)
                    message = message.strip()
                    matched = True
                    break # Restart loop to check for more media patterns
            if not matched:
                break # Exit loop if no media patterns matched
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

    # Save amended DataFrame to [filename]_filtered.csv
    filename_base = config["current"].rsplit('.', 1)[0] # Remove extension
    output_csv = processed / f"{filename_base}_filtered.csv"
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved amended DataFrame to {output_csv}")

    # Filter messages with emojis
    emoji_msgs = df[df["has_emoji"] == True]

    # Define emojis to ignore (skin tone modifiers)
    ignore_emojis = {chr(int(code, 16)) for code in ['1F3FB', '1F3FC', '1F3FD', '1F3FE', '1F3FF']}

    # Initialize dictionary for counts
    count_once = {}

    # Process each message
    for message in emoji_msgs["message"]:
        # Extract all emojis in the message, excluding ignored ones
        emojis = [char for char in message if char in emoji.EMOJI_DATA and char not in ignore_emojis]
        # Count unique emojis once per message
        unique_emojis = set(emojis)
        for e in unique_emojis:
            count_once[e] = count_once.get(e, 0) + 1

    # Create DataFrame for emoji counts
    emoji_counts_df = pd.DataFrame({
        "emoji": list(count_once.keys()),
        "count_once": list(count_once.values())
    })

    # Log emoji counts
    logger.info("Emoji usage counts (before sorting):")
    logger.info(emoji_counts_df)

    # Calculate percentages
    total_once = emoji_counts_df["count_once"].sum()
    emoji_counts_df["percent_once"] = (emoji_counts_df["count_once"] / total_once) * 100

    # Add Unicode code and name
    emoji_counts_df["unicode_code"] = emoji_counts_df["emoji"].apply(lambda x: f"U+{ord(x):04X}")
    emoji_counts_df["unicode_name"] = emoji_counts_df["emoji"].apply(
        lambda x: emoji.demojize(x).strip(":").replace("_", " ").title()
    )

    # Sort by count_once
    emoji_counts_df = emoji_counts_df.sort_values(by="count_once", ascending=False)

    # Calculate cumulative percentages
    cumulative_once = emoji_counts_df["percent_once"].cumsum()

    # Plotting count_once
    fig, ax = plt.subplots(figsize=(len(emoji_counts_df) * 0.3, 6))
    ax2 = ax.twinx() # Secondary y-axis for cumulative percentage
    x_positions = np.arange(len(emoji_counts_df))
    bars = ax.bar(x_positions, emoji_counts_df["percent_once"], color="#ADD8E6", align='edge', width=0.5)
    ax.set_ylabel("Percentage (%)", fontsize=10, labelpad=20)
    ax.set_title("Emoji Usage: Once per Message")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 20))
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xlim(-0.5, len(emoji_counts_df))
    ylim_bottom, ylim_top = ax.get_ylim()
    ax.set_ylim(ylim_bottom - 3, ylim_top)
    # Calculate x for 75%
    cum_once_np = np.array(cumulative_once)
    idx_once = np.where(cum_once_np >= 75)[0][0]
    x_once = idx_once + 1
    y_once = len(emoji_counts_df)
    # Add vertical dashed line
    ax.axvline(x=idx_once + 0.5, color="orange", linestyle="--", linewidth=1)
    # Add texts for x and y emojis
    left_mid = idx_once / 2
    right_mid = (idx_once + 0.5) + (y_once - idx_once - 1) / 2
    y_text = ylim_bottom - 1.5
    ax.text(left_mid, y_text, f"{x_once} emojis", ha='center', fontsize=8)
    ax.text(right_mid, y_text, f"{y_once} emojis", ha='center', fontsize=8)
    ax.set_xticks([])  # No x-ticks
    # Plot cumulative line and 75% dashed line
    ax2.plot(x_positions + 0.25, cumulative_once, color="orange", label="Cumulative %")
    ax2.axhline(y=75, color="orange", linestyle="--", linewidth=1, xmin=-0.5, xmax=len(emoji_counts_df) + 0.5)
    ax2.set_ylabel("Cumulative Percentage (%)", fontsize=10, labelpad=20)
    ax2.set_ylim(0, 100)
    ax2.set_yticks(np.arange(0, 101, 10))
    ax2.spines['right'].set_position(('outward', 20))
    ax2.tick_params(axis='y', labelsize=10, colors='orange')
    ax2.spines['right'].set_color('orange')
    ax2.set_title("Top 25:", loc='center', y=0.45, fontsize=12)
    # Add table for top 25 emojis (count_once), with rank first
    top_25_once = emoji_counts_df.head(25)
    cum_once_top = top_25_once["percent_once"].cumsum()
    table_data = [
        [str(i+1) for i in range(25)],
        list(top_25_once["emoji"]),
        [f"{count:.0f}" for count in top_25_once["count_once"]],
        [f"{cum:.1f}%" for cum in cum_once_top]
    ]
    table = ax.table(cellText=table_data,
                     rowLabels=["Rank", "Emoji", "Count", "Cum"],
                     colWidths=[0.05] * 25,
                     loc='center',
                     bbox=[0.1, 0.25, 0.8, 0.2], # Position between center and bottom
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax2.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9)
    output_path = Path("img/emoji_counts_once.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved count_once plot: {output_path}")
    plt.show()

    # Save emoji counts to CSV with Unicode code and name
    output_csv = Path("data/emoji_counts_summary.csv")
    emoji_counts_df[["emoji", "count_once", "percent_once", "unicode_code", "unicode_name"]].to_csv(output_csv, index=False)
    logger.info(f"Saved emoji counts summary to {output_csv}")

if __name__ == "__main__":
    main()