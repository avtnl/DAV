import sys
import tomllib
import warnings
from pathlib import Path

import emoji
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

# Configure logger to write to a file
logger.add(
    "logs/app_{time}.log",
    rotation="1 MB",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)
warnings.simplefilter(action="ignore", category=FutureWarning)


def main() -> None:
    # Set font to Segoe UI Emoji for emoji support
    try:
        plt.rcParams["font.family"] = "Segoe UI Emoji"
    except:
        logger.warning(
            "Segoe UI Emoji font not found. Falling back to default font. Some emojis may not render correctly."
        )
        plt.rcParams["font.family"] = "DejaVu Sans"

    # Read configuration
    logger.debug("Loading configuration from config.toml")
    configfile = Path("config.toml").resolve()
    try:
        with configfile.open("rb") as f:
            config = tomllib.load(f)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.exception(f"Failed to load config.toml: {e}")
        sys.exit(1)

    # Define processed directory
    processed = Path("data/processed")
    logger.debug(f"Processed directory set to: {processed}")

    # Load and validate data files
    datafile = processed / config["current"]
    if not datafile.exists():
        logger.warning(
            "Datafile does not exist. First run src/preprocess.py, and check the timestamp!"
        )
        sys.exit(1)
    df = pd.read_parquet(datafile)
    logger.info(df)

    # Filter out group name and reset index
    df = df[df["author"] != "MAAP"].reset_index(drop=True)

    # Emoji count analysis
    # Filter messages with emojis
    emoji_msgs = df[df["has_emoji"]]

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
    emoji_counts_df = pd.DataFrame(
        {
            "emoji": list(count_all.keys()),
            "count_all": list(count_all.values()),
            "count_once": [count_once.get(e, 0) for e in count_all.keys()],
        }
    )

    # Log emoji counts
    logger.info("Emoji usage counts (before sorting):")
    logger.info(emoji_counts_df)

    # Filter to top 25 emojis by count_all
    emoji_counts_top = emoji_counts_df.sort_values(by="count_all", ascending=False).head(25)

    # Plotting combined count_all and count_once
    _fig, ax = plt.subplots(figsize=(len(emoji_counts_top) * 0.4, 6))
    x_positions = np.arange(len(emoji_counts_top))  # Evenly spaced x-positions
    bar_width = 0.4  # Width for each bar (two bars per emoji)

    # Plot count_all bars (gray)
    ax.bar(
        x_positions - bar_width / 2,
        emoji_counts_top["count_all"],
        width=bar_width,
        color="#808080",
        label="All Occurrences",
    )

    # Plot count_once bars (light blue)
    ax.bar(
        x_positions + bar_width / 2,
        emoji_counts_top["count_once"],
        width=bar_width,
        color="#ADD8E6",
        label="Once per Message",
    )

    # Customize plot
    ax.set_xticks(x_positions)
    ax.set_xticklabels(emoji_counts_top["emoji"], rotation=0)
    ax.set_ylabel("Count")
    ax.set_title("Top 25 Emojis: Total vs. Unique Message Occurrences")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend()

    # Adjust layout and save
    plt.tight_layout()
    output_path = Path("img/emoji_counts_combined.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved combined emoji counts plot: {output_path}")
    plt.show()

    # Save emoji counts to CSV
    output_csv = Path("data/emoji_counts_summary.csv")
    emoji_counts_df.to_csv(output_csv, index=False)
    logger.info(f"Saved emoji counts summary to {output_csv}")


if __name__ == "__main__":
    main()
