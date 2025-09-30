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

    # Define authors and their short labels
    authors = {
        "Anthony van Tilburg": "AvT",
        "Anja Berkemeijer": "AB",
        "Phons Berkemeijer": "PB",
        "Madeleine": "M"
    }

    # Filter out group name and reset index
    df = df[df["author"] != "MAAP"].reset_index(drop=True)

    # Count messages with emojis per author
    emoji_counts = df[df["has_emoji"] == True].groupby("author").size().reset_index(name="emoji_messages")
    
    # Count total messages per author
    total_counts = df.groupby("author").size().reset_index(name="total_messages")

    # Create a DataFrame with all authors, filling missing ones with 0
    emoji_summary = pd.DataFrame({
        "author": list(authors.keys()),
        "label": list(authors.values())
    })
    emoji_summary = emoji_summary.merge(emoji_counts, on="author", how="left").fillna({"emoji_messages": 0})
    emoji_summary = emoji_summary.merge(total_counts, on="author", how="left").fillna({"total_messages": 0})
    emoji_summary["emoji_messages"] = emoji_summary["emoji_messages"].astype(int)
    emoji_summary["total_messages"] = emoji_summary["total_messages"].astype(int)

    # Sort by emoji_messages in descending order
    emoji_summary = emoji_summary.sort_values(by="emoji_messages", ascending=False).reset_index(drop=True)

    # Log the emoji and total message counts
    logger.info("Emoji and total message counts per author (sorted by emoji messages):")
    logger.info(emoji_summary)

    # Plotting author-based bars
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.25  # Width for each bar (to keep bars touching)
    x = [i * bar_width for i in range(len(authors))]  # Positions for gray bars
    
    # Define shades of gray (decreasing darkness) and blue (decreasing darkness)
    gray_shades = ["#404040", "#606060", "#808080", "#A0A0A0"]
    blue_shades = ["#00008B", "#0000CD", "#4169E1", "#6495ED"]

    # Plot total messages (gray bars) first
    bars_total = ax.bar(x, emoji_summary["total_messages"], width=bar_width, color=gray_shades, align='edge', label="Total Messages")
    
    # Plot emoji messages (blue bars) on top, offset to start at midpoint of gray bars
    bars_emoji = ax.bar([i + bar_width / 2 for i in x], emoji_summary["emoji_messages"], width=bar_width, color=blue_shades, align='edge', label="Messages with Emojis")
    
    # Add author names on blue bars
    for i, bar in enumerate(bars_emoji):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height / 2, 
                emoji_summary["label"].iloc[i], 
                ha="center", va="center", color="white", fontsize=10, rotation="vertical")

    # Customize plot
    ax.set_xticks([i * bar_width + bar_width for i in range(len(authors))])
    ax.set_xticklabels(emoji_summary["label"])
    ax.set_ylabel("Number of Messages")
    ax.set_title("Total Messages and Emoji Usage by Author (Sorted by Emoji Messages)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend()

    # Save and show author-based plot
    output_path = Path("img/emoji_usage_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()

    # Emoji count analysis
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

    # Plotting count_all (top 25)
    emoji_counts_all = emoji_counts_df.sort_values(by="count_all", ascending=False).head(25)
    fig1 = plt.figure(figsize=(6, len(emoji_counts_all) * 0.4))
    ax1 = fig1.add_subplot(111)
    y_positions = range(len(emoji_counts_all))  # Evenly spaced y-positions
    ax1.barh(y_positions, emoji_counts_all["count_all"], height=0.8, color="#808080")
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(emoji_counts_all["emoji"])
    ax1.set_xlabel("Count (All Occurrences)")
    ax1.set_title("Top 25 Emojis by Total Occurrences")
    ax1.invert_yaxis()  # Highest count at top
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.tick_params(axis='y', labelsize=12)
    plt.tight_layout()
    output_path = Path("img/emoji_counts_all.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved count_all plot: {output_path}")
    plt.show()

    # Plotting count_once (top 25)
    emoji_counts_once = emoji_counts_df.sort_values(by="count_once", ascending=False).head(25)
    fig2 = plt.figure(figsize=(6, len(emoji_counts_once) * 0.4))
    ax2 = fig2.add_subplot(111)
    y_positions = range(len(emoji_counts_once))  # Evenly spaced y-positions
    ax2.barh(y_positions, emoji_counts_once["count_once"], height=0.8, color="#808080")
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(emoji_counts_once["emoji"])
    ax2.set_xlabel("Count (Once per Message)")
    ax2.set_title("Top 25 Emojis by Unique Message Occurrences")
    ax2.invert_yaxis()  # Highest count at top
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='x', labelsize=10)
    ax2.tick_params(axis='y', labelsize=12)
    plt.tight_layout()
    output_path = Path("img/emoji_counts_once.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved count_once plot: {output_path}")
    plt.show()

    # Save emoji counts to CSV
    output_csv = Path("data/emoji_counts_summary.csv")
    emoji_counts_df.to_csv(output_csv, index=False)
    logger.info(f"Saved emoji counts summary to {output_csv}")

if __name__ == "__main__":
    main()