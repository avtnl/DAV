import string
import sys
import tomllib
import warnings
from pathlib import Path

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
    # Set font to Segoe UI Emoji for character support, with DejaVu Sans as fallback
    try:
        plt.rcParams["font.family"] = ["Segoe UI Emoji", "DejaVu Sans"]
    except:
        logger.warning(
            "Segoe UI Emoji font not found. Falling back to DejaVu Sans. Some characters may not render correctly."
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

    # Punctuation count analysis
    # Filter messages with emojis
    punctuation_msgs = df[df["has_emoji"]]

    # Initialize dictionaries for counts
    count_all = {}
    count_once = {}
    none_count = 0  # Count messages with no punctuation

    # Process each message
    for message in punctuation_msgs["message"]:
        # Extract all punctuation marks in the message
        punctuations = [char for char in message if char in string.punctuation]
        if not punctuations:
            none_count += 1  # Increment if no punctuation
        else:
            # Count all occurrences
            for p in punctuations:
                count_all[p] = count_all.get(p, 0) + 1
            # Count unique punctuation marks once per message
            unique_punctuations = set(punctuations)
            for p in unique_punctuations:
                count_once[p] = count_once.get(p, 0) + 1

    # Create DataFrame for punctuation counts
    punctuation_counts_df = pd.DataFrame(
        {
            "punctuation": list(count_all.keys()),
            "count_all": list(count_all.values()),
            "count_once": [count_once.get(p, 0) for p in count_all.keys()],
        }
    )

    # Add "None" category
    none_row = pd.DataFrame(
        {
            "punctuation": ["None"],
            "count_all": [none_count],
            "count_once": [0],  # count_once is undefined for None
        }
    )

    # Append "None" to DataFrame
    punctuation_counts_df = pd.concat([punctuation_counts_df, none_row], ignore_index=True)

    # Sort by count_all descending
    punctuation_counts_df = punctuation_counts_df.sort_values(
        by="count_all", ascending=False
    ).reset_index(drop=True)

    # Log punctuation counts
    logger.info("Punctuation usage counts (sorted by count_all, including None):")
    logger.info(punctuation_counts_df)

    # Plotting combined count_all and count_once
    _fig, ax = plt.subplots(figsize=(len(punctuation_counts_df) * 0.4, 6))
    x_positions = np.arange(len(punctuation_counts_df))  # Evenly spaced x-positions
    bar_width = 0.4  # Width for each bar (two bars per punctuation)

    # Plot count_all bars (gray)
    ax.bar(
        x_positions - bar_width / 2,
        punctuation_counts_df["count_all"],
        width=bar_width,
        color="#808080",
        label="All Occurrences",
    )

    # Plot count_once bars (light blue), skipping "None"
    count_once_data = punctuation_counts_df["count_once"].copy()
    count_once_data[punctuation_counts_df["punctuation"] == "None"] = (
        0  # No count_once bar for None
    )
    ax.bar(
        x_positions + bar_width / 2,
        count_once_data,
        width=bar_width,
        color="#ADD8E6",
        label="Once per Message",
    )

    # Customize plot
    ax.set_xticks(x_positions)
    ax.set_xticklabels(punctuation_counts_df["punctuation"], rotation=0)
    ax.set_ylabel("Count")
    ax.set_title("Punctuation Marks and None: Total vs. Unique Message Occurrences")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend()

    # Adjust layout and save
    plt.tight_layout()
    output_path = Path("img/punctuation_counts_combined.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved combined punctuation counts plot: {output_path}")
    plt.show()

    # Save punctuation counts to CSV
    output_csv = Path("data/punctuation_counts_summary.csv")
    punctuation_counts_df.to_csv(output_csv, index=False)
    logger.info(f"Saved punctuation counts summary to {output_csv}")


if __name__ == "__main__":
    main()
