import sys
import tomllib
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
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

    # Define authors and their short labels
    authors = {
        "Anthony van Tilburg": "AvT",
        "Anja Berkemeijer": "AB",
        "Phons Berkemeijer": "PB",
        "Madeleine": "M",
    }

    # Filter out group name and reset index
    df = df[df["author"] != "MAAP"].reset_index(drop=True)

    # Count messages with emojis per author
    emoji_counts = df[df["has_emoji"]].groupby("author").size().reset_index(name="emoji_messages")

    # Count total messages per author
    total_counts = df.groupby("author").size().reset_index(name="total_messages")

    # Create a DataFrame with all authors, filling missing ones with 0
    emoji_summary = pd.DataFrame({"author": list(authors.keys()), "label": list(authors.values())})
    emoji_summary = emoji_summary.merge(emoji_counts, on="author", how="left").fillna(
        {"emoji_messages": 0}
    )
    emoji_summary = emoji_summary.merge(total_counts, on="author", how="left").fillna(
        {"total_messages": 0}
    )
    emoji_summary["emoji_messages"] = emoji_summary["emoji_messages"].astype(int)
    emoji_summary["total_messages"] = emoji_summary["total_messages"].astype(int)

    # Sort by emoji_messages in descending order
    emoji_summary = emoji_summary.sort_values(by="emoji_messages", ascending=False).reset_index(
        drop=True
    )

    # Log the emoji and total message counts
    logger.info("Emoji and total message counts per author (sorted by emoji messages):")
    logger.info(emoji_summary)

    # Plotting
    _fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.25  # Width for each bar (to keep bars touching)
    x = [i * bar_width for i in range(len(authors))]  # Positions for gray bars

    # Define shades of gray (decreasing darkness) and blue (decreasing darkness)
    gray_shades = ["#404040", "#606060", "#808080", "#A0A0A0"]
    blue_shades = ["#00008B", "#0000CD", "#4169E1", "#6495ED"]  # Dark to light blue

    # Plot total messages (gray bars) first
    ax.bar(
        x,
        emoji_summary["total_messages"],
        width=bar_width,
        color=gray_shades,
        align="edge",
        label="Total Messages",
    )

    # Plot emoji messages (blue bars) on top, offset to start at midpoint of gray bars
    bars_emoji = ax.bar(
        [i + bar_width / 2 for i in x],
        emoji_summary["emoji_messages"],
        width=bar_width,
        color=blue_shades,
        align="edge",
        label="Messages with Emojis",
    )

    # Add author names on blue bars
    for i, bar in enumerate(bars_emoji):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height / 2,
            emoji_summary["label"].iloc[i],
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            rotation="vertical",
        )

    # Customize plot
    ax.set_xticks([i * bar_width + bar_width for i in range(len(authors))])
    ax.set_xticklabels(emoji_summary["label"])
    ax.set_ylabel("Number of Messages")
    ax.set_title("Total Messages and Emoji Usage by Author (Sorted by Emoji Messages)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend()

    # Save and show plot
    output_path = Path("img/emoji_usage_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.tight_layout()

    # Save summary to CSV
    output_csv = Path("data/emoji_usage_summary.csv")
    emoji_summary.to_csv(output_csv, index=False)
    logger.info(f"Saved emoji usage summary to {output_csv}")

    # Save image(s)
    output_path = Path("img/04-distribution/emoji_usage_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
