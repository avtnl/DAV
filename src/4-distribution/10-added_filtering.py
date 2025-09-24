import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from loguru import logger
import warnings
import tomllib
import re
import numpy as np
import string


# Configure logger to write to a file
logger.add("logs/app_{time}.log", rotation="1 MB", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
warnings.simplefilter(action="ignore", category=FutureWarning)

def main():
    # Set font to Segoe UI Emoji for character support, with DejaVu Sans as fallback
    try:
        plt.rcParams['font.family'] = ['Segoe UI Emoji', 'DejaVu Sans']
    except:
        logger.warning("Segoe UI Emoji font not found. Falling back to DejaVu Sans. Some characters may not render correctly.")
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

    # Punctuation count analysis
    count_all = {}
    count_once = {}
    none_count = 0  # Count messages with no punctuation
    
    # Process each cleaned message
    for message in df["message_cleaned"]:
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
    punctuation_counts_df = pd.DataFrame({
        "punctuation": list(count_all.keys()),
        "count_all": list(count_all.values()),
        "count_once": [count_once.get(p, 0) for p in count_all.keys()]
    })
    
    # Add "None" category
    none_row = pd.DataFrame({
        "punctuation": ["None"],
        "count_all": [none_count],
        "count_once": [0]  # count_once is undefined for None
    })
    
    # Append "None" to DataFrame
    punctuation_counts_df = pd.concat([punctuation_counts_df, none_row], ignore_index=True)
    
    # Sort by count_all descending
    punctuation_counts_df = punctuation_counts_df.sort_values(by="count_all", ascending=False).reset_index(drop=True)
    
    # Log punctuation counts
    logger.info("Punctuation usage counts (sorted by count_all, including None):")
    logger.info(punctuation_counts_df)

    # Calculate percentages
    total_count_all = punctuation_counts_df["count_all"].sum()
    total_count_once = punctuation_counts_df[punctuation_counts_df["punctuation"] != "None"]["count_once"].sum()
    punctuation_counts_df["percent_all"] = (punctuation_counts_df["count_all"] / total_count_all * 100).round(2) if total_count_all > 0 else 0.0
    punctuation_counts_df["percent_once"] = 0.0
    punctuation_counts_df.loc[punctuation_counts_df["punctuation"] != "None", "percent_once"] = (punctuation_counts_df["count_once"] / total_count_once * 100).round(2) if total_count_once > 0 else 0.0
    
    # Plotting combined count_all and count_once
    fig, ax = plt.subplots(figsize=(len(punctuation_counts_df) * 0.4, 6))
    x_positions = np.arange(len(punctuation_counts_df))  # Evenly spaced x-positions
    bar_width = 0.4  # Width for each bar (two bars per punctuation)

    # Plot count_all bars (gray)
    ax.bar(x_positions - bar_width / 2, punctuation_counts_df["count_all"], width=bar_width, color="#808080", label="All Occurrences")
    
    # Plot count_once bars (light blue), skipping "None"
    count_once_data = punctuation_counts_df["count_once"].copy()
    count_once_data[punctuation_counts_df["punctuation"] == "None"] = 0  # No count_once bar for None
    ax.bar(x_positions + bar_width / 2, count_once_data, width=bar_width, color="#ADD8E6", label="Once per Message")
    
    # Customize plot
    ax.set_xticks(x_positions)
    ax.set_xticklabels(punctuation_counts_df["punctuation"], rotation=0)
    ax.set_ylabel("Count")
    ax.set_title("Punctuation Marks and None: Total vs. Unique Message Occurrences")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend()

    # Adjust layout and save
    plt.tight_layout()
    output_path = Path("img/punctuation_counts_combined.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved combined punctuation counts plot: {output_path}")
    plt.show()

    # Plotting count_all percentages
    fig1, ax1 = plt.subplots(figsize=(len(punctuation_counts_df) * 0.4, 6))
    x_positions = np.arange(len(punctuation_counts_df))  # Evenly spaced x-positions
    ax1.bar(x_positions, punctuation_counts_df["percent_all"], width=0.8, color="#808080")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(punctuation_counts_df["punctuation"], rotation=0)
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Punctuation Marks and None: Percentage of Total Occurrences")
    ax1.set_ylim(0, max(punctuation_counts_df["percent_all"]) + 5 if total_count_all > 0 else 100)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=10)
    plt.tight_layout()
    output_path = Path("img/punctuation_percent_all.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved count_all percentage plot: {output_path}")
    plt.show()

    # Plotting count_once percentages (excluding None)
    punctuation_counts_once = punctuation_counts_df[punctuation_counts_df["punctuation"] != "None"].sort_values(by="count_once", ascending=False).reset_index(drop=True)
    fig2, ax2 = plt.subplots(figsize=(len(punctuation_counts_once) * 0.4, 6))
    x_positions = np.arange(len(punctuation_counts_once))
    ax2.bar(x_positions, punctuation_counts_once["percent_once"], width=0.8, color="#ADD8E6")
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(punctuation_counts_once["punctuation"], rotation=0)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Punctuation Marks: Percentage of Unique Message Occurrences")
    ax2.set_ylim(0, max(punctuation_counts_once["percent_once"]) + 5 if total_count_once > 0 else 100)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=10)
    plt.tight_layout()
    output_path = Path("img/punctuation_percent_once.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved count_once percentage plot: {output_path}")
    plt.show()

    # Save punctuation counts to CSV
    output_csv = Path("data/punctuation_counts_summary.csv")
    punctuation_counts_df.to_csv(output_csv, index=False)
    logger.info(f"Saved punctuation counts summary to {output_csv}")

if __name__ == "__main__":
    main()