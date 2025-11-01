import re
import sys
import tomllib
import warnings
from pathlib import Path

import emoji
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

    # Add changes, has_number, has_emoji, and has_name columns
    df["changes"] = ""
    df["has_number"] = False
    df["has_emoji"] = False
    df["has_name"] = False

    # Define regex patterns for cleaning
    non_media_patterns = [
        (r"Dit bericht is verwijderd\.", "message deleted", re.IGNORECASE),
        (
            r"(?:Anthony van Tilburg|Anja Berkemeijer|Phons Berkemeijer|Madeleine) heeft de groepsafbeelding gewijzigd",
            "grouppicture",
            re.IGNORECASE,
        ),
    ]
    media_patterns = [
        (r"afbeelding\s*weggelaten", "picture deleted", re.IGNORECASE),
        (r"video\s*weggelaten", "video deleted", re.IGNORECASE),
        (r"audio\s*weggelaten", "audio deleted", re.IGNORECASE),
        (r"GIF\s*weggelaten", "GIF deleted", re.IGNORECASE),
        (r"sticker\s*weggelaten", "sticker deleted", re.IGNORECASE),
        (r"document\s*weggelaten", "document deleted", re.IGNORECASE),
        (r"videonotitie\s*weggelaten", "video note deleted", re.IGNORECASE),
    ]
    fallback_pattern = r"\s*[\u200e\u200f]*\[\d{2}-\d{2}-\d{4},\s*\d{2}:\d{2}:\d{2}\]\s*(?:Anthony van Tilburg|Anja Berkemeijer|Phons Berkemeijer|Madeleine)[\s\u200e\u200f]*:.*"

    # Define regex patterns for names with typo forgiveness
    name_patterns = [
        re.compile(r"\b[Aa][a-z]?n[a-z]?t[a-z]?h[a-z]?o[a-z]?n[a-z]?y\b", re.IGNORECASE),  # Anthony
        re.compile(
            r"\b[Mm][a-z]?a[a-z]?d[a-z]?[ae][a-z]?l[a-z]?e[a-z]?[iy][a-z]?n[a-z]?e\b", re.IGNORECASE
        ),  # Madeleine, Madaleine, Madelaine
        re.compile(r"\b[Pp][a-z]?h[a-z]?o[a-z]?n[a-z]?s\b", re.IGNORECASE),  # Phons
        re.compile(r"\b[Aa][a-z]?n[a-z]?j[a-z]?[ae]\b", re.IGNORECASE),  # Anja, Anje
        re.compile(r"\b[Bb][a-z]?o\b", re.IGNORECASE),  # Bo
        re.compile(r"\b[Ll][a-z]?a[a-z]?r[a-z]?s\b", re.IGNORECASE),  # Lars
        re.compile(r"\b[Mm][a-z]?a[a-z]?t[a-z]?s\b", re.IGNORECASE),  # Mats
        re.compile(
            r"\b[Mm][a-z]?a[a-z]?r[a-z]?j[a-z]?o[a-z]?l[a-z]?e[a-z]?i[a-z]?n\b", re.IGNORECASE
        ),  # Marjolein
        re.compile(r"\b[Ee][a-z]?v[a-z]?e[a-z]?l[a-z]?i[a-z]?n[a-z]?e\b", re.IGNORECASE),  # Eveline
    ]

    # Clean messages and update columns
    def clean_message(row):
        message = row["message"]
        changes = []

        # Apply non-media patterns first
        for pattern, change, flags in non_media_patterns:
            if re.search(pattern, message, flags=flags):
                logger.debug(f"Matched non-media pattern '{pattern}' in message: {message}")
                if change not in changes:
                    changes.append(change)
                message = re.sub(pattern, "", message, flags=flags).strip()

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
                    bracket_idx = message.rfind("[", 0, start_idx)
                    if bracket_idx != -1:
                        # Check for space before '['
                        if bracket_idx > 0 and message[bracket_idx - 1] == " ":
                            remove_start = bracket_idx - 1  # Include space
                        else:
                            remove_start = bracket_idx
                        # Remove from bracket_idx (or bracket_idx - 1) to end of match
                        message = message[:remove_start] + message[match.end() :]
                    else:
                        # If no '[', just remove the matched media phrase
                        message = re.sub(pattern, "", message, flags=flags)
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
            message = re.sub(fallback_pattern, "", message, flags=re.IGNORECASE).strip()

        # If message is empty, contains only spaces, or is None, set to "completely removed"
        if message is None or message == "" or message.strip() == "":
            message = "completely removed"

        # Check for digits in cleaned message
        if message != "completely removed" and re.search(r"\d+", message):
            row["has_number"] = True

        # Check for emojis in cleaned message
        if message != "completely removed" and any(char in emoji.EMOJI_DATA for char in message):
            row["has_emoji"] = True

        # Check for names in cleaned message
        if message != "completely removed":
            for pattern in name_patterns:
                if pattern.search(message):
                    row["has_name"] = True
                    break

        # Update changes column
        row["changes"] = ", ".join(changes) if changes else row["changes"]
        row["message_cleaned"] = message
        return row

    df = df.apply(clean_message, axis=1)
    logger.info(
        f"Cleaned messages: {df[['message', 'message_cleaned', 'changes', 'has_number', 'has_emoji', 'has_name']].head(10).to_string()}"
    )

    # Create summary table for all combinations of has_emoji, has_number, and has_name
    total_messages = len(df)
    # Create all possible combinations
    all_combinations = pd.MultiIndex.from_product(
        [[True, False], [True, False], [True, False]], names=["has_emoji", "has_number", "has_name"]
    )
    # Convert to DataFrame
    summary = pd.DataFrame(index=all_combinations).reset_index()
    # Group by the three columns and count
    counts = df.groupby(["has_emoji", "has_number", "has_name"]).size().reset_index(name="count")
    # Merge with all combinations, filling missing counts with 0
    summary = summary.merge(counts, on=["has_emoji", "has_number", "has_name"], how="left")
    summary["count"] = summary["count"].fillna(0).astype(int)
    # Calculate percentage
    summary["percentage"] = (summary["count"] / total_messages * 100).round(2)
    logger.info(f"Summary table with all combinations:\n{summary.to_string()}")

    # Save amended DataFrame to [filename]_filtered.csv
    filename_base = config["current"].rsplit(".", 1)[0]  # Remove extension
    output_csv = processed / f"{filename_base}_filtered.csv"
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved amended DataFrame to {output_csv}")

    # Save summary table to CSV
    summary_csv = processed / f"{filename_base}_summary.csv"
    summary.to_csv(summary_csv, index=False)
    logger.info(f"Saved summary table to {summary_csv}")


if __name__ == "__main__":
    main()
