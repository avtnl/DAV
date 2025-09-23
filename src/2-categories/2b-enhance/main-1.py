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

# Configure logger to write to a file
logger.add("logs/app_{time}.log", rotation="1 MB", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

warnings.simplefilter(action="ignore", category=FutureWarning)

def main():
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

    # Dictionary to map datafile keys to WhatsApp group names
    group_mapping = {
        "current_1": "maap",
        "current_2a": "golfmaten",
        "current_2b": "golfmaten",
        "current_3": "dac",
        "current_4": "tillies"
    }

    # Load and validate data files
    dataframes = {}
    for key in group_mapping:
        datafile = processed / config[key]
        logger.debug(f"Attempting to load {datafile}")
        if not datafile.exists():
            logger.warning(f"{datafile} does not exist. Run preprocess.py or check timestamp!")
            continue
        try:
            df = pd.read_parquet(datafile)
            df["whatsapp_group"] = group_mapping[key]
            dataframes[key] = df
            logger.info(f"Loaded {datafile} with {len(df)} rows")
        except Exception as e:
            logger.exception(f"Failed to load {datafile}: {e}")
            continue

    # Check if any DataFrames were loaded
    if not dataframes:
        logger.error("No valid data files were loaded. Exiting.")
        exit(1)

    # Concatenate DataFrames
    logger.debug("Concatenating DataFrames")
    try:
        df = pd.concat(dataframes.values(), ignore_index=True)
        logger.info(f"Concatenated DataFrame with {len(df)} rows and columns: {df.columns.tolist()}")
    except Exception as e:
        logger.exception(f"Failed to concatenate DataFrames: {e}")
        exit(1)

    # Verify the result
    logger.info(f"Unique WhatsApp groups: {df['whatsapp_group'].unique().tolist()}")
    logger.debug(f"DataFrame head:\n{df.head().to_string()}")
    logger.debug(f"DataFrame dtypes:\n{df.dtypes}")

    # Debug: Check author counts per group
    logger.info("Author counts per WhatsApp group:")
    logger.info(df.groupby(["whatsapp_group", "author"]).size().to_string())

    # Filter out group names and reset index
    rows_before = len(df)
    df = df.loc[df["author"] != "MAAP"]
    df = df.loc[df["author"] != "Golfmaten"]
    df = df.loc[df["author"] != "What's up with golf"]
    df = df.loc[df["author"] != "DAC cie"]
    df = df.loc[df["author"] != "Tillies & co"]
    df = df.reset_index(drop=True)
    rows_after = len(df)
    logger.info(f"DataFrame filtered: {rows_before} rows reduced to {rows_after} rows")

    # Save processed DataFrame
    now = datetime.now(tz=pytz.timezone('Europe/Amsterdam')).strftime("%Y%m%d-%H%M%S")
    output = processed / f"whatsapp_all-{now}"
    try:
        df.to_csv(output.with_suffix(".csv"), index=False)
        df.to_parquet(output.with_suffix(".parq"), index=False)
        logger.info(f"DataFrame saved as: {output.with_suffix('.csv')} and {output.with_suffix('.parq')}")
    except Exception as e:
        logger.exception(f"Failed to save DataFrame: {e}")
        exit(1)

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Extract date, year, month, and year-month
    df["date"] = df["timestamp"].dt.date
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month  # Add month column
    df["isoweek"] = df["timestamp"].dt.isocalendar().week
    df["year-week"] = df["timestamp"].dt.strftime("%Y-%W")
    df["year-month"] = df["timestamp"].dt.strftime("%Y-%m")
    logger.debug(f"Added date, year, month, isoweek, year-week, and year-month columns. DataFrame columns: {df.columns.tolist()}")

    # Count messages per day per WhatsApp group per author
    daily_counts = df.groupby(["date", "whatsapp_group", "author"]).size().reset_index(name="message_count")
    logger.info(f"Messages counted per day per WhatsApp group per Author. Resulting shape: {daily_counts.shape}")
    logger.debug(f"Daily counts head:\n{daily_counts.head().to_string()}")

    # Calculate average message count per day per WhatsApp group, excluding "Anthony van Tilburg"
    avg_counts = daily_counts[daily_counts["author"] != "Anthony van Tilburg"].groupby(["date", "whatsapp_group"])["message_count"].mean().reset_index(name="avg_message_count")
    logger.info(f"Average message counts computed. Resulting shape: {avg_counts.shape}")
    logger.debug(f"Average counts head:\n{avg_counts.head().to_string()}")

    # Merge average counts back into daily_counts
    daily_counts = daily_counts.merge(avg_counts, on=["date", "whatsapp_group"], how="left")
    daily_counts["avg_message_count"] = daily_counts["avg_message_count"].fillna(0)
    logger.info(f"Merged average message counts into daily_counts. New columns: {daily_counts.columns.tolist()}")

    # Add year to daily_counts for consistency
    daily_counts = daily_counts.merge(df[["date", "year"]].drop_duplicates(), on="date", how="left")
    logger.debug(f"Added year to daily_counts. New columns: {daily_counts.columns.tolist()}")

    # Print table of messages per year with WhatsApp groups as rows
    group_authors = df.groupby("whatsapp_group")["author"].unique().to_dict()
    logger.info("Authors per WhatsApp group:")
    for group, auths in group_authors.items():
        logger.info(f"{group}: {auths.tolist()}")

    for group in df["whatsapp_group"].unique():
        # Filter data for the group
        group_df = df[df["whatsapp_group"] == group]
        # Get authors for this group, excluding Anthony
        authors = {auth: auth[:2] for auth in group_authors[group] if auth != "Anthony van Tilburg"}
        authors["Anthony van Tilburg"] = "AvT" if "Anthony van Tilburg" in group_authors[group] else None
        ordered_labels = [label for auth, label in authors.items() if auth != "Anthony van Tilburg"] + ["AvT"] if authors.get("Anthony van Tilburg") else []

        # Count messages per year and author
        yearly_counts = group_df.groupby(["whatsapp_group", "year", "author"]).size().reset_index(name="message_count")
        logger.debug(f"Yearly counts for {group}:\n{yearly_counts.head().to_string()}")

        # Pivot to get authors as columns
        yearly_pivot = yearly_counts.pivot_table(
            index="whatsapp_group",
            columns=["year", "author"],
            values="message_count",
            fill_value=0
        ).astype(int)

        # Calculate average for non-Anthony authors
        non_anthony_authors = [auth for auth in group_authors[group] if auth != "Anthony van Tilburg"]
        non_anthony_counts = yearly_counts[yearly_counts["author"].isin(non_anthony_authors)]
        avg_yearly = non_anthony_counts.groupby(["whatsapp_group", "year"])["message_count"].mean().reset_index(name="Avg_Non_Anthony")
        avg_pivot = avg_yearly.pivot_table(
            index="whatsapp_group",
            columns="year",
            values="Avg_Non_Anthony",
            fill_value=0
        )

        # Combine pivots
        yearly_pivot.columns = [f"{year}_{author}" for year, author in yearly_pivot.columns]
        avg_pivot.columns = [f"{year}_Avg_Non_Anthony" for year in avg_pivot.columns]
        combined_pivot = pd.concat([yearly_pivot, avg_pivot], axis=1)

        # Reorder columns: non-Anthony authors, average, Anthony
        ordered_columns = []
        for year in range(2015, 2026):
            for label in ordered_labels[:-1] if "Anthony van Tilburg" in group_authors[group] else ordered_labels:
                if f"{year}_{label}" in combined_pivot.columns:
                    ordered_columns.append(f"{year}_{label}")
            if f"{year}_Avg_Non_Anthony" in combined_pivot.columns:
                ordered_columns.append(f"{year}_Avg_Non_Anthony")
            if "Anthony van Tilburg" in group_authors[group] and f"{year}_AvT" in combined_pivot.columns:
                ordered_columns.append(f"{year}_AvT")
        combined_pivot = combined_pivot[ordered_columns]

        logger.info(f"\nMessages per Year for {group}:")
        logger.info(combined_pivot.to_string())

    # Plot messages per month for each WhatsApp group
    monthly_counts = df.groupby(["whatsapp_group", "year", "month", "author"]).size().reset_index(name="message_count")
    monthly_counts["month"] = monthly_counts["month"].astype(int)
    logger.debug(f"Monthly counts head:\n{monthly_counts.head().to_string()}")

    for group in df["whatsapp_group"].unique():
        group_data = monthly_counts[monthly_counts["whatsapp_group"] == group]
        fig, ax = plt.subplots(figsize=(12, 6))

        # Get authors for this group
        non_anthony_authors = [auth for auth in group_authors[group] if auth != "Anthony van Tilburg"]

        # Plot non-Anthony authors
        non_anthony_data = group_data[group_data["author"].isin(non_anthony_authors)]
        for author in non_anthony_data["author"].unique():
            author_data = non_anthony_data[non_anthony_data["author"] == author]
            ax.plot(author_data["year"] + author_data["month"] / 12, author_data["message_count"], 
                    color="lightgray", alpha=0.5, label="_nolegend_")

        # Plot average of non-Anthony authors
        avg_monthly = non_anthony_data.groupby(["whatsapp_group", "year", "month"])["message_count"].mean().reset_index(name="avg_message_count")
        ax.plot(avg_monthly["year"] + avg_monthly["month"] / 12, avg_monthly["avg_message_count"], 
                color="black", linewidth=2, label="Avg (Non-Anthony)")

        # Plot Anthony van Tilburg
        anthony_data = group_data[group_data["author"] == "Anthony van Tilburg"]
        if not anthony_data.empty:
            ax.plot(anthony_data["year"] + anthony_data["month"] / 12, anthony_data["message_count"], 
                    color="red", linewidth=2, label="Anthony van Tilburg")

        # Set x-axis with year-month labels
        years = range(2015, 2026)
        year_months = [f"{y}-{m:02d}" for y in years for m in range(1, 13)]
        x_ticks = [y + m / 12 for y in years for m in range(1, 13)]
        ax.set_xticks(x_ticks[::6])  # Show every 6th month (e.g., Jan, Jul)
        ax.set_xticklabels(year_months[::6], rotation=45)
        ax.set_xlabel("Year-Month")
        ax.set_ylabel("Messages per Month")
        ax.set_title(f"Monthly Message Counts for {group}")
        ax.legend()
        plt.tight_layout()
        output_path = Path(f"img/monthly_messages_{group}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved monthly plot: {output_path}")
        plt.show()

if __name__ == "__main__":
    main()