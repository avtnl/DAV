import sys
import tomllib
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytz
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

    # Dictionary to map datafile keys to WhatsApp group names
    group_mapping = {
        "current_1": "maap",
        "current_2a": "golfmaten",
        "current_2b": "golfmaten",
        "current_3": "dac",
        "current_4": "tillies",
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
        sys.exit(1)

    # Concatenate DataFrames
    logger.debug("Concatenating DataFrames")
    try:
        df = pd.concat(dataframes.values(), ignore_index=True)
        logger.info(
            f"Concatenated DataFrame with {len(df)} rows and columns: {df.columns.tolist()}"
        )
    except Exception as e:
        logger.exception(f"Failed to concatenate DataFrames: {e}")
        sys.exit(1)

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
    now = datetime.now(tz=pytz.timezone("Europe/Amsterdam")).strftime("%Y%m%d-%H%M%S")
    output = processed / f"whatsapp_all-{now}"
    try:
        df.to_csv(output.with_suffix(".csv"), index=False)
        df.to_parquet(output.with_suffix(".parq"), index=False)
        logger.info(
            f"DataFrame saved as: {output.with_suffix('.csv')} and {output.with_suffix('.parq')}"
        )
    except Exception as e:
        logger.exception(f"Failed to save DataFrame: {e}")
        sys.exit(1)

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Extract date, year, month, and year-month
    df["date"] = df["timestamp"].dt.date
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["isoweek"] = df["timestamp"].dt.isocalendar().week
    df["year-week"] = df["timestamp"].dt.strftime("%Y-%W")
    df["year-month"] = df["timestamp"].dt.strftime("%Y-%m")
    logger.debug(
        f"Added date, year, month, isoweek, year-week, and year-month columns. DataFrame columns: {df.columns.tolist()}"
    )

    # Count messages per day per WhatsApp group per author
    daily_counts = (
        df.groupby(["date", "whatsapp_group", "author"]).size().reset_index(name="message_count")
    )
    logger.info(
        f"Messages counted per day per WhatsApp group per Author. Resulting shape: {daily_counts.shape}"
    )
    logger.debug(f"Daily counts head:\n{daily_counts.head().to_string()}")

    # Calculate average message count per day per WhatsApp group, excluding "Anthony van Tilburg"
    avg_counts = (
        daily_counts[daily_counts["author"] != "Anthony van Tilburg"]
        .groupby(["date", "whatsapp_group"])["message_count"]
        .mean()
        .reset_index(name="avg_message_count")
    )
    logger.info(f"Average message counts computed. Resulting shape: {avg_counts.shape}")
    logger.debug(f"Average counts head:\n{avg_counts.head().to_string()}")

    # Merge average counts back into daily_counts
    daily_counts = daily_counts.merge(avg_counts, on=["date", "whatsapp_group"], how="left")
    daily_counts["avg_message_count"] = daily_counts["avg_message_count"].fillna(0)
    logger.info(
        f"Merged average message counts into daily_counts. New columns: {daily_counts.columns.tolist()}"
    )

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
        authors["Anthony van Tilburg"] = (
            "AvT" if "Anthony van Tilburg" in group_authors[group] else None
        )
        ordered_labels = (
            [label for auth, label in authors.items() if auth != "Anthony van Tilburg"] + ["AvT"]
            if authors.get("Anthony van Tilburg")
            else []
        )

        # Count messages per year and author
        yearly_counts = (
            group_df.groupby(["whatsapp_group", "year", "author"])
            .size()
            .reset_index(name="message_count")
        )
        logger.debug(f"Yearly counts for {group}:\n{yearly_counts.head().to_string()}")

        # Pivot to get authors as columns
        yearly_pivot = yearly_counts.pivot_table(
            index="whatsapp_group", columns=["year", "author"], values="message_count", fill_value=0
        ).astype(int)

        # Calculate average for non-Anthony authors
        non_anthony_authors = [
            auth for auth in group_authors[group] if auth != "Anthony van Tilburg"
        ]
        non_anthony_counts = yearly_counts[yearly_counts["author"].isin(non_anthony_authors)]
        avg_yearly = (
            non_anthony_counts.groupby(["whatsapp_group", "year"])["message_count"]
            .mean()
            .reset_index(name="Avg_Non_Anthony")
        )
        avg_pivot = avg_yearly.pivot_table(
            index="whatsapp_group", columns="year", values="Avg_Non_Anthony", fill_value=0
        )

        # Combine pivots
        yearly_pivot.columns = [f"{year}_{author}" for year, author in yearly_pivot.columns]
        avg_pivot.columns = [f"{year}_Avg_Non_Anthony" for year in avg_pivot.columns]
        combined_pivot = pd.concat([yearly_pivot, avg_pivot], axis=1)

        # Reorder columns: non-Anthony authors, average, Anthony
        ordered_columns = []
        for year in range(2015, 2026):
            for label in (
                ordered_labels[:-1]
                if "Anthony van Tilburg" in group_authors[group]
                else ordered_labels
            ):
                if f"{year}_{label}" in combined_pivot.columns:
                    ordered_columns.append(f"{year}_{label}")
            if f"{year}_Avg_Non_Anthony" in combined_pivot.columns:
                ordered_columns.append(f"{year}_Avg_Non_Anthony")
            if (
                "Anthony van Tilburg" in group_authors[group]
                and f"{year}_AvT" in combined_pivot.columns
            ):
                ordered_columns.append(f"{year}_AvT")
        combined_pivot = combined_pivot[ordered_columns]

        logger.info(f"\nMessages per Year for {group}:")
        logger.info(combined_pivot.to_string())

    # **Combined plot for all WhatsApp groups**
    # Ensure all years are present
    all_years = pd.DataFrame({"year": range(2015, 2026)})
    yearly_counts = (
        df.groupby(["whatsapp_group", "year", "author"]).size().reset_index(name="message_count")
    )
    yearly_counts = all_years.merge(yearly_counts, on="year", how="left").fillna(
        {
            "message_count": 0,
            "whatsapp_group": yearly_counts["whatsapp_group"].mode().iloc[0]
            if not yearly_counts.empty
            else "Unknown",
            "author": "Unknown",
        }
    )
    yearly_counts["message_count"] = yearly_counts["message_count"].astype(int)
    logger.debug(f"Yearly counts head (with all years):\n{yearly_counts.head().to_string()}")

    _fig, ax = plt.subplots(figsize=(12, 6))

    # Get all authors
    non_anthony_authors = [
        auth for auth in df["author"].unique() if auth not in {"Anthony van Tilburg", "Unknown"}
    ]
    logger.debug(f"Non-Anthony authors: {non_anthony_authors}")

    # Plot all non-Anthony authors (light gray)
    non_anthony_data = yearly_counts[yearly_counts["author"].isin(non_anthony_authors)]
    for author in non_anthony_data["author"].unique():
        author_data = non_anthony_data[non_anthony_data["author"] == author]
        author_data = all_years.merge(author_data, on="year", how="left").fillna(
            {"message_count": 0}
        )
        ax.plot(
            author_data["year"],
            author_data["message_count"],
            color="lightgray",
            alpha=0.5,
            label="_nolegend_",
        )

    # Plot group-specific averages (gray)
    for group in df["whatsapp_group"].unique():
        group_data = yearly_counts[yearly_counts["whatsapp_group"] == group]
        group_non_anthony = group_data[group_data["author"].isin(non_anthony_authors)]
        group_avg = (
            group_non_anthony.groupby(["whatsapp_group", "year"])["message_count"]
            .mean()
            .reset_index(name="group_avg")
        )
        group_avg = all_years.merge(group_avg, on="year", how="left").fillna(
            {"group_avg": 0, "whatsapp_group": group}
        )
        ax.plot(
            group_avg["year"],
            group_avg["group_avg"],
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label=f"Avg {group} (Non-Anthony)",
        )

    # Plot Anthony per group (dark gray)
    anthony_data = yearly_counts[yearly_counts["author"] == "Anthony van Tilburg"]
    for group in df["whatsapp_group"].unique():
        group_anthony = anthony_data[anthony_data["whatsapp_group"] == group]
        group_anthony = all_years.merge(group_anthony, on="year", how="left").fillna(
            {"message_count": 0, "whatsapp_group": group, "author": "Anthony van Tilburg"}
        )
        if group_anthony["message_count"].sum() > 0:
            ax.plot(
                group_anthony["year"],
                group_anthony["message_count"],
                color="darkgray",
                linewidth=1.5,
                label=f"Anthony ({group})",
            )

    # Plot overall average (non-Anthony, black)
    overall_avg = (
        non_anthony_data.groupby("year")["message_count"].mean().reset_index(name="overall_avg")
    )
    overall_avg = all_years.merge(overall_avg, on="year", how="left").fillna({"overall_avg": 0})
    ax.plot(
        overall_avg["year"],
        overall_avg["overall_avg"],
        color="black",
        linewidth=2,
        label="Overall Avg (Non-Anthony)",
    )

    # Plot overall Anthony average (blue)
    overall_anthony = (
        anthony_data.groupby("year")["message_count"].mean().reset_index(name="anthony_avg")
    )
    overall_anthony = all_years.merge(overall_anthony, on="year", how="left").fillna(
        {"anthony_avg": 0}
    )
    if overall_anthony["anthony_avg"].sum() > 0:
        ax.plot(
            overall_anthony["year"],
            overall_anthony["anthony_avg"],
            color="blue",
            linewidth=2,
            label="Anthony Avg (All Groups)",
        )

    # Fill area between overall average and Anthony's average
    x = overall_avg["year"]
    y1 = overall_avg["overall_avg"]
    y2 = overall_anthony["anthony_avg"]
    condition = y2 > y1
    logger.debug(f"Fill condition for overall average: {condition.tolist()}")
    ax.fill_between(x, y1, y2, where=condition, color="green", alpha=0.3, interpolate=True)
    ax.fill_between(x, y1, y2, where=~condition, color="red", alpha=0.3, interpolate=True)

    # Set x-axis with year labels
    years = range(2015, 2026)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45)
    ax.set_xlabel("Year")
    ax.set_ylabel("Messages per Year")
    ax.set_title("Yearly Message Counts Across All WhatsApp Groups")
    ax.legend()
    plt.tight_layout()
    output_path = Path("img/yearly_messages_combined.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved combined plot: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
