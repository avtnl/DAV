import sys
import tomllib
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

    # Ensure timestamp is datetime and extract year for tables
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year
    logger.debug(f"Added year column. DataFrame columns: {df.columns.tolist()}")

    # Debug: Log active years per author per group
    active_years = (
        df.groupby(["whatsapp_group", "author"])["year"].agg(["min", "max"]).reset_index()
    )
    logger.info("Active years per author per group:")
    logger.info(active_years.to_string())
    # Flag authors who left early (max year < 2025 in period)
    filter_df = df[(df["timestamp"] >= "2020-07-01") & (df["timestamp"] <= "2025-07-31")]
    active_years_period = (
        filter_df.groupby(["whatsapp_group", "author"])["year"].agg(["min", "max"]).reset_index()
    )
    early_leavers = active_years_period[
        (active_years_period["max"] < 2025)
        & (active_years_period["author"] != "Anthony van Tilburg")
    ]
    logger.info("Authors who left early (max year < 2025 in July 2020 - July 2025):")
    logger.info(early_leavers.to_string() if not early_leavers.empty else "No authors left early.")

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

    # Bar chart for July 2020 - July 2025
    # Filter for period July 2020 - July 2025
    filter_df = df[(df["timestamp"] >= "2020-07-01") & (df["timestamp"] <= "2025-07-31")]
    logger.info(f"Filtered DataFrame for July 2020 - July 2025: {len(filter_df)} rows")

    # Calculate total messages per group for sorting
    group_total = filter_df.groupby("whatsapp_group").size().reset_index(name="total_messages")
    sorted_groups = group_total.sort_values("total_messages", ascending=False)[
        "whatsapp_group"
    ].tolist()
    logger.info(f"Sorted groups by total messages: {sorted_groups}")

    # Calculate average messages per non-Anthony author per group
    non_anthony = filter_df[filter_df["author"] != "Anthony van Tilburg"]
    # Count messages per author per group
    non_anthony_counts = (
        non_anthony.groupby(["whatsapp_group", "author"]).size().reset_index(name="messages")
    )
    # Compute average messages per author per group
    non_anthony_group = (
        non_anthony_counts.groupby("whatsapp_group")["messages"]
        .mean()
        .reset_index(name="non_anthony_avg")
    )
    # Count number of non-Anthony authors per group
    non_anthony_authors_count = (
        non_anthony_counts.groupby("whatsapp_group")["author"]
        .nunique()
        .reset_index(name="num_authors")
    )
    non_anthony_group = non_anthony_group.merge(
        non_anthony_authors_count, on="whatsapp_group", how="left"
    ).fillna({"num_authors": 0})
    non_anthony_group = (
        non_anthony_group.set_index("whatsapp_group")
        .reindex(sorted_groups)
        .reset_index()
        .fillna({"non_anthony_avg": 0, "num_authors": 0})
    )
    logger.info(
        f"Non-Anthony average messages and author counts per group:\n{non_anthony_group.to_string()}"
    )

    # Anthony messages per group (total)
    anthony = filter_df[filter_df["author"] == "Anthony van Tilburg"]
    anthony_group = anthony.groupby("whatsapp_group").size().reset_index(name="anthony_messages")
    anthony_group = (
        anthony_group.set_index("whatsapp_group")
        .reindex(sorted_groups)
        .reset_index()
        .fillna({"anthony_messages": 0})
    )

    # Bar chart
    _fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.4  # Reduced width for overlap**
    positions = np.arange(len(sorted_groups))

    # Light gray bars for non-Anthony average
    ax.bar(
        positions,
        non_anthony_group["non_anthony_avg"],
        width=bar_width,
        color="lightgray",
        label="Average number of messages Non-Anthony",
    )

    # Blue bars overlapping 50% to the right
    ax.bar(
        positions + bar_width / 2,
        anthony_group["anthony_messages"],
        width=bar_width,
        color="blue",
        label="Number of messages Anthony",
    )

    # Add red arrow for maap
    maap_idx = sorted_groups.index("maap") if "maap" in sorted_groups else None
    if maap_idx is not None:
        x_pos = positions[maap_idx] + bar_width  # Upper middle of light gray bar
        y_start = non_anthony_group["non_anthony_avg"].iloc[maap_idx]  # Top of light gray bar
        y_end = anthony_group["anthony_messages"].iloc[maap_idx]  # Top of blue bar
        ax.annotate(
            "",
            xy=(x_pos, y_end),
            xytext=(x_pos, y_start),
            arrowprops={"arrowstyle": "<->", "color": "red", "lw": 2},
        )
        logger.info(
            f"Red arrow for maap: from (x={x_pos:.2f}, y={y_start:.2f}) to (x={x_pos:.2f}, y={y_end:.2f})"
        )

    # Customize x-axis labels to include number of authors
    xtick_labels = [
        f"{group} ({num_authors:.1f})"
        for group, num_authors in zip(sorted_groups, non_anthony_group["num_authors"], strict=False)
    ]
    ax.set_xticks(positions + bar_width / 2)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel("WhatsApp Group (Number of Non-Anthony Authors)")
    ax.set_ylabel("Messages (July 2020 - July 2025)")
    ax.set_title(
        "Anthony's participation ratio in whatsapp_group 'maap' is significant lower than other groups"
    )
    ax.legend()
    plt.tight_layout()
    output_path = Path("img/yearly_bar_chart_combined.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved bar chart: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
