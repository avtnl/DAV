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

    # Load and validate data files
    datafile = processed / config["current"]
    if not datafile.exists():
        logger.warning(
            "Datafile does not exist. First run src/preprocess.py, and check the timestamp!"
        )
    df = pd.read_parquet(datafile)
    logger.info(df)

    # Extract year and week information
    df["date"] = df["timestamp"].dt.date
    df["year"] = df["timestamp"].dt.year
    df["isoweek"] = df["timestamp"].dt.isocalendar().week
    df["year-week"] = df["timestamp"].dt.strftime("%Y-%W")
    logger.info(df.head())

    # Reindex in order to fill the missing weeks
    df = df.drop(index=[0])

    # Group by year and week for plotting
    p = df.groupby(["year", "isoweek"]).size().reset_index(name="count")
    logger.info(p.head())

    # Ensure all weeks (1 to 52) are present for each year
    all_weeks = pd.DataFrame({"isoweek": range(1, 53)})  # Weeks 1 to 52
    years = p["year"].unique()
    full_data = []
    for year in years:
        year_data = all_weeks.copy()
        year_data["year"] = year
        year_data = year_data.merge(
            p[p["year"] == year][["isoweek", "count"]], on="isoweek", how="left"
        ).fillna({"count": 0})
        full_data.append(year_data)
    p = pd.concat(full_data, ignore_index=True)
    logger.info(p.head())

    # Calculate average across all years for each week
    average_all = p.groupby("isoweek")["count"].mean().reset_index(name="avg_count_all")
    logger.info(average_all.head())

    # Calculate average across all years excluding 2020
    average_no_2020 = p[p["year"] != 2020].groupby("isoweek")["count"].mean().reset_index(name="avg_count_no_2020")
    logger.info(average_no_2020.head())

    # Calculate and log average message counts (excluding 2020) for specified week ranges
    weeks_1_12_35_53_no_2020 = average_no_2020[
        (average_no_2020["isoweek"].between(1, 12)) | (average_no_2020["isoweek"].between(35, 53))
    ]["avg_count_no_2020"].mean()
    weeks_12_19_no_2020 = average_no_2020[
        average_no_2020["isoweek"].between(12, 19)
    ]["avg_count_no_2020"].mean()
    weeks_19_35_no_2020 = average_no_2020[
        average_no_2020["isoweek"].between(19, 35)
    ]["avg_count_no_2020"].mean()

    logger.info(f"Average message count (excl. 2020) for weeks 1-12 and 35-53: {weeks_1_12_35_53_no_2020:.2f}")
    logger.info(f"Average message count (excl. 2020) for weeks 12-19: {weeks_12_19_no_2020:.2f}")
    logger.info(f"Average message count (excl. 2020) for weeks 19-35: {weeks_19_35_no_2020:.2f}")

    # Calculate average message counts (including 2020) for specified week ranges
    weeks_1_12_35_53_all = average_all[
        (average_all["isoweek"].between(1, 12)) | (average_all["isoweek"].between(35, 53))
    ]["avg_count_all"].mean()
    weeks_12_19_all = average_all[
        average_all["isoweek"].between(12, 19)
    ]["avg_count_all"].mean()
    weeks_19_35_all = average_all[
        average_all["isoweek"].between(19, 35)
    ]["avg_count_all"].mean()

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 6))

    # Add vertical lines just before weeks 12, 19, and 35
    vline_weeks = [11.5, 18.5, 34.5]  # Just before weeks 12, 19, 35
    for week in vline_weeks:
        ax.axvline(
            x=week,
            color="gray",
            linestyle="--",
            alpha=0.5,
            zorder=1,  # Behind data lines
            # No label to exclude from legend
        )

    # Add horizontal lines for average message counts (excluding 2020) in blue
    # Weeks 1–12 and 35–53 (two segments)
    ax.hlines(
        y=weeks_1_12_35_53_no_2020,
        xmin=1,
        xmax=11.5,
        colors="blue",
        linestyles="--",
        alpha=0.7,
        zorder=4,  # Above data lines
        # No label to exclude from legend
    )
    ax.hlines(
        y=weeks_1_12_35_53_no_2020,
        xmin=34.5,
        xmax=52,
        colors="blue",
        linestyles="--",
        alpha=0.7,
        zorder=4,
    )
    # Weeks 12–19
    ax.hlines(
        y=weeks_12_19_no_2020,
        xmin=11.5,
        xmax=18.5,
        colors="blue",
        linestyles="--",
        alpha=0.7,
        zorder=4,
        # No label to exclude from legend
    )
    # Weeks 19–35
    ax.hlines(
        y=weeks_19_35_no_2020,
        xmin=18.5,
        xmax=34.5,
        colors="blue",
        linestyles="--",
        alpha=0.7,
        zorder=4,
        # No label to exclude from legend
    )

    # Add horizontal lines for average message counts (including 2020) in lightgray
    # Weeks 1–12 and 35–53 (two segments)
    ax.hlines(
        y=weeks_1_12_35_53_all,
        xmin=1,
        xmax=11.5,
        colors="lightgray",
        linestyles="--",
        alpha=0.7,
        zorder=5,  # Above blue horizontal lines
        # No label to exclude from legend
    )
    ax.hlines(
        y=weeks_1_12_35_53_all,
        xmin=34.5,
        xmax=52,
        colors="lightgray",
        linestyles="--",
        alpha=0.7,
        zorder=5,
    )
    # Weeks 12–19
    ax.hlines(
        y=weeks_12_19_all,
        xmin=11.5,
        xmax=18.5,
        colors="lightgray",
        linestyles="--",
        alpha=0.7,
        zorder=5,
        # No label to exclude from legend
    )
    # Weeks 19–35
    ax.hlines(
        y=weeks_19_35_all,
        xmin=18.5,
        xmax=34.5,
        colors="lightgray",
        linestyles="--",
        alpha=0.7,
        zorder=5,
        # No label to exclude from legend
    )

    # Plot average across all years in lightgray
    sns.lineplot(
        data=average_all,
        x="isoweek",
        y="avg_count_all",
        ax=ax,
        color="lightgray",
        label="Average (All Years)",
        linewidth=2.5,
        zorder=2,  # Above vertical lines
    )

    # Plot average excluding 2020 in blue
    sns.lineplot(
        data=average_no_2020,
        x="isoweek",
        y="avg_count_no_2020",
        ax=ax,
        color="blue",
        label="Average (Excl. 2020)",
        linewidth=2.5,
        zorder=3,  # Above lightgray line
    )

    # Set x-axis to show weeks 1 to 52
    ax.set_xticks(range(1, 53, 4))  # Show every 4th week for clarity
    ax.set_xticklabels(range(1, 53, 4), ha="right", fontsize=10)
    ax.set_xlabel("Week of Year")
    ax.set_ylabel("Message Count")
    plt.title("Our weekly golf tournament follows the sun")

    # Ensure legend includes only averages and vertical lines
    ax.legend(title="Averages")

    output_path = Path("img/average_messages_by_week_with_vlines_hlines_all_years.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()    