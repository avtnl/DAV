import sys
import tomllib
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
    average_no_2020 = (
        p[p["year"] != 2020]
        .groupby("isoweek")["count"]
        .mean()
        .reset_index(name="avg_count_no_2020")
    )
    logger.info(average_no_2020.head())

    # Plotting
    _fig, ax = plt.subplots(figsize=(14, 6))

    # Add vertical lines just before weeks 12, 19, 35, and 38
    vline_weeks = [11.5, 18.5, 34.5]  # Just before weeks 12, 19, 35
    vline_labels = ["Week 12", "Week 19", "Week 35"]
    for week, label in zip(vline_weeks, vline_labels, strict=False):
        ax.axvline(
            x=week,
            color="gray",
            linestyle="--",
            alpha=0.5,
            zorder=1,  # Behind data lines
            label=label
            if week == 11.5
            else None,  # Add label only for first line to avoid legend clutter
        )

    # Plot average across all years in black
    sns.lineplot(
        data=average_all,
        x="isoweek",
        y="avg_count_all",
        ax=ax,
        color="black",
        label="Average (All Years)",
        linewidth=2.5,
        zorder=2,  # Lower zorder
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
        zorder=3,  # Higher zorder to plot on top
    )

    # Set x-axis to show weeks 1 to 52
    ax.set_xticks(range(1, 53, 4))  # Show every 4th week for clarity
    ax.set_xticklabels(range(1, 53, 4), rotation=45, ha="right", fontsize=10)
    ax.set_xlabel("Week of Year")
    ax.set_ylabel("Message Count")
    plt.title("Average Messages per Week (All Years vs. Excluding 2020)")

    # Ensure legend shows both averages
    ax.legend(title="Averages")

    output_path = Path("img/average_messages_by_week_all_vs_no_2020.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
