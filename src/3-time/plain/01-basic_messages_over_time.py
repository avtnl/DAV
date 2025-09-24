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

    # Let's extract some more info from the timestamp:
    df["date"] = df["timestamp"].dt.date
    df["isoweek"] = df["timestamp"].dt.isocalendar().week
    df["year-week"] = df["timestamp"].dt.strftime("%Y-%W")
    logger.info(df.head())

    # Reindex in order to fill the missing weeks.
    df = df.drop(index=[0])

    p = df.groupby("year-week").count()
    logger.info(p.head())

    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()
    new_index = pd.date_range(
        start=min_ts, end=max_ts, freq="W", name="year-week"
    ).strftime("%Y-%W")
    p = p.reindex(new_index, fill_value=0)
    logger.info(p.head())

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 6))  # Increased width to reduce crowding
    sns.scatterplot(data=p, x=p.index, y="timestamp", ax=ax)
    p["moving_avg"] = p["timestamp"].rolling(window=1).mean()
    sns.lineplot(data=p, x=p.index, y="moving_avg", ax=ax)

    # Set x-axis ticks to show one tick per year
    # Get unique years and select the first week of each year (e.g., YYYY-01)
    years = sorted(set(idx.split("-")[0] for idx in p.index))  # Unique years
    year_starts = []
    for year in years:
        # Find the first week of the year (prefer "-01", fall back to first available)
        try:
            first_week = next(idx for idx in p.index if idx.startswith(f"{year}-01"))
        except StopIteration:
            first_week = next(idx for idx in p.index if idx.startswith(f"{year}-"))
        year_starts.append(first_week)
    
    # Log selected ticks for debugging
    logger.info(f"Selected ticks: {year_starts}")
    logger.info(f"Tick labels: {years}")

    # Set ticks and labels
    ax.set_xticks(year_starts)
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=10)  # Use years as labels

    plt.title("Messages over time")
    output_path = Path("img/first_plot_regarding_time_for_dac.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()

