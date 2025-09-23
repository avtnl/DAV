# import requests
from io import StringIO
from pathlib import Path
from loguru import logger
import warnings
import tomllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose


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
        exit(1)

    df = pd.read_parquet(datafile)
    logger.info(df)

    # Extract year and week information
    df["date"] = df["timestamp"].dt.date
    df["year"] = df["timestamp"].dt.year
    df["isoweek"] = df["timestamp"].dt.isocalendar().week
    df["year-week"] = df["timestamp"].dt.strftime("%Y-%W")
    logger.info(df.head())

    # Reindex to fill missing weeks
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

    # Create a time series index for decomposition
    p["date"] = pd.to_datetime(p["year"].astype(str) + "-" + p["isoweek"].astype(str) + "-1", format="%Y-%W-%w")
    p = p.sort_values("date").set_index("date")

    # Decompose the time series
    try:
        logger.debug("Decomposing time series")
        if len(p) < 52:  # Ensure enough data for weekly decomposition
            logger.warning("Data length too short for meaningful decomposition with period=52")
            return
        result = seasonal_decompose(p["count"], model="additive", period=52)  # Weekly period
        
        # Create figure with subplots
        plt.figure(figsize=(10, 8))
        result.plot()
        plt.suptitle("Time Series Decomposition of Message Counts")
        plt.tight_layout()

        # Save and show the plot
        output_path = Path("img/decomposition-week.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved decomposition-week plot: {output_path}")
        plt.show()
    except Exception as e:
        logger.exception(f"Failed to decompose or plot time series: {e}")

if __name__ == "__main__":
    main()