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

    # Extract year and month information
    df["date"] = pd.to_datetime(df["timestamp"])  # Convert timestamp to datetime
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["year-month"] = df["date"].dt.strftime("%Y-%m")
    logger.info(df.head())

    # Reindex to fill missing months
    df = df.drop(index=[0])  # Remove the first row if needed

    # Group by year and month for plotting
    p = df.groupby(["year", "month"]).size().reset_index(name="count")
    logger.info(p.head())

    # Ensure all months (1 to 12) are present for each year
    all_months = pd.DataFrame({"month": range(1, 13)})  # Months 1 to 12
    years = p["year"].unique()
    full_data = []
    for year in years:
        month_data = all_months.copy()
        month_data["year"] = year
        month_data = month_data.merge(
            p[p["year"] == year][["month", "count"]], on="month", how="left"
        ).fillna({"count": 0})
        full_data.append(month_data)
    p = pd.concat(full_data, ignore_index=True)
    logger.info(p.head())

    # Create a time series index for decomposition
    p["date"] = pd.to_datetime(p["year"].astype(str) + "-" + p["month"].astype(str) + "-01", format="%Y-%m-%d")
    p = p.sort_values("date").set_index("date")

    # Decompose the time series
    try:
        logger.debug("Decomposing time series")
        if len(p) < 12:  # Ensure enough data for monthly decomposition
            logger.warning("Data length too short for meaningful decomposition with period=12")
            return
        result = seasonal_decompose(p["count"], model="additive", period=12)  # Monthly period
        
        # Create figure with subplots
        plt.figure(figsize=(10, 8))
        result.plot()
        plt.suptitle("Time Series Decomposition of Message Counts (Monthly)")
        plt.tight_layout()

        # Save and show the plot
        output_path = Path("img/decomposition-month.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved decomposition-month plot: {output_path}")
        plt.show()
    except Exception as e:
        logger.exception(f"Failed to decompose or plot time series: {e}")

    # Assuming df_monthly['value'] is your monthly series
    result_monthly = seasonal_decompose(df_monthly["count"], model="additive", period=12)
    seasonal_variance = np.var(result_monthly.seasonal) / np.var(df_monthly["count"])
    logger.info((f"Monthly Seasonal Variance Proportion: {seasonal_variance:.3f}"))

if __name__ == "__main__":
    main()