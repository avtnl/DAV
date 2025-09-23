# import requests
from io import StringIO
from pathlib import Path
from loguru import logger
import warnings
import tomllib
import pandas as pd
import numpy as np
from numpy.fft import fft, fftfreq  # Added missing import for fft and fftfreq
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import savgol_filter, butter, filtfilt

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

   # Prepare time series data
    y = p["count"].values  # Monthly counts
    n = len(y)
    time = np.arange(n)  # Time vector in months (0 to n-1)

    # Apply Savitzky-Golay filter
    window_size = 7  # Window size (odd number, e.g., 9 months)
    poly_order = 3  # Polynomial order
    if window_size > n or window_size % 2 == 0:
        logger.warning(f"Window size {window_size} is invalid for {n} data points. Using 7 instead.")
        window_size = 7
    savitzky_golay_filtered = savgol_filter(y, window_size, poly_order)

    # Apply Low-pass Butterworth filter
    cutoff_frequency = 1.0 / 6  # Cutoff at 1 cycle per year (1/6 of sampling rate)
    filter_order = 3  # Filter order
    nyquist = 0.5  # Nyquist frequency (half the sampling rate, 1 sample per month)
    normalized_cutoff = cutoff_frequency / nyquist  # Normalize cutoff frequency
    b, a = butter(filter_order, normalized_cutoff, btype="low", analog=False)
    butterworth_filtered = filtfilt(b, a, y)

    # Plot the original and filtered signals
    try:
        plt.figure(figsize=(14, 8))
        plt.plot(time, y, label="Original Monthly Counts", color="lightgray")
        plt.plot(time, savitzky_golay_filtered, label="Savitzky-Golay Filtered", color="red")
        plt.plot(time, butterworth_filtered, label="Low-pass Butterworth Filtered", color="blue")
        plt.legend()
        plt.title("Monthly Message Counts Smoothing with Filters")
        plt.xlabel("Time (months)")
        plt.ylabel("Count")
        plt.grid(True)
        plt.show()

        # Save and show the plot
        output_path = Path("img/scipy-month.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved scipy-month plot: {output_path}")
    except Exception as e:
        logger.exception(f"Failed to plot filtered signals: {e}")

if __name__ == "__main__":
    main()