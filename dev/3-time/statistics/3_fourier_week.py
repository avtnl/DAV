# import requests
import sys
import tomllib
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from numpy.fft import fft, fftfreq  # Added missing import for fft and fftfreq

# Configure logger to write to a file
logger.add(
    "logs/app_{time}.log",
    rotation="1 MB",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)
warnings.simplefilter(action="ignore", category=FutureWarning)


def main() -> None:
    def fourier_model(timeseries, k):
        # Calculate the number of data points in the timeseries
        t = 1.0
        n = len(timeseries)
        # Generate a time vector 'x' from 0 to n*t (excluding the endpoint) evenly spaced
        x = np.linspace(0.0, n * t, n, endpoint=False)
        # Perform the Fourier Transform of the timeseries
        yf = fft(timeseries)
        # Generate the frequency bins for the first half of the Fourier Transform result
        # This represents the positive frequencies up to the Nyquist frequency
        # the nyquist frequency is the highest frequency that can be represented in the fourier transform
        # it is half of the sampling frequency
        xf = fftfreq(n, t)[: n // 2]
        # Identify indices of the 'k' largest frequencies by their magnitude in the first half of the Fourier spectrum
        # the largest frequencies are the most important components of the signal
        indices = np.argsort(np.abs(yf[0 : n // 2]))[-k:]
        # Extract the frequencies corresponding to the 'k' largest magnitudes
        frequencies = xf[indices]
        # Calculate the amplitudes of these frequencies as twice the magnitude divided by n
        # This accounts for the symmetry of the Fourier Transform for real signals
        amplitudes = 2.0 / n * np.abs(yf[indices])
        # Extract the phases of these frequencies and adjust by adding pi/2 to align phases
        phases = np.angle(yf[indices]) + 1 / 2 * np.pi
        # Return a dictionary of the model parameters: 'x', 'frequencies', 'amplitudes', 'phases'
        return {
            "x": x,
            "frequencies": frequencies,
            "amplitudes": amplitudes,
            "phases": phases,
        }

    def model(parameters):
        # Extract the time vector 'x' from the parameters
        x = parameters["x"]
        # Extract the frequencies, amplitudes, and phases from the parameters
        frequencies = parameters["frequencies"]
        amplitudes = parameters["amplitudes"]
        phases = parameters["phases"]

        # Initialize a zero array 'y' of the same shape as 'x' to store the model output
        y = np.zeros_like(x)

        # Add each sine wave component to 'y' based on the extracted frequencies, amplitudes, and phases
        for freq, amp, phase in zip(frequencies, amplitudes, phases, strict=False):
            y += amp * np.sin(2.0 * np.pi * freq * x + phase)

        # Return the composite signal 'y' as the sum of the sine wave components
        return y

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

    # Extract year and week information
    df["date"] = pd.to_datetime(
        df["timestamp"]
    )  # Convert timestamp to datetime for better handling
    df["year"] = df["date"].dt.year
    df["isoweek"] = df["date"].dt.isocalendar().week
    df["year-week"] = df["date"].dt.strftime("%Y-%W")
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

    # Assign the time series data to y
    y = p["count"].values  # Convert to numpy array for Fourier transform
    k = 2  # Number of Fourier components
    parameters = fourier_model(y, k)
    x = parameters["x"]
    y_model = model(parameters)

    def plot_model(x, y, y_model) -> None:
        plt.figure(figsize=(15, 6))
        plt.plot(x, y, label="Original Data")
        plt.plot(x, y_model, label="Modeled with Fourier", linestyle="--")
        plt.xlabel("Time (weeks)")  # Changed to weeks
        plt.ylabel("Count")
        plt.legend()
        plt.title("Fourier Model of Weekly Message Counts")
        plt.grid(True)
        plt.show()

    plot_model(x, y, y_model)

    # Save and show the plot
    output_path = Path("img/fourier-week.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved fourier-week plot: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
