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
        average_data = p.groupby("isoweek")["count"].mean().reset_index(name="avg_count")
        logger.info(average_data.head())

        # Plotting
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = sns.color_palette("husl", len(years))  # Distinct colors for each year

        # Plot each year's data
        for i, year in enumerate(sorted(years)):
            year_data = p[p["year"] == year]
            sns.scatterplot(
                data=year_data,
                x="isoweek",
                y="count",
                ax=ax,
                color=colors[i],
                label=str(year),  # Year in legend
                zorder=2,  # Lower zorder for individual years
            )
            # Compute moving average for the year
            year_data["moving_avg"] = year_data["count"].rolling(window=1, min_periods=1).mean()
            sns.lineplot(
                data=year_data,
                x="isoweek",
                y="moving_avg",
                ax=ax,
                color=colors[i],
                label=None,  # Avoid duplicate legend entries
                zorder=2,  # Lower zorder for individual years
            )

        # Plot average across all years in black
        sns.lineplot(
            data=average_data,
            x="isoweek",
            y="avg_count",
            ax=ax,
            color="black",
            label="Average",
            linewidth=2.5,  # Thicker line for emphasis
            zorder=3,  # Higher zorder to plot on top
        )

        # Set x-axis to show weeks 1 to 52
        ax.set_xticks(range(1, 53, 4))  # Show every 4th week for clarity
        ax.set_xticklabels(range(1, 53, 4), rotation=45, ha="right", fontsize=10)
        ax.set_xlabel("Week of Year")
        ax.set_ylabel("Message Count")
        plt.title("Messages over Time by Year with Average")

        # Ensure legend shows years and average
        ax.legend(title="Year")

        output_path = Path("img/messages_by_week_per_year_with_average.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot: {output_path}")
        plt.show()

if __name__ == "__main__":
    main()