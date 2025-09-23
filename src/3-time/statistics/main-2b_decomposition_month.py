from pathlib import Path
from loguru import logger
import warnings
import tomllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Configure logger
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

    # Load data
    processed = Path("data/processed")
    datafile = processed / config["current"]
    if not datafile.exists():
        logger.warning("Datafile does not exist. Run src/preprocess.py first!")
        exit(1)
    
    df = pd.read_parquet(datafile)
    logger.info(f"Loaded data:\n{df.head()}")

    # Aggregate to monthly data
    df["date"] = pd.to_datetime(df["timestamp"])
    df["year-month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("year-month").size().reset_index(name="count")
    monthly["date"] = pd.to_datetime(monthly["year-month"].astype(str) + "-01")
    monthly = monthly.set_index("date")["count"]

    # Ensure complete monthly index (fill missing months with 0)
    full_index = pd.date_range(start=monthly.index.min(), end=monthly.index.max(), freq="MS")
    monthly = monthly.reindex(full_index, fill_value=0)
    logger.info(f"Monthly data prepared:\n{monthly.head()}")

    # Decompose time series
    try:
        logger.debug("Decomposing time series (monthly, period=12)")
        if len(monthly) < 24:  # Require at least 2 years for stable decomposition
            logger.warning("Data length too short for reliable decomposition (need >= 24 months)")
            return
        result = seasonal_decompose(monthly, model="additive", period=12)

        # Quantify seasonality strength
        seasonal_variance = np.var(result.seasonal) / np.var(monthly)
        logger.info(f"Monthly Seasonal Variance Proportion: {seasonal_variance:.3f}")

        # Plot decomposition
        plt.figure(figsize=(10, 8))
        result.plot()
        plt.suptitle("Time Series Decomposition of Monthly Counts (Period=12)")
        plt.tight_layout()
        output_path = Path("img/decomposition-month.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved decomposition plot: {output_path}")
        plt.close()

        # Validate decomposition: Residual ACF and Ljung-Box test
        logger.debug("Validating decomposition with residual ACF and Ljung-Box test")
        plt.figure(figsize=(10, 4))
        plot_acf(result.resid.dropna(), lags=24, title="Monthly Residual ACF")
        output_path_acf = Path("img/residual_acf.png")
        plt.savefig(output_path_acf, dpi=300, bbox_inches="tight")
        logger.info(f"Saved residual ACF plot: {output_path_acf}")
        plt.close()

        lb_test = acorr_ljungbox(result.resid.dropna(), lags=[12], return_df=True)
        logger.info(f"Ljung-Box p-value at lag 12: {lb_test['lb_pvalue'].iloc[0]:.3f}")

    except Exception as e:
        logger.exception(f"Failed to decompose or validate time series: {e}")

if __name__ == "__main__":
    main()