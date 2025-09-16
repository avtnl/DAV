import wa_analyzer.preprocess as preprocessor
import pandas as pd
import tomllib
import re
from pathlib import Path
from loguru import logger
from datetime import datetime

def main():
    # Call preprocessor
    # preprocessor.main(["--device", "ios"])

    # Read in the file
    configfile = Path("config.toml").resolve()
    with configfile.open("rb") as f:
        config = tomllib.load(f)
    processed = Path("data/processed")
    datafile = processed / config["inputpath"]
    if not datafile.exists():
        logger.warning(
            f"{datafile} does not exist. Maybe first run src/preprocess.py, or check the timestamp!"
        )

    # Convert the "timestamp" column from a string to a datetime object.
    df = pd.read_csv(datafile, parse_dates=["timestamp"])
    logger.info(df.head())

    # Sometimes, author names have a tilde in front
    clean_tilde = r"^~\u202f"
    df["author"] = df["author"].apply(lambda x: re.sub(clean_tilde, "", x))

if __name__ == "__main__":
    main()