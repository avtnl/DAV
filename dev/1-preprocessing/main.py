import re
import tomllib
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
import wa_analyzer.preprocess as preprocessor
from loguru import logger


def main() -> None:
    def find_name_csv(timestamp) -> None:
        pass

    # Read in the file
    configfile = Path("config.toml").resolve()
    with configfile.open("rb") as f:
        config = tomllib.load(f)
    processed = Path(config["processed"])
    # preprocess = config.get("preprocess", False)
    preprocess = config["preprocess"]
    if preprocess:
        now = datetime.now(tz=pytz.timezone("Europe/Amsterdam")).strftime("%Y%m%d-%H%M%S")
        logger.info(now)
        preprocessor.main(["--device", "ios"])
        find_name_csv(now)
    else:
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

    # Let's find emojis in the text and add that as a feature.
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U00002702-\U000027b0"  # Dingbats
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )

    def has_emoji(text):
        return bool(emoji_pattern.search(text))

    df["has_emoji"] = df["message"].apply(has_emoji)

    # Create a timestamp for a new, unique, filename
    now = datetime.now(tz=pytz.timezone("Europe/Amsterdam")).strftime("%Y%m%d-%H%M%S")
    logger.info(now)

    # Create unique filename
    output = processed / f"whatsapp-{now}.csv"
    logger.info(output)

    # Save to CSV and Parquet
    df.to_csv(output, index=False)
    df.to_parquet(output.with_suffix(".parq"), index=False)


if __name__ == "__main__":
    main()
