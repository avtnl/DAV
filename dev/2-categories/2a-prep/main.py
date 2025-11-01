import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger


def main() -> None:
    # Read configuration
    configfile = Path("config.toml").resolve()
    with configfile.open("rb") as f:
        config = tomllib.load(f)

    # Read data
    root = Path(__file__).parent.parent.resolve()  # Points to DAV
    processed = root / Path(config["processed"])
    datafile = processed / config["current"]
    if not datafile.exists():
        logger.warning(
            f"{datafile} does not exist. First run src/preprocess.py, and check the timestamp!"
        )
        return  # Exit if file doesn't exist
    df = pd.read_parquet(datafile)
    print(df.dtypes)

    # Filter out group name and reset index
    df = df[df["author"] != "MAAP"].reset_index(drop=True)

    # Extract date and count messages per day
    df["date"] = df["timestamp"].dt.date
    daily_counts = df.groupby("date").size().reset_index(name="message_count")
    daily_counts["year"] = pd.to_datetime(daily_counts["date"]).dt.year

    # Categorize message volume
    freq_ranges = [0, 5, 10, 20, float("inf")]
    categories = ["some", "more", "intense", "crazy"]
    daily_counts["volume_category"] = pd.cut(
        daily_counts["message_count"], bins=freq_ranges, labels=categories, right=False
    )

    # Summarize days per category per year
    summary = (
        daily_counts.groupby(["year", "volume_category"])
        .size()
        .reset_index(name="day_count")
        .pivot(index="year", columns="volume_category", values="day_count")
        .fillna(0)
        .astype(int)
    )
    all_years = pd.Index(range(2019, 2026), name="year")
    summary = summary.reindex(all_years, fill_value=0)
    print("Summary: Number of Days per Category per Year")
    print(summary)

    # Define authors and their short labels
    authors = {
        "Anthony van Tilburg": "AvT",
        "Anja Berkemeijer": "AB",
        "Phons Berkemeijer": "PB",
        "Madeleine": "M",
    }

    # Create a copy of daily_counts to start
    daily_counts_extended = daily_counts.copy()

    # Initialize dictionary to hold each author's zero-participation summary
    zero_summaries = {}

    for full_name, label in authors.items():
        # Step 1: Filter messages by author
        author_msgs = df[df["author"] == full_name]

        # Step 2: Count messages per day
        author_counts = author_msgs.groupby("date").size().reset_index(name=f"{label}_count")

        # Step 3: Merge with daily_counts
        daily_counts_extended = daily_counts_extended.merge(author_counts, on="date", how="left")
        daily_counts_extended[f"{label}_count"] = (
            daily_counts_extended[f"{label}_count"].fillna(0).astype(int)
        )

        # Step 4: Flag zero participation
        daily_counts_extended[f"{label}_zero"] = daily_counts_extended[f"{label}_count"] == 0

        # Step 5: Count zero-participation days per category per year
        zero_summary = (
            daily_counts_extended[daily_counts_extended[f"{label}_zero"]]
            .groupby(["year", "volume_category"])
            .size()
            .reset_index(name="zero_days")
            .pivot(index="year", columns="volume_category", values="zero_days")
            .fillna(0)
            .astype(int)
        )

        # Step 6: Reindex to include all years
        zero_summary = zero_summary.reindex(all_years, fill_value=0)

        # Step 7: Rename columns
        zero_summary.columns = [f"{col}-{label}" for col in zero_summary.columns]

        # Store for later merge
        zero_summaries[label] = zero_summary

    # Step 8: Combine all summaries into one DataFrame
    pd.concat(zero_summaries.values(), axis=1)

    # Step 9: Display each author's zero-participation summary separately
    for label, summary_df in zero_summaries.items():
        print(f"\n📊 Days with Zero Participation — {label}")
        print(summary_df)

    # Step 10: Save each author's table to a separate CSV file
    for label, summary_df in zero_summaries.items():
        file_path = (
            rf"c:\Users\avtnl\Documents\HU\Data Analytics\My_Project\zero_participation_{label}.csv"
        )
        summary_df.to_csv(file_path, index=True)
        print(f"✅ Saved: {file_path}")

    # Calculate participation metrics
    years_span = 6.125

    total_days = (daily_counts.groupby("volume_category").size().reindex(categories)) / years_span

    ordered_labels = ["M", "AB", "PB", "AvT"]
    author_participation = {}

    for label in ordered_labels:
        zero_days = (
            daily_counts_extended[daily_counts_extended[f"{label}_zero"]]
            .groupby("volume_category")
            .size()
            .reindex(categories)
            .fillna(0)
        ) / years_span
        participation_days = total_days - zero_days
        author_participation[label] = participation_days

    # Build plot DataFrame
    df_plot = pd.DataFrame(author_participation)
    df_plot.insert(0, "Total Days", total_days)
    df_plot = df_plot[["Total Days", "M", "AB", "PB", "AvT"]]
    print("\ndf_plot contents:")
    print(df_plot)
    print("AB values:", df_plot["AB"].tolist())
    print("PB values:", df_plot["PB"].tolist())

    # Plotting
    _fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.15
    x = range(len(categories))

    ax.bar([i - 2 * bar_width for i in x], df_plot["Total Days"], width=bar_width, color="blue")
    ax.bar([i - bar_width for i in x], df_plot["M"], width=bar_width, color="#C0C0C0")
    ax.bar(x, df_plot["AB"], width=bar_width, color="#808080")
    ax.bar([i + bar_width for i in x], df_plot["PB"], width=bar_width, color="#404040")
    ax.bar([i + 2 * bar_width for i in x], df_plot["AvT"], width=bar_width, color="red")

    # Add names in "some" category
    some_index = categories.index("some")
    bar_positions = [
        some_index - bar_width,
        some_index,
        some_index + bar_width,
        some_index + 2 * bar_width,
    ]
    names = ["Madeleine", "Anja", "Phons", "Anthony"]
    for pos, name, height in zip(
        bar_positions, names, [df_plot.loc["some", lbl] for lbl in ordered_labels], strict=False
    ):
        ax.text(
            pos,
            height / 2,
            name,
            ha="center",
            va="center",
            color="black",
            fontsize=9,
            rotation="vertical",
        )

    # Customize plot
    custom_labels = {
        "some": "Some chatting",
        "more": "More chatting",
        "intense": "Intense chatting",
        "crazy": "Insane chatting",
    }
    ax.set_xticks(x)
    ax.set_xticklabels([custom_labels[cat] for cat in categories])
    ax.set_ylabel("Average Number of Days per Year")
    ax.text(
        0.03,
        1.05,
        "Anthony",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=22,
        color="red",
    )
    ax.text(
        0.16,
        1.05,
        "is 'efficient' in chatting, no matter the intensity",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=22,
        color="black",
    )
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend().set_visible(False)

    plt.tight_layout()
    output_path = Path("img/chat_activity_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
