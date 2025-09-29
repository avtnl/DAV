import pandas as pd
from loguru import logger
import emoji

class DataPreparation:
    def build_visual_categories(self, df):
        """
        Prepare DataFrame for visualization by adding year column, computing active years,
        early leavers, and message counts per group, year, and author.

        Args:
            df (pandas.DataFrame): Input DataFrame with 'timestamp', 'author', and 'whatsapp_group' columns.

        Returns:
            tuple: (pandas.DataFrame, dict, pandas.DataFrame, pandas.DataFrame, list) -
                   Modified DataFrame, group authors dict, non-Anthony average DataFrame,
                   Anthony messages DataFrame, sorted groups list.
        """
        if df is None or df.empty:
            logger.error("No valid DataFrame provided for visualization preparation")
            return None, None, None, None, None

        try:
            # Ensure timestamp is datetime and extract year
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["year"] = df["timestamp"].dt.year
            logger.debug(f"Added year column. DataFrame columns: {df.columns.tolist()}")

            # Log active years per author per group
            active_years = df.groupby(['whatsapp_group', 'author'])['year'].agg(['min', 'max']).reset_index()
            logger.info("Active years per author per group:")
            logger.info(active_years.to_string())

            # Flag authors who left early (max year < 2025 in period)
            filter_df = df[(df['timestamp'] >= '2020-07-01') & (df['timestamp'] <= '2025-07-31')]
            active_years_period = filter_df.groupby(['whatsapp_group', 'author'])['year'].agg(['min', 'max']).reset_index()
            early_leavers = active_years_period[(active_years_period['max'] < 2025) & (active_years_period['author'] != "Anthony van Tilburg")]
            logger.info("Authors who left early (max year < 2025 in July 2020 - July 2025):")
            logger.info(early_leavers.to_string() if not early_leavers.empty else "No authors left early.")

            # Get authors per group
            group_authors = df.groupby("whatsapp_group")["author"].unique().to_dict()
            logger.info("Authors per WhatsApp group:")
            for group, auths in group_authors.items():
                logger.info(f"{group}: {auths.tolist()}")

            # Prepare data for visualization
            filter_df = df[(df['timestamp'] >= '2020-07-01') & (df['timestamp'] <= '2025-07-31')]
            logger.info(f"Filtered DataFrame for July 2020 - July 2025: {len(filter_df)} rows")

            # Calculate total messages per group for sorting
            group_total = filter_df.groupby('whatsapp_group').size().reset_index(name='total_messages')
            sorted_groups = group_total.sort_values('total_messages', ascending=False)['whatsapp_group'].tolist()
            logger.info(f"Sorted groups by total messages: {sorted_groups}")

            # Calculate average messages per non-Anthony author per group
            non_anthony = filter_df[filter_df['author'] != "Anthony van Tilburg"]
            non_anthony_counts = non_anthony.groupby(['whatsapp_group', 'author']).size().reset_index(name='messages')
            non_anthony_group = non_anthony_counts.groupby('whatsapp_group')['messages'].mean().reset_index(name='non_anthony_avg')
            non_anthony_authors_count = non_anthony_counts.groupby('whatsapp_group')['author'].nunique().reset_index(name='num_authors')
            non_anthony_group = non_anthony_group.merge(non_anthony_authors_count, on='whatsapp_group', how='left').fillna({'num_authors': 0})
            non_anthony_group = non_anthony_group.set_index('whatsapp_group').reindex(sorted_groups).reset_index().fillna({'non_anthony_avg': 0, 'num_authors': 0})
            logger.info(f"Non-Anthony average messages and author counts per group:\n{non_anthony_group.to_string()}")

            # Anthony messages per group
            anthony = filter_df[filter_df['author'] == "Anthony van Tilburg"]
            anthony_group = anthony.groupby('whatsapp_group').size().reset_index(name='anthony_messages')
            anthony_group = anthony_group.set_index('whatsapp_group').reindex(sorted_groups).reset_index().fillna({'anthony_messages': 0})

            return df, group_authors, non_anthony_group, anthony_group, sorted_groups
        except Exception as e:
            logger.exception(f"Failed to prepare visualization categories: {e}")
            return None, None, None, None, None

    def build_visual_time(self, df):
        """
        Prepare DataFrame for time-based visualization by extracting year and week information
        and calculating message counts per week.

        Args:
            df (pandas.DataFrame): Input DataFrame with 'timestamp' and 'whatsapp_group' columns.

        Returns:
            tuple: (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame) -
                   Modified DataFrame with additional columns, grouped DataFrame by year and week,
                   average message counts per week across all years.
        """
        if df is None or df.empty:
            logger.error("No valid DataFrame provided for time-based visualization preparation")
            return None, None, None

        try:
            # Ensure timestamp is datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Extract year and week information
            df["date"] = df["timestamp"].dt.date
            df["year"] = df["timestamp"].dt.year
            df["isoweek"] = df["timestamp"].dt.isocalendar().week
            df["year-week"] = df["timestamp"].dt.strftime("%Y-%W")
            logger.info(f"DataFrame head after adding time columns:\n{df.head()}")

            # Group by year and week for plotting
            p = df.groupby(["year", "isoweek"]).size().reset_index(name="count")
            logger.info(f"Grouped DataFrame by year and week:\n{p.head()}")

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
            logger.info(f"DataFrame with all weeks filled:\n{p.head()}")

            # Calculate average across all years for each week
            average_all = p.groupby("isoweek")["count"].mean().reset_index(name="avg_count_all")
            logger.info(f"Average message counts per week:\n{average_all.head()}")

            # Calculate average across all years excluding 2020 (for logging only)
            average_no_2020 = p[p["year"] != 2020].groupby("isoweek")["count"].mean().reset_index(name="avg_count_no_2020")
            logger.info(f"Average message counts per week (excluding 2020):\n{average_no_2020.head()}")

            # Calculate and log average message counts (excluding 2020) for specified week ranges
            weeks_1_12_35_53_no_2020 = average_no_2020[
                (average_no_2020["isoweek"].between(1, 12)) | (average_no_2020["isoweek"].between(35, 53))
            ]["avg_count_no_2020"].mean()
            weeks_12_19_no_2020 = average_no_2020[
                average_no_2020["isoweek"].between(12, 19)
            ]["avg_count_no_2020"].mean()
            weeks_19_35_no_2020 = average_no_2020[
                average_no_2020["isoweek"].between(19, 35)
            ]["avg_count_no_2020"].mean()
            logger.info(f"Average message count (excl. 2020) for weeks 1-12 and 35-53: {weeks_1_12_35_53_no_2020:.2f}")
            logger.info(f"Average message count (excl. 2020) for weeks 12-19: {weeks_12_19_no_2020:.2f}")
            logger.info(f"Average message count (excl. 2020) for weeks 19-35: {weeks_19_35_no_2020:.2f}")

            return df, p, average_all
        except Exception as e:
            logger.exception(f"Failed to prepare time-based visualization data: {e}")
            return None, None, None

    def build_visual_distribution(self, df):
        """
        Prepare DataFrame for emoji distribution visualization by counting emojis.

        Args:
            df (pandas.DataFrame): Input DataFrame with 'message', 'author', and 'has_emoji' columns.

        Returns:
            tuple: (pandas.DataFrame, pandas.DataFrame) -
                   Modified DataFrame with cleaned messages and changes, DataFrame with emoji counts.
        """
        if df is None or df.empty:
            logger.error("No valid DataFrame provided for distribution visualization preparation")
            return None, None

        try:
            # Filter messages with emojis
            emoji_msgs = df[df["has_emoji"] == True]

            # Define emojis to ignore (skin tone modifiers)
            ignore_emojis = {chr(int(code, 16)) for code in ['1F3FB', '1F3FC', '1F3FD', '1F3FE', '1F3FF']}

            # Initialize dictionary for counts
            count_once = {}
            # Process each message
            for message in emoji_msgs["message"]:
                # Extract all emojis in the message, excluding ignored ones
                emojis = [char for char in message if char in emoji.EMOJI_DATA and char not in ignore_emojis]
                # Count unique emojis once per message
                unique_emojis = set(emojis)
                for e in unique_emojis:
                    count_once[e] = count_once.get(e, 0) + 1

            # Create DataFrame for emoji counts
            emoji_counts_df = pd.DataFrame({
                "emoji": list(count_once.keys()),
                "count_once": list(count_once.values())
            })

            # Log emoji counts
            logger.info("Emoji usage counts (before sorting):")
            logger.info(emoji_counts_df.to_string())

            # Calculate percentages
            total_once = emoji_counts_df["count_once"].sum()
            emoji_counts_df["percent_once"] = (emoji_counts_df["count_once"] / total_once) * 100

            # Add Unicode code and name
            emoji_counts_df["unicode_code"] = emoji_counts_df["emoji"].apply(lambda x: f"U+{ord(x):04X}")
            emoji_counts_df["unicode_name"] = emoji_counts_df["emoji"].apply(
                lambda x: emoji.demojize(x).strip(":").replace("_", " ").title()
            )

            # Sort by count_once
            emoji_counts_df = emoji_counts_df.sort_values(by="count_once", ascending=False)

            return df, emoji_counts_df
        except Exception as e:
            logger.exception(f"Failed to prepare distribution visualization data: {e}")
            return None, None