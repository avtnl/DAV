import pandas as pd
import re
from loguru import logger
import emoji
import itertools

class DataPreparation:
    """A class for preparing WhatsApp message data for visualization, including category,
    time-based, distribution, and relationship analyses."""

    def __init__(self, data_editor=None):
        """Initialize DataPreparation with a DataEditor instance and emoji pattern.

        Args:
            data_editor (DataEditor, optional): Instance of DataEditor for emoji handling.

        Attributes:
            data_editor (DataEditor): Stored DataEditor instance for accessing emoji-related methods.
            emoji_pattern (re.Pattern): Regex pattern to match sequences of emojis (one or more).
        """
        self.data_editor = data_editor  # Store DataEditor instance for emoji handling
        self.emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f1e0-\U0001f1ff"  # flags (iOS)
            "\U00002702-\U000027b0"  # Dingbats
            "\U000024c2-\U0001f251"
            "\U0001f900-\U0001f9ff"  # supplemental symbols & pictographs
            "]+",
            flags=re.UNICODE,
        )

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
            df (pandas.DataFrame): Input DataFrame with 'message_cleaned', 'author', and 'has_emoji' columns.

        Returns:
            tuple: (pandas.DataFrame, pandas.DataFrame) -
                   Modified DataFrame with cleaned messages and changes, DataFrame with emoji counts.
        """
        if df is None or df.empty:
            logger.error("No valid DataFrame provided for distribution visualization preparation")
            return None, None

        if self.data_editor is None:
            logger.error("No DataEditor instance provided for emoji handling")
            return None, None

        try:
            # Filter messages with emojis
            emoji_msgs = df[df["has_emoji"] == True]

            # Initialize dictionary for counts
            count_once = {}
            # Process each message
            for message in emoji_msgs["message_cleaned"]:
                # Extract all emojis in the message, excluding ignored ones (using DataEditor logic)
                emojis = [char for char in message if char in emoji.EMOJI_DATA and char not in self.data_editor.ignore_emojis]
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

    def build_visual_relationships(self, df_group, authors):
        """
        Build tables showing relationships between first words and authors in a WhatsApp group.

        Args:
            df_group (pandas.DataFrame): Filtered DataFrame for a specific group with 'message_cleaned' and 'author' columns.
            authors (list): List of unique authors in the group.

        Returns:
            pandas.DataFrame or None: Numerical DataFrame for the second table (highest >= MIN_HIGHEST and total >= MIN_TOTAL, sorted descending), or None if no data.
        """
        MIN_TOTAL = 10
        MIN_HIGHEST = 70  # in percent

        if df_group.empty:
            logger.error("Empty DataFrame provided for building visual relationships.")
            return None

        try:
            # Extract first_word from message_cleaned
            def get_first_word(message):
                if isinstance(message, str) and message.strip():
                    parts = message.split()
                    if parts:
                        return parts[0]
                return None

            df_group['first_word'] = df_group['message_cleaned'].apply(get_first_word)
            df_group = df_group.dropna(subset=['first_word'])

            if df_group.empty:
                logger.info("No valid first words found after extraction.")
                return None

            # Get counts per first_word and author
            counts = df_group.groupby(['first_word', 'author']).size().reset_index(name='count')
            pivot = counts.pivot(index='first_word', columns='author', values='count').fillna(0)

            # Calculate total
            pivot['total'] = pivot.sum(axis=1)

            # Filter out rows where total == 0 (though unlikely)
            pivot = pivot[pivot['total'] > 0]

            # Authors columns
            authors = sorted([a for a in authors if a in pivot.columns])
            if not authors:
                logger.error("No matching authors found in pivot table.")
                return None

            # Calculate percentages (numerical)
            percentages = pivot[authors].div(pivot['total'], axis=0) * 100
            percentages['highest'] = percentages.max(axis=1)

            # Combine into full numerical table: total + percentages (authors + highest)
            full_table_num = pd.concat([pivot[['total']], percentages], axis=1)

            # Create string version for logging
            full_table_str = full_table_num.copy()
            for col in authors + ['highest']:
                full_table_str[col] = full_table_str[col].apply(lambda x: f"{int(x)}%")

            # Table 1: total >= MIN_TOTAL, sorted by total descending
            table1_num = full_table_num[full_table_num['total'] >= MIN_TOTAL].sort_values('total', ascending=False)
            if not table1_num.empty:
                table1_str = full_table_str.loc[table1_num.index]
                logger.info(f"Table 1 (total >= {MIN_TOTAL}, sorted by total desc) for group {df_group['whatsapp_group'].iloc[0]}:\n{table1_str.to_string()}")
            else:
                logger.info(f"No first words with total >= {MIN_TOTAL} for group {df_group['whatsapp_group'].iloc[0]}.")

            # Table 2: highest >= MIN_HIGHEST and total >= MIN_TOTAL, sorted by highest descending
            table2_num = full_table_num[(full_table_num['highest'] >= MIN_HIGHEST) & (full_table_num['total'] >= MIN_TOTAL)].sort_values('highest', ascending=False)
            if not table2_num.empty:
                table2_str = full_table_str.loc[table2_num.index]
                logger.info(f"Table 2 (highest >= {MIN_HIGHEST}% and total >= {MIN_TOTAL}, sorted by highest desc) for group {df_group['whatsapp_group'].iloc[0]}:\n{table2_str.to_string()}")
            else:
                logger.info(f"No first words with highest >= {MIN_HIGHEST}% and total >= {MIN_TOTAL} for group {df_group['whatsapp_group'].iloc[0]}.")
                return None

            return table2_num
        except Exception as e:
            logger.exception(f"Failed to build visual relationships: {e}")
            return None

    def build_visual_relationships_2(self, df_group, authors):
        """
        Build tables showing relationships between emoji sequences and authors in a WhatsApp group.

        Args:
            df_group (pandas.DataFrame): Filtered DataFrame for a specific group with 'message_cleaned' and 'author' columns.
            authors (list): List of unique authors in the group.

        Returns:
            tuple: (pandas.DataFrame or None, pandas.DataFrame or None) - Numerical DataFrames for table1 and table2.
        """
        MIN_TOTAL = 10
        MIN_HIGHEST = 60  # in percent

        if df_group.empty:
            logger.error("Empty DataFrame provided for building visual relationships_2.")
            return None, None

        try:
            # Extract emoji sequences from message_cleaned
            sequences = []
            for _, row in df_group.iterrows():
                message = row['message_cleaned']
                author = row['author']
                if isinstance(message, str):
                    emoji_sequences = self.emoji_pattern.findall(message)
                    for seq in emoji_sequences:
                        sequences.append({'sequence': seq, 'author': author})

            if not sequences:
                logger.info("No emoji sequences found in the group.")
                return None, None

            seq_df = pd.DataFrame(sequences)
            counts = seq_df.groupby(['sequence', 'author']).size().reset_index(name='count')
            pivot = counts.pivot(index='sequence', columns='author', values='count').fillna(0)

            # Calculate total
            pivot['total'] = pivot.sum(axis=1)

            # Filter out rows where total == 0 (though unlikely)
            pivot = pivot[pivot['total'] > 0]

            # Authors columns
            authors = sorted([a for a in authors if a in pivot.columns])
            if not authors:
                logger.error("No matching authors found in pivot table.")
                return None, None

            # Convert counts to int
            pivot[authors] = pivot[authors].astype(int)
            pivot['total'] = pivot['total'].astype(int)

            # Calculate percentages (numerical)
            percentages = pivot[authors].div(pivot['total'], axis=0) * 100
            percentages['highest'] = percentages.max(axis=1)

            # Count number of authors with non-zero percentages
            non_zero_authors = (percentages[authors] > 0).sum(axis=1)

            # Combine into full numerical table: total + percentages (authors + highest)
            full_table_num = pd.concat([pivot[['total']], percentages], axis=1)

            # Create string version for logging
            full_table_str = full_table_num.copy()
            for col in authors + ['highest']:
                full_table_str[col] = full_table_str[col].apply(lambda x: f"{int(x)}%")

            # Table 1: total >= MIN_TOTAL, sorted by total descending
            table1_num = full_table_num[full_table_num['total'] >= MIN_TOTAL].sort_values('total', ascending=False)
            if not table1_num.empty:
                table1_str = full_table_str.loc[table1_num.index]
                logger.info(f"Table 1 (total >= {MIN_TOTAL}, sorted by total desc) for group {df_group['whatsapp_group'].iloc[0]}:\n{table1_str.to_string()}")
            else:
                logger.info(f"No emoji sequences with total >= {MIN_TOTAL} for group {df_group['whatsapp_group'].iloc[0]}.")

            # Table 2: highest >= MIN_HIGHEST, total >= MIN_TOTAL, and max 2 authors with >0%, sorted by highest descending
            table2_num = full_table_num[
                (full_table_num['highest'] >= MIN_HIGHEST) & 
                (full_table_num['total'] >= MIN_TOTAL) & 
                (non_zero_authors <= 2)
            ].sort_values('highest', ascending=False)
            if not table2_num.empty:
                table2_str = full_table_str.loc[table2_num.index]
                logger.info(f"Table 2 (highest >= {MIN_HIGHEST}%, total >= {MIN_TOTAL}, max 2 authors with >0%, sorted by highest desc) for group {df_group['whatsapp_group'].iloc[0]}:\n{table2_str.to_string()}")
            else:
                logger.info(f"No emoji sequences with highest >= {MIN_HIGHEST}%, total >= {MIN_TOTAL}, and max 2 authors with >0% for group {df_group['whatsapp_group'].iloc[0]}.")
                return table1_num, None

            return table1_num, table2_num
        except Exception as e:
            logger.exception(f"Failed to build visual relationships_2: {e}")
            return None, None

    def build_visual_relationships_2(self, df_group, authors):
        """
        Build tables showing relationships between emoji sequences and authors in a WhatsApp group.

        Args:
            df_group (pandas.DataFrame): Filtered DataFrame for a specific group with 'message_cleaned' and 'author' columns.
            authors (list): List of unique authors in the group.

        Returns:
            tuple: (pandas.DataFrame or None, pandas.DataFrame or None) - Numerical DataFrames for table1 and table2.
        """
        MIN_TOTAL = 10
        MIN_HIGHEST = 60  # in percent

        if df_group.empty:
            logger.error("Empty DataFrame provided for building visual relationships_2.")
            return None, None

        try:
            # Extract emoji sequences from message_cleaned
            sequences = []
            for _, row in df_group.iterrows():
                message = row['message_cleaned']
                author = row['author']
                if isinstance(message, str):
                    emoji_sequences = self.emoji_pattern.findall(message)
                    for seq in emoji_sequences:
                        sequences.append({'sequence': seq, 'author': author})

            if not sequences:
                logger.info("No emoji sequences found in the group.")
                return None, None

            seq_df = pd.DataFrame(sequences)
            counts = seq_df.groupby(['sequence', 'author']).size().reset_index(name='count')
            pivot = counts.pivot(index='sequence', columns='author', values='count').fillna(0)

            # Calculate total
            pivot['total'] = pivot.sum(axis=1)

            # Filter out rows where total == 0 (though unlikely)
            pivot = pivot[pivot['total'] > 0]

            # Authors columns
            authors = sorted([a for a in authors if a in pivot.columns])
            if not authors:
                logger.error("No matching authors found in pivot table.")
                return None, None

            # Convert counts to int
            pivot[authors] = pivot[authors].astype(int)
            pivot['total'] = pivot['total'].astype(int)

            # Calculate percentages (numerical)
            percentages = pivot[authors].div(pivot['total'], axis=0) * 100
            percentages['highest'] = percentages.max(axis=1)

            # Count number of authors with non-zero percentages
            non_zero_authors = (percentages[authors] > 0).sum(axis=1)

            # Combine into full numerical table: total + percentages (authors + highest)
            full_table_num = pd.concat([pivot[['total']], percentages], axis=1)

            # Create string version for logging
            full_table_str = full_table_num.copy()
            for col in authors + ['highest']:
                full_table_str[col] = full_table_str[col].apply(lambda x: f"{int(x)}%")

            # Table 1: total >= MIN_TOTAL, sorted by total descending
            table1_num = full_table_num[full_table_num['total'] >= MIN_TOTAL].sort_values('total', ascending=False)
            if not table1_num.empty:
                table1_str = full_table_str.loc[table1_num.index]
                logger.info(f"Table 1 (total >= {MIN_TOTAL}, sorted by total desc) for group {df_group['whatsapp_group'].iloc[0]}:\n{table1_str.to_string()}")
            else:
                logger.info(f"No emoji sequences with total >= {MIN_TOTAL} for group {df_group['whatsapp_group'].iloc[0]}.")

            # Table 2: highest >= MIN_HIGHEST, total >= MIN_TOTAL, and max 2 authors with >0%, sorted by highest descending
            table2_num = full_table_num[
                (full_table_num['highest'] >= MIN_HIGHEST) & 
                (full_table_num['total'] >= MIN_TOTAL) & 
                (non_zero_authors <= 2)
            ].sort_values('highest', ascending=False)
            if not table2_num.empty:
                table2_str = full_table_str.loc[table2_num.index]
                logger.info(f"Table 2 (highest >= {MIN_HIGHEST}%, total >= {MIN_TOTAL}, max 2 authors with >0%, sorted by highest desc) for group {df_group['whatsapp_group'].iloc[0]}:\n{table2_str.to_string()}")
            else:
                logger.info(f"No emoji sequences with highest >= {MIN_HIGHEST}%, total >= {MIN_TOTAL}, and max 2 authors with >0% for group {df_group['whatsapp_group'].iloc[0]}.")
                return table1_num, None

            return table1_num, table2_num
        except Exception as e:
            logger.exception(f"Failed to build visual relationships_2: {e}")
            return None, None

    def build_visual_relationships_3(self, df_group, authors):
        """
        Analyze daily participation in a WhatsApp group and combine results into a single table.

        Args:
            df_group (pandas.DataFrame): Filtered DataFrame for the group with 'timestamp' and 'author' columns.
            authors (list): List of unique authors in the group.

        Returns:
            pandas.DataFrame or None: Combined DataFrame with columns 'type', 'author', 'num_days', 'total_messages', '#participants', and author-specific columns.
        """
        if df_group is None or df_group.empty:
            logger.error("No valid DataFrame provided for relationships_3 preparation")
            return None

        try:
            # Ensure timestamp is datetime and extract date
            df_group["timestamp"] = pd.to_datetime(df_group["timestamp"])
            df_group["date"] = df_group["timestamp"].dt.date

            # Get unique sorted authors
            authors = sorted(authors)

            # Calculate message counts, lengths, and percentages for the entire group
            total_messages = len(df_group)
            message_counts = df_group.groupby('author').size()
            message_percentages = (message_counts / total_messages * 100).round(0).astype(int)
            df_group['message_length'] = df_group['message_cleaned'].apply(lambda x: len(str(x)) if isinstance(x, str) else 0)
            total_length = df_group['message_length'].sum()
            length_counts = df_group.groupby('author')['message_length'].sum()
            length_percentages = (length_counts / total_length * 100).round(0).astype(int)
            # Calculate average message length per author for the entire group
            avg_message_length = (length_counts / message_counts).round(0).astype(int).fillna(0)

            # Daily message counts per author
            daily_counts = df_group.groupby(["date", "author"]).size().unstack(fill_value=0).reindex(columns=authors, fill_value=0)

            # Total messages per day
            daily_total = daily_counts.sum(axis=1)

            # Number of participants per day
            daily_participants = (daily_counts > 0).sum(axis=1)

            # Overall period
            min_date = df_group["date"].min()
            max_date = df_group["date"].max()
            total_days = (max_date - min_date).days + 1
            days_with_messages = len(daily_counts)
            days_no_messages = total_days - days_with_messages

            # Initialize list for combined table
            combined_data = []

            # Messages (%) block
            msg_row = {
                "type": "Messages (%)",
                "author": None,
                "num_days": 0,
                "total_messages": 0,
                "#participants": 0
            }
            for author in authors:
                msg_pct = message_percentages.get(author, 0)
                avg_len = avg_message_length.get(author, 0)
                msg_row[author] = f"{msg_pct}%/{avg_len}"
            combined_data.append(msg_row)

            # Message Length (%) block
            len_row = {
                "type": "Message Length (%)",
                "author": None,
                "num_days": 0,
                "total_messages": 0,
                "#participants": 0
            }
            for author in authors:
                len_pct = length_percentages.get(author, 0)
                avg_len = avg_message_length.get(author, 0)
                len_row[author] = f"{len_pct}%/{avg_len}"
            combined_data.append(len_row)

            # Period (overall)
            combined_data.append({
                "type": "Period",
                "author": None,
                "num_days": total_days,
                "total_messages": 0,
                "#participants": 0,
                **{author: 0 for author in authors}
            })
            combined_data.append({
                "type": "Period",
                "author": "None",
                "num_days": days_no_messages,
                "total_messages": 0,
                "#participants": 0,
                **{author: 0 for author in authors}
            })

            # Days with only 1 participant, details per author
            for author in authors:
                other_authors = [a for a in authors if a != author]
                mask = (daily_counts[author] > 0) & (daily_counts[other_authors] == 0).all(axis=1)
                num_days = mask.sum()
                total_msg = daily_total[mask].sum() if num_days > 0 else 0
                if num_days > 0:
                    # Filter df_group for these days
                    active_dates = daily_counts[mask].index
                    active_df = df_group[df_group['date'].isin(active_dates) & (df_group['author'] == author)]
                    # Calculate average message length for this author
                    active_df['message_length'] = active_df['message_cleaned'].apply(lambda x: len(str(x)) if isinstance(x, str) else 0)
                    active_length = active_df['message_length'].sum()
                    active_message_count = len(active_df)
                    avg_len = int(round(active_length / active_message_count)) if active_message_count > 0 else 0
                    author_values = {a: 0 for a in authors}
                    author_values[author] = f"100%/100%({avg_len})"
                else:
                    author_values = {a: 0 for a in authors}
                combined_data.append({
                    "type": "Single",
                    "author": author,
                    "num_days": num_days,
                    "total_messages": total_msg,
                    "#participants": 1,
                    **author_values
                })

            # Days with only 2 participants, details per combination
            for comb in itertools.combinations(authors, 2):
                pair_str = " & ".join(sorted(comb))
                other_authors = [a for a in authors if a not in comb]
                mask = (daily_counts[list(comb)] > 0).all(axis=1) & (daily_counts[other_authors] == 0).all(axis=1)
                num_days = mask.sum()
                total_msg = daily_total[mask].sum() if num_days > 0 else 0
                if num_days > 0:
                    # Filter df_group for these days
                    active_dates = daily_counts[mask].index
                    active_df = df_group[df_group['date'].isin(active_dates)]
                    # Calculate message percentages for active authors
                    active_message_counts = active_df[active_df['author'].isin(comb)].groupby('author').size()
                    active_total_messages = active_message_counts.sum()
                    message_pct = (active_message_counts / active_total_messages * 100).round(0).astype(int)
                    # Calculate message length percentages for active authors
                    active_df['message_length'] = active_df['message_cleaned'].apply(lambda x: len(str(x)) if isinstance(x, str) else 0)
                    active_length_counts = active_df[active_df['author'].isin(comb)].groupby('author')['message_length'].sum()
                    active_total_length = active_length_counts.sum()
                    length_pct = (active_length_counts / active_total_length * 100).round(0).astype(int)
                    # Calculate average message length for active authors
                    active_avg_length = (active_length_counts / active_message_counts).round(0).astype(int).fillna(0)
                    # Combine percentages into strings
                    author_values = {author: 0 for author in authors}
                    for p in comb:
                        msg_pct = message_pct.get(p, 0)
                        len_pct = length_pct.get(p, 0)
                        avg_len = active_avg_length.get(p, 0)
                        author_values[p] = f"{msg_pct}%/{len_pct}%({avg_len})"
                else:
                    author_values = {author: 0 for author in authors}
                combined_data.append({
                    "type": "Pairs",
                    "author": pair_str,
                    "num_days": num_days,
                    "total_messages": total_msg,
                    "#participants": 2,
                    **author_values
                })

            # Days with only 3 participants, details per non-participant
            for non_part in authors:
                participants = [a for a in authors if a != non_part]
                mask = (daily_counts[participants] > 0).all(axis=1) & (daily_counts[non_part] == 0)
                num_days = mask.sum()
                total_msg = daily_total[mask].sum() if num_days > 0 else 0
                if num_days > 0:
                    # Filter df_group for these days
                    active_dates = daily_counts[mask].index
                    active_df = df_group[df_group['date'].isin(active_dates)]
                    # Calculate message percentages for active authors
                    active_message_counts = active_df[active_df['author'].isin(participants)].groupby('author').size()
                    active_total_messages = active_message_counts.sum()
                    message_pct = (active_message_counts / active_total_messages * 100).round(0).astype(int)
                    # Calculate message length percentages for active authors
                    active_df['message_length'] = active_df['message_cleaned'].apply(lambda x: len(str(x)) if isinstance(x, str) else 0)
                    active_length_counts = active_df[active_df['author'].isin(participants)].groupby('author')['message_length'].sum()
                    active_total_length = active_length_counts.sum()
                    length_pct = (active_length_counts / active_total_length * 100).round(0).astype(int)
                    # Calculate average message length for active authors
                    active_avg_length = (active_length_counts / active_message_counts).round(0).astype(int).fillna(0)
                    # Combine percentages into strings
                    author_values = {author: 0 for author in authors}
                    for p in participants:
                        msg_pct = message_pct.get(p, 0)
                        len_pct = length_pct.get(p, 0)
                        avg_len = active_avg_length.get(p, 0)
                        author_values[p] = f"{msg_pct}%/{len_pct}%({avg_len})"
                else:
                    author_values = {author: 0 for author in authors}
                combined_data.append({
                    "type": "Non-participant",
                    "author": non_part,
                    "num_days": num_days,
                    "total_messages": total_msg,
                    "#participants": 3,
                    **author_values
                })

            # Days with 4 participants
            mask = (daily_counts > 0).all(axis=1)
            num_days = mask.sum()
            total_msg = daily_total[mask].sum() if num_days > 0 else 0
            if num_days > 0:
                # Filter df_group for these days
                active_dates = daily_counts[mask].index
                active_df = df_group[df_group['date'].isin(active_dates)]
                # Calculate message percentages for all authors
                active_message_counts = active_df.groupby('author').size()
                active_total_messages = active_message_counts.sum()
                message_pct = (active_message_counts / active_total_messages * 100).round(0).astype(int)
                # Calculate message length percentages for all authors
                active_df['message_length'] = active_df['message_cleaned'].apply(lambda x: len(str(x)) if isinstance(x, str) else 0)
                active_length_counts = active_df.groupby('author')['message_length'].sum()
                active_total_length = active_length_counts.sum()
                length_pct = (active_length_counts / active_total_length * 100).round(0).astype(int)
                # Calculate average message length for all authors
                active_avg_length = (active_length_counts / active_message_counts).round(0).astype(int).fillna(0)
                # Combine percentages into strings
                author_values = {author: 0 for author in authors}
                for p in authors:
                    msg_pct = message_pct.get(p, 0)
                    len_pct = length_pct.get(p, 0)
                    avg_len = active_avg_length.get(p, 0)
                    author_values[p] = f"{msg_pct}%/{len_pct}%({avg_len})"
            else:
                author_values = {author: 0 for author in authors}
            combined_data.append({
                "type": "All",
                "author": "All",
                "num_days": num_days,
                "total_messages": total_msg,
                "#participants": 4,
                **author_values
            })

            # Create combined DataFrame
            combined_df = pd.DataFrame(combined_data)
            combined_df = combined_df.sort_values(by=["#participants", "num_days"], ascending=[True, False])
            logger.info(f"Combined participation table for group {df_group['whatsapp_group'].iloc[0]}:\n{combined_df.to_string(index=False)}")

            return combined_df
        except Exception as e:
            logger.exception(f"Failed to prepare relationships_3 data: {e}")
            return None