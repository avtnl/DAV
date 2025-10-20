import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
from loguru import logger
import warnings
import matplotlib.font_manager as fm
import networkx as nx
import itertools
import emoji
import re
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class DataPreparation:
    """A class for preparing WhatsApp message data for visualization, including category,
    time-based, distribution, and relationship analyses."""

    class SequenceHandler:
        """
        Subclass for handling sequence-specific analysis in chat data.
        Generates daily sequence DataFrames, computes scores (alternation, rhythm, not-married),
        and detects married couples based on interaction patterns.
        """
        
        def __init__(self, gender_map, married_couples=None):
            """
            Initialize SequenceHandler with required mappings.
            
            Args:
                gender_map (dict): Mapping of authors to genders, e.g., {'Anthony van Tilburg': 'M', ...}.
                married_couples (list of tuples, optional): Known married pairs, e.g., [('M1', 'F1')].
            """
            self.gender_map = gender_map
            self.married_couples = married_couples or []

        def build_sequence_scores(self, df_group, authors, include_married_alternation=False):
            """
            Generate a DataFrame with daily sequence scores for a group.
            
            Args:
                df_group (pandas.DataFrame): Filtered DataFrame for a single group.
                authors (list): List of unique authors in the group.
                include_married_alternation (bool): Whether to compute score_married_alternation.
            
            Returns:
                pandas.DataFrame: Columns: date, total_messages, total_participants, sequence,
                                  score_alternating_MF, score_rhythm, score_not_married,
                                  [score_married_alternation].
            """
            if df_group.empty:
                logger.error("Empty DataFrame provided for sequence scoring.")
                return pd.DataFrame()
            
            try:
                df_group['timestamp'] = pd.to_datetime(df_group['timestamp'])
                df_group['date'] = df_group['timestamp'].dt.date
                
                # Log author activity
                author_counts = df_group.groupby('author').size()
                logger.debug(f"Author message counts in group {df_group['whatsapp_group'].iloc[0]}:\n{author_counts.to_string()}")
                
                # Filter out days with insufficient messages
                daily_counts = df_group.groupby('date').size()
                valid_dates = daily_counts[daily_counts >= 3].index  # At least 3 messages
                df_group = df_group[df_group['date'].isin(valid_dates)]
                if df_group.empty:
                    logger.warning(f"No days with >=3 messages for group {df_group['whatsapp_group'].iloc[0]}.")
                    return pd.DataFrame()
                
                daily_data = []
                for date, daily_df in df_group.groupby('date'):
                    sequence = daily_df['author'].tolist()
                    total_messages = len(sequence)
                    total_participants = len(set(sequence))
                    
                    # Log sequences with specific authors for debugging
                    if any(a in sequence for a in ['Phons Berkemeijer', 'Anja Berkemeijer']):
                        logger.debug(f"Date {date}: Sequence = {sequence}")
                    
                    score_alternating_MF = self._compute_alternating_MF(sequence)
                    score_rhythm = self._compute_rhythm_autocorrelation(sequence)
                    score_not_married = self._compute_not_married(sequence, authors)
                    score_married_alternation = self._compute_alternating_married(sequence) if include_married_alternation and self.married_couples else 0.0
                    
                    row = {
                        'date': date,
                        'total_messages': total_messages,
                        'total_participants': total_participants,
                        'sequence': sequence,
                        'score_alternating_MF': score_alternating_MF,
                        'score_rhythm': score_rhythm,
                        'score_not_married': score_not_married
                    }
                    if include_married_alternation:
                        row['score_married_alternation'] = score_married_alternation
                    daily_data.append(row)
                
                scores_df = pd.DataFrame(daily_data)
                logger.info(f"Generated sequence scores DataFrame for group {df_group['whatsapp_group'].iloc[0]} with {len(scores_df)} days.")
                return scores_df
            except Exception as e:
                logger.exception(f"Failed to build sequence scores: {e}")
                return pd.DataFrame()

        def detect_married_couples(self, df_scores):
            """
            Aggregate scores to detect likely married couples based on low interaction.
            Prioritizes the pair with the lowest interaction score, then assigns the remaining pair.
            
            Args:
                df_scores (pandas.DataFrame): Output from build_sequence_scores.
            
            Returns:
                dict: Detected couples, e.g., {'M1': 'F1', 'M2': 'F2'}.
            """
            if df_scores.empty:
                logger.error("Empty scores DataFrame provided for couple detection.")
                return {}
            
            try:
                # Aggregate not-married scores
                aggregate_interactions = defaultdict(float)
                for _, row in df_scores.iterrows():
                    for pair, score in row['score_not_married'].items():
                        aggregate_interactions[pair] += score
                num_days = len(df_scores)
                for pair in aggregate_interactions:
                    aggregate_interactions[pair] /= num_days if num_days > 0 else 1
                
                logger.debug(f"Aggregated interaction scores: {dict(aggregate_interactions)}")
                
                # Filter M-F pairs
                mf_pairs = {pair: score for pair, score in aggregate_interactions.items() if self._is_mf_pair(pair)}
                if not mf_pairs:
                    logger.error("No M-F pairs found for couple detection.")
                    return {}
                
                # Select the pair with the lowest interaction score (likely married)
                sorted_pairs = sorted(mf_pairs.items(), key=lambda x: x[1])  # Lowest score first
                top_pair = sorted_pairs[0][0]  # e.g., 'Anthony van Tilburg-Madeleine'
                top_m, top_f = top_pair.split('-')
                
                # Find remaining male and female
                males = [a for a in self.gender_map if self.gender_map[a] == 'M']
                females = [a for a in self.gender_map if self.gender_map[a] == 'F']
                remaining_males = [m for m in males if m != top_m]
                remaining_females = [f for f in females if f != top_f]
                
                if len(remaining_males) != 1 or len(remaining_females) != 1:
                    logger.warning(f"Unexpected number of remaining males ({len(remaining_males)}) or females ({len(remaining_females)}).")
                    return {top_m: top_f}
                
                # Form the second couple
                detected = {top_m: top_f, remaining_males[0]: remaining_females[0]}
                
                # Validate with alternation score
                self.married_couples = [(m, f) for m, f in detected.items()]
                alt_scores = df_scores['sequence'].apply(self._compute_alternating_married)
                avg_alt_score = alt_scores.mean() if not alt_scores.empty else 0
                logger.info(f"Detected couples: {detected} with avg alternation score {avg_alt_score:.3f}")
                return detected
            except Exception as e:
                logger.exception(f"Failed to detect married couples: {e}")
                return {}

        def _compute_alternating_MF(self, sequence):
            """Compute gender alternation score (0-1): Fraction of transitions where gender changes."""
            if len(sequence) < 2:
                return 0.0
            transitions = list(zip(sequence[:-1], sequence[1:]))
            alternations = sum(1 for prev, next in transitions if self.gender_map.get(prev, '') != self.gender_map.get(next, ''))
            return alternations / len(transitions)

        def _compute_rhythm_autocorrelation(self, sequence, max_lag=5):
            """Compute rhythm score using autocorrelation: Average similarity over lags."""
            if len(sequence) < 2:
                return 0.0
            author_codes = {author: idx for idx, author in enumerate(set(sequence))}
            codes = [author_codes[author] for author in sequence]
            codes = np.array(codes)
            autocorrs = []
            n = len(codes)
            
            # Suppress NumPy warnings for this block
            with np.errstate(all='warn'):
                for lag in range(1, min(max_lag + 1, n)):
                    if n - lag >= 2:  # Ensure enough data for correlation
                        # Check for non-zero variance
                        if np.std(codes[:-lag]) > 0 and np.std(codes[lag:]) > 0:
                            corr = np.corrcoef(codes[:-lag], codes[lag:])[0, 1]
                            autocorrs.append(corr if not np.isnan(corr) else 0)
                        else:
                            autocorrs.append(0)  # Zero correlation for constant sequences
                    else:
                        autocorrs.append(0)  # Zero for short sequences
            
            return np.mean(autocorrs) if autocorrs else 0.0

        def _compute_not_married(self, sequence, authors):
            """Compute pair-wise interaction scores (higher = more likely not married)."""
            if len(sequence) < 2:
                return {f"{m}-{f}": 0.0 for m in authors for f in authors if self._is_mf_pair(f"{m}-{f}")}
            
            transitions = Counter(zip(sequence[:-1], sequence[1:]))
            mf_interactions = defaultdict(float)
            males = [a for a in authors if self.gender_map.get(a, '') == 'M']
            females = [a for a in authors if self.gender_map.get(a, '') == 'F']
            
            for m in males:
                for f in females:
                    pair_key = f"{m}-{f}"
                    mf_interactions[pair_key] = transitions.get((m, f), 0) + transitions.get((f, m), 0)
            
            # Normalize by max interaction to avoid suppressing low-activity pairs
            total_mf = max(mf_interactions.values(), default=1)
            if total_mf > 0:
                for pair in mf_interactions:
                    mf_interactions[pair] = 1 - (mf_interactions[pair] / total_mf)  # Invert: low interaction = high score
            
            logger.debug(f"Not-married scores for sequence {sequence[:10]}...: {dict(mf_interactions)}")
            return {pair: score for pair, score in mf_interactions.items()}

        def _is_mf_pair(self, pair):
            """Check if a pair string (e.g., 'M1-F1') is M-F."""
            if '-' in pair:
                m, f = pair.split('-')
                return self.gender_map.get(m, '') == 'M' and self.gender_map.get(f, '') == 'F'
            return False

        def _compute_alternating_married(self, sequence):
            """Alternation score factoring in marriage: Lower if transitions are within married pairs."""
            if len(sequence) < 2 or not self.married_couples:
                return 0.0
            transitions = list(zip(sequence[:-1], sequence[1:]))
            married_trans = sum(1 for prev, next in transitions if (prev, next) in self.married_couples or (next, prev) in self.married_couples)
            return 1 - (married_trans / len(transitions)) if len(transitions) > 0 else 0.0

        def _compute_rhythm_ngram(self, sequence, n=2):
            """Alternative rhythm score: 1 - (unique n-grams / total n-grams)."""
            if len(sequence) <= n:
                return 0.0
            ngrams = list(zip(*[sequence[i:] for i in range(n)]))
            unique = len(set(ngrams))
            total = len(ngrams)
            return 1 - (unique / total) if total > 0 else 0.0

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
            # Ensure timestamp is datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Use DataEditor methods to add year, active_years, and early_leaver columns
            df["year"] = self.data_editor.get_year(df)
            df["active_years"] = self.data_editor.active_years(df)
            df["early_leaver"] = self.data_editor.early_leaver(df)
            logger.debug(f"Added year, active_years, and early_leaver columns. DataFrame columns: {df.columns.tolist()}")

            # Log active years per author per group
            active_years = df.groupby(['whatsapp_group', 'author'])['year'].agg(['min', 'max']).reset_index()
            active_years['active_years'] = active_years.apply(lambda x: f"{x['min']}-{x['max']}", axis=1)
            logger.info("Active years per author per group:")
            logger.info(active_years[['whatsapp_group', 'author', 'active_years']].to_string())

            # Log early leavers
            early_leavers = df[df['early_leaver'] == True][['whatsapp_group', 'author']].drop_duplicates()
            logger.info("Authors who left early (max year < 2025 in July 2015 - July 2025):")
            logger.info(early_leavers.to_string() if not early_leavers.empty else "No authors left early.")

            # Get authors per group
            group_authors = df.groupby("whatsapp_group")["author"].unique().to_dict()
            logger.info("Authors per WhatsApp group:")
            for group, auths in group_authors.items():
                logger.info(f"{group}: {auths.tolist()}")

            # Prepare data for visualization
            filter_df = df[(df['timestamp'] >= '2015-07-01') & (df['timestamp'] <= '2025-07-31')]
            logger.info(f"Filtered DataFrame for July 2015 - July 2025: {len(filter_df)} rows")

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

    def build_visual_relationships_arc(self, df_group, authors):
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
        
    def build_visual_relationships_bubble(self, df, groups=None):
        """
        Prepare data for a bubble plot: average words vs average punctuations per message,
        with bubble size as number of messages, split by group and has_emoji.
        
        Args:
            df (pandas.DataFrame): The full DataFrame with WhatsApp data.
            groups (list, optional): List of 2 group names to include. Defaults to first 2 unique groups.
        
        Returns:
            pandas.DataFrame: Aggregated data with columns: whatsapp_group, author, has_emoji,
                            message_count, avg_words, avg_punct.
        """
        try:
            if groups is None:
                groups = df['whatsapp_group'].unique()[:2]
            df_filtered = df[df['whatsapp_group'].isin(groups)].copy()
            
            def count_words(message):
                if not isinstance(message, str):
                    return 0
                # Add space before emoji sequences if not preceded by space
                message = re.sub(r'([^\s])(' + self.emoji_pattern.pattern + r')', r'\1 \2', message)
                # Replace sequences of emojis with a single 'EMOJI'
                message = self.emoji_pattern.sub('EMOJI', message)
                # Handle currency with decimals as one word (e.g., €5.50 or $5.50)
                message = re.sub(r'([€$]\s*\d+[.,]\d+)', lambda m: m.group(0).replace(' ', ''), message)
                # Split on spaces
                words = re.split(r'\s+', message.strip())
                return len([w for w in words if w])
            
            def count_punctuations(message):
                if not isinstance(message, str):
                    return 0
                # Replace repeated punctuation with a single instance
                message = re.sub(r'([!?.,;:])\1+', r'\1', message)
                # Count ! ? . , ; : but exclude decimal points in numbers
                punctuations = re.findall(r'(?<![\d])[!?.,;:](?![\d])', message)
                return len(punctuations)
            
            df_filtered['word_count'] = df_filtered['message_cleaned'].apply(count_words)
            df_filtered['punct_count'] = df_filtered['message_cleaned'].apply(count_punctuations)
            
            agg_df = df_filtered.groupby(['whatsapp_group', 'author', 'has_emoji']).agg(
                message_count=('message_cleaned', 'count'),
                avg_words=('word_count', 'mean'),
                avg_punct=('punct_count', 'mean')
            ).reset_index()
            
            logger.info(f"Prepared bubble plot data for groups {groups}:\n{agg_df.to_string(index=False)}")
            return agg_df
        except Exception as e:
            logger.exception(f"Failed to prepare bubble plot data: {e}")
            return None
        
    def build_interaction_features(self, df, group_authors):
        """
        Build a feature matrix for interaction and network dynamics analysis.
        Features include normalized reply frequencies, mention frequencies, centrality measures,
        and cross-group participation. Computed per author-year, with separate rows for
        'Anthony van Tilburg' for each group per year and an overall row across all groups.
        
        Args:
            df (pandas.DataFrame): Full cleaned DataFrame with 'whatsapp_group' column.
            group_authors (dict): Dictionary of group names to lists of authors.
        
        Returns:
            pandas.DataFrame: Feature matrix with 'author_year' or 'author_year_group' index and 'whatsapp_group' column.
        """
        try:
            all_authors = sorted(set(a for authors in group_authors.values() for a in authors))
            first_names = {a: a.split()[0] for a in all_authors}
            first_name_to_author = {v: k for k, v in first_names.items()}
            
            df = df.copy()
            df['year'] = df['timestamp'].dt.year
            df = df.sort_values(['whatsapp_group', 'timestamp'])
            df['prev_author'] = df.groupby('whatsapp_group')['author'].shift(1)
            df['mentions'] = df['message'].apply(lambda x: self.data_editor.extract_mentions(x, first_name_to_author))
            feature_list = []
            
            # Group by author and year
            for (author, year), author_year_df in df.groupby(['author', 'year']):
                if author == 'Anthony van Tilburg':
                    # Create separate rows for each group and an overall row
                    for group in author_year_df['whatsapp_group'].unique():
                        sub_df = author_year_df[author_year_df['whatsapp_group'] == group]
                        if sub_df.empty:
                            continue
                        features = {
                            'author_year': f"{author}_{year}_{group}",
                            'whatsapp_group': group
                        }
                        reply_counts = sub_df[sub_df['prev_author'].notna()]['prev_author'].value_counts(normalize=True)
                        for tgt in all_authors:
                            features[f'reply_to_{tgt.replace(" ", "_")}'] = reply_counts.get(tgt, 0.0)
                        mention_flat = [m for mentions in sub_df['mentions'] for m in mentions]
                        mention_counts = Counter(mention_flat)
                        total_mentions = sum(mention_counts.values())
                        for tgt in all_authors:
                            features[f'mention_{tgt.replace(" ", "_")}'] = mention_counts[tgt] / total_mentions if total_mentions > 0 else 0.0
                        year_df = df[df['year'] == year].copy()
                        G = nx.Graph()
                        for _, row in year_df[year_df['prev_author'].notna()].iterrows():
                            G.add_edge(row['author'], row['prev_author'])
                        if len(G) > 0 and author in G:
                            features['degree_centrality'] = nx.degree_centrality(G)[author]
                            features['betweenness_centrality'] = nx.betweenness_centrality(G)[author]
                            features['closeness_centrality'] = nx.closeness_centrality(G)[author]
                        else:
                            features['degree_centrality'] = 0.0
                            features['betweenness_centrality'] = 0.0
                            features['closeness_centrality'] = 0.0
                        features['num_groups_participated'] = author_year_df['whatsapp_group'].nunique()
                        total_msgs = len(sub_df)
                        features['avg_msgs_per_group'] = total_msgs / features['num_groups_participated'] if features['num_groups_participated'] > 0 else 0.0
                        threading_features = self._compute_threading_features(sub_df, author)
                        features.update(threading_features)
                        feature_list.append(features)
                    
                    # Add overall row for Anthony
                    sub_df = author_year_df  # Use all data for this author-year
                    features = {
                        'author_year': f"{author}_{year}",
                        'whatsapp_group': 'overall'
                    }
                    reply_counts = sub_df[sub_df['prev_author'].notna()]['prev_author'].value_counts(normalize=True)
                    for tgt in all_authors:
                        features[f'reply_to_{tgt.replace(" ", "_")}'] = reply_counts.get(tgt, 0.0)
                    mention_flat = [m for mentions in sub_df['mentions'] for m in mentions]
                    mention_counts = Counter(mention_flat)
                    total_mentions = sum(mention_counts.values())
                    for tgt in all_authors:
                        features[f'mention_{tgt.replace(" ", "_")}'] = mention_counts[tgt] / total_mentions if total_mentions > 0 else 0.0
                    year_df = df[df['year'] == year].copy()
                    G = nx.Graph()
                    for _, row in year_df[year_df['prev_author'].notna()].iterrows():
                        G.add_edge(row['author'], row['prev_author'])
                    if len(G) > 0 and author in G:
                        features['degree_centrality'] = nx.degree_centrality(G)[author]
                        features['betweenness_centrality'] = nx.betweenness_centrality(G)[author]
                        features['closeness_centrality'] = nx.closeness_centrality(G)[author]
                    else:
                        features['degree_centrality'] = 0.0
                        features['betweenness_centrality'] = 0.0
                        features['closeness_centrality'] = 0.0
                    features['num_groups_participated'] = author_year_df['whatsapp_group'].nunique()
                    total_msgs = len(sub_df)
                    features['avg_msgs_per_group'] = total_msgs / features['num_groups_participated'] if features['num_groups_participated'] > 0 else 0.0
                    threading_features = self._compute_threading_features(sub_df, author)
                    features.update(threading_features)
                    feature_list.append(features)
                else:
                    # For other authors, use primary group (most messages)
                    group_counts = author_year_df['whatsapp_group'].value_counts()
                    primary_group = group_counts.index[0] if not group_counts.empty else 'unknown'
                    sub_df = author_year_df[author_year_df['whatsapp_group'] == primary_group]
                    features = {
                        'author_year': f"{author}_{year}",
                        'whatsapp_group': primary_group
                    }
                    reply_counts = sub_df[sub_df['prev_author'].notna()]['prev_author'].value_counts(normalize=True)
                    for tgt in all_authors:
                        features[f'reply_to_{tgt.replace(" ", "_")}'] = reply_counts.get(tgt, 0.0)
                    mention_flat = [m for mentions in sub_df['mentions'] for m in mentions]
                    mention_counts = Counter(mention_flat)
                    total_mentions = sum(mention_counts.values())
                    for tgt in all_authors:
                        features[f'mention_{tgt.replace(" ", "_")}'] = mention_counts[tgt] / total_mentions if total_mentions > 0 else 0.0
                    year_df = df[df['year'] == year].copy()
                    G = nx.Graph()
                    for _, row in year_df[year_df['prev_author'].notna()].iterrows():
                        G.add_edge(row['author'], row['prev_author'])
                    if len(G) > 0 and author in G:
                        features['degree_centrality'] = nx.degree_centrality(G)[author]
                        features['betweenness_centrality'] = nx.betweenness_centrality(G)[author]
                        features['closeness_centrality'] = nx.closeness_centrality(G)[author]
                    else:
                        features['degree_centrality'] = 0.0
                        features['betweenness_centrality'] = 0.0
                        features['closeness_centrality'] = 0.0
                    features['num_groups_participated'] = author_year_df['whatsapp_group'].nunique()
                    total_msgs = len(sub_df)
                    features['avg_msgs_per_group'] = total_msgs / features['num_groups_participated'] if features['num_groups_participated'] > 0 else 0.0
                    threading_features = self._compute_threading_features(sub_df, author)
                    features.update(threading_features)
                    feature_list.append(features)
            
            feature_df = pd.DataFrame(feature_list).set_index('author_year')
            logger.info(f"Built interaction feature matrix with shape {feature_df.shape}")
            logger.debug(f"Feature matrix columns: {feature_df.columns.tolist()}")
            logger.debug(f"Feature matrix preview:\n{feature_df.head().to_string()}")
            return feature_df
        except Exception as e:
            logger.exception(f"Failed to build interaction features: {e}")
            return None

    def _compute_threading_features(self, sub_df, author):
        """
        Compute threading-related features for an author in their sub-DataFrame.
        
        Args:
            sub_df (pandas.DataFrame): Sub-DataFrame for author-year.
            author (str): Author name.
        
        Returns:
            dict: Threading features (num_initiated, avg_depth_initiated, num_joined, avg_position).
        """
        features = {
            'num_initiated': 0,
            'avg_depth_initiated': 0.0,
            'num_joined': 0,
            'avg_position': 0.0
        }
        try:
            if sub_df.empty or len(sub_df) < 2:
                logger.debug(f"Sub-DataFrame for {author} is empty or too small for threading analysis.")
                return features
            
            # Ensure timestamp is datetime
            sub_df = sub_df.copy()
            sub_df['timestamp'] = pd.to_datetime(sub_df['timestamp'])
            
            # Reset index to avoid alignment issues
            sub_df = sub_df.reset_index(drop=True)
            
            # Compute time differences within groups
            group_dfs = []
            for group, group_df in sub_df.groupby('whatsapp_group'):
                if len(group_df) < 2:
                    logger.debug(f"Group {group} for {author} has fewer than 2 messages, skipping.")
                    continue
                group_df = group_df.sort_values('timestamp')
                group_df['time_diff'] = group_df['timestamp'].diff().dt.total_seconds() / 3600.0
                group_df['thread_id'] = (group_df['time_diff'] > 1).cumsum()
                group_dfs.append(group_df)
            
            if not group_dfs:
                logger.debug(f"No valid groups with sufficient messages for {author}.")
                return features
            
            # Concatenate group DataFrames
            sub_df = pd.concat(group_dfs, ignore_index=True)
            
            # Count threads initiated by this author (first message in a thread)
            thread_starts = sub_df[sub_df['thread_id'].diff() != 0]  # First message of each thread
            thread_starts = thread_starts[thread_starts['author'] == author]
            features['num_initiated'] = len(thread_starts)
            
            # Placeholder for other features due to complexity of thread depth/position
            # For simplicity, assume avg_depth_initiated=1.0 and avg_position=0.5
            features['avg_depth_initiated'] = 1.0 if features['num_initiated'] > 0 else 0.0
            features['num_joined'] = len(sub_df) - features['num_initiated']
            features['avg_position'] = 0.5 if len(sub_df) > 0 else 0.0
            
            logger.debug(f"Threading features for {author}: {features}")
            return features
        except Exception as e:
            logger.warning(f"Failed to compute threading features for {author}: {e}")
            return features

    def build_visual_not_message_content(self, df):
        """
        Prepare feature DataFrame for non-message content visualization by aggregating features per author-month-year-group.

        Args:
            df (pandas.DataFrame): Organized DataFrame with relevant columns including 'whatsapp_group', 'month', and 'year'.

        Returns:
            pandas.DataFrame or None: Feature matrix with 'author_month_year_group' index, 'author', 'month', 'year', 'whatsapp_group' columns, and aggregated features, or None if preparation fails.
        """
        if df is None or df.empty:
            logger.error("No valid DataFrame provided for non-message content preparation")
            return None

        try:
            feature_list = []
            #for (author, month, year, group), sub_group in df.groupby(['author', 'month', 'year', 'whatsapp_group']):
            for (author, week, year, group), sub_group in df.groupby(['author', 'week', 'year', 'whatsapp_group']):
                total_messages = len(sub_group)
                if total_messages == 0:
                    continue
                features = {
                    #'author_month_year_group': f"{author}_{month:02d}_{year}_{group}",
                    'author_week_year_group': f"{author}_{week:02d}_{year}_{group}",
                    'author': author,
                    'week': week,
                    #'month': month,
                    'year': year,
                    'whatsapp_group': group,
                    'total_messages': total_messages,
                    'mean_number_of_chats_that_day': sub_group['number_of_chats_that_day'].mean(),
                    'mean_length_chat': sub_group['length_chat'].mean(),
                    'mean_response_time': sub_group['response_time'].mean(),
                    'proportion_link': (sub_group['has_link'] == 'link').sum() / total_messages,
                    'proportion_deleted': (sub_group['was_deleted'] == 'deleted').sum() / total_messages,
                    'sum_pictures_deleted': sub_group['pictures_deleted'].sum(),
                    'sum_videos_deleted': sub_group['videos_deleted'].sum(),
                    'sum_audios_deleted': sub_group['audios_deleted'].sum(),
                    'sum_gifs_deleted': sub_group['gifs_deleted'].sum(),
                    'sum_stickers_deleted': sub_group['stickers_deleted'].sum(),
                    'sum_documents_deleted': sub_group['documents_deleted'].sum(),
                    'sum_videonotes_deleted': sub_group['videonotes_deleted'].sum(),
                    'mean_number_of_emojis': sub_group['number_of_emojis'].mean(),
                    'proportion_has_emoji': (sub_group['has_emoji'] == 'emoji(s)').sum() / total_messages,
                    'proportion_ends_with_emoji': (sub_group['ends_with_emoji'] == 'ends_with_emoji').sum() / total_messages,
                    'mean_number_of_punctuations': sub_group['number_of_punctuations'].mean(),
                    'proportion_has_punctuation': (sub_group['has_punctuation'] == 'punctuation(s)').sum() / total_messages,
                    'proportion_ends_with_punctuation': (sub_group['ends_with_punctuation'] == 'ends_with_punctuation').sum() / total_messages
                }
                feature_list.append(features)
            #feature_df = pd.DataFrame(feature_list).set_index('author_month_year_group')
            feature_df = pd.DataFrame(feature_list).set_index('author_week_year_group')
            logger.info(f"Built non-message content feature matrix with shape {feature_df.shape}")
            logger.debug(f"Feature matrix columns: {feature_df.columns.tolist()}")
            logger.debug(f"Feature matrix preview:\n{feature_df.head().to_string()}")
            return feature_df
        except Exception as e:
            logger.exception(f"Failed to build non-message content features: {e}")
            return None

    def compute_month_correlations(self, feature_df):
        """
        Compute Pearson correlations between 'month' and numerical features in the feature DataFrame.

        Args:
            feature_df (pandas.DataFrame): Feature matrix with 'author_month_year_group' index, 'month', and numerical features.

        Returns:
            pandas.Series or None: Series of correlation coefficients for numerical features against 'month', or None if computation fails.
        """
        if feature_df is None or feature_df.empty:
            logger.error("No valid feature DataFrame provided for month correlations")
            return None

        try:
            if 'month' not in feature_df.columns:
                logger.error("Column 'month' not found in feature DataFrame")
                return None
            # Select numerical columns (excluding 'author', 'month', 'year', 'whatsapp_group')
            numerical_cols = [col for col in feature_df.columns 
                            if col not in ['author', 'month', 'year', 'whatsapp_group'] 
                            and feature_df[col].dtype in ['int64', 'float64']]
            if not numerical_cols:
                logger.warning("No numerical columns found for correlation analysis")
                return None
            # Compute correlations
            correlations = feature_df[numerical_cols + ['month']].corr()['month'].drop('month')
            logger.info(f"Computed correlations with 'month':\n{correlations.to_string()}")
            return correlations
        except Exception as e:
            logger.exception(f"Failed to compute month correlations: {e}")
            return None
      