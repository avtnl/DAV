import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
from loguru import logger
import warnings
import matplotlib.font_manager as fm
import networkx as nx
import itertools
import emoji

# Suppress FutureWarning from seaborn/pandas
warnings.simplefilter(action="ignore", category=FutureWarning)

class PlotManager:
    def __init__(self):
        # Set font to Segoe UI Emoji for emoji support
        try:
            plt.rcParams['font.family'] = 'Segoe UI Emoji'
        except:
            logger.warning("Segoe UI Emoji font not found. Falling back to default font. Some emojis may not render correctly.")
            plt.rcParams['font.family'] = 'DejaVu Sans'

    def build_visual_categories(self, group_authors, non_anthony_group, anthony_group, sorted_groups):
        """
        Create a bar chart comparing non-Anthony average messages and Anthony's messages
        per WhatsApp group for July 2020 - July 2025, with a horizontal line for overall average messages.

        Args:
            group_authors (dict): Dictionary of group names to lists of authors.
            non_anthony_group (pandas.DataFrame): DataFrame with non-Anthony average messages and author counts.
            anthony_group (pandas.DataFrame): DataFrame with Anthony's message counts.
            sorted_groups (list): List of group names sorted by total messages.

        Returns:
            matplotlib.figure.Figure or None: Figure object for the bar chart, or None if creation fails.
        """
        try:
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bar_width = 0.4
            positions = np.arange(len(sorted_groups))

            # Plot non-Anthony average messages
            ax.bar(positions, non_anthony_group['non_anthony_avg'], width=bar_width, color='lightgray', label='Average number of messages Non-Anthony')

            # Plot Anthony's messages
            ax.bar(positions + bar_width / 2, anthony_group['anthony_messages'], width=bar_width, color='blue', label='Number of messages Anthony')

            # Calculate overall average messages across all groups
            overall_avg = (non_anthony_group['non_anthony_avg'] * non_anthony_group['num_authors'] + anthony_group['anthony_messages']).sum() / (non_anthony_group['num_authors'].sum() + len(sorted_groups))
            ax.axhline(y=overall_avg, color='black', linestyle='--', linewidth=1.5, label='Overall average messages per Author')
            logger.info(f"Overall average messages across all groups: {overall_avg:.2f}")

            # Add block arrows for maap
            maap_idx = sorted_groups.index('maap') if 'maap' in sorted_groups else None
            if maap_idx is not None:
                x_pos = positions[maap_idx] + 0.75 * bar_width
                y_start = non_anthony_group['non_anthony_avg'].iloc[maap_idx]
                y_end = anthony_group['anthony_messages'].iloc[maap_idx]
                # Draw block arrows (using two arrows for thicker effect)
                ax.annotate('',
                            xy=(x_pos, y_end), xytext=(x_pos, y_start),
                            arrowprops=dict(arrowstyle='-|>', color='red', lw=5, mutation_scale=20))
                ax.annotate('',
                            xy=(x_pos, y_start), xytext=(x_pos, y_end),
                            arrowprops=dict(arrowstyle='-|>', color='red', lw=5, mutation_scale=20))
                logger.info(f"Block arrows for maap: from (x={x_pos:.2f}, y={y_start:.2f}) to (x={x_pos:.2f}, y={y_end:.2f})")

            # Customize x-axis labels
            xtick_labels = [f"{group} ({num_authors:.1f})" for group, num_authors in zip(sorted_groups, non_anthony_group['num_authors'])]
            ax.set_xticks(positions + bar_width / 2)
            ax.set_xticklabels(xtick_labels)
            ax.set_xlabel("WhatsApp Group (Number of Non-Anthony Authors)")
            ax.set_ylabel("Messages (July 2020 - July 2025)")

            # Add two-part title
            ax.text(0.5, 1.08, "Too much to handle or too much crap?",
                    fontsize=16, ha='center', va='bottom', transform=ax.transAxes)
            ax.text(0.5, 1.02, "Anthony's participation is significant lower for 1st whatsapp group",
                    fontsize=12, ha='center', va='bottom', transform=ax.transAxes)            

            ax.legend()
            plt.tight_layout()
            plt.show()

            return fig
        except Exception as e:
            logger.exception(f"Failed to build bar chart: {e}")
            return None

    def build_visual_categories_2(self, df):
        """
        Create a bar chart showing the number of messages by attachment type, including a 'No attachments' category.

        Args:
            df (pandas.DataFrame): The DataFrame containing message data with relevant columns.

        Returns:
            matplotlib.figure.Figure or None: Figure object for the bar chart, or None if creation fails.
        """
        try:
            # Compute counts for each attachment type
            counts = {}
            counts['Links'] = df['has_link'].sum()
            counts['Pictures'] = (df['pictures_deleted'] > 0).sum()
            counts['Videos'] = (df['videos_deleted'] > 0).sum()
            counts['Audios'] = (df['audios_deleted'] > 0).sum()
            counts['Gifs'] = (df['gifs_deleted'] > 0).sum()
            counts['Stickers'] = (df['stickers_deleted'] > 0).sum()
            counts['Documents'] = (df['documents_deleted'] > 0).sum()
            counts['Videonotes'] = (df['videonotes_deleted'] > 0).sum()

            # Compute 'No attachments': messages where none of the above conditions are true
            no_attach_mask = ~df['has_link'] & \
                            (df['pictures_deleted'] == 0) & \
                            (df['videos_deleted'] == 0) & \
                            (df['audios_deleted'] == 0) & \
                            (df['gifs_deleted'] == 0) & \
                            (df['stickers_deleted'] == 0) & \
                            (df['documents_deleted'] == 0) & \
                            (df['videonotes_deleted'] == 0)
            counts['No attachments'] = no_attach_mask.sum()

            # Create bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            categories = list(counts.keys())
            values = list(counts.values())
            ax.bar(categories, values, color='skyblue')
            ax.set_xlabel("Attachment Type")
            ax.set_ylabel("Number of Messages")
            ax.set_title("Messages by Attachment Type (July 2020 - July 2025)")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

            return fig
        except Exception as e:
            logger.exception(f"Failed to build attachment categories bar chart: {e}")
            return None

    def build_visual_time(self, p, average_all):
        """
        Create a line plot showing average message counts per week for the 'dac' WhatsApp group.

        Args:
            p (pandas.DataFrame): DataFrame with year, isoweek, and message counts.
            average_all (pandas.DataFrame): DataFrame with average message counts per week across all years.

        Returns:
            matplotlib.figure.Figure or None: Figure object for the line plot, or None if creation fails.
        """
        try:
            # Calculate average message counts (including 2020) for specified week ranges
            weeks_1_12_35_53_all = average_all[
                (average_all["isoweek"].between(1, 12)) | (average_all["isoweek"].between(35, 53))
            ]["avg_count_all"].mean()
            weeks_12_19_all = average_all[
                average_all["isoweek"].between(12, 19)
            ]["avg_count_all"].mean()
            weeks_19_35_all = average_all[
                (average_all["isoweek"].between(19, 35))
            ]["avg_count_all"].mean()

            # Create line plot
            fig, ax = plt.subplots(figsize=(14, 6))

            # Add vertical lines just before weeks 12, 19, and 35
            vline_weeks = [11.5, 18.5, 34.5]  # Just before weeks 12, 19, 35
            for week in vline_weeks:
                ax.axvline(
                    x=week,
                    color="gray",
                    linestyle="--",
                    alpha=0.5,
                    zorder=1,  # Behind data lines
                )

            # Add horizontal lines for average message counts (including 2020) in black
            # Weeks 1–12 and 35–53 (two segments)
            ax.hlines(
                y=weeks_1_12_35_53_all,
                xmin=1,
                xmax=11.5,
                colors="black",
                linestyles="--",
                alpha=0.7,
                zorder=5,  # Above data lines
            )
            ax.hlines(
                y=weeks_1_12_35_53_all,
                xmin=34.5,
                xmax=52,
                colors="black",
                linestyles="--",
                alpha=0.7,
                zorder=5,
            )
            # Weeks 12–19
            ax.hlines(
                y=weeks_12_19_all,
                xmin=11.5,
                xmax=18.5,
                colors="black",
                linestyles="--",
                alpha=0.7,
                zorder=5,
            )
            # Weeks 19–35
            ax.hlines(
                y=weeks_19_35_all,
                xmin=18.5,
                xmax=34.5,
                colors="black",
                linestyles="--",
                alpha=0.7,
                zorder=5,
            )

            # Plot average across all years in black
            sns.lineplot(
                data=average_all,
                x="isoweek",
                y="avg_count_all",
                ax=ax,
                color="black",
                linewidth=2.5,
                zorder=2,  # Above vertical lines
            )

            # Add labels for Rest, Prep, and Play periods at 90% y-axis
            y_min, y_max = ax.get_ylim()  # Get y-axis limits
            y_label = y_min + 0.9 * (y_max - y_min)  # 90% of y-axis range
            ax.text(
                x=5,  # Midpoint of weeks 1–12
                y=y_label,
                s="---------Rest---------",
                ha="center",
                va="center",
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.0, edgecolor=None),
                zorder=7,  # Above all elements
            )
            ax.text(
                x=15,  # Midpoint of weeks 12–19
                y=y_label,
                s="---Prep---",
                ha="center",
                va="center",
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.0, edgecolor="gray"),
                zorder=7,  # Above all elements
            )
            ax.text(
                x=26.5,  # Midpoint of weeks 19–35
                y=y_label,
                s="---------Play---------",
                ha="center",
                va="center",
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="green", alpha=0.0, edgecolor=None),
                zorder=7,  # Above all elements
            )
            ax.text(
                x=45,  # Midpoint of weeks 35–53
                y=y_label,
                s="---------Rest---------",
                ha="center",
                va="center",
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.0, edgecolor="gray"),
                zorder=7,  # Above all elements
            )

            # Fill weeks 12–19 with light green
            ax.axvspan(
                xmin=11.5,
                xmax=18.5,
                color="lightgreen",
                alpha=0.3,
                zorder=0,  # Behind all elements
            )
            # Fill weeks 19–35 with green
            ax.axvspan(
                xmin=18.5,
                xmax=34.5,
                color="green",
                alpha=0.3,
                zorder=0,  # Behind all elements
            )

            # Set x-axis with week numbers and month names
            week_ticks = [1, 5, 9, 14, 18, 23, 27, 31, 36, 40, 44, 49]  # Ticks for each month
            month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            combined_labels = [f"{week}\n{month}" for week, month in zip(week_ticks, month_labels)]
            ax.set_xticks(week_ticks)
            ax.set_xticklabels(combined_labels, ha="right", fontsize=8)
            ax.set_xlabel("Week/ Month of Year", fontsize=8)
            ax.set_ylabel("Average message count per week (2017 - 2025)", fontsize=8)
            plt.title("Golf season, decoded by WhatsApp heartbeat", fontsize=24)

            plt.show()
            return fig
        except Exception as e:
            logger.exception(f"Failed to build time-based plot: {e}")
            return None

    def build_visual_distribution(self, emoji_counts_df):
        """
        Create a bar plot showing the distribution of individual emoji usage in the 'maap' WhatsApp group.

        Args:
            emoji_counts_df (pandas.DataFrame): DataFrame with columns 'emoji', 'count_once', 'percent_once', 'unicode_code', 'unicode_name'.

        Returns:
            matplotlib.figure.Figure or None: Figure object for the bar plot, or None if creation fails.
        """
        try:
            # Check if emoji_counts_df is empty or lacks required columns
            required_columns = ['emoji', 'count_once', 'percent_once']
            if emoji_counts_df is None or emoji_counts_df.empty or not all(col in emoji_counts_df.columns for col in required_columns):
                logger.error("No valid emoji_counts_df or required columns missing. Skipping distribution plot.")
                return None

            # Log emoji counts
            logger.info(f"Emoji usage counts:\n{emoji_counts_df.to_string()}")

            # If no emojis found, skip plotting
            if emoji_counts_df.empty:
                logger.error("No emojis found in 'maap' group. Skipping distribution plot.")
                return None

            # Create bar plot with dynamic figsize for all emojis
            num_emojis = len(emoji_counts_df)
            fig, ax = plt.subplots(figsize=(max(num_emojis * 0.2, 8), 8))
            ax2 = ax.twinx()  # Secondary y-axis for cumulative percentage
            x_positions = np.arange(num_emojis)
            bars = ax.bar(x_positions, emoji_counts_df['percent_once'], color='purple', align='edge', width=0.5)
            ax.set_ylabel("Likelihood (%) of finding an Emoji in a random chosen message", fontsize=12, labelpad=20)
            ax.set_title("The Long Tail of Emotion: Few Speak for Many", fontsize=20)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_position(('outward', 20))
            ax.tick_params(axis='y', labelsize=10)
            ax.set_xlim(-0.5, num_emojis)
            ylim_bottom, ylim_top = ax.get_ylim()
            ax.set_ylim(ylim_bottom - 3, ylim_top)
            ax.set_xticks([])  # No x-ticks

            # Calculate cumulative percentages
            cumulative_once = emoji_counts_df['percent_once'].cumsum()

            # Find index where cumulative percentage reaches 75%
            cum_once_np = np.array(cumulative_once)
            idx_once = None
            if len(cum_once_np) > 0 and np.any(cum_once_np >= 75):
                idx_once = np.where(cum_once_np >= 75)[0][0]
                x_once = idx_once + 1
                y_once = len(emoji_counts_df)
                # Add orange background from left edge to vertical dashed line
                ax.axvspan(-0.5, idx_once + 0.5, facecolor="lightgreen", alpha=0.2)
                # Add vertical dashed line
                ax.axvline(x=idx_once + 0.5, color="orange", linestyle="--", linewidth=1)
                # Add texts for x and y emojis
                left_mid = idx_once / 2
                right_mid = (idx_once + 0.5) + (y_once - idx_once - 1) / 2
                y_text = ylim_bottom - 1.5
                ax.text(left_mid, y_text, f"<-- {x_once} emojis -->", ha='center', fontsize=12)
                ax.text(right_mid, y_text, f"<-- {y_once} emojis -->", ha='center', fontsize=12)

            # Plot cumulative line and 75% dashed line (if applicable)
            ax2.plot(x_positions + 0.25, cumulative_once, color="orange", label="Cumulative %")
            if idx_once is not None:
                ax2.axhline(y=75, color="orange", linestyle="--", linewidth=1, xmin=-0.5, xmax=num_emojis + 0.5)
            ax2.set_ylabel("Cumulative Percentage (%)", fontsize=12, labelpad=20)
            ax2.set_ylim(0, 100)
            ax2.set_yticks(np.arange(0, 101, 10))
            ax2.spines['right'].set_position(('outward', 20))
            ax2.tick_params(axis='y', labelsize=10, colors='orange')
            ax2.spines['right'].set_color('orange')

            # Add table for top 25 emojis (count_once)
            top_25_once = emoji_counts_df.head(25)
            cum_once_top = top_25_once['percent_once'].cumsum()
            table_data = [
                [str(i+1) for i in range(len(top_25_once))],
                [f"{row['emoji']}" for _, row in top_25_once.iterrows()],
                [f"{count:.0f}" for count in top_25_once['count_once']],
                [f"{cum:.1f}%" for cum in cum_once_top]
            ]
            col_width = 0.8 / len(top_25_once)  # Dynamic column width
            table = ax.table(cellText=table_data,
                            rowLabels=["Rank", "Emoji", "Count", "Cum"],
                            colWidths=[col_width] * len(top_25_once),
                            loc='bottom',
                            bbox=[0.1, -0.45, 0.8, 0.3])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)

            # Add "Top 25:" label above the table
            fig.text(0.5, 0.27, "Top 25:", ha='center', fontsize=12)
            ax2.legend(loc='upper left', fontsize=8)
            plt.tight_layout()
            plt.subplots_adjust(left=0.1, right=0.9, bottom=0.35)

            plt.show()
            return fig
        except Exception as e:
            logger.exception(f"Failed to build distribution plot: {e}")
            return None

    def build_visual_distribution_2(self, df):
        """
        Create a scatterplot showing the number of messages containing each of the top 25 emojis
        for each author in the 'maap' WhatsApp group. Each author is represented by a different color.

        Args:
            df (pandas.DataFrame): DataFrame filtered for 'maap' group, containing 'author' and 'message' columns.

        Returns:
            matplotlib.figure.Figure or None: Figure object for the scatterplot, or None if creation fails.
        """
        if df.empty:
            logger.error("Empty DataFrame provided for build_visual_distribution_2.")
            return None

        try:
            # Recreate ignore_emojis (skin tone modifiers) from DataEditor
            ignore_emojis = {chr(int(code, 16)) for code in ['1F3FB', '1F3FC', '1F3FD', '1F3FE', '1F3FF']}

            # Extract all emojis from messages, excluding ignored ones
            all_emojis = []
            for message in df['message']:
                if isinstance(message, str):
                    all_emojis.extend([c for c in message if c in emoji.EMOJI_DATA and c not in ignore_emojis])

            if not all_emojis:
                logger.warning("No emojis found in the messages for 'maap' group.")
                return None

            # Get top 25 emojis by occurrence frequency
            emoji_freq = Counter(all_emojis)
            top_25_emojis = [e for e, _ in emoji_freq.most_common(25)]

            # Get sorted unique authors (assuming 4 authors)
            authors = sorted(df['author'].unique())
            if len(authors) != 4:
                logger.warning(f"Expected 4 authors in 'maap' group, found {len(authors)}.")

            # Define colors for each author
            colors = ['blue', 'red', 'green', 'purple']  # Distinct colors for up to 4 authors
            color_map = {author: color for author, color in zip(authors, colors)}

            # Compute counts: number of messages containing each top emoji per author
            counts = {author: [] for author in authors}
            for emoji_char in top_25_emojis:
                for author in authors:
                    author_msgs = df[df['author'] == author]['message']
                    count = sum(1 for msg in author_msgs if isinstance(msg, str) and emoji_char in msg)
                    counts[author].append(count)

            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))
            x_pos = np.arange(len(top_25_emojis))

            # Plot scatter points for each author
            max_y = 0
            for author in authors:
                y = counts[author]
                ax.scatter(x_pos, y, color=color_map[author], label=author, s=50)
                max_y = max(max_y, max(y) if y else 0)

            # Customize plot
            ax.set_xticks(x_pos)
            ax.set_xticklabels(top_25_emojis, fontsize=14)
            ax.set_xlabel('Top 25 Emojis (Ordered by Overall Frequency)')
            ax.set_ylabel('Number of Messages')
            ax.set_title('Number of Messages Containing Top 25 Emojis per Author in "maap" Group (July 2020 - July 2025)')
            ax.set_ylim(0, max_y * 1.1 if max_y > 0 else 1)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()

            plt.tight_layout()
            plt.show()

            return fig
        except Exception as e:
            logger.exception(f"Failed to build visual distribution 2: {e}")
            return None

    def build_visual_relationships(self, table2, group):
            """
            Create a horizontal bar chart showing total counts and highest author counts for first words
            with high association to one author, including the author with the highest percentage.

            Args:
                table2 (pandas.DataFrame): Numerical DataFrame from build_visual_relationships (highest >= MIN_HIGHEST and total >= MIN_TOTAL, sorted desc).
                group (str): WhatsApp group name for the title.

            Returns:
                matplotlib.figure.Figure or None: Figure object for the bar chart, or None if creation fails.
            """
            if table2 is None or table2.empty:
                logger.error("No valid table provided for building visual relationships plot.")
                return None

            try:
                # Sort by highest descending, reverse for plotting (highest at top)
                table2 = table2.sort_values('highest', ascending=True)  # Ascending for barh (bottom to top)
                first_words = table2.index

                # Identify the author with the highest percentage for each first word
                author_columns = [col for col in table2.columns if col not in ['total', 'highest']]
                highest_authors = table2[author_columns].idxmax(axis=1)

                # Combine first word and highest author for y-axis labels
                y_labels = [f"{word} ({author})" for word, author in zip(first_words, highest_authors)]

                # Create figure
                fig, ax = plt.subplots(figsize=(10, max(4, len(first_words) * 0.5)))

                # Black bars for total
                ax.barh(y_labels, table2['total'], color='black', label='Total Count')

                # Orange bars for highest count (highest % * total / 100)
                highest_counts = (table2['highest'] / 100) * table2['total']
                ax.barh(y_labels, highest_counts, color='orange', label='Highest Author Count')

                ax.set_xlabel('Count')
                ax.set_ylabel('First Word (Author with Highest %)')
                ax.set_title(f"First Words Strongly Associated with One Author in {group}")
                ax.legend()
                plt.tight_layout()

                plt.show()
                return fig
            except Exception as e:
                logger.exception(f"Failed to build visual relationships plot: {e}")
                return None

    def build_visual_relationships_2(self, table2, group):
        """
        Create a horizontal bar chart showing total counts and highest author counts for emoji sequences
        with high association to one author, including the author with the highest percentage.

        Args:
            table2 (pandas.DataFrame): Numerical DataFrame from build_visual_relationships_2 (highest >= MIN_HIGHEST and total >= MIN_TOTAL, sorted desc).
            group (str): WhatsApp group name for the title.

        Returns:
            matplotlib.figure.Figure or None: Figure object for the bar chart, or None if creation fails.
        """
        if table2 is None or table2.empty:
            logger.error("No valid table provided for building visual relationships_2 plot.")
            return None

        try:
            # Sort by highest descending, reverse for plotting (highest at top)
            table2 = table2.sort_values('highest', ascending=True)  # Ascending for barh (bottom to top)
            sequences = table2.index

            # Identify the author with the highest percentage for each sequence
            author_columns = [col for col in table2.columns if col not in ['total', 'highest']]
            highest_authors = table2[author_columns].idxmax(axis=1)

            # Combine sequence and highest author for y-axis labels
            y_labels = [f"{seq} ({author})" for seq, author in zip(sequences, highest_authors)]

            # Create figure
            fig, ax = plt.subplots(figsize=(10, max(4, len(sequences) * 0.5)))

            # Black bars for total
            ax.barh(y_labels, table2['total'], color='black', label='Total Count')

            # Orange bars for highest count (highest % * total / 100)
            highest_counts = (table2['highest'] / 100) * table2['total']
            ax.barh(y_labels, highest_counts, color='orange', label='Highest Author Count')

            ax.set_xlabel('Count')
            ax.set_ylabel('Emoji Sequence (Author with Highest %)')
            ax.set_title(f"Emoji Sequences Strongly Associated with One Author in {group}")
            ax.legend()
            plt.tight_layout()

            plt.show()
            return fig
        except Exception as e:
            logger.exception(f"Failed to build visual relationships_2 plot: {e}")
            return None

    def build_visual_relationships_3(self, combined_df, group):
        """
        Create a network diagram showing interactions between authors in the 'maap' group.

        Args:
            combined_df (pandas.DataFrame): DataFrame from build_visual_relationships_3 with 'Pairs' rows containing interaction data.
            group (str): WhatsApp group name ('maap').

        Returns:
            matplotlib.figure.Figure or None: Figure object for the network diagram, or None if creation fails.
        """
        if nx is None:
            logger.error("Cannot create network diagram: NetworkX is not installed.")
            return None
        if combined_df is None or combined_df.empty:
            logger.error("No valid DataFrame provided for building visual relationships_3 plot.")
            return None
        try:
            # Extract Pairs data
            pairs_df = combined_df[combined_df['type'] == 'Pairs'][['author', 'total_messages']]
            if pairs_df.empty:
                logger.error("No Pairs data found in combined_df for building network diagram.")
                return None

            # Create graph
            G = nx.Graph()
            authors = set()
            for pair in pairs_df['author']:
                a1, a2 = pair.split(' & ')
                a1, a2 = a1.strip(), a2.strip()
                authors.add(a1)
                authors.add(a2)
                G.add_edge(a1, a2, weight=pairs_df[pairs_df['author'] == pair]['total_messages'].iloc[0])

            # Define married couples (assumption based on common naming conventions)
            married_edges = [('Anja Berkemeijer', 'Phons Berkemeijer'), ('Madeleine', 'Anthony van Tilburg')]

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            pos = nx.spring_layout(G, k=0.5, iterations=50)

            # Draw edges (non-married in blue, married in red)
            max_weight = max(nx.get_edge_attributes(G, 'weight').values(), default=1)
            for u, v in G.edges():
                edge_data = G.get_edge_data(u, v)
                if edge_data is None:
                    logger.warning(f"Edge ({u}, {v}) not found in graph. Skipping.")
                    continue
                weight = edge_data['weight']
                color = 'red' if (u, v) in married_edges or (v, u) in married_edges else 'blue'
                width = 1 + 5 * (weight / max_weight)  # Scale width between 1 and 6
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, edge_color=color, ax=ax)

            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

            # Add edge labels
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

            ax.set_title(f"Messaging Interactions in {group} Group\n(Red: Married Couples, Blue: Others)")
            plt.axis('off')
            plt.tight_layout()

            plt.show()
            return fig
        except Exception as e:
            logger.exception(f"Failed to build network diagram: {e}")
            return None
        
    def build_visual_relationships_4(self, combined_df, group):
            """
            Create an arc diagram showing interactions between authors in the 'maap' group.
            Includes pair interactions (gray), triple contributions (lightgray), and total messages (red for married, blue for others).
            Only total message arcs are labeled, with Anthony & Phons label lowered to avoid overlap with Anja & Madeleine.
            All line thicknesses are amplified by a constant APPLIFIER set to 3.
            For Anthony & Phons, lightgray and gray lines are shifted left to ensure visibility.

            Args:
                combined_df (pandas.DataFrame): DataFrame from build_visual_relationships_3 with 'Pairs' and 'Non-participant' rows containing interaction data.
                group (str): WhatsApp group name ('maap').

            Returns:
                matplotlib.figure.Figure or None: Figure object for the arc diagram, or None if creation fails.
            """
            APPLIFIER = 3  # Constant to amplify line thickness
            if combined_df is None or combined_df.empty:
                logger.error("No valid DataFrame provided for building visual relationships_4 plot.")
                return None
            try:
                # Get unique authors
                authors = set()
                for _, row in combined_df.iterrows():
                    if row['type'] == 'Pairs':
                        a1, a2 = row['author'].split(' & ')
                        authors.add(a1.strip())
                        authors.add(a2.strip())
                    elif row['type'] == 'Non-participant':
                        participant_cols = [col for col in combined_df.columns if col not in ['type', 'author', 'num_days', 'total_messages', '#participants']]
                        participants = [col for col in participant_cols if row[col] != 0]
                        for p in participants:
                            authors.add(p)
                authors = sorted(list(authors))

                # Create figure
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.set_aspect('equal')

                # Position authors around a circle
                n = len(authors)
                angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
                radius = 1.0
                pos = {author: (radius * np.cos(angle), radius * np.sin(angle)) for author, angle in zip(authors, angles)}

                # Initialize dictionaries for edge weights
                pair_weights = {}  # For Pairs (gray)
                triple_weights = {}  # For triple contributions (lightgray)
                total_weights = {}  # For sum of Pairs + Triples (red/blue)

                # Process Pairs rows
                pairs_df = combined_df[combined_df['type'] == 'Pairs']
                for _, row in pairs_df.iterrows():
                    a1, a2 = row['author'].split(' & ')
                    a1, a2 = a1.strip(), a2.strip()
                    key = frozenset([a1, a2])
                    weight = row['total_messages']
                    pair_weights[key] = weight
                    total_weights[key] = total_weights.get(key, 0) + weight

                # Process Non-participant rows (triples)
                triples_df = combined_df[combined_df['type'] == 'Non-participant']
                for _, row in triples_df.iterrows():
                    participant_cols = [col for col in combined_df.columns if col not in ['type', 'author', 'num_days', 'total_messages', '#participants']]
                    participants = [col for col in participant_cols if row[col] != 0]
                    if len(participants) != 3:
                        logger.warning(f"Unexpected number of participants in triple row: {len(participants)}. Skipping.")
                        continue

                    total_msg = row['total_messages']
                    pct_dict = {}
                    for p in participants:
                        val_str = row[p]
                        if isinstance(val_str, str):
                            try:
                                msg_pct_str = val_str.split('%')[0]
                                pct_dict[p] = int(msg_pct_str) / 100
                            except (ValueError, IndexError):
                                logger.warning(f"Invalid percentage format for {p}: {val_str}. Skipping.")
                                continue

                    for i, j in itertools.combinations(participants, 2):
                        if i in pct_dict and j in pct_dict:
                            pair_weight = (pct_dict[i] + pct_dict[j]) * total_msg
                            key = frozenset([i, j])
                            triple_weights[key] = triple_weights.get(key, 0) + pair_weight
                            total_weights[key] = total_weights.get(key, 0) + pair_weight

                if not total_weights:
                    logger.error("No edges found after processing pairs and triples.")
                    return None

                # Define married couples
                married_edges = [('Anja Berkemeijer', 'Phons Berkemeijer'), ('Madeleine', 'Anthony van Tilburg')]

                # Get max weight for scaling linewidths
                max_weight = max(total_weights.values(), default=1)

                # Draw arcs in order: lightgray (triples), gray (pairs), red/blue (total)
                for arc_type, weights, color, height_offset, zorder in [
                    ('triple', triple_weights, 'lightgray', 0.4, 1),
                    ('pair', pair_weights, 'gray', 0.55, 2),
                    ('total', total_weights, None, 0.7, 3)  # Color set per edge
                ]:
                    for key, weight in weights.items():
                        a1, a2 = list(key)
                        x1, y1 = pos[a1]
                        x2, y2 = pos[a2]
                        # Calculate arc control point
                        xm = (x1 + x2) / 2
                        ym = (y1 + y2) / 2
                        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        height = dist * height_offset
                        # Set color
                        if arc_type == 'total':
                            color = 'red' if (a1, a2) in married_edges or (a2, a1) in married_edges else 'blue'
                        # Apply horizontal shift for Anthony & Phons for lightgray and gray lines
                        x_offset = 0
                        if set([a1, a2]) == set(['Anthony van Tilburg', 'Phons Berkemeijer']):
                            if arc_type == 'triple':
                                x_offset = -0.1  # Shift lightgray line left
                            elif arc_type == 'pair':
                                x_offset = -0.2  # Shift gray line further left
                        # Scale line width and amplify by APPLIFIER
                        width = (1 + 5 * (weight / max_weight)) * APPLIFIER  # Scale between 1 and 6, then multiply by APPLIFIER
                        # Draw arc with x-offset
                        t = np.linspace(0, 1, 100)
                        x = (1 - t)**2 * x1 + 2 * (1 - t) * t * (xm + x_offset) + t**2 * x2
                        y = (1 - t)**2 * y1 + 2 * (1 - t) * t * (ym + height) + t**2 * y2
                        ax.plot(x, y, color=color, linewidth=width, zorder=zorder)
                        # Add label only for total arcs
                        if arc_type == 'total':
                            label_x = (x1 + x2) / 2
                            label_y = (y1 + y2) / 2 + height * 0.5
                            # Lower the label for Anthony & Phons
                            if set([a1, a2]) == set(['Anthony van Tilburg', 'Phons Berkemeijer']):
                                label_y -= 0.5  # Move label downward
                            ax.text(label_x, label_y, f"{int(round(weight))}", ha='center', va='center', fontsize=8,
                                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), zorder=zorder + 1)

                # Draw nodes
                for author, (x, y) in pos.items():
                    ax.scatter([x], [y], s=2000, color='lightblue', edgecolors='black', zorder=4)
                    ax.text(x, y, author, ha='center', va='center', fontsize=10, fontweight='bold', zorder=5)

                ax.set_title(f"Messaging Interactions in {group} Group\n(Red: Married Couples, Blue: Others, Gray: Pairs, Lightgray: Triples)")
                ax.axis('off')
                plt.tight_layout()

                plt.show()
                return fig
            except Exception as e:
                logger.exception(f"Failed to build arc diagram: {e}")
                return None

    def build_visual_relationships_bubble(self, agg_df):
        """
        Create a bubble plot from prepared data: avg_words vs avg_punct,
        bubble size = message_count, colored by group, alpha by has_emoji.
        
        Args:
            agg_df (pandas.DataFrame): Aggregated data from DataPreparation with columns:
                                      whatsapp_group, author, has_emoji, message_count, avg_words, avg_punct.
        
        Returns:
            matplotlib.figure.Figure or None: The plot figure.
        """
        if agg_df is None or agg_df.empty:
            logger.error("No data provided for bubble plot.")
            return None
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            groups = agg_df['whatsapp_group'].unique()
            # Color mapping for groups (using hex codes)
            colors = {
                groups[0]: '#FFA500' if len(groups) > 0 else '#808080',  # Orange for first group (e.g., maap)
                groups[1]: '#90EE90' if len(groups) > 1 else '#808080',   # Light green for second group (e.g., dac)
                groups[2]: '#0000FF' if len(groups) > 2 else '#808080',   # Blue for third group (e.g., golfmaten)
                groups[3]: '#800080' if len(groups) > 3 else '#808080'    # Purple for fourth group (e.g., tillies)
            }
            
            # Create a set to track legend entries
            legend_added = set()
            
            for _, row in agg_df.iterrows():
                group = row['whatsapp_group']
                has_emoji = row['has_emoji']
                author = row['author']
                # Create author initials for labeling
                initials = ''.join(word[0].upper() for word in author.split() if word)
                color = colors.get(group, '#808080')
                alpha = 0.7 if has_emoji else 0.3
                size = row['message_count'] * 10
                
                # Plot scatter point
                ax.scatter(
                    row['avg_words'],
                    row['avg_punct'],
                    s=size,
                    color=color,
                    alpha=alpha
                )
                
                # Add author initials as label
                ax.text(
                    row['avg_words'],
                    row['avg_punct'],
                    initials,
                    fontsize=8,
                    ha='center',
                    va='center',
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
                )
                
                # Add legend entry only once per group/emoji combination
                legend_key = (group, has_emoji)
                if legend_key not in legend_added:
                    label = f"{group} ({'with' if has_emoji else 'without'} emojis)"
                    ax.scatter([], [], s=100, color=color, alpha=alpha, label=label)
                    legend_added.add(legend_key)
            
            # Add trendlines for averages across all groups for each has_emoji category
            for has_emoji, trend_color in [(True, 'red'), (False, 'black')]:
                subset = agg_df[agg_df['has_emoji'] == has_emoji]
                if len(subset) > 1:  # Need at least 2 points for regression
                    x = subset['avg_words']
                    y = subset['avg_punct']
                    slope, intercept = np.polyfit(x, y, 1)  # Linear fit
                    line_x = np.linspace(x.min(), x.max(), 100)
                    line_y = slope * line_x + intercept
                    ax.plot(line_x, line_y, color=trend_color, linestyle='--', label=f"Trendline ({'with' if has_emoji else 'without'} emojis)")
            
            ax.set_xlabel('Average Number of Words per Message')
            ax.set_ylabel('Average Number of Punctuations per Message')
            ax.set_title('Bubble Plot: Words vs Punctuations by Group, Author, and Emoji Usage\n(Bubble Size: Number of Messages)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
            return fig
        except Exception as e:
            logger.exception(f"Failed to build bubble plot: {e}")
            return None

    def build_visual_sequence_comparison(self, sequence_df, group):
        """
        Create a line plot comparing sequence scores (alternating_MF, rhythm, married_alternation) over time.
        
        Args:
            sequence_df (pandas.DataFrame): DataFrame with columns: date, score_alternating_MF, score_rhythm,
                                            score_not_married, [score_married_alternation].
            group (str): WhatsApp group name (e.g., 'maap').
        
        Returns:
            matplotlib.figure.Figure or None: Figure object for the line plot, or None if creation fails.
        """
        if sequence_df.empty:
            logger.error(f"No data provided for sequence comparison plot for group {group}.")
            return None
        
        try:
            # Ensure date is datetime
            sequence_df['date'] = pd.to_datetime(sequence_df['date'])
            
            # Check for valid data in score columns
            score_columns = ['score_alternating_MF', 'score_rhythm']
            if 'score_married_alternation' in sequence_df.columns:
                score_columns.append('score_married_alternation')
            else:
                logger.warning("score_married_alternation not found in DataFrame. Excluding from plot.")
            
            # Log data summary
            logger.debug(f"Sequence DataFrame for {group}:\n{sequence_df[score_columns + ['date']].describe().to_string()}")
            if sequence_df[score_columns].isna().all().all():
                logger.error(f"All score columns ({score_columns}) contain only NaN values.")
                return None
            
            # Create line plot
            fig, ax = plt.subplots(figsize=(12, 6))
            for col in score_columns:
                # Skip if column is all NaN or zero
                if sequence_df[col].isna().all() or (sequence_df[col] == 0).all():
                    logger.warning(f"Skipping {col} due to all NaN or zero values.")
                    continue
                sns.lineplot(data=sequence_df, x='date', y=col, label=col.replace('score_', '').replace('_', ' ').title(), ax=ax)
            
            if not ax.has_data():
                logger.error("No data plotted. Check if all score columns are empty or invalid.")
                return None
            
            ax.set_title(f"Sequence Scores Over Time for {group} Group")
            ax.set_xlabel("Date")
            ax.set_ylabel("Score (0-1)")
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Ensure plot displays
            plt.show(block=False)  # Non-blocking show for compatibility
            logger.info(f"Created sequence comparison plot for group {group}.")
            return fig
        except Exception as e:
            logger.exception(f"Failed to create sequence comparison plot for group {group}: {e}")
            return None                     