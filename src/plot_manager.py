import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from loguru import logger
import warnings
import matplotlib.font_manager as fm

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
        per WhatsApp group for July 2020 - July 2025.

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

            # Add red arrow for maap
            maap_idx = sorted_groups.index('maap') if 'maap' in sorted_groups else None
            if maap_idx is not None:
                x_pos = positions[maap_idx] + bar_width
                y_start = non_anthony_group['non_anthony_avg'].iloc[maap_idx]
                y_end = anthony_group['anthony_messages'].iloc[maap_idx]
                ax.annotate('',
                            xy=(x_pos, y_end), xytext=(x_pos, y_start),
                            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
                logger.info(f"Red arrow for maap: from (x={x_pos:.2f}, y={y_start:.2f}) to (x={x_pos:.2f}, y={y_end:.2f})")

            # Customize x-axis labels
            xtick_labels = [f"{group} ({num_authors:.1f})" for group, num_authors in zip(sorted_groups, non_anthony_group['num_authors'])]
            ax.set_xticks(positions + bar_width / 2)
            ax.set_xticklabels(xtick_labels)
            ax.set_xlabel("WhatsApp Group (Number of Non-Anthony Authors)")
            ax.set_ylabel("Messages (July 2020 - July 2025)")
            ax.set_title("Anthony's participation ratio in whatsapp_group 'maap' is significant lower than other groups")
            ax.legend()
            plt.tight_layout()
            plt.show()

            return fig
        except Exception as e:
            logger.exception(f"Failed to build bar chart: {e}")
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