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

    def build_visual_distribution(self, emoji_counts_df):
        """
        Create a bar plot showing the distribution of emoji usage in the 'maap' WhatsApp group.

        Args:
            emoji_counts_df (pandas.DataFrame): DataFrame with emoji counts, percentages, Unicode codes, and names.

        Returns:
            matplotlib.figure.Figure or None: Figure object for the bar plot, or None if creation fails.
        """
        try:
            # Calculate cumulative percentages
            cumulative_once = emoji_counts_df["percent_once"].cumsum()

            # Create bar plot
            fig, ax = plt.subplots(figsize=(len(emoji_counts_df) * 0.3, 8))
            ax2 = ax.twinx()  # Secondary y-axis for cumulative percentage
            x_positions = np.arange(len(emoji_counts_df))
            bars = ax.bar(x_positions, emoji_counts_df["percent_once"], color="purple", align='edge', width=0.5)
            ax.set_ylabel("Likelihood (%) of finding an Emoji in a random chosen message", fontsize=12, labelpad=20)
            ax.set_title("The Long Tail of Emotion: Few Speak for Many", fontsize=20)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_position(('outward', 20))
            ax.tick_params(axis='y', labelsize=10)
            ax.set_xlim(-0.5, len(emoji_counts_df))
            ylim_bottom, ylim_top = ax.get_ylim()
            ax.set_ylim(ylim_bottom - 3, ylim_top)

            # Calculate x for 75%
            cum_once_np = np.array(cumulative_once)
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
            ax.set_xticks([])  # No x-ticks

            # Plot cumulative line and 75% dashed line
            ax2.plot(x_positions + 0.25, cumulative_once, color="orange", label="Cumulative %")
            ax2.axhline(y=75, color="orange", linestyle="--", linewidth=1, xmin=-0.5, xmax=len(emoji_counts_df) + 0.5)
            ax2.set_ylabel("Cumulative Percentage (%)", fontsize=12, labelpad=20)
            ax2.set_ylim(0, 100)
            ax2.set_yticks(np.arange(0, 101, 10))
            ax2.spines['right'].set_position(('outward', 20))
            ax2.tick_params(axis='y', labelsize=10, colors='orange')
            ax2.spines['right'].set_color('orange')

            # Add table for top 25 emojis (count_once)
            top_25_once = emoji_counts_df.head(25)
            cum_once_top = top_25_once["percent_once"].cumsum()
            table_data = [
                [str(i+1) for i in range(25)],
                list(top_25_once["emoji"]),
                [f"{count:.0f}" for count in top_25_once["count_once"]],
                [f"{cum:.1f}%" for cum in cum_once_top]
            ]
            table = ax.table(cellText=table_data,
                            rowLabels=["Rank", "Emoji", "Count", "Cum"],
                            colWidths=[0.05] * 25,
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