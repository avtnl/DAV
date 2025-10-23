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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
from scipy.stats import chi2

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

    def build_visual_relationships_arc(self, combined_df, group):
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
        Create a bubble plot from prepared data: avg_words vs avg_punct, bubble size = message_count.
        Groups/ Colors: has_emoji=True in shades of green (darkest for largest, top 5), 
        has_emoji=False in shades of red (darkest for largest, top 5),
        with two trendlines: red (has_emoji=False), green (has_emoji=True).

        Args:
            agg_df (pandas.DataFrame): Aggregated data from DataPreparation with columns:
                                    whatsapp_group, author, has_emoji, message_count, avg_words, avg_punct.

        Returns:
            matplotlib.figure.Figure or None: Figure object for the bubble plot, or None if creation fails.
        """
        if agg_df is None or agg_df.empty:
            logger.error("No data provided for bubble plot.")
            return None

        try:
            # Define color shades (10 shades, using first 5 for top 5, darkest for 1st, lightest for 5th; 10th for others)
            green_shades = ['#00CC00', '#1AFF1A', '#33FF33', '#4DFF4D', '#66FF66', '#80FF80', '#99FF99', '#B3FFB3', '#CCFFCC', '#E6FFE6']  # Dark to light green
            red_shades = ['#CC0000', '#FF1A1A', '#FF3333', '#FF4D4D', '#FF6666', '#FF8080', '#FF9999', '#FFB3B3', '#FFCCCC', '#FFE6E6']  # Dark to light red

            # Plot: has_emoji=True in green shades, has_emoji=False in orange shades for top 5 combinations
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get top 5 group/author combinations for has_emoji=True and has_emoji=False
            emoji_true = agg_df[agg_df['has_emoji'] == True]
            emoji_false = agg_df[agg_df['has_emoji'] == False]
            
            top_5_true = emoji_true.sort_values('message_count', ascending=False).head(5)[['whatsapp_group', 'author']].values.tolist()
            top_5_true_keys = [f"{group}_{author}" for group, author in top_5_true]
            top_5_false = emoji_false.sort_values('message_count', ascending=False).head(5)[['whatsapp_group', 'author']].values.tolist()
            top_5_false_keys = [f"{group}_{author}" for group, author in top_5_false]
            
            # Log to check top 5
            logger.debug(f"Top 5 group/author combinations for has_emoji=True: {top_5_true}")
            logger.debug(f"Top 5 group/author combinations for has_emoji=False: {top_5_false}")
            
            for _, row in agg_df.iterrows():
                group = row['whatsapp_group']
                author = row['author']
                has_emoji = row['has_emoji']
                group_author = f"{group}_{author}"
                size = row['message_count'] * 10
                initials = f"{group[0].upper()}{''.join(word[0].upper() for word in author.split() if word)}"
                
                # Assign color based on has_emoji and top 5 combinations
                if has_emoji and group_author in top_5_true_keys:
                    rank = top_5_true_keys.index(group_author)
                    color = green_shades[rank]
                elif not has_emoji and group_author in top_5_false_keys:
                    rank = top_5_false_keys.index(group_author)
                    color = red_shades[rank]
                else:
                    color = green_shades[6] if has_emoji else red_shades[6]  # Lightest green/red for others
                
                # Plot scatter point
                ax.scatter(
                    row['avg_words'],
                    row['avg_punct'],
                    s=size,
                    color=color,
                    alpha=0.5
                )
                
                # Add author initials
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
            
            # Add trendlines
            if len(agg_df) > 1:
                # Define line_x for trendlines based on the range of avg_words
                line_x = np.linspace(agg_df['avg_words'].min(), agg_df['avg_words'].max(), 100)
                
                # Trendline for has_emoji=False (red)
                if len(emoji_false) > 1:
                    x_false = emoji_false['avg_words']
                    y_false = emoji_false['avg_punct']
                    slope_false, intercept_false = np.polyfit(x_false, y_false, 1)
                    line_y_false = slope_false * line_x + intercept_false
                    ax.plot(line_x, line_y_false, color='red', linestyle='--')
                
                # Trendline for has_emoji=True (green)
                if len(emoji_true) > 1:
                    x_true = emoji_true['avg_words']
                    y_true = emoji_true['avg_punct']
                    slope_true, intercept_true = np.polyfit(x_true, y_true, 1)
                    line_y_true = slope_true * line_x + intercept_true
                    ax.plot(line_x, line_y_true, color='green', linestyle='--')
            
            # Customize plot
            ax.set_xlabel('Average Number of Words per Message')
            ax.set_ylabel('Average Number of Punctuations per Message')
            ax.set_title("More words = More Punctuations, but Emojis reduce number of Punctuations!",  fontsize=20, x=0.6, ha='center')
            
            # Create custom legend
            legend_elements = []
            # Calculate total message
            total_false_msgs = int(emoji_false['message_count'].sum()) if not emoji_false.empty else 0
            total_true_msgs = int(emoji_true['message_count'].sum()) if not emoji_true.empty else 0
            legend_elements.append(plt.scatter([], [], s=100, color=red_shades[0], label=f'Without emojis\nTotal: {total_false_msgs:,} msgs'))
            legend_elements.append(plt.scatter([], [], s=100, color=green_shades[0], label=f'With emojis\nTotal: {total_true_msgs:,} msgs'))
            legend_elements.append(plt.scatter([], [], s=0, label='\nSize of bubble reflects number of msgs\n', alpha=0))
            legend_elements.append(plt.plot([], [], color='red', linestyle='--', label='Trend line without emojis')[0])
            legend_elements.append(plt.plot([], [], color='green', linestyle='--', label='Trend line with emojis')[0])
            
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.show()
            
            logger.info("Created bubble plot successfully.")
            return fig  # Return single figure instead of a list
        except Exception as e:
            logger.exception(f"Failed to build bubble plot: {e}")
            return None

    def build_visual_relationships_bubble_2(self, agg_df):
        """
        Create an enhanced bubble plot from prepared data: avg_words vs avg_punct, bubble size = message_count.
        Groups/Colors: has_emoji=True in shades of green (darkest for largest, top 5),
        has_emoji=False in shades of red (darkest for largest, top 5),
        with two trendlines: red (has_emoji=False), green (has_emoji=True).
        Adds horizontal and vertical lines at y=1,2,3 with corresponding x-value labels, and filled areas.

        Args:
            agg_df (pandas.DataFrame): Aggregated data from DataPreparation with columns:
                                    whatsapp_group, author, has_emoji, message_count, avg_words, avg_punct.

        Returns:
            matplotlib.figure.Figure or None: Figure object for the bubble plot, or None if creation fails.
        """
        if agg_df is None or agg_df.empty:
            logger.error("No data provided for enhanced bubble plot.")
            return None

        try:
            # Define color shades (10 shades, using first 5 for top 5, darkest for 1st, lightest for 5th; 6th for others)
            green_shades = ['#00CC00', '#1AFF1A', '#33FF33', '#4DFF4D', '#66FF66', '#80FF80', '#99FF99', '#B3FFB3', '#CCFFCC', '#E6FFE6']  # Dark to light green
            red_shades = ['#CC0000', '#FF1A1A', '#FF3333', '#FF4D4D', '#FF6666', '#FF8080', '#FF9999', '#FFB3B3', '#FFCCCC', '#FFE6E6']  # Dark to light red
            gray_shades = ['#1A1A1A', '#333333', '#4D4D4D', '#666666', '#808080', '#999999', '#B3B3B3', '#CCCCCC', '#E6E6E6', '#F2F2F2']  # Dark to light gray

            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get top 5 group/author combinations for has_emoji=True and has_emoji=False
            emoji_true = agg_df[agg_df['has_emoji'] == True]
            emoji_false = agg_df[agg_df['has_emoji'] == False]
            
            top_5_true = emoji_true.sort_values('message_count', ascending=False).head(5)[['whatsapp_group', 'author']].values.tolist()
            top_5_true_keys = [f"{group}_{author}" for group, author in top_5_true]
            top_5_false = emoji_false.sort_values('message_count', ascending=False).head(5)[['whatsapp_group', 'author']].values.tolist()
            top_5_false_keys = [f"{group}_{author}" for group, author in top_5_false]
            
            # Log to check top 5
            logger.debug(f"Top 5 group/author combinations for has_emoji=True: {top_5_true}")
            logger.debug(f"Top 5 group/author combinations for has_emoji=False: {top_5_false}")
            
            # Compute trendlines first to get slopes and intercepts
            slope_false, intercept_false, slope_true, intercept_true = None, None, None, None
            if len(agg_df) > 1:
                # Trendline for has_emoji=False (red)
                if len(emoji_false) > 1:
                    x_false = emoji_false['avg_words']
                    y_false = emoji_false['avg_punct']
                    slope_false, intercept_false = np.polyfit(x_false, y_false, 1)
                
                # Trendline for has_emoji=True (green)
                if len(emoji_true) > 1:
                    x_true = emoji_true['avg_words']
                    y_true = emoji_true['avg_punct']
                    slope_true, intercept_true = np.polyfit(x_true, y_true, 1)
            
            # Calculate intersection points for y=1,2,3
            ks = [1, 2, 3]
            x_reds = []
            x_greens = []
            for k in ks:
                x_red = (k - intercept_false) / slope_false if slope_false and slope_false != 0 else 0
                x_green = (k - intercept_true) / slope_true if slope_true and slope_true != 0 else 0
                # Clip to non-negative
                x_red = max(0, x_red)
                x_green = max(0, x_green)
                x_reds.append(x_red)
                x_greens.append(x_green)

            # Calculate additional y values where x_green[1] and x_green[2] hit the red trendline
            additional_ys = []
            for k in [0, 1]:
                # Use x_green[1] (for y=1) and x_green[2] (for y=2) from green trendline
                x_green = x_greens[k]
                # Calculate y at x_green on red trendline
                y_next = slope_false * x_green + intercept_false if slope_false is not None and intercept_false is not None else 0
                additional_ys.append(y_next)
            
            # Log the additional_ys values
            logger.info(f"Additional y values where x_green[1] and x_green[2] hit red trendline: {additional_ys}")

            # Set y-axis to start at 0 and ensure y=3.5 is visible
            ax.set_ylim(bottom=0, top=max(3.5, agg_df['avg_punct'].max() * 1.1))

            # Fill areas first (grays under red, greens between red and green, additional areas)
            # Grays (stepped polygons under red)
            gray_colors = [gray_shades[3], gray_shades[5], gray_shades[7]]
            prev_x_red = 0
            prev_k = 0
            for i, (k, x_red, color) in enumerate(zip(ks, x_reds, gray_colors)):
                ax.fill(
                    [0, 0, x_red, prev_x_red],
                    [prev_k, k, k, prev_k],
                    color=color,
                    alpha=0.5,
                    zorder=0
                )
                prev_x_red = x_red
                prev_k = k

            # Grays (rectangles from y=0 to k, x_red to x_green)
            gray_colors = [gray_shades[3], gray_shades[5], gray_shades[7]]            
            for i, (k, x_red, x_green, color) in enumerate(zip(ks, x_reds, x_greens, gray_colors)):
                ax.fill(
                    [x_red, x_red, x_green, x_green],
                    [0, k, k, 0],
                    color=color,
                    alpha=0.5,
                    zorder=0
                )

            # Additional areas
            # Polygon: (x_green[1], additional_ys[1]), (x_red[2], 3), (x_red[2], 0), (x_green[1], 0)
            ax.fill(
                [x_greens[1], x_reds[2], x_reds[2], x_greens[1]],
                [additional_ys[1], 3, 0, 0],
                color=gray_shades[7],
                alpha=0.5,
                zorder=0
            )

            # Polygon: (x_green[0], additional_ys[0]), (x_red[1], 2), (x_red[1], 0), (x_green[0], 0)
            ax.fill(
                [x_greens[0], x_reds[1], x_reds[1], x_greens[0]],
                [additional_ys[0], 2, 0, 0],
                color=gray_shades[5],
                alpha=0.5,
                zorder=0
            )

            # Triangle: (0, 0), (x_red[0], 1), (x_red[0], 0)
            ax.fill(
                [0, x_reds[0], x_reds[0]],
                [0, 1, 0],
                color=gray_shades[3],
                alpha=0.5,
                zorder=0
            )

            # Triangle: (x_red[1], 2), (x_green[1], additional_ys[1]), (x_green[1], 2)
            ax.fill(
                [x_reds[1], x_greens[1], x_greens[1]],
                [2, additional_ys[1], 2],
                color=gray_shades[7],
                alpha=0.5,
                zorder=0
            )

            # Plot bubbles on top
            for _, row in agg_df.iterrows():
                group = row['whatsapp_group']
                author = row['author']
                has_emoji = row['has_emoji']
                group_author = f"{group}_{author}"
                size = row['message_count'] * 10
                initials = f"{group[0].upper()}{''.join(word[0].upper() for word in author.split() if word)}"
                
                # Assign color based on has_emoji and top 5 combinations
                if has_emoji and group_author in top_5_true_keys:
                    rank = top_5_true_keys.index(group_author)
                    color = green_shades[6]
                elif not has_emoji and group_author in top_5_false_keys:
                    rank = top_5_false_keys.index(group_author)
                    color = red_shades[6]
                else:
                    color = green_shades[6] if has_emoji else red_shades[6]  # Lightest green/red for others
                
                # Plot scatter point
                ax.scatter(
                    row['avg_words'],
                    row['avg_punct'],
                    s=size,
                    color=color,
                    alpha=0.5
                )
                
                # Add author initials
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
            
            # Add trendlines
            if len(agg_df) > 1:
                # Define line_x for trendlines based on the range of avg_words
                line_x = np.linspace(agg_df['avg_words'].min(), agg_df['avg_words'].max(), 100)
                
                # Trendline for has_emoji=False (red)
                if len(emoji_false) > 1:
                    line_y_false = slope_false * line_x + intercept_false
                    ax.plot(line_x, line_y_false, color='red', linestyle='--')
                
                # Trendline for has_emoji=True (green)
                if len(emoji_true) > 1:
                    line_y_true = slope_true * line_x + intercept_true
                    ax.plot(line_x, line_y_true, color='green', linestyle='--')
            
            # Add horizontal and vertical lines with x-value labels
            for i, k in enumerate(ks):
                x_red = x_reds[i]
                x_green = x_greens[i]
                
                # Horizontals
                ax.hlines(y=k, xmin=0, xmax=x_red, color='black', linestyle='--')
                ax.hlines(y=k, xmin=x_red, xmax=x_green, color='black', linestyle=':')
                
                # Verticals
                ax.vlines(x=x_red, ymin=0, ymax=k, color='red', linestyle='--')
                ax.vlines(x=x_green, ymin=0, ymax=k, color='green', linestyle=':')
                
                # Add x-value labels
                ax.text(x_red, 0.2, f"{x_red:.1f}", color='red', ha='center', va='top', fontsize=12)
                ax.text(x_green, 0.2, f"{x_green:.1f}", color='green', ha='center', va='top', fontsize=12)

            # Add red and green dots at trendline intersections
            for i, k in enumerate(ks):
                # Red dots at (x_red[i], k)
                ax.scatter([x_reds[i]], [k], s=100, color=red_shades[0], zorder=5)
                ax.scatter([x_reds[i]], 0.05, s=100, color=red_shades[0], zorder=5)
                # Green dots at (x_green[i], k)
                ax.scatter([x_greens[i]], [k], s=100, color=green_shades[0], zorder=5)
                ax.scatter([x_greens[i]], 0.05, s=100, color=green_shades[0], zorder=5)

            # # Add black rectangle with thicker lines
            # rect = patches.Rectangle(
            #     (5, 0),  # Bottom-left corner (x, y)
            #     22,   # Width
            #     0.25,       # Height (from y=1 to y=3)
            #     linewidth=3,  # Thicker line
            #     edgecolor='black',
            #     facecolor='none',  # No fill, just the outline
            #     zorder=1
            # )
            # ax.add_patch(rect)

            # Add block arrows at y=0.25 between x_red[k] and x_green[k] for k=0,1,2
            for i, k in enumerate(ks):
                # Draw block arrow from x_reds[i] to x_greens[i] at y=0.25
                ax.annotate(
                    '',
                    xy=(x_greens[i], 0.25),  # End point (arrowhead)
                    xytext=(x_reds[i], 0.25),  # Start point
                    arrowprops=dict(
                        arrowstyle='-|>',
                        color='black',
                        linewidth=3,
                        mutation_scale=20
                    ),
                    zorder=1
                )
                # Draw reverse block arrow for thicker effect (optional, mimicking build_visual_categories)
                ax.annotate(
                    '',
                    xy=(x_reds[i], 0.25),  # End point (arrowhead)
                    xytext=(x_greens[i], 0.25),  # Start point
                    arrowprops=dict(
                        arrowstyle='-|>',
                        color='black',
                        linewidth=3,
                        mutation_scale=20
                    ),
                    zorder=1
                )

            # Customize plot
            ax.set_xlabel('Average Number of Words per Message')
            ax.set_ylabel('Average Number of Punctuations per Message')
            ax.set_title("More words = More Punctuations ... and Emojis reduce the number of Punctuations!", fontsize=20, x=0.65, ha='center')

            # Create custom legend
            legend_elements = []
            total_false_msgs = int(emoji_false['message_count'].sum()) if not emoji_false.empty else 0
            total_true_msgs = int(emoji_true['message_count'].sum()) if not emoji_true.empty else 0
            legend_elements.append(plt.scatter([], [], s=100, color=red_shades[6], label=f'Without emojis\nTotal: {total_false_msgs:,} msgs'))
            legend_elements.append(plt.scatter([], [], s=100, color=green_shades[6], label=f'With emojis\nTotal: {total_true_msgs:,} msgs'))
            legend_elements.append(plt.scatter([], [], s=0, label='\nSize of bubble reflects number of msgs\n', alpha=0))
            legend_elements.append(plt.plot([], [], color='red', linestyle='--', label='Trend line without emojis')[0])
            legend_elements.append(plt.plot([], [], color='green', linestyle='--', label='Trend line with emojis')[0])
            
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.show()
            
            logger.info("Created enhanced bubble plot successfully.")
            return fig
        except Exception as e:
            logger.exception(f"Failed to build enhanced bubble plot: {e}")
            return None

    def build_visual_model(self, sequence_df, group):
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
        
    def build_visual_multi_dimensional(self, feature_df, method='tsne'):
        """
        Create a 2D visualization of WhatsApp groups based on golf-relatedness using PCA or t-SNE.

        Args:
            feature_df (pandas.DataFrame): DataFrame with 'whatsapp_group' as the first column and keyword frequencies as subsequent columns.
            method (str): Dimensionality reduction method ('pca' or 'tsne'). Default: 'tsne'.

        Returns:
            matplotlib.figure.Figure or None: Figure object for the 2D plot, or None if creation fails.
        """
        try:
            # Extract group names and feature matrix
            groups = feature_df['whatsapp_group'].values
            X_normalized = feature_df.drop('whatsapp_group', axis=1).values

            # Apply dimensionality reduction
            if method.lower() == 'pca':
                reducer = PCA(n_components=2)
                X_reduced = reducer.fit_transform(X_normalized)
                loadings = pd.DataFrame(reducer.components_.T, index=feature_df.drop('whatsapp_group', axis=1).columns, columns=['Component 1', 'Component 2'])
                logger.info(f"PCA Loadings:\n{loadings.to_string()}")                
            elif method.lower() == 'tsne':
                reducer = TSNE(n_components=2, perplexity=2, random_state=42)  # Low perplexity for few points
            else:
                logger.error(f"Invalid method '{method}'. Use 'pca' or 'tsne'.")
                return None

            X_reduced = reducer.fit_transform(X_normalized)

            # Create plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c='blue')

            # Annotate group names
            for i, group in enumerate(groups):
                ax.annotate(group, (X_reduced[i, 0], X_reduced[i, 1]), fontsize=12, ha='right')

            ax.set_title(f"Golf-Relatedness of WhatsApp Groups ({method.upper()})")
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            plt.tight_layout()
            plt.show()

            logger.info(f"Created {method.upper()} visualization successfully.")
            return fig
        except Exception as e:
            logger.exception(f"Failed to build multi-dimensional visualization: {e}")
            return None

    def build_visual_multi_dimensional_2(self, feature_df, method='tsne'):
        """
        Create a 2D visualization of group/author combinations based on golf-relatedness using PCA or t-SNE.

        Args:
            feature_df (pandas.DataFrame): DataFrame with 'group_author' as the first column and keyword frequencies as subsequent columns.
            method (str): Dimensionality reduction method ('pca' or 'tsne'). Default: 'tsne'.

        Returns:
            matplotlib.figure.Figure or None: Figure object for the 2D plot, or None if creation fails.
        """
        try:
            # Extract group/author labels and feature matrix
            group_authors = feature_df['group_author'].values
            X_normalized = feature_df.drop('group_author', axis=1).values

            # Apply dimensionality reduction
            if method.lower() == 'pca':
                reducer = PCA(n_components=2)
            elif method.lower() == 'tsne':
                reducer = TSNE(n_components=2, perplexity=min(5, len(group_authors)-1), random_state=42)  # Adjust perplexity
            else:
                logger.error(f"Invalid method '{method}'. Use 'pca' or 'tsne'.")
                return None

            X_reduced = reducer.fit_transform(X_normalized)

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 8))  # Slightly larger for more points
            # Color-code by group for clarity
            groups = [ga.split('/')[0] for ga in group_authors]
            unique_groups = list(set(groups))
            colors = sns.color_palette("husl", len(unique_groups))
            group_color_map = dict(zip(unique_groups, colors))

            for i, group_author in enumerate(group_authors):
                group = group_author.split('/')[0]
                ax.scatter(X_reduced[i, 0], X_reduced[i, 1], c=[group_color_map[group]], label=group if i == groups.index(group) else None)
                # ax.annotate(group_author, (X_reduced[i, 0], X_reduced[i, 1]), fontsize=8, ha='right')  # Smaller font

            ax.set_title(f"Golf-Relatedness of Group/Author Combinations ({method.upper()})")
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.legend(title="WhatsApp Group")
            plt.tight_layout()
            plt.show()

            # Log PCA loadings if using PCA
            if method.lower() == 'pca':
                loadings = pd.DataFrame(reducer.components_.T, index=feature_df.drop('group_author', axis=1).columns, columns=['Component 1', 'Component 2'])
                logger.info(f"PCA Loadings:\n{loadings.to_string()}")

            logger.info(f"Created {method.upper()} visualization for group/author combinations successfully.")
            return fig
        except Exception as e:
            logger.exception(f"Failed to build multi-dimensional visualization: {e}")
            return None

    def build_visual_interactions(self, feature_df, method='tsne'):
            """
            Specialized 2D visualization for interaction features using PCA or t-SNE.
            Colors by author for evolution over years.
            
            Args:
                feature_df (pandas.DataFrame): Feature matrix with 'author_year' index.
                method (str): 'pca' or 'tsne'.
            
            Returns:
                matplotlib.figure.Figure or None: The plot figure.
            """
            try:
                labels = feature_df.index.values
                X = feature_df.values

                if method.lower() == 'pca':
                    reducer = PCA(n_components=2)
                elif method.lower() == 'tsne':
                    reducer = TSNE(n_components=2, perplexity=min(5, len(labels)-1), random_state=42)
                else:
                    raise ValueError(f"Invalid method '{method}'.")

                X_reduced = reducer.fit_transform(X)

                # Extract authors from labels (author_year)
                authors = [label.split('_')[0] for label in labels]  # Assuming no '_' in names
                unique_authors = list(set(authors))
                colors = sns.color_palette("husl", len(unique_authors))
                author_color_map = dict(zip(unique_authors, colors))

                fig, ax = plt.subplots(figsize=(10, 8))
                for i, label in enumerate(labels):
                    auth = label.split('_')[0]
                    ax.scatter(X_reduced[i, 0], X_reduced[i, 1], c=[author_color_map[auth]], label=auth if authors.index(auth) == i else None)
                    # ax.annotate(label, (X_reduced[i, 0], X_reduced[i, 1]), fontsize=8, ha='right')

                ax.set_title(f"Interaction Dynamics of Authors Over Years ({method.upper()})")
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.legend(title="Author")
                plt.tight_layout()
                plt.show()

                logger.info(f"Created interaction visualization with {method.upper()}.")
                return fig
            except Exception as e:
                logger.exception(f"Failed to build interaction visualization: {e}")
                return None

    def build_visual_interactions_2(self, feature_df, method='tsne'):
        """
        Create two 2D visualizations for interaction features using PCA or t-SNE.
        First plot: Colors by author, with 'Anthony van Tilburg' points for each group-year and overall.
        Second plot: Colors by group ('maap': blue, 'golfmaten': red, 'dac': green) and
        'Anthony van Tilburg' points by group (light blue for 'maap', light red for 'golfmaten',
        light green for 'dac', gray for overall).
        
        Args:
            feature_df (pandas.DataFrame): Feature matrix with 'author_year' or 'author_year_group' index and 'whatsapp_group' column.
            method (str): 'pca' or 'tsne'.
        
        Returns:
            tuple: (matplotlib.figure.Figure, matplotlib.figure.Figure) or (None, None) if creation fails.
        """
        try:
            labels = feature_df.index.values
            X = feature_df.drop(['whatsapp_group'], axis=1, errors='ignore').values
            if method.lower() == 'pca':
                reducer = PCA(n_components=2)
            elif method.lower() == 'tsne':
                reducer = TSNE(n_components=2, perplexity=min(5, len(labels)-1), random_state=42)
            else:
                raise ValueError(f"Invalid method '{method}'.")
            X_reduced = reducer.fit_transform(X)

            # First plot: Color by author
            authors = [label.split('_')[0] for label in labels]  # Extract author from 'author_year' or 'author_year_group'
            unique_authors = list(set(authors))
            colors = sns.color_palette("husl", len(unique_authors))
            author_color_map = dict(zip(unique_authors, colors))
            fig1, ax1 = plt.subplots(figsize=(10, 8))
            for i, label in enumerate(labels):
                auth = label.split('_')[0]
                ax1.scatter(X_reduced[i, 0], X_reduced[i, 1], c=[author_color_map[auth]], label=auth if authors.index(auth) == i else None)
                ax1.annotate(label, (X_reduced[i, 0], X_reduced[i, 1]), fontsize=8, ha='right')
            ax1.set_title(f"Interaction Dynamics of Authors Over Years ({method.upper()})")
            ax1.set_xlabel('Component 1')
            ax1.set_ylabel('Component 2')
            ax1.legend(title="Author")
            plt.tight_layout()
            plt.show()

            # Second plot: Color by group, with special colors for Anthony van Tilburg
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            group_color_map = {
                'maap': 'blue',
                'golfmaten': 'red',
                'dac': 'green',
                'maap_anthony': 'lightblue',
                'golfmaten_anthony': 'lightcoral',
                'dac_anthony': 'lightgreen',
                'overall': 'gray'
            }
            legend_elements = [
                patches.Patch(color='blue', label='maap'),
                patches.Patch(color='red', label='golfmaten'),
                patches.Patch(color='green', label='dac'),
                patches.Patch(color='lightblue', label='Anthony van Tilburg (maap)'),
                patches.Patch(color='lightcoral', label='Anthony van Tilburg (golfmaten)'),
                patches.Patch(color='lightgreen', label='Anthony van Tilburg (dac)'),
                patches.Patch(color='gray', label='Anthony van Tilburg (overall)')
            ]
            for i, label in enumerate(labels):
                author = label.split('_')[0]
                group = feature_df.iloc[i]['whatsapp_group']
                if author == 'Anthony van Tilburg':
                    # Check if the label is group-specific or overall
                    if group == 'overall':
                        color = group_color_map['overall']
                    else:
                        color = group_color_map.get(f"{group}_anthony", 'black')  # Use group-specific color for Anthony
                else:
                    color = group_color_map.get(group, 'black')  # Fallback for non-Anthony authors
                ax2.scatter(X_reduced[i, 0], X_reduced[i, 1], c=[color], label=None)
                # ax2.annotate(label, (X_reduced[i, 0], X_reduced[i, 1]), fontsize=8, ha='right')
            ax2.set_title(f"Interaction Dynamics by Group ({method.upper()})")
            ax2.set_xlabel('Component 1')
            ax2.set_ylabel('Component 2')
            ax2.legend(handles=legend_elements, title="Group Membership")
            plt.tight_layout()
            plt.show()

            if method.lower() == 'pca':
                loadings = pd.DataFrame(reducer.components_.T, index=feature_df.drop(['whatsapp_group'], axis=1, errors='ignore').columns, columns=['Component 1', 'Component 2'])
                logger.info(f"PCA Loadings:\n{loadings.to_string()}")

            logger.info(f"Created interaction visualizations with {method.upper()}.")
            return fig1, fig2
        except Exception as e:
            logger.exception(f"Failed to build interaction visualizations: {e}")
            return None, None

    def draw_confidence_ellipse(self, data, ax, alpha=0.60, facecolor='none', edgecolor='black', zorder=10):
        """
        Draw a confidence ellipse covering approximately 'alpha' proportion of the data points.

        Args:
            data (np.array): 2D array of points (n_samples, 2).
            ax (matplotlib.axes.Axes): Axes object to draw the ellipse on.
            alpha (float): Proportion of points to cover (e.g., 0.60 for 60%).
            facecolor (str): Fill color of the ellipse.
            edgecolor (str): Edge color of the ellipse.
            zorder (int): Drawing order of the ellipse.
        """
        if len(data) < 2:
            return
        cov = np.cov(data, rowvar=False)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        mean_x = np.mean(data[:, 0])
        mean_y = np.mean(data[:, 1])
        # Scale for alpha coverage (using chi2 for Gaussian assumption)
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        scale = np.sqrt(chi2.ppf(alpha, 2))  # For specified alpha coverage in 2D
        ellipse = Ellipse(xy=(mean_x, mean_y),
                        width=lambda_[0] * scale * 2, height=lambda_[1] * scale * 2,
                        angle=np.rad2deg(np.arccos(v[0, 0])),
                        edgecolor=edgecolor, facecolor=facecolor, alpha=0.3, zorder=zorder)
        ax.add_patch(ellipse)
        logger.debug(f"Drew {alpha*100:.0f}% confidence ellipse with center ({mean_x:.2f}, {mean_y:.2f}) and scale {scale:.2f}")

    def build_visual_not_message_content(self, feature_df, draw_ellipse=False, alpha_per_group=0.6, alpha_global=0.6, plot_type='both', groupby_period='week'):
        """
        Create multiple 2D visualizations for non-message content features:
        - Per group t-SNE and PCA, colored by author, with optional confidence ellipses.
        - Global t-SNE and PCA, colored by group, with Anthony having distinct colors per group, with optional confidence ellipses.

        Args:
            feature_df (pandas.DataFrame): Feature matrix with 'author_period_year_group' index, 'author', period, 'year', 'whatsapp_group', and features.
            draw_ellipse (bool): Whether to draw confidence ellipses for clusters.
            alpha_per_group (float): Confidence level for per-group ellipses (e.g., 0.60 for 60%).
            alpha_global (float): Confidence level for global plot ellipses (e.g., 0.60 for 60%).
            plot_type (str): Type of plots to generate ('per_group', 'global', or 'both').
            groupby_period (str): Period used for grouping ('week', 'month', or 'year').

        Returns:
            list of dict: List of {'fig': Figure, 'filename': str} for each plot, or None if creation fails.
        """
        try:
            figs = []
            unique_groups = feature_df['whatsapp_group'].unique()
            drop_columns = ['author', 'year', 'whatsapp_group']
            if groupby_period in ['week', 'month']:
                drop_columns.append(groupby_period)
            drop_columns = [col for col in drop_columns if col in feature_df.columns]
            numerical_features = feature_df.drop(drop_columns, axis=1)

            # Feature selection based on variance
            variances = numerical_features.var()
            logger.info(f"Feature variances:\n{variances.sort_values(ascending=False).to_string()}")
            top_features = variances.nlargest(15).index  # Select top 15 features by variance
            if len(top_features) < numerical_features.shape[1]:
                logger.info(f"Selected top {len(top_features)} features by variance: {list(top_features)}")
                numerical_features = numerical_features[top_features]
            else:
                logger.info("Using all features (less than 15 or equal variance)")

            # # Normalize features
            # numerical_features = StandardScaler().fit_transform(numerical_features)
            # logger.info(f"Normalized numerical features with shape {numerical_features.shape}")

            for method in ['pca', 'tsne']:
                # Reduce dimensions
                if method == 'pca':
                    reducer = PCA(n_components=2)
                elif method == 'tsne':
                    reducer = TSNE(n_components=2, perplexity=min(30, len(feature_df)-1), random_state=42)
                X_reduced = reducer.fit_transform(numerical_features)

                # Log pairwise distances to diagnose cluster overlap
                distances = pairwise_distances(X_reduced, metric='euclidean')
                logger.info(f"{method.upper()} embedding: Mean pairwise distance: {distances.mean():.2f}, Std: {distances.std():.2f}")

                # 1. Per group plots, color by author
                if plot_type in ['per_group', 'both']:
                    for group in unique_groups:
                        mask = feature_df['whatsapp_group'] == group
                        if not mask.any():
                            continue
                        X_group = X_reduced[mask]
                        group_df = feature_df[mask]
                        authors = group_df['author']
                        unique_authors = list(set(authors))
                        colors = sns.color_palette("husl", len(unique_authors))
                        author_color_map = dict(zip(unique_authors, colors))

                        fig, ax = plt.subplots(figsize=(10, 8))
                        for i in range(len(X_group)):
                            auth = authors.iloc[i]
                            ax.scatter(X_group[i, 0], X_group[i, 1], c=[author_color_map[auth]], label=auth if list(authors).index(auth) == i else None)
                        
                        # Add confidence ellipses per author if draw_ellipse is True
                        if draw_ellipse:
                            for auth in unique_authors:
                                auth_mask = group_df['author'] == auth
                                if auth_mask.sum() < 2:
                                    logger.warning(f"Skipping ellipse for {auth} in {group}: insufficient points")
                                    continue
                                auth_points = X_group[auth_mask]
                                self.draw_confidence_ellipse(auth_points, ax, alpha=alpha_per_group, facecolor=author_color_map[auth], edgecolor='black', zorder=0)

                        ax.set_title(f"Author Clustering in {group} ({method.upper()})")
                        ax.set_xlabel('Component 1', labelpad=2)
                        ax.set_ylabel('Component 2', labelpad=2)
                        ax.legend(title="Author")
                        plt.tight_layout(pad=0.5)
                        figs.append({'fig': fig, 'filename': f"not_message_content_per_group_{group}_{method}"})
                        plt.show()

                # 2. Global plot, color by group, Anthony special
                if plot_type in ['global', 'both']:
                    group_color_map = {
                        'maap': 'blue',
                        'golfmaten': 'red',
                        'dac': 'green'
                    }
                    anthony_color_map = {
                        'maap': 'lightblue',
                        'golfmaten': 'lightcoral',
                        'dac': 'lightgreen'
                    }
                    group_ellipse_color_map = {
                        'maap': 'blue',
                        'golfmaten': 'red',
                        'dac': 'green'
                    }
                    fig, ax = plt.subplots(figsize=(10, 8))
                    for i in range(len(X_reduced)):
                        group = feature_df['whatsapp_group'].iloc[i]
                        auth = feature_df['author'].iloc[i]
                        if auth == 'Anthony van Tilburg':
                            color = anthony_color_map.get(group, 'gray')
                        else:
                            color = group_color_map.get(group, 'black')
                        ax.scatter(X_reduced[i, 0], X_reduced[i, 1], c=[color], label=None, alpha=0.6)

                    # Add confidence ellipses per group if draw_ellipse is True
                    if draw_ellipse:
                        for group in unique_groups:
                            mask = feature_df['whatsapp_group'] == group
                            if not mask.any() or len(X_reduced[mask]) < 2:
                                logger.warning(f"Skipping ellipse for group {group}: insufficient points")
                                continue
                            group_points = X_reduced[mask]
                            self.draw_confidence_ellipse(group_points, ax, alpha=alpha_global, facecolor=group_ellipse_color_map[group], edgecolor='black', zorder=0)

                    # Create legend
                    legend_elements = [patches.Patch(color=v, label=k) for k, v in group_color_map.items()]
                    legend_elements += [patches.Patch(color=v, label=f"Anthony ({k})") for k, v in anthony_color_map.items()]
                    ax.set_title("Riding the Wave of WhatsApp: Group Patterns in Messaging Behavior")
                    ax.set_xlabel('Component 1', labelpad=2)
                    ax.set_ylabel('Component 2', labelpad=2)
                    ax.legend(handles=legend_elements, title="Group / Anthony")
                    plt.tight_layout(pad=0.5)
                    figs.append({'fig': fig, 'filename': f"not_message_content_global_{method}"})
                    plt.show()

            logger.info(f"Created {len(figs)} visualizations for non-message content.")
            return figs
        except Exception as e:
            logger.exception(f"Failed to build non-message content visualizations: {e}")
            return None

    def build_visual_not_message_content_2(self, feature_df):
        """
        Create multiple 2D visualizations for non-message content features:
        - Per group t-SNE and PCA, colored by author.
        - Global t-SNE and PCA, colored by group, with Anthony having distinct colors per group.

        Args:
            feature_df (pandas.DataFrame): Feature matrix with 'author_month_year_group' index, 'author', 'month', 'year', 'whatsapp_group', and features.

        Returns:
            list of dict: List of {'fig': Figure, 'filename': str} for each plot, or None if creation fails.
        """
        try:
            figs = []
            unique_groups = feature_df['whatsapp_group'].unique()
            numerical_features = feature_df.drop(['author', 'month', 'year', 'whatsapp_group'], axis=1)

            # Feature selection based on variance
            variances = numerical_features.var()
            logger.info(f"Feature variances:\n{variances.sort_values(ascending=False).to_string()}")
            top_features = variances.nlargest(10).index  # Select top 10 features by variance
            if len(top_features) < numerical_features.shape[1]:
                logger.info(f"Selected top {len(top_features)} features by variance: {list(top_features)}")
                numerical_features = numerical_features[top_features]
            else:
                logger.info("Using all features (less than 10 or equal variance)")

            # Normalize features
            numerical_features = StandardScaler().fit_transform(numerical_features)
            logger.info(f"Normalized numerical features with shape {numerical_features.shape}")

            for method in ['pca', 'tsne']:
                # Reduce dimensions
                if method == 'pca':
                    reducer = PCA(n_components=2)
                elif method == 'tsne':
                    #reducer = TSNE(n_components=2, perplexity=100, random_state=42, metric='cosine')  # Increased perplexity
                    reducer = TSNE(n_components=2, perplexity=min(30, len(feature_df)-1), random_state=42, metric='cosine')
                X_reduced = reducer.fit_transform(numerical_features)

                # Log pairwise distances to diagnose cluster overlap
                distances = pairwise_distances(X_reduced, metric='euclidean')
                logger.info(f"{method.upper()} embedding: Mean pairwise distance: {distances.mean():.2f}, Std: {distances.std():.2f}")

                # 1. Per group plots, color by author
                for group in unique_groups:
                    mask = feature_df['whatsapp_group'] == group
                    if not mask.any():
                        continue
                    X_group = X_reduced[mask]
                    group_df = feature_df[mask]
                    authors = group_df['author']
                    unique_authors = list(set(authors))
                    colors = sns.color_palette("husl", len(unique_authors))
                    author_color_map = dict(zip(unique_authors, colors))

                    fig, ax = plt.subplots(figsize=(10, 8))
                    for i in range(len(X_group)):
                        auth = authors.iloc[i]
                        ax.scatter(X_group[i, 0], X_group[i, 1], c=[author_color_map[auth]], label=auth if list(authors).index(auth) == i else None)
                    ax.set_title(f"Author Clustering in {group} ({method.upper()})")
                    ax.set_xlabel('Component 1', labelpad=2)
                    ax.set_ylabel('Component 2', labelpad=2)
                    ax.legend(title="Author")
                    plt.tight_layout(pad=0.5)
                    figs.append({'fig': fig, 'filename': f"not_message_content_per_group_{group}_{method}"})
                    plt.show()

                # 2. Global plot, color by group, Anthony special
                group_color_map = {
                    'maap': 'blue',
                    'golfmaten': 'red',
                    'dac': 'green',
                    'tillies': 'yellow'
                }
                anthony_color_map = {
                    'maap': 'lightblue',
                    'golfmaten': 'lightcoral',
                    'dac': 'lightgreen',
                    'tillies': 'lightyellow'
                }
                fig, ax = plt.subplots(figsize=(10, 8))
                for i in range(len(X_reduced)):
                    group = feature_df['whatsapp_group'].iloc[i]
                    auth = feature_df['author'].iloc[i]
                    if auth == 'Anthony van Tilburg':
                        color = anthony_color_map.get(group, 'gray')
                    else:
                        color = group_color_map.get(group, 'black')
                    ax.scatter(X_reduced[i, 0], X_reduced[i, 1], c=[color], label=None)
                # Create legend
                legend_elements = [patches.Patch(color=v, label=k) for k, v in group_color_map.items()]
                legend_elements += [patches.Patch(color=v, label=f"Anthony ({k})") for k, v in anthony_color_map.items()]
                ax.set_title(f"Global Author Clustering ({method.upper()})")
                ax.set_xlabel('Component 1', labelpad=2)
                ax.set_ylabel('Component 2', labelpad=2)
                ax.legend(handles=legend_elements, title="Group / Anthony")
                plt.tight_layout(pad=0.5)
                figs.append({'fig': fig, 'filename': f"not_message_content_global_{method}"})
                plt.show()

            logger.info(f"Created {len(figs)} visualizations for non-message content.")
            return figs
        except Exception as e:
            logger.exception(f"Failed to build non-message content visualizations: {e}")
            return None

    def plot_month_correlations(self, correlations):
        """
        Create a bar plot visualizing correlations between 'month' and numerical features.

        Args:
            correlations (pandas.Series): Series of correlation coefficients with feature names as index.

        Returns:
            matplotlib.figure.Figure or None: Figure object for the bar plot, or None if creation fails.
        """
        if correlations is None or correlations.empty:
            logger.error("No valid correlations provided for plotting")
            return None

        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(correlations.index, correlations.values, color='skyblue')
            ax.set_title("Correlation of Features with Month")
            ax.set_xlabel("Features")
            ax.set_ylabel("Pearson Correlation Coefficient")
            ax.set_ylim(-1, 1)
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)

            # Add correlation values on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height,
                        f'{height:.3f}', ha='center', va='bottom' if height >= 0 else 'top')

            # Rotate x-axis labels for readability
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

            logger.info("Created month correlations bar plot")
            return fig
        except Exception as e:
            logger.exception(f"Failed to create month correlations plot: {e}")
            return None

    def plot_feature_trends(self, feature_df, feature_name):
        """
        Create a box plot visualizing the distribution of a numerical feature by month.

        Args:
            feature_df (pandas.DataFrame): Feature DataFrame with 'month' and numerical feature columns.
            feature_name (str): Name of the numerical feature to plot.

        Returns:
            matplotlib.figure.Figure or None: Figure object for the box plot, or None if creation fails.
        """
        if feature_df is None or feature_df.empty or feature_name not in feature_df.columns:
            logger.error(f"No valid DataFrame or feature '{feature_name}' provided for trend plotting")
            return None

        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='month', y=feature_name, data=feature_df, ax=ax)
            ax.set_title(f"{feature_name} Distribution by Month")
            ax.set_xlabel("Month")
            ax.set_ylabel(feature_name)
            plt.tight_layout()
            plt.show()
            logger.info(f"Created trend box plot for {feature_name}")
            return fig
        except Exception as e:
            logger.exception(f"Failed to plot {feature_name} trends: {e}")
            return None               
                                                