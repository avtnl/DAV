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
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict, Any

# Suppress FutureWarning from seaborn/pandas
warnings.simplefilter(action="ignore", category=FutureWarning)

class PlotSettings(BaseModel):
    """Base settings for all plots."""
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    figsize: Tuple[int, int] = (10, 6)
    rotation: int = 0
    legend_title: Optional[str] = None

class ColorSettings(PlotSettings):
    """Extended settings for colored plots."""
    color_palette: str = "coolwarm"

class DimReductionSettings(BaseModel):
    """Settings for dimensionality reduction."""
    n_top_features: int = 15
    perplexity: int = 30
    metric: str = "euclidean"

class NoMessageContentSettings(ColorSettings):
    """Settings for non-message content visualizations."""
    group_color_map: Dict[str, str] = {
        'maap': 'blue',
        'golfmaten': 'red',
        'dac': 'green'
    }
    anthony_color_map: Dict[str, str] = {
        'maap': 'lightblue',
        'golfmaten': 'lightcoral',
        'dac': 'lightgreen'
    }
    draw_ellipse: bool = False
    alpha_per_group: float = 0.6
    alpha_global: float = 0.6
    plot_type: str = 'both'  # 'per_group', 'global', or 'both'

class CategoriesPlotSettings(ColorSettings):
    """Settings for categories bar chart."""
    bar_width: float = 0.4
    overall_avg_label: str = 'Overall average messages per Author'
    arrow_color: str = 'red'
    arrow_lw: int = 5
    arrow_mutation_scale: int = 20

class TimePlotSettings(PlotSettings):
    """Settings for time-based line plot."""
    vline_weeks: List[float] = [11.5, 18.5, 34.5]
    week_ticks: List[int] = [1, 5, 9, 14, 18, 23, 27, 31, 36, 40, 44, 49]
    month_labels: List[str] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    rest_label: str = "---------Rest---------"
    prep_label: str = "---Prep---"
    play_label: str = "---------Play---------"
    line_color: str = "black"
    linewidth: float = 2.5

class DistributionPlotSettings(PlotSettings):
    """Settings for emoji distribution plot."""
    bar_color: str = 'purple'
    cumulative_color: str = "orange"
    cum_threshold: int = 75
    top_n: int = 25

class BubbleNewPlotSettings(PlotSettings):
    """Settings for the new bubble plot."""
    group_colors: Dict[str, str] = Field(default_factory=lambda: {
        'maap': 'lightblue',
        'dac': 'lightgreen',
        'golfmaten': 'orange',
        'tillies': 'gray'
    })
    trendline_color: str = 'red'
    bubble_alpha: float = 0.6
    trendline_alpha: float = 0.8
    min_bubble_size: int = 50
    max_bubble_size: int = 500
    legend_scale_factor: float = 1.0 / 3.0  # Scale legend sizes to 1/3 of plot bubble sizes

class ArcPlotSettings(ColorSettings):
    amplifier: int = 3
    married_couples: List[Tuple[str, str]] = Field(default_factory=lambda: [("Anja Berkemeijer", "Phons Berkemeijer"), ("Madeleine", "Anthony van Tilburg")])
    arc_types: List[Tuple[str, str | None, float, int]] = Field(default_factory=lambda: [
        ("triple", "lightgray", 0.4, 1),
        ("pair", "gray", 0.55, 2),
        ("total", None, 0.7, 3)
    ])
    total_colors: Dict[str, str] = Field(default_factory=lambda: {"married": "red", "other": "blue"})
    special_x_offsets: Dict[Tuple[str, str, str], float] = Field(default_factory=lambda: {
        ("Anthony van Tilburg", "Phons Berkemeijer", "triple"): -0.1,
        ("Anthony van Tilburg", "Phons Berkemeijer", "pair"): -0.2
    })
    special_label_y_offsets: Dict[Tuple[str, str], float] = Field(default_factory=lambda: {
        ("Anthony van Tilburg", "Phons Berkemeijer"): -0.5
    })
    excluded_columns: List[str] = Field(default_factory=lambda: ['type', 'author', 'num_days', 'total_messages', '#participants'])
    node_size: int = 2000
    node_color: str = 'lightblue'
    node_edge_color: str = 'black'
    node_fontsize: int = 10
    node_fontweight: str = 'bold'
    label_fontsize: int = 8
    label_bbox: Dict = Field(default_factory=lambda: dict(facecolor='white', alpha=0.8, edgecolor='none'))
    title_template: str = "Messaging Interactions in {group} Group\n(Red: Married Couples, Blue: Others, Gray: Pairs, Lightgray: Triples)"

class BasePlot:
    """Base class for creating plots with common behavior."""
    def __init__(self, settings: PlotSettings):
        self.settings = settings
        self.fig = None
        self.ax = None

    def create_figure(self):
        """Create and configure figure based on settings."""
        self.fig, self.ax = plt.subplots(figsize=self.settings.figsize)
        self.ax.set_xlabel(self.settings.xlabel)
        self.ax.set_ylabel(self.settings.ylabel)
        self.ax.set_title(self.settings.title)
        if self.settings.legend_title:
            self.ax.legend(title=self.settings.legend_title)
        plt.xticks(rotation=self.settings.rotation)
        plt.tight_layout()
        return self.fig, self.ax

    def get_figure(self):
        """Return figure, creating if needed."""
        if self.fig is None:
            self.create_figure()
        return self.fig

class PlotManager:
    def __init__(self):
        # Set font to Segoe UI Emoji for emoji support
        try:
            plt.rcParams['font.family'] = 'Segoe UI Emoji'
        except:
            logger.warning("Segoe UI Emoji font not found. Falling back to default font. Some emojis may not render correctly.")
            plt.rcParams['font.family'] = 'DejaVu Sans'

    def _prepare_features(self, feature_df, groupby_period=None, settings: DimReductionSettings = DimReductionSettings()):
        """Prepare features: drop non-numeric, select top by variance, no normalization."""
        drop_columns = ['author', 'year', 'whatsapp_group']
        if groupby_period and groupby_period in ['week', 'month', 'year']:
            drop_columns.append(groupby_period)
        drop_columns = [col for col in drop_columns if col in feature_df.columns]
        numerical_features = feature_df.drop(drop_columns, axis=1)
        variances = numerical_features.var()
        logger.info(f"Feature variances:\n{variances.sort_values(ascending=False).to_string()}")
        top_features = variances.nlargest(settings.n_top_features).index
        if len(top_features) < numerical_features.shape[1]:
            logger.info(f"Selected top {len(top_features)} features by variance: {list(top_features)}")
        else:
            logger.info("Using all features")
        numerical_features = numerical_features[top_features]
        logger.info(f"Prepared numerical features (no scaling) with shape {numerical_features.shape}")
        return numerical_features.values  # Convert to NumPy array for compatibility

    def _get_reducer(self, method, n_samples, settings: DimReductionSettings = DimReductionSettings()):
        """Get reducer based on method and settings."""
        perplexity = min(settings.perplexity, n_samples - 1)
        if method == 'pca':
            return PCA(n_components=2)
        elif method == 'tsne':
            return TSNE(n_components=2, perplexity=perplexity, random_state=42, metric=settings.metric)
        else:
            raise ValueError(f"Unknown reduction method: {method}")

    def _plot_per_group(self, X_reduced, feature_df, method, settings: NoMessageContentSettings):
        """Per-group scatter plots colored by author, with optional ellipses."""
        figs = []
        unique_groups = feature_df['whatsapp_group'].unique()
        for group in unique_groups:
            mask = feature_df['whatsapp_group'] == group
            if not mask.any():
                continue
            X_group = X_reduced[mask]
            group_df = feature_df[mask]
            authors = group_df['author']
            unique_authors = list(set(authors))
            colors = sns.color_palette(settings.color_palette, len(unique_authors))
            author_color_map = dict(zip(unique_authors, colors))
            fig, ax = plt.subplots(figsize=settings.figsize)
            for i in range(len(X_group)):
                auth = authors.iloc[i]
                ax.scatter(X_group[i, 0], X_group[i, 1], c=[author_color_map[auth]], label=auth if list(authors).index(auth) == i else None)
            if settings.draw_ellipse:
                for auth in unique_authors:
                    auth_mask = group_df['author'] == auth
                    if auth_mask.sum() < 2:
                        logger.warning(f"Skipping ellipse for {auth} in {group}: insufficient points")
                        continue
                    auth_points = X_group[auth_mask]
                    self.draw_confidence_ellipse(auth_points, ax, alpha=settings.alpha_per_group, facecolor=author_color_map[auth], edgecolor='black', zorder=0)
            ax.set_title(f"Author Clustering in {group} ({method.upper()})")
            ax.set_xlabel('Component 1', labelpad=2)
            ax.set_ylabel('Component 2', labelpad=2)
            ax.legend(title="Author")
            plt.tight_layout(pad=0.5)
            figs.append({'fig': fig, 'filename': f"no_message_content_per_group_{group}_{method}"})
            plt.show()
        return figs

    def _plot_global(self, X_reduced, feature_df, method, settings: NoMessageContentSettings):
        """Global scatter plot colored by group, with Anthony special and optional ellipses."""
        fig, ax = plt.subplots(figsize=settings.figsize)
        for i in range(len(X_reduced)):
            group = feature_df['whatsapp_group'].iloc[i]
            auth = feature_df['author'].iloc[i]
            if auth == 'Anthony van Tilburg':
                color = settings.anthony_color_map.get(group, 'gray')
            else:
                color = settings.group_color_map.get(group, 'black')
            ax.scatter(X_reduced[i, 0], X_reduced[i, 1], c=[color], label=None, alpha=0.6)
        if settings.draw_ellipse:
            unique_groups = feature_df['whatsapp_group'].unique()
            for group in unique_groups:
                mask = feature_df['whatsapp_group'] == group
                if not mask.any() or len(X_reduced[mask]) < 2:
                    logger.warning(f"Skipping ellipse for group {group}: insufficient points")
                    continue
                group_points = X_reduced[mask]
                self.draw_confidence_ellipse(group_points, ax, alpha=settings.alpha_global, facecolor=settings.group_color_map.get(group, 'gray'), edgecolor='black', zorder=0)
        legend_elements = [patches.Patch(color=v, label=k) for k, v in settings.group_color_map.items()]
        legend_elements += [patches.Patch(color=v, label=f"Anthony ({k})") for k, v in settings.anthony_color_map.items()]
        ax.set_title("Riding the Wave of WhatsApp: Group Patterns in Messaging Behavior")
        ax.set_xlabel('Component 1', labelpad=2)
        ax.set_ylabel('Component 2', labelpad=2)
        ax.legend(handles=legend_elements, title="Group / Anthony")
        plt.tight_layout(pad=0.5)
        plt.show()
        return {'fig': fig, 'filename': f"no_message_content_global_{method}"}

    def build_visual_categories(self, group_authors, non_anthony_group, anthony_group, sorted_groups, settings: CategoriesPlotSettings = CategoriesPlotSettings()):
        """
        Create a bar chart comparing non-Anthony average messages and Anthony's messages.
        """
        try:
            fig, ax = plt.subplots(figsize=settings.figsize)
            positions = np.arange(len(sorted_groups))

            ax.bar(positions, non_anthony_group['non_anthony_avg'], width=settings.bar_width, color='lightgray', label='Average number of messages Non-Anthony')
            ax.bar(positions + settings.bar_width / 2, anthony_group['anthony_messages'], width=settings.bar_width, color='blue', label='Number of messages Anthony')

            overall_avg = (non_anthony_group['non_anthony_avg'] * non_anthony_group['num_authors'] + anthony_group['anthony_messages']).sum() / (non_anthony_group['num_authors'].sum() + len(sorted_groups))
            ax.axhline(y=overall_avg, color='black', linestyle='--', linewidth=1.5, label=settings.overall_avg_label)
            logger.info(f"Overall average messages across all groups: {overall_avg:.2f}")

            maap_idx = sorted_groups.index('maap') if 'maap' in sorted_groups else None
            if maap_idx is not None:
                x_pos = positions[maap_idx] + 0.75 * settings.bar_width
                y_start = non_anthony_group['non_anthony_avg'].iloc[maap_idx]
                y_end = anthony_group['anthony_messages'].iloc[maap_idx]
                ax.annotate('', xy=(x_pos, y_end), xytext=(x_pos, y_start),
                            arrowprops=dict(arrowstyle='-|>', color=settings.arrow_color, lw=settings.arrow_lw, mutation_scale=settings.arrow_mutation_scale))
                ax.annotate('', xy=(x_pos, y_start), xytext=(x_pos, y_end),
                            arrowprops=dict(arrowstyle='-|>', color=settings.arrow_color, lw=settings.arrow_lw, mutation_scale=settings.arrow_mutation_scale))
                logger.info(f"Block arrows for maap: from (x={x_pos:.2f}, y={y_start:.2f}) to (x={x_pos:.2f}, y={y_end:.2f})")

            xtick_labels = [f"{group} ({num_authors:.1f})" for group, num_authors in zip(sorted_groups, non_anthony_group['num_authors'])]
            ax.set_xticks(positions + settings.bar_width / 2)
            ax.set_xticklabels(xtick_labels)
            ax.set_xlabel(settings.xlabel)
            ax.set_ylabel(settings.ylabel)

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

    def build_visual_time(self, df: pd.DataFrame, weekly_avg: pd.DataFrame, settings: TimePlotSettings = TimePlotSettings()) -> Optional[plt.Figure]:
        """
        Create a line plot showing average message counts per week for 'dac'.
        
        Args:
            df (pd.DataFrame): Original DataFrame with 'whatsapp_group', 'week', 'year'.
            weekly_avg (pd.DataFrame): DataFrame with 'week', 'avg_messages' for 'dac'.
            settings (TimePlotSettings): Plot settings for customization.
        
        Returns:
            plt.Figure: Matplotlib figure, or None if creation fails.
        """
        if not isinstance(settings, TimePlotSettings):
            logger.warning("Settings must be an instance of TimePlotSettings. Using default.")
            settings = TimePlotSettings()
        
        try:
            if not isinstance(weekly_avg, pd.DataFrame) or weekly_avg.empty:
                logger.error("Invalid or empty weekly_avg DataFrame")
                return None
            
            # Compute period averages
            rest_period_avg = weekly_avg[(weekly_avg['week'].between(1, 14)) | (weekly_avg['week'].between(35, 53))]['avg_messages'].mean()
            prep_period_avg = weekly_avg[weekly_avg['week'].between(14, 18)]['avg_messages'].mean()  # 1st April–1st May
            play_period_avg = weekly_avg[weekly_avg['week'].between(18, 35)]['avg_messages'].mean()  # 1st May–1st Sep
            
            # Debug data
            logger.debug(f"weekly_avg range: weeks={weekly_avg['week'].min()}-{weekly_avg['week'].max()}, avg_messages={weekly_avg['avg_messages'].min()}-{weekly_avg['avg_messages'].max()}")
            logger.debug(f"Period averages: Rest={rest_period_avg:.2f}, Prep={prep_period_avg:.2f}, Play={play_period_avg:.2f}")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot weekly averages
            ax.plot(weekly_avg['week'], weekly_avg['avg_messages'], color=settings.line_color or 'green', linewidth=settings.linewidth, zorder=2)
            
            # Add vertical lines for key dates (1st April ~week 14, 1st May ~week 18, 1st Sep ~week 35)
            for week in settings.vline_weeks:
                ax.axvline(x=week, color='gray', linestyle='--', alpha=0.5, zorder=1)
            
            # Add horizontal lines for period averages
            ax.hlines(y=rest_period_avg, xmin=1, xmax=14, colors='black', linestyles='--', alpha=0.7, zorder=5)
            ax.hlines(y=rest_period_avg, xmin=35, xmax=53, colors='black', linestyles='--', alpha=0.7, zorder=5)
            ax.hlines(y=prep_period_avg, xmin=14, xmax=18, colors='black', linestyles='--', alpha=0.7, zorder=5)
            ax.hlines(y=play_period_avg, xmin=18, xmax=35, colors='black', linestyles='--', alpha=0.7, zorder=5)
            
            # Add shaded regions
            ax.axvspan(xmin=1, xmax=14, color='white', alpha=1.0, zorder=0)    # Rest: 1st Jan–1st April
            ax.axvspan(xmin=14, xmax=18, color='lightgreen', alpha=0.3, zorder=0)  # Prep: 1st April–1st May
            ax.axvspan(xmin=18, xmax=35, color='green', alpha=0.3, zorder=0)   # Play: 1st May–1st Sep
            ax.axvspan(xmin=35, xmax=53, color='white', alpha=1.0, zorder=0)   # Rest: 1st Sep–31st Dec
            
            # Add annotations
            y_min, y_max = ax.get_ylim()
            y_label = y_min + 0.9 * (y_max - y_min)
            ax.text(7, y_label, settings.rest_label, ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            ax.text(16, y_label, settings.prep_label, ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            ax.text(26.5, y_label, settings.play_label, ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
            ax.text(44, y_label, settings.rest_label, ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            # Set x-axis to key dates
            ax.set_xticks([14, 18, 35])
            ax.set_xticklabels(['1st April', '1st May', '1st Sep'], fontsize=12)
            ax.set_ylabel('Average Message Count per Week (2017–2025)', fontsize=12)
            ax.set_title('Golf Season, Decoded by WhatsApp Heartbeat', fontsize=16)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend
            ax.legend(['Weekly Avg Messages'], loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=10)
            
            plt.tight_layout(rect=[0, 0.1, 1, 0.95])
            plt.show()
            logger.info("Created time visualization for 'dac' group.")
            logger.debug(f"Plot settings: line_color={settings.line_color or 'green'}, figsize=(14, 8)")
            return fig
        except Exception as e:
            logger.exception(f"Failed to build time visualization: {e}")
            return None

    def build_visual_distribution(self, df_emojis: pd.DataFrame, settings: DistributionPlotSettings = DistributionPlotSettings()) -> Optional[plt.Figure]:
        """
        Create a bar/line plot for emoji distribution, using count_once for messages containing each emoji.
        
        Args:
            df_emojis (pandas.DataFrame): DataFrame with ['emoji', 'count_once', 'cumulative_percentage'].
            settings (DistributionPlotSettings): Plot settings (e.g., top_n, colors, threshold).
        
        Returns:
            plt.Figure: Matplotlib figure with the plot, or None if creation fails.
        """
        if not isinstance(settings, DistributionPlotSettings):
            logger.warning("Settings must be an instance of DistributionPlotSettings. Using default.")
            settings = DistributionPlotSettings()
        
        try:
            # Check for required columns
            required_cols = ['emoji', 'count_once', 'cumulative_percentage']
            if df_emojis.empty or not all(col in df_emojis.columns for col in required_cols):
                missing_cols = [col for col in required_cols if col not in df_emojis.columns]
                logger.error(f"No valid emoji DataFrame or missing columns: {missing_cols}")
                return None
            
            # Select top N based on count_once
            top_n = settings.top_n
            top_emojis = df_emojis.head(top_n)
            
            # Create table_data for logging
            table_data = [
                [str(i+1) for i in range(len(top_emojis))],
                [row['emoji'] for _, row in top_emojis.iterrows()],
                [f"{count:.0f}" for count in top_emojis['count_once']],
                [f"{cum:.1f}%" if not pd.isna(cum) else '0.0%' for cum in top_emojis['cumulative_percentage']]
            ]
            logger.debug(f"Emoji table data:\n{table_data}")
            
            # Check if count_once is all zeros
            if top_emojis['count_once'].sum() == 0:
                logger.warning("All count_once values are zero. Plot will be empty.")
            
            # Create figure with two y-axes
            fig, ax1 = plt.subplots(figsize=settings.figsize)
            
            # Bar plot for count_once
            ax1.bar(range(len(top_emojis)), top_emojis['count_once'], color=settings.bar_color)
            ax1.set_xlabel('Emoji Rank')
            ax1.set_ylabel('Messages Containing Emoji', color=settings.bar_color)
            ax1.tick_params(axis='y', labelcolor=settings.bar_color)
            
            # Line plot for cumulative percentage
            ax2 = ax1.twinx()
            ax2.plot(range(len(top_emojis)), top_emojis['cumulative_percentage'], color=settings.cumulative_color, marker='o')
            ax2.set_ylabel('Cumulative Percentage (%)', color=settings.cumulative_color)
            ax2.tick_params(axis='y', labelcolor=settings.cumulative_color)
            
            # Set x-ticks to emojis
            ax1.set_xticks(range(len(top_emojis)))
            ax1.set_xticklabels(top_emojis['emoji'], rotation=settings.rotation)
            
            # Set title
            fig.suptitle(settings.title or "Emoji Distribution in 'maap' Group")
            
            # Threshold line
            threshold_rank = top_emojis[top_emojis['cumulative_percentage'] >= settings.cum_threshold].index.min()
            if pd.notna(threshold_rank):
                threshold_index = top_emojis.index.get_loc(threshold_rank)
                ax1.axvline(x=threshold_index + 0.5, color='red', linestyle='--')
                ax1.text(threshold_index + 1, max(top_emojis['count_once'], default=1) * 0.9,
                        f'{settings.cum_threshold}% at rank {threshold_index + 1}', color='red')
            
            plt.tight_layout()
            plt.show()
            logger.info("Created emoji distribution visualization for 'maap' group.")
            return fig
        except Exception as e:
            logger.exception(f"Failed to build emoji distribution visualization: {e}")
            return None

    def build_visual_relationships_arc(self, combined_df, group, settings: ArcPlotSettings = ArcPlotSettings()):
        """Arc diagram for relationships."""
        if combined_df is None or combined_df.empty:
            logger.error("No valid DataFrame provided for building visual relationships_4 plot.")
            return None
        try:
            # Extract authors
            participant_cols = [col for col in combined_df.columns if col not in settings.excluded_columns]
            authors = sorted(set(participant_cols))
            # Create figure
            fig, ax = plt.subplots(figsize=settings.figsize)
            ax.set_aspect('equal')
            # Position authors around a circle
            n = len(authors)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            radius = 1.0
            pos = {author: (radius * np.cos(angle), radius * np.sin(angle)) for author, angle in zip(authors, angles)}
            # Initialize weights
            pair_weights = {}
            triple_weights = {}
            total_weights = {}
            # Process Pairs
            pairs_df = combined_df[combined_df['type'] == 'Pairs']
            for _, row in pairs_df.iterrows():
                a1, a2 = [a.strip() for a in row['author'].split(' & ')]
                key = frozenset([a1, a2])
                weight = row['total_messages']
                pair_weights[key] = weight
                total_weights[key] = total_weights.get(key, 0) + weight
            # Process Non-participant (triples)
            triples_df = combined_df[combined_df['type'] == 'Non-participant']
            for _, row in triples_df.iterrows():
                participants = [col for col in participant_cols if row[col] != 0]
                if len(participants) != len(authors) - 1:
                    logger.warning(f"Unexpected number of participants in row: {len(participants)}. Skipping.")
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
            # Get max weight for scaling linewidths
            max_weight = max(total_weights.values(), default=1)
            # Weights dict
            weights_dict = {'pair': pair_weights, 'triple': triple_weights, 'total': total_weights}
            # Draw arcs
            for arc_type, color, height_offset, zorder in settings.arc_types:
                weights = weights_dict.get(arc_type, {})
                for key, weight in weights.items():
                    a1, a2 = list(key)
                    x1, y1 = pos[a1]
                    x2, y2 = pos[a2]
                    xm = (x1 + x2) / 2
                    ym = (y1 + y2) / 2
                    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    height = dist * height_offset
                    # Set color for total
                    if arc_type == 'total':
                        pair_tuple = (a1, a2) if a1 < a2 else (a2, a1)
                        is_married = pair_tuple in settings.married_couples or (a2, a1) in settings.married_couples
                        color = settings.total_colors['married'] if is_married else settings.total_colors['other']
                    # X offset
                    sorted_pair = tuple(sorted([a1, a2]))
                    offset_key = (*sorted_pair, arc_type)
                    x_offset = settings.special_x_offsets.get(offset_key, 0)
                    # Scale line width
                    width = (1 + 5 * (weight / max_weight)) * settings.amplifier
                    # Draw arc
                    t = np.linspace(0, 1, 100)
                    x = (1 - t)**2 * x1 + 2 * (1 - t) * t * (xm + x_offset) + t**2 * x2
                    y = (1 - t)**2 * y1 + 2 * (1 - t) * t * (ym + height) + t**2 * y2
                    ax.plot(x, y, color=color, linewidth=width, zorder=zorder)
                    # Add label only for total arcs
                    if arc_type == 'total':
                        label_x = (x1 + x2) / 2
                        label_y = (y1 + y2) / 2 + height * 0.5
                        label_offset_key = tuple(sorted([a1, a2]))
                        label_y += settings.special_label_y_offsets.get(label_offset_key, 0)
                        ax.text(label_x, label_y, f"{int(round(weight))}", ha='center', va='center', fontsize=settings.label_fontsize,
                                bbox=settings.label_bbox, zorder=zorder + 1)
            # Draw nodes
            for author, (x, y) in pos.items():
                ax.scatter([x], [y], s=settings.node_size, color=settings.node_color, edgecolors=settings.node_edge_color, zorder=4)
                ax.text(x, y, author, ha='center', va='center', fontsize=settings.node_fontsize, fontweight=settings.node_fontweight, zorder=5)
            ax.set_title(settings.title_template.format(group=group))
            ax.axis('off')
            plt.tight_layout()
            plt.show()
            return fig
        except Exception as e:
            logger.exception(f"Failed to build arc diagram: {e}")
            return None

    def build_visual_relationships_bubble(self, feature_df, settings: BubbleNewPlotSettings = BubbleNewPlotSettings()):
        """
        Create a bubble plot of average words vs average punctuation, with bubble size as message count,
        one bubble per author per WhatsApp group, and a single red trendline. Legend bubble sizes are scaled
        to match the original scale, while plot bubbles are three times larger.
        
        Args:
            feature_df (pandas.DataFrame): DataFrame with 'whatsapp_group', 'author', 'avg_words', 'avg_punct', 'message_count' columns.
            settings (BubbleNewPlotSettings): Plot settings including group colors and trendline style.
        
        Returns:
            matplotlib.figure.Figure or None: Figure object for the bubble plot, or None if creation fails.
        """
        if feature_df is None or feature_df.empty:
            logger.error("No valid DataFrame provided for building bubble plot.")
            return None
        try:
            # Check required columns with aliases
            required_cols = {
                'whatsapp_group': 'whatsapp_group',
                'author': 'author',
                'avg_words': ['avg_words'],
                'avg_punct': ['avg_punct'],
                'message_count': ['message_count']
            }
            missing_cols = []
            for expected, aliases in required_cols.items():
                if isinstance(aliases, list):
                    if not any(alias in feature_df.columns for alias in aliases):
                        missing_cols.append(expected)
                elif aliases not in feature_df.columns:
                    missing_cols.append(expected)
            if missing_cols:
                logger.error(f"Missing required columns. Expected {list(required_cols.keys())}, found {feature_df.columns.tolist()}")
                return None

            # Map aliases to actual columns
            x_col = next(col for col in required_cols['avg_words'] if col in feature_df.columns)
            y_col = next(col for col in required_cols['avg_punct'] if col in feature_df.columns)
            size_col = next(col for col in required_cols['message_count'] if col in feature_df.columns)

            # Create figure
            fig, ax = plt.subplots(figsize=settings.figsize)
            
            # Normalize bubble sizes based on message_count and scale by 3 for plot
            min_size, max_size = settings.min_bubble_size, settings.max_bubble_size
            message_counts = feature_df[size_col]
            size_scale = (message_counts - message_counts.min()) / (message_counts.max() - message_counts.min()) if message_counts.max() != message_counts.min() else 1.0
            bubble_sizes = (min_size + (max_size - min_size) * size_scale) * 3  # Three times larger for plot

            # Plot bubbles for each author per group
            for group in settings.group_colors.keys():
                group_df = feature_df[feature_df['whatsapp_group'] == group]
                if not group_df.empty:
                    scatter = ax.scatter(group_df[x_col], group_df[y_col],
                                        s=bubble_sizes[group_df.index], alpha=settings.bubble_alpha,
                                        color=settings.group_colors[group], label=group)

            # Add trendline (linear fit) across all data
            x = feature_df[x_col]
            y = feature_df[y_col]
            coefficients = np.polyfit(x, y, 1)  # Linear fit (degree 1)
            trendline = np.poly1d(coefficients)
            ax.plot(x, trendline(x), color=settings.trendline_color, alpha=settings.trendline_alpha)

            # Customize plot and legend
            ax.set_title(settings.title or "Average Words vs Punctuation by Author per Group")
            ax.set_xlabel(settings.xlabel or "Average Words per Message")
            ax.set_ylabel(settings.ylabel or "Average Punctuation per Message")
            # Create legend with scaled sizes
            legend_handles = [plt.scatter([], [], s=min_size * settings.legend_scale_factor, c=color, alpha=settings.bubble_alpha, label=group)
                            for group, color in settings.group_colors.items()]
            ax.legend(handles=legend_handles, title="WhatsApp Group", bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.show()
            return fig
        except Exception as e:
            logger.exception(f"Failed to build bubble plot: {e}")
            return None

    def draw_confidence_ellipse(self, data, ax, alpha=0.95, facecolor='none', edgecolor='black', zorder=0):
        """Draw confidence ellipse (unchanged, as it's already abstract)."""
        if len(data) < 2:
            return
        cov = np.cov(data, rowvar=False)
        mean_x = np.mean(data[:, 0])
        mean_y = np.mean(data[:, 1])
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        scale = np.sqrt(chi2.ppf(alpha, 2))
        ellipse = Ellipse(xy=(mean_x, mean_y),
                          width=lambda_[0] * scale * 2, height=lambda_[1] * scale * 2,
                          angle=np.rad2deg(np.arccos(v[0, 0])),
                          edgecolor=edgecolor, facecolor=facecolor, alpha=0.3, zorder=zorder)
        ax.add_patch(ellipse)
        logger.debug(f"Drew {alpha*100:.0f}% confidence ellipse with center ({mean_x:.2f}, {mean_y:.2f}) and scale {scale:.2f}")

    def build_visual_no_message_content(self, feature_df, plot_type: str = 'both', dr_settings: DimReductionSettings = DimReductionSettings(), nmc_settings: NoMessageContentSettings = NoMessageContentSettings(), settings: Optional[PlotSettings] = None):
        """
        Non-message content visualizations using configs.

        Args:
            feature_df (pandas.DataFrame): Feature matrix with relevant columns.
            plot_type (str): Dimensionality reduction method ('both', 'pca', or 'tsne').
            dr_settings (DimReductionSettings): Dimensionality reduction settings.
            nmc_settings (NoMessageContentSettings): Settings for group and Anthony color maps.
            settings (Optional[PlotSettings]): Legacy settings for backward compatibility (optional).

        Returns:
            list: List of dictionaries containing figures and filenames, or None if creation fails.
        """
        if settings is not None and not isinstance(settings, (DimReductionSettings, NoMessageContentSettings)):
            logger.warning("Received legacy 'settings' parameter. Using default dr_settings and nmc_settings instead. Update caller to use dr_settings and nmc_settings.")
            dr_settings = DimReductionSettings()
            nmc_settings = NoMessageContentSettings()

        # Validate plot_type
        if plot_type not in ['both', 'pca', 'tsne']:
            logger.error(f"Invalid plot_type: {plot_type}. Must be 'both', 'pca', or 'tsne'.")
            return None

        try:
            figs = []
            numerical_features = self._prepare_features(feature_df, settings=dr_settings)
            # Determine methods based on plot_type
            methods = ['pca', 'tsne'] if plot_type == 'both' else [plot_type]
            for method in methods:
                reducer = self._get_reducer(method, len(feature_df), dr_settings)
                X_reduced = reducer.fit_transform(numerical_features)
                distances = pairwise_distances(X_reduced, metric=dr_settings.metric)
                logger.info(f"{method.upper()} embedding: Mean pairwise distance: {distances.mean():.2f}, Std: {distances.std():.2f}")
                if nmc_settings.plot_type in ['per_group', 'both']:
                    figs.extend(self._plot_per_group(X_reduced, feature_df, method, nmc_settings))
                if nmc_settings.plot_type in ['global', 'both']:
                    figs.append(self._plot_global(X_reduced, feature_df, method, nmc_settings))
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

    def plot_feature_trends(self, feature_df, feature_name, settings: PlotSettings = PlotSettings()):
        """Feature trends box plot using config."""
        if feature_df is None or feature_df.empty or feature_name not in feature_df.columns:
            logger.error(f"No valid DataFrame or feature '{feature_name}' provided for trend plotting")
            return None
        try:
            fig, ax = plt.subplots(figsize=settings.figsize)
            sns.boxplot(x='month', y=feature_name, data=feature_df, ax=ax)
            ax.set_title(settings.title)
            ax.set_xlabel(settings.xlabel)
            ax.set_ylabel(settings.ylabel)
            plt.tight_layout()
            plt.show()
            logger.info(f"Created trend box plot for {feature_name}")
            return fig
        except Exception as e:
            logger.exception(f"Failed to plot {feature_name} trends: {e}")
            return None

    def build_visual_interactions(self, feature_df, method='tsne', settings: DimReductionSettings = DimReductionSettings(), nmc_settings: NoMessageContentSettings = NoMessageContentSettings()):
        """
        Specialized 2D visualization for interaction features using PCA or t-SNE.
        Colors by author for evolution over years.
    
        Args:
            feature_df (pandas.DataFrame): Feature matrix with 'author_year' index.
            method (str): 'pca' or 'tsne'.
            settings (DimensionalityReductionSettings): Dimensionality reduction settings.
    
        Returns:
            matplotlib.figure.Figure or None: The plot figure.
        """
        if not isinstance(settings, DimReductionSettings):
            logger.warning("Settings must be an instance of DimReductionSettings. Using default DimReductionSettings.")
            settings = DimReductionSettings()
        
        try:
            labels = feature_df.index.values
            # Drop non-numeric columns to ensure only numeric data is used for reduction
            X = feature_df.drop(['whatsapp_group'], axis=1, errors='ignore').values
            reducer = self._get_reducer(method, len(labels), settings)
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

    def build_visual_interactions_2(self, feature_df, method='tsne', settings: DimReductionSettings = DimReductionSettings(), nmc_settings: NoMessageContentSettings = NoMessageContentSettings()):
        """
        Create two 2D visualizations for interaction features using PCA or t-SNE.
        First plot: Colors by author, with 'Anthony van Tilburg' points for each group-year and overall.
        Second plot: Colors by group ('maap': blue, 'golfmaten': red, 'dac': green) and
        'Anthony van Tilburg' points by group (light blue for 'maap', light red for 'golfmaten',
        light green for 'dac', gray for overall).
    
        Args:
            feature_df (pandas.DataFrame): Feature matrix with 'author_year' or 'author_year_group' index and 'whatsapp_group' column.
            method (str): 'pca' or 'tsne'.
            settings (DimReductionSettings): Dimensionality reduction settings.
            nmc_settings (NoMessageContentSettings): Settings for group and Anthony color maps.
    
        Returns:
            tuple: (matplotlib.figure.Figure, matplotlib.figure.Figure) or (None, None) if creation fails.
        """
        if not isinstance(settings, DimReductionSettings):
            logger.warning("Settings must be an instance of DimReductionSettings. Using default DimReductionSettings.")
            settings = DimReductionSettings()
        
        try:
            labels = feature_df.index.values
            X = feature_df.drop(['whatsapp_group'], axis=1, errors='ignore').values
            reducer = self._get_reducer(method, len(labels), settings)
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
            group_color_map = nmc_settings.group_color_map
            anthony_color_map = nmc_settings.anthony_color_map
            legend_elements = [
                patches.Patch(color=v, label=k) for k, v in group_color_map.items()
            ]
            legend_elements += [
                patches.Patch(color=v, label=f"Anthony van Tilburg ({k})") for k, v in anthony_color_map.items()
            ]
            legend_elements.append(patches.Patch(color='gray', label='Anthony van Tilburg (overall)'))
            for i, label in enumerate(labels):
                author = label.split('_')[0]
                group = feature_df.iloc[i]['whatsapp_group']
                if author == 'Anthony van Tilburg':
                    # Check if the label is group-specific or overall
                    if group == 'overall':
                        color = 'gray'
                    else:
                        color = anthony_color_map.get(group, 'black')  # Use group-specific color for Anthony
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