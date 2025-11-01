# === Module Docstring ===
"""
Plot manager for the WhatsApp Chat Analyzer.

Creates all visualisations used in the analysis pipeline:
* Bar charts (categories)
* Time-series line plots
* Emoji distribution plots
* Arc diagrams of relationships
* Bubble plots (words vs punctuation)
* Dimensionality-reduction scatter plots (PCA / t-SNE)

All column references are resolved through :class:`constants.Columns`.
All UI strings (titles, axis labels, legends) are derived from the
``.human`` property of the same enum – guaranteeing a single source of
truth for both data access and presentation.

Examples
--------
>>> from plot_manager import PlotManager
>>> pm = PlotManager()
>>> fig = pm.build_visual_categories(...)
"""

# === Imports ===
import itertools
import warnings
from typing import List, Dict, Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib import patches
from matplotlib.patches import Ellipse
from pydantic import BaseModel, Field
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

from constants import Columns, Groups, PlotType, GroupByPeriod, PlotFeed

# Suppress FutureWarning from seaborn/pandas
warnings.simplefilter(action="ignore", category=FutureWarning)


# === Settings Models ===
class PlotSettings(BaseModel):
    """Base settings for every plot (figure size, titles, axis labels)."""

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    figsize: Tuple[int, int] = (10, 6)
    rotation: int = 0
    legend_title: Optional[str] = None


class ColorSettings(PlotSettings):
    """Adds a colour palette to the base settings."""

    color_palette: str = "coolwarm"


class DimReductionSettings(BaseModel):
    """Hyper-parameters for dimensionality reduction."""

    n_top_features: int = 15
    perplexity: int = 30
    metric: str = "euclidean"


class PMNoMessageContentSettings(ColorSettings):
    """Settings for non-message-content scatter plots."""

    group_color_map: Dict[str, str] = {
        Groups.MAAP.value: "blue",
        Groups.GOLFMATEN.value: "red",
        Groups.DAC.value: "green",
    }
    anthony_color_map: Dict[str, str] = {
        Groups.MAAP.value: "lightblue",
        Groups.GOLFMATEN.value: "lightcoral",
        Groups.DAC.value: "lightgreen",
    }
    draw_ellipse: bool = False
    alpha_per_group: float = 0.6
    alpha_global: float = 0.6
    plot_type: str = PlotFeed.BOTH.value  # 'per_group', 'global', or 'both'


class CategoriesPlotSettings(ColorSettings):
    """Bar-chart settings for the categories visualisation."""

    bar_width: float = 0.4
    overall_avg_label: str = "Overall average messages per Author"
    arrow_color: str = "red"
    arrow_lw: int = 5
    arrow_mutation_scale: int = 20


class TimePlotSettings(PlotSettings):
    """Line-plot settings for the weekly activity visualisation."""

    vline_weeks: List[float] = [11.5, 18.5, 34.5]
    week_ticks: List[int] = [1, 5, 9, 14, 18, 23, 27, 31, 36, 40, 44, 49]
    month_labels: List[str] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    rest_label: str = "---------Rest---------"
    prep_label: str = "---Prep---"
    play_label: str = "---------Play---------"
    line_color: str = "black"
    linewidth: float = 2.5


class DistributionPlotSettings(PlotSettings):
    """Settings for the emoji distribution bar + cumulative line plot."""

    bar_color: str = "purple"
    cumulative_color: str = "orange"
    cum_threshold: int = 75
    top_n: int = 25


class BubbleNewPlotSettings(PlotSettings):
    """Settings for the new bubble plot (words vs punctuation)."""

    group_colors: Dict[str, str] = Field(
        default_factory=lambda: {
            Groups.MAAP.value: "lightblue",
            Groups.DAC.value: "lightgreen",
            Groups.GOLFMATEN.value: "orange",
            Groups.TILLIES.value: "gray",
        }
    )
    trendline_color: str = "red"
    bubble_alpha: float = 0.6
    trendline_alpha: float = 0.8
    min_bubble_size: int = 50
    max_bubble_size: int = 500
    legend_scale_factor: float = 1.0 / 3.0  # Legend bubbles = 1/3 of plot bubbles


class ArcPlotSettings(ColorSettings):
    """Arc-diagram settings for relationship visualisation."""

    amplifier: int = 3
    married_couples: List[Tuple[str, str]] = Field(
        default_factory=lambda: [
            ("Anja Berkemeijer", "Phons Berkemeijer"),
            ("Madeleine", "Anthony van Tilburg"),
        ]
    )
    arc_types: List[Tuple[str, Optional[str], float, int]] = Field(
        default_factory=lambda: [
            ("triple", "lightgray", 0.4, 1),
            ("pair", "gray", 0.55, 2),
            ("total", None, 0.7, 3),
        ]
    )
    total_colors: Dict[str, str] = Field(
        default_factory=lambda: {"married": "red", "other": "blue"}
    )
    special_x_offsets: Dict[Tuple[str, str, str], float] = Field(
        default_factory=lambda: {
            ("Anthony van Tilburg", "Phons Berkemeijer", "triple"): -0.1,
            ("Anthony van Tilburg", "Phons Berkemeijer", "pair"): -0.2,
        }
    )
    special_label_y_offsets: Dict[Tuple[str, str], float] = Field(
        default_factory=lambda: {("Anthony van Tilburg", "Phons Berkemeijer"): -0.5}
    )
    excluded_columns: List[str] = Field(
        default_factory=lambda: [
            "type",
            Columns.AUTHOR.value,
            "num_days",
            "total_messages",
            "#participants",
        ]
    )
    node_size: int = 2000
    node_color: str = "lightblue"
    node_edge_color: str = "black"
    node_fontsize: int = 10
    node_fontweight: str = "bold"
    label_fontsize: int = 8
    label_bbox: Dict[str, Any] = Field(
        default_factory=lambda: {
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": "none",
        }
    )
    title_template: str = "Messaging Interactions in {group} Group\n(Red: Married Couples, Blue: Others, Gray: Pairs, Lightgray: Triples)"


# === Helper Classes ===
class BasePlot:
    """Thin wrapper that creates a Matplotlib figure with common settings."""

    def __init__(self, settings: PlotSettings) -> None:
        self.settings = settings
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None

    def create_figure(self) -> Tuple[plt.Figure, plt.Axes]:
        """Create the figure and axes, apply title / labels."""
        self.fig, self.ax = plt.subplots(figsize=self.settings.figsize)
        self.ax.set_xlabel(self.settings.xlabel)
        self.ax.set_ylabel(self.settings.ylabel)
        self.ax.set_title(self.settings.title)
        if self.settings.legend_title:
            self.ax.legend(title=self.settings.legend_title)
        plt.xticks(rotation=self.settings.rotation)
        plt.tight_layout()
        return self.fig, self.ax

    def get_figure(self) -> plt.Figure:
        """Return the figure, creating it if necessary."""
        if self.fig is None:
            self.create_figure()
        return self.fig  # type: ignore[return-value]


# === Main Plot Manager ===
class PlotManager:
    """Orchestrates every visualisation defined in the analysis pipeline."""

    def __init__(self) -> None:
        # Set emoji-capable font
        try:
            plt.rcParams["font.family"] = "Segoe UI Emoji"
        except Exception:  # pragma: no cover
            logger.warning(
                "Segoe UI Emoji not available – falling back to DejaVu Sans."
            )
            plt.rcParams["font.family"] = "DejaVu Sans"

    # --------------------------------------------------------------------- #
    # === Private helpers – feature preparation & reduction ===
    # --------------------------------------------------------------------- #
    def _prepare_features(
        self,
        feature_df: pd.DataFrame,
        groupby_period: Optional[str] = None,
        settings: DimReductionSettings = DimReductionSettings(),
    ) -> np.ndarray:
        """
        Drop identifier columns, keep top-variance numeric features.

        Args:
            feature_df: Input DataFrame with raw features.
            groupby_period: Optional ``week`` / ``month`` / ``year`` column to drop.
            settings: Dimensionality-reduction hyper-parameters.

        Returns:
            Numpy array of selected numeric features (no scaling).
        """
        drop_columns = [
            Columns.AUTHOR.value,
            Columns.YEAR.value,
            Columns.WHATSAPP_GROUP.value,
        ]
        if groupby_period in {
            GroupByPeriod.WEEK.value,
            GroupByPeriod.MONTH.value,
            GroupByPeriod.YEAR.value,
        }:
            drop_columns.append(groupby_period)

        drop_columns = [c for c in drop_columns if c in feature_df.columns]
        numerical_features = feature_df.drop(columns=drop_columns)

        variances = numerical_features.var()
        logger.info(
            f"Feature variances:\n{variances.sort_values(ascending=False).to_string()}"
        )

        top_features = variances.nlargest(settings.n_top_features).index
        if len(top_features) < numerical_features.shape[1]:
            logger.info(
                f"Selected top {len(top_features)} features: {list(top_features)}"
            )
        else:
            logger.info("Using all numeric features")

        numerical_features = numerical_features[top_features]
        logger.info(
            f"Prepared numerical features (shape {numerical_features.shape})"
        )
        return numerical_features.values  # type: ignore[no-any-return]

    def _get_reducer(
        self,
        method: str,
        n_samples: int,
        settings: DimReductionSettings = DimReductionSettings(),
    ):
        """
        Instantiate PCA or t-SNE reducer with safe perplexity.

        Args:
            method: ``pca`` or ``tsne``.
            n_samples: Number of observations.
            settings: Reduction hyper-parameters.

        Returns:
            Fitted reducer instance.
        """
        perplexity = min(settings.perplexity, n_samples - 1)
        if method == PlotType.PCA.value:
            return PCA(n_components=2)
        if method == PlotType.TSNE.value:
            return TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42,
                metric=settings.metric,
            )
        raise ValueError(f"Unknown reduction method: {method}")

    # --------------------------------------------------------------------- #
    # === Private scatter-plot helpers (per-group / global) ===
    # --------------------------------------------------------------------- #
    def _plot_per_group(
        self,
        X_reduced: np.ndarray,
        feature_df: pd.DataFrame,
        method: str,
        settings: PMNoMessageContentSettings,
    ) -> List[Dict[str, Any]]:
        """
        Scatter points per WhatsApp group, colour by author, optional ellipses.

        Returns:
            List of dicts ``{'fig': fig, 'filename': name}``.
        """
        figs: List[Dict[str, Any]] = []
        for group in feature_df[Columns.WHATSAPP_GROUP.value].unique():
            mask = feature_df[Columns.WHATSAPP_GROUP.value] == group
            if not mask.any():
                continue

            X_group = X_reduced[mask]
            group_df = feature_df[mask]
            authors = group_df[Columns.AUTHOR.value]
            unique_authors = list(set(authors))
            palette = sns.color_palette(settings.color_palette, len(unique_authors))
            author_color_map = dict(zip(unique_authors, palette, strict=False))

            fig, ax = plt.subplots(figsize=settings.figsize)
            for i, auth in enumerate(authors):
                ax.scatter(
                    X_group[i, 0],
                    X_group[i, 1],
                    c=[author_color_map[auth]],
                    label=auth if authors.tolist().index(auth) == i else None,
                )

            if settings.draw_ellipse:
                for auth in unique_authors:
                    auth_mask = group_df[Columns.AUTHOR.value] == auth
                    if auth_mask.sum() < 2:
                        logger.warning(
                            f"Ellipse skipped for {auth} in {group}: <2 points"
                        )
                        continue
                    auth_points = X_group[auth_mask]
                    self.draw_confidence_ellipse(
                        auth_points,
                        ax,
                        alpha=settings.alpha_per_group,
                        facecolor=author_color_map[auth],
                        edgecolor="black",
                        zorder=0,
                    )

            ax.set_title(f"Author Clustering in {group} ({method.upper()})")
            ax.set_xlabel("Component 1", labelpad=2)
            ax.set_ylabel("Component 2", labelpad=2)
            ax.legend(title="Author")
            plt.tight_layout(pad=0.5)
            figs.append(
                {
                    "fig": fig,
                    "filename": f"no_message_content_per_group_{group}_{method}",
                }
            )
            plt.show()
        return figs

    def _plot_global(
        self,
        X_reduced: np.ndarray,
        feature_df: pd.DataFrame,
        method: str,
        settings: PMNoMessageContentSettings,
    ) -> Dict[str, Any]:
        """
        Global scatter coloured by group (Anthony special handling).

        Returns:
            Dict ``{'fig': fig, 'filename': name}``.
        """
        fig, ax = plt.subplots(figsize=settings.figsize)

        for i in range(len(X_reduced)):
            group = feature_df.iloc[i][Columns.WHATSAPP_GROUP.value]
            auth = feature_df.iloc[i][Columns.AUTHOR.value]

            if auth == "Anthony van Tilburg":
                color = settings.anthony_color_map.get(group, "gray")
            else:
                color = settings.group_color_map.get(group, "black")

            ax.scatter(
                X_reduced[i, 0], X_reduced[i, 1], c=[color], label=None, alpha=0.6
            )

        if settings.draw_ellipse:
            for group in feature_df[Columns.WHATSAPP_GROUP.value].unique():
                mask = feature_df[Columns.WHATSAPP_GROUP.value] == group
                if not mask.any() or len(X_reduced[mask]) < 2:
                    logger.warning(f"Ellipse skipped for group {group}: <2 points")
                    continue
                group_points = X_reduced[mask]
                self.draw_confidence_ellipse(
                    group_points,
                    ax,
                    alpha=settings.alpha_global,
                    facecolor=settings.group_color_map.get(group, "gray"),
                    edgecolor="black",
                    zorder=0,
                )

        legend_elements = [
            patches.Patch(color=v, label=k) for k, v in settings.group_color_map.items()
        ]
        legend_elements += [
            patches.Patch(color=v, label=f"Anthony ({k})")
            for k, v in settings.anthony_color_map.items()
        ]

        ax.set_title(
            "Riding the Wave of WhatsApp: Group Patterns in Messaging Behavior"
        )
        ax.set_xlabel("Component 1", labelpad=2)
        ax.set_ylabel("Component 2", labelpad=2)
        ax.legend(handles=legend_elements, title="Group / Anthony")
        plt.tight_layout(pad=0.5)
        plt.show()

        return {"fig": fig, "filename": f"no_message_content_global_{method}"}

    # --------------------------------------------------------------------- #
    # === Public visualisation builders ===
    # --------------------------------------------------------------------- #
    def build_visual_categories(
        self,
        group_authors: pd.DataFrame,
        non_anthony_group: pd.DataFrame,
        anthony_group: pd.DataFrame,
        sorted_groups: List[str],
        settings: CategoriesPlotSettings = CategoriesPlotSettings(),
    ) -> Optional[plt.Figure]:
        """
        Bar chart comparing non-Anthony average vs Anthony messages per group.

        Args:
            group_authors: Unused (kept for API compatibility).
            non_anthony_group: DataFrame with ``non_anthony_avg`` and ``num_authors``.
            anthony_group: DataFrame with ``anthony_messages``.
            sorted_groups: Ordered list of group identifiers.
            settings: Visual styling.

        Returns:
            Matplotlib figure or ``None`` on error.
        """
        try:
            fig, ax = plt.subplots(figsize=settings.figsize)
            positions = np.arange(len(sorted_groups))

            # Non-Anthony bars
            ax.bar(
                positions,
                non_anthony_group["non_anthony_avg"],
                width=settings.bar_width,
                color="lightgray",
                label="Average messages (non-Anthony)",
            )
            # Anthony bars
            ax.bar(
                positions + settings.bar_width / 2,
                anthony_group["anthony_messages"],
                width=settings.bar_width,
                color="blue",
                label="Anthony messages",
            )

            # Overall average line
            overall_avg = (
                non_anthony_group["non_anthony_avg"]
                * non_anthony_group["num_authors"]
                + anthony_group["anthony_messages"]
            ).sum() / (non_anthony_group["num_authors"].sum() + len(sorted_groups))
            ax.axhline(
                y=overall_avg,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=settings.overall_avg_label,
            )
            logger.info(f"Overall average messages: {overall_avg:.2f}")

            # Special arrow for the 'maap' group
            if Groups.MAAP.value in sorted_groups:
                maap_idx = sorted_groups.index(Groups.MAAP.value)
                x_pos = positions[maap_idx] + 0.75 * settings.bar_width
                y_start = non_anthony_group["non_anthony_avg"].iloc[maap_idx]
                y_end = anthony_group["anthony_messages"].iloc[maap_idx]

                for y_from, y_to in [(y_start, y_end), (y_end, y_start)]:
                    ax.annotate(
                        "",
                        xy=(x_pos, y_to),
                        xytext=(x_pos, y_from),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=settings.arrow_color,
                            lw=settings.arrow_lw,
                            mutation_scale=settings.arrow_mutation_scale,
                        ),
                    )

            # X-axis labels with author count
            xtick_labels = [
                f"{g} ({n:.0f})"
                for g, n in zip(
                    sorted_groups, non_anthony_group["num_authors"], strict=False
                )
            ]
            ax.set_xticks(positions + settings.bar_width / 2)
            ax.set_xticklabels(xtick_labels)
            ax.set_xlabel(settings.xlabel or "WhatsApp Group")
            ax.set_ylabel(settings.ylabel or "Messages")

            # Subtitle annotations
            ax.text(
                0.5,
                1.08,
                "Too much to handle or too much crap?",
                fontsize=16,
                ha="center",
                va="bottom",
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                1.02,
                "Anthony's participation is significantly lower for the first group",
                fontsize=12,
                ha="center",
                va="bottom",
                transform=ax.transAxes,
            )

            ax.legend()
            plt.tight_layout()
            plt.show()
            return fig
        except Exception as e:  # pragma: no cover
            logger.exception(f"Categories plot failed: {e}")
            return None

    def build_visual_time(
        self,
        p: pd.DataFrame,
        average_all: pd.DataFrame,
        settings: TimePlotSettings = TimePlotSettings(),
    ) -> Optional[plt.Figure]:
        """
        Weekly activity line plot with shaded periods (rest / prep / play).

        Args:
            p: Unused (kept for API compatibility).
            average_all: DataFrame with ``isoweek`` and ``avg_count_all``.
            settings: Styling.

        Returns:
            Matplotlib figure or ``None`` on error.
        """
        try:
            # Pre-compute period averages
            weeks_rest = (
                average_all["isoweek"].between(1, 12)
                | average_all["isoweek"].between(35, 53)
            )
            weeks_prep = average_all["isoweek"].between(12, 19)
            weeks_play = average_all["isoweek"].between(19, 35)

            avg_rest = average_all.loc[weeks_rest, "avg_count_all"].mean()
            avg_prep = average_all.loc[weeks_prep, "avg_count_all"].mean()
            avg_play = average_all.loc[weeks_play, "avg_count_all"].mean()

            fig, ax = plt.subplots(figsize=settings.figsize)

            # Vertical period separators
            for week in settings.vline_weeks:
                ax.axvline(x=week, color="gray", linestyle="--", alpha=0.5, zorder=1)

            # Horizontal average lines per period
            for xmin, xmax, yval in [
                (1, 11.5, avg_rest),
                (34.5, 52, avg_rest),
                (11.5, 18.5, avg_prep),
                (18.5, 34.5, avg_play),
            ]:
                ax.hlines(
                    y=yval,
                    xmin=xmin,
                    xmax=xmax,
                    colors="black",
                    linestyles="--",
                    alpha=0.7,
                    zorder=5,
                )

            # Main time series
            sns.lineplot(
                data=average_all,
                x=Columns.ISOWEEK.value,
                y="avg_count_all",
                ax=ax,
                color=settings.line_color,
                linewidth=settings.linewidth,
                zorder=2,
            )

            # Period shading
            ax.axvspan(11.5, 18.5, color="lightgreen", alpha=0.3, zorder=0)
            ax.axvspan(18.5, 34.5, color="green", alpha=0.3, zorder=0)

            # Period labels (centered vertically)
            y_min, y_max = ax.get_ylim()
            y_label = y_min + 0.9 * (y_max - y_min)
            for x, txt in [(5, settings.rest_label), (15, settings.prep_label), (26.5, settings.play_label), (45, settings.rest_label)]:
                ax.text(
                    x,
                    y_label,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=12,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.0),
                    zorder=7,
                )

            # X-axis with week + month
            combined_labels = [
                f"{w}\n{m}"
                for w, m in zip(settings.week_ticks, settings.month_labels, strict=False)
            ]
            ax.set_xticks(settings.week_ticks)
            ax.set_xticklabels(combined_labels, ha="right", fontsize=8)
            ax.set_xlabel("Week / Month of Year", fontsize=8)
            ax.set_ylabel("Average messages per week (2017-2025)", fontsize=8)
            ax.set_title("Golf season, decoded by WhatsApp heartbeat", fontsize=24)

            plt.show()
            return fig
        except Exception as e:  # pragma: no cover
            logger.exception(f"Time plot failed: {e}")
            return None

    def build_visual_distribution(
        self,
        emoji_counts_df: pd.DataFrame,
        settings: DistributionPlotSettings = DistributionPlotSettings(),
    ) -> Optional[plt.Figure]:
        """
        Bar + cumulative line plot showing emoji usage distribution.

        Args:
            emoji_counts_df: Must contain ``emoji``, ``count_once``, ``percent_once``.
            settings: Styling.

        Returns:
            Matplotlib figure or ``None`` on error.
        """
        required = ["emoji", "count_once", "percent_once"]
        if not all(col in emoji_counts_df.columns for col in required):
            logger.error("emoji_counts_df missing required columns")
            return None

        try:
            n = len(emoji_counts_df)
            fig, ax = plt.subplots(figsize=(max(n * 0.2, 8), 8))
            ax2 = ax.twinx()

            x_pos = np.arange(n)
            ax.bar(
                x_pos,
                emoji_counts_df["percent_once"],
                color=settings.bar_color,
                align="edge",
                width=0.5,
            )
            ax.set_ylabel(
                "Likelihood (%) of an emoji in a random message",
                fontsize=12,
                labelpad=20,
            )
            ax.set_title(settings.title or "Emoji Distribution", fontsize=20)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_position(("outward", 20))
            ax.set_xlim(-0.5, n)
            ax.set_xticks([])

            cum = emoji_counts_df["percent_once"].cumsum()
            ax2.plot(
                x_pos + 0.25,
                cum,
                color=settings.cumulative_color,
                label="Cumulative %",
            )

            # Highlight threshold
            idx_thresh = None
            if (cum >= settings.cum_threshold).any():
                idx_thresh = np.where(cum >= settings.cum_threshold)[0][0]
                ax.axvspan(-0.5, idx_thresh + 0.5, facecolor="lightgreen", alpha=0.2)
                ax.axvline(
                    idx_thresh + 0.5,
                    color=settings.cumulative_color,
                    linestyle="--",
                    linewidth=1,
                )
                left_mid = idx_thresh / 2
                right_mid = (idx_thresh + 0.5) + (n - idx_thresh - 1) / 2
                y_txt = ax.get_ylim()[0] - 1.5
                ax.text(left_mid, y_txt, f"<-- {idx_thresh+1} emojis -->", ha="center", fontsize=12)
                ax.text(right_mid, y_txt, f"<-- {n} emojis -->", ha="center", fontsize=12)

            if idx_thresh is not None:
                ax2.axhline(
                    settings.cum_threshold,
                    color=settings.cumulative_color,
                    linestyle="--",
                    linewidth=1,
                )

            ax2.set_ylabel("Cumulative %", fontsize=12, labelpad=20)
            ax2.set_ylim(0, 100)
            ax2.set_yticks(np.arange(0, 101, 10))
            ax2.spines["right"].set_position(("outward", 20))
            ax2.tick_params(axis="y", labelsize=10, colors=settings.cumulative_color)
            ax2.spines["right"].set_color(settings.cumulative_color)

            # Top-N table
            top = emoji_counts_df.head(settings.top_n)
            cum_top = top["percent_once"].cumsum()
            table_data = [
                [str(i + 1) for i in range(len(top))],
                [row["emoji"] for _, row in top.iterrows()],
                [f"{c:.0f}" for c in top["count_once"]],
                [f"{c:.1f}%" for c in cum_top],
            ]
            col_w = 0.8 / len(top)
            ax.table(
                cellText=table_data,
                rowLabels=["Rank", "Emoji", "Count", "Cum"],
                colWidths=[col_w] * len(top),
                loc="bottom",
                bbox=[0.1, -0.45, 0.8, 0.3],
            ).auto_set_font_size(False)
            fig.text(0.5, 0.27, "Top 25:", ha="center", fontsize=12)
            ax2.legend(loc="upper left", fontsize=8)

            plt.tight_layout()
            plt.subplots_adjust(left=0.1, right=0.9, bottom=0.35)
            plt.show()
            return fig
        except Exception as e:  # pragma: no cover
            logger.exception(f"Distribution plot failed: {e}")
            return None

    def build_visual_relationships_arc(
        self,
        combined_df: pd.DataFrame,
        group: str,
        settings: ArcPlotSettings = ArcPlotSettings(),
    ) -> Optional[plt.Figure]:
        """
        Arc diagram visualising pair-wise message volumes.

        Args:
            combined_df: Must contain ``type``, ``author`` and participant columns.
            group: Human-readable group name for the title.
            settings: Geometry & styling.

        Returns:
            Matplotlib figure or ``None`` on error.
        """
        if combined_df is None or combined_df.empty:
            logger.error("Empty DataFrame for arc diagram")
            return None

        try:
            participant_cols = [
                c
                for c in combined_df.columns
                if c not in settings.excluded_columns
            ]
            authors = sorted(set(participant_cols))
            n = len(authors)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            radius = 1.0
            pos = {
                a: (radius * np.cos(ang), radius * np.sin(ang))
                for a, ang in zip(authors, angles, strict=False)
            }

            pair_weights: Dict[frozenset, float] = {}
            triple_weights: Dict[frozenset, float] = {}
            total_weights: Dict[frozenset, float] = {}

            # ----- Pairs -----
            pairs = combined_df[combined_df["type"] == "Pairs"]
            for _, row in pairs.iterrows():
                a1, a2 = [s.strip() for s in row[Columns.AUTHOR.value].split(" & ")]
                key = frozenset([a1, a2])
                pair_weights[key] = row["total_messages"]
                total_weights[key] = total_weights.get(key, 0) + row["total_messages"]

            # ----- Triples (non-participant) -----
            triples = combined_df[combined_df["type"] == "Non-participant"]
            for _, row in triples.iterrows():
                participants = [c for c in participant_cols if row[c] != 0]
                if len(participants) != len(authors) - 1:
                    continue
                total_msg = row["total_messages"]
                pct = {}
                for p in participants:
                    val = row[p]
                    if isinstance(val, str):
                        try:
                            pct[p] = int(val.split("%")[0]) / 100
                        except Exception:
                            continue
                for i, j in itertools.combinations(participants, 2):
                    if i in pct and j in pct:
                        w = (pct[i] + pct[j]) * total_msg
                        key = frozenset([i, j])
                        triple_weights[key] = triple_weights.get(key, 0) + w
                        total_weights[key] = total_weights.get(key, 0) + w

            if not total_weights:
                logger.error("No edges after processing")
                return None

            max_w = max(total_weights.values(), default=1)
            weights_dict = {
                "pair": pair_weights,
                "triple": triple_weights,
                "total": total_weights,
            }

            fig, ax = plt.subplots(figsize=settings.figsize)
            ax.set_aspect("equal")

            for arc_type, color, h_off, z in settings.arc_types:
                for key, w in weights_dict.get(arc_type, {}).items():
                    a1, a2 = list(key)
                    x1, y1 = pos[a1]
                    x2, y2 = pos[a2]
                    xm = (x1 + x2) / 2
                    ym = (y1 + y2) / 2
                    dist = np.hypot(x2 - x1, y2 - y1)
                    height = dist * h_off

                    # Total-arc colour logic
                    if arc_type == "total":
                        pair_t = (a1, a2) if a1 < a2 else (a2, a1)
                        married = pair_t in settings.married_couples or (
                            a2,
                            a1,
                        ) in settings.married_couples
                        color = (
                            settings.total_colors["married"]
                            if married
                            else settings.total_colors["other"]
                        )

                    # X-offset for crowded labels
                    sorted_pair = tuple(sorted([a1, a2]))
                    x_off = settings.special_x_offsets.get(
                        (*sorted_pair, arc_type), 0
                    )

                    # Line width scaling
                    lw = (1 + 5 * (w / max_w)) * settings.amplifier

                    t = np.linspace(0, 1, 100)
                    x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * (xm + x_off) + t**2 * x2
                    y = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * (ym + height) + t**2 * y2
                    ax.plot(x, y, color=color, linewidth=lw, zorder=z)

                    if arc_type == "total":
                        lbl_x = (x1 + x2) / 2
                        lbl_y = (y1 + y2) / 2 + height * 0.5
                        lbl_y += settings.special_label_y_offsets.get(
                            tuple(sorted([a1, a2])), 0
                        )
                        ax.text(
                            lbl_x,
                            lbl_y,
                            f"{round(w)}",
                            ha="center",
                            va="center",
                            fontsize=settings.label_fontsize,
                            bbox=settings.label_bbox,
                            zorder=z + 1,
                        )

            # Nodes
            for auth, (x, y) in pos.items():
                ax.scatter(
                    [x],
                    [y],
                    s=settings.node_size,
                    color=settings.node_color,
                    edgecolors=settings.node_edge_color,
                    zorder=4,
                )
                ax.text(
                    x,
                    y,
                    auth,
                    ha="center",
                    va="center",
                    fontsize=settings.node_fontsize,
                    fontweight=settings.node_fontweight,
                    zorder=5,
                )

            ax.set_title(settings.title_template.format(group=group))
            ax.axis("off")
            plt.tight_layout()
            plt.show()
            return fig
        except Exception as e:  # pragma: no cover
            logger.exception(f"Arc diagram failed: {e}")
            return None

    def build_visual_relationships_bubble(
        self,
        feature_df: pd.DataFrame,
        settings: BubbleNewPlotSettings = BubbleNewPlotSettings(),
    ) -> Optional[plt.Figure]:
        """
        Bubble plot: words vs punctuation, size = message count.

        Args:
            feature_df: Must contain ``whatsapp_group``, ``author``,
                        ``avg_words``, ``avg_punct``, ``message_count``.
            settings: Colours, scaling, trendline.

        Returns:
            Matplotlib figure or ``None`` on error.
        """
        required = {
            Columns.WHATSAPP_GROUP.value,
            Columns.AUTHOR.value,
            Columns.AVG_WORDS.value,
            Columns.AVG_PUNCT.value,
            Columns.MESSAGE_COUNT.value,
        }
        if not required.issubset(feature_df.columns):
            logger.error("Bubble plot missing required columns")
            return None

        try:
            fig, ax = plt.subplots(figsize=settings.figsize)

            msg = feature_df[Columns.MESSAGE_COUNT.value]
            size_scale = (
                (msg - msg.min()) / (msg.max() - msg.min())
                if msg.max() != msg.min()
                else 1.0
            )
            bubble_sizes = (
                settings.min_bubble_size
                + (settings.max_bubble_size - settings.min_bubble_size) * size_scale
            ) * 3  # Plot bubbles 3x larger

            for grp, col in settings.group_colors.items():
                mask = feature_df[Columns.WHATSAPP_GROUP.value] == grp
                sub = feature_df[mask]
                if sub.empty:
                    continue
                ax.scatter(
                    sub[Columns.AVG_WORDS.value],
                    sub[Columns.AVG_PUNCT.value],
                    s=bubble_sizes[mask],
                    alpha=settings.bubble_alpha,
                    color=col,
                    label=grp,
                )

            # Trendline (linear fit across *all* points)
            x = feature_df[Columns.AVG_WORDS.value]
            y = feature_df[Columns.AVG_PUNCT.value]
            coef = np.polyfit(x, y, 1)
            trend = np.poly1d(coef)
            ax.plot(x, trend(x), color=settings.trendline_color, alpha=settings.trendline_alpha)

            ax.set_title(
                settings.title
                or f"{Columns.AVG_WORDS.human} vs {Columns.AVG_PUNCT.human}"
            )
            ax.set_xlabel(settings.xlabel or Columns.AVG_WORDS.human)
            ax.set_ylabel(settings.ylabel or Columns.AVG_PUNCT.human)

            # Legend with scaled-down bubbles
            legend_handles = [
                plt.scatter(
                    [],
                    [],
                    s=settings.min_bubble_size * settings.legend_scale_factor,
                    c=col,
                    alpha=settings.bubble_alpha,
                    label=grp,
                )
                for grp, col in settings.group_colors.items()
            ]
            ax.legend(
                handles=legend_handles,
                title="WhatsApp Group",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )
            ax.grid(True, linestyle="--", alpha=0.7)

            plt.tight_layout()
            plt.show()
            return fig
        except Exception as e:  # pragma: no cover
            logger.exception(f"Bubble plot failed: {e}")
            return None

    def draw_confidence_ellipse(
        self,
        data: np.ndarray,
        ax: plt.Axes,
        alpha: float = 0.95,
        facecolor: str = "none",
        edgecolor: str = "black",
        zorder: int = 0,
    ) -> None:
        """Add a confidence ellipse to *ax* (95 % default)."""
        if len(data) < 2:
            return
        cov = np.cov(data, rowvar=False)
        mean = data.mean(axis=0)
        eig_val, eig_vec = np.linalg.eig(cov)
        scale = np.sqrt(chi2.ppf(alpha, 2))
        ellipse = Ellipse(
            xy=mean,
            width=np.sqrt(eig_val[0]) * scale * 2,
            height=np.sqrt(eig_val[1]) * scale * 2,
            angle=np.rad2deg(np.arccos(eig_vec[0, 0])),
            edgecolor=edgecolor,
            facecolor=facecolor,
            alpha=0.3,
            zorder=zorder,
        )
        ax.add_patch(ellipse)

    def build_visual_no_message_content(
        self,
        feature_df: pd.DataFrame,
        plot_type: str = PlotFeed.BOTH.value,
        dr_settings: DimReductionSettings = DimReductionSettings(),
        nmc_settings: PMNoMessageContentSettings = PMNoMessageContentSettings(),
        settings: Optional[PlotSettings] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        PCA / t-SNE visualisations of non-message features.

        Args:
            feature_df: Numeric + identifier columns.
            plot_type: ``both`` / ``pca`` / ``tsne``.
            dr_settings: Reduction hyper-parameters.
            nmc_settings: Colour / ellipse options.
            settings: Legacy – ignored (kept for backward compatibility).

        Returns:
            List of ``{'fig': fig, 'filename': name}`` or ``None``.
        """
        if settings is not None:
            logger.warning("Legacy `settings` ignored – use dr_settings / nmc_settings")

        if plot_type not in {
            PlotFeed.BOTH.value,
            PlotType.PCA.value,
            PlotType.TSNE.value,
        }:
            logger.error(f"Invalid plot_type: {plot_type}")
            return None

        try:
            X = self._prepare_features(feature_df, settings=dr_settings)
            methods = (
                [PlotType.PCA.value, PlotType.TSNE.value]
                if plot_type == PlotFeed.BOTH.value
                else [plot_type]
            )
            results: List[Dict[str, Any]] = []

            for method in methods:
                reducer = self._get_reducer(method, len(feature_df), dr_settings)
                X_red = reducer.fit_transform(X)

                dist = pairwise_distances(X_red, metric=dr_settings.metric)
                logger.info(
                    f"{method.upper()} – mean distance {dist.mean():.2f} (±{dist.std():.2f})"
                )

                if nmc_settings.plot_type in {PlotFeed.PER_GROUP.value, PlotFeed.BOTH.value}:
                    results.extend(
                        self._plot_per_group(X_red, feature_df, method, nmc_settings)
                    )
                if nmc_settings.plot_type in {PlotFeed.GLOBAL.value, PlotFeed.BOTH.value}:
                    results.append(
                        self._plot_global(X_red, feature_df, method, nmc_settings)
                    )

            logger.info(f"Created {len(results)} non-message-content plots")
            return results
        except Exception as e:  # pragma: no cover
            logger.exception(f"Non-message-content visualisation failed: {e}")
            return None

    # --------------------------------------------------------------------- #
    # === Additional public helpers (correlation, trends, interactions) ===
    # --------------------------------------------------------------------- #
    def plot_month_correlations(self, correlations: pd.Series) -> Optional[plt.Figure]:
        """Bar plot of Pearson correlation between month and numeric features."""
        if correlations is None or correlations.empty:
            logger.error("No correlations to plot")
            return None
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(
                correlations.index, correlations.values, color="skyblue"
            )
            ax.set_title("Correlation of Features with Month")
            ax.set_xlabel("Features")
            ax.set_ylabel("Pearson r")
            ax.set_ylim(-1, 1)
            ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
            ax.grid(True, axis="y", linestyle="--", alpha=0.7)

            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom" if h >= 0 else "top",
                )

            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()
            return fig
        except Exception as e:  # pragma: no cover
            logger.exception(f"Month-correlation plot failed: {e}")
            return None

    def plot_feature_trends(
        self,
        feature_df: pd.DataFrame,
        feature_name: str,
        settings: PlotSettings = PlotSettings(),
    ) -> Optional[plt.Figure]:
        """Box-plot of a numeric feature across months."""
        if feature_name not in feature_df.columns:
            logger.error(f"Feature {feature_name} not in DataFrame")
            return None
        try:
            fig, ax = plt.subplots(figsize=settings.figsize)
            sns.boxplot(
                x=Columns.MONTH.value, y=feature_name, data=feature_df, ax=ax
            )
            ax.set_title(settings.title or f"{feature_name} by Month")
            ax.set_xlabel(settings.xlabel or Columns.MONTH.human)
            ax.set_ylabel(settings.ylabel or feature_name)
            plt.tight_layout()
            plt.show()
            return fig
        except Exception as e:  # pragma: no cover
            logger.exception(f"Feature-trend plot failed: {e}")
            return None

    def build_visual_interactions(
        self,
        feature_df: pd.DataFrame,
        method: str = PlotType.TSNE.value,
        settings: DimReductionSettings = DimReductionSettings(),
        nmc_settings: PMNoMessageContentSettings = PMNoMessageContentSettings(),
    ) -> Tuple[Optional[plt.Figure], Optional[plt.Figure]]:
        """
        Two interaction visualisations (author-coloured & group-coloured).

        Returns:
            (author_fig, group_fig) – either may be ``None`` on error.
        """
        try:
            # Drop non-numeric identifier
            X = feature_df.drop(
                columns=[Columns.WHATSAPP_GROUP.value], errors="ignore"
            ).values
            reducer = self._get_reducer(method, len(feature_df), settings)
            X_red = reducer.fit_transform(X)

            labels = feature_df.index.values
            authors = [lbl.split("_")[0] for lbl in labels]
            uniq_auth = list(set(authors))
            pal = sns.color_palette("husl", len(uniq_auth))
            auth_map = dict(zip(uniq_auth, pal, strict=False))

            # ---- Author plot ----
            fig1, ax1 = plt.subplots(figsize=(10, 8))
            for i, lbl in enumerate(labels):
                auth = lbl.split("_")[0]
                ax1.scatter(
                    X_red[i, 0],
                    X_red[i, 1],
                    c=[auth_map[auth]],
                    label=auth if authors.index(auth) == i else None,
                )
            ax1.set_title(f"Interaction Dynamics – Authors ({method.upper()})")
            ax1.set_xlabel("Component 1")
            ax1.set_ylabel("Component 2")
            ax1.legend(title="Author")
            plt.tight_layout()
            plt.show()

            # ---- Group plot (Anthony special) ----
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            legend_elems = [
                patches.Patch(color=v, label=k) for k, v in nmc_settings.group_color_map.items()
            ]
            legend_elems += [
                patches.Patch(color=v, label=f"Anthony ({k})")
                for k, v in nmc_settings.anthony_color_map.items()
            ]
            legend_elems.append(
                patches.Patch(color="gray", label="Anthony (overall)")
            )

            for i, lbl in enumerate(labels):
                auth = lbl.split("_")[0]
                grp = feature_df.iloc[i][Columns.WHATSAPP_GROUP.value]
                if auth == "Anthony van Tilburg":
                    color = (
                        "gray"
                        if grp == "overall"
                        else nmc_settings.anthony_color_map.get(grp, "black")
                    )
                else:
                    color = nmc_settings.group_color_map.get(grp, "black")
                ax2.scatter(X_red[i, 0], X_red[i, 1], c=[color])

            ax2.set_title(f"Interaction Dynamics – Groups ({method.upper()})")
            ax2.set_xlabel("Component 1")
            ax2.set_ylabel("Component 2")
            ax2.legend(handles=legend_elems, title="Group")
            plt.tight_layout()
            plt.show()

            if method == PlotType.PCA.value:
                loads = pd.DataFrame(
                    reducer.components_.T,
                    index=feature_df.drop(
                        columns=[Columns.WHATSAPP_GROUP.value], errors="ignore"
                    ).columns,
                    columns=["Component 1", "Component 2"],
                )
                logger.info(f"PCA loadings:\n{loads}")

            return fig1, fig2
        except Exception as e:  # pragma: no cover
            logger.exception(f"Interaction visualisations failed: {e}")
            return None, None