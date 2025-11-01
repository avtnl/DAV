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
from typing import Any

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

from constants import Columns, GroupByPeriod, Groups, PlotFeed, PlotType

# Suppress FutureWarning from seaborn/pandas
warnings.simplefilter(action="ignore", category=FutureWarning)


# === Settings Models ===
class PlotSettings(BaseModel):
    """Base settings for every plot (figure size, titles, axis labels)."""

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    figsize: tuple[int, int] = (10, 6)
    rotation: int = 0
    legend_title: str | None = None


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

    group_color_map: dict[str, str] = {
        Groups.MAAP.value: "blue",
        Groups.GOLFMATEN.value: "red",
        Groups.DAC.value: "green",
    }
    anthony_color_map: dict[str, str] = {
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

    vline_weeks: list[float] = [11.5, 18.5, 34.5]
    week_ticks: list[int] = [1, 5, 9, 14, 18, 23, 27, 31, 36, 40, 44, 49]
    month_labels: list[str] = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
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

    group_colors: dict[str, str] = Field(
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
    married_couples: list[tuple[str, str]] = Field(
        default_factory=lambda: [
            ("Anja Berkemeijer", "Phons Berkemeijer"),
            ("Madeleine", "Anthony van Tilburg"),
        ]
    )
    arc_types: list[tuple[str, str | None, float, int]] = Field(
        default_factory=lambda: [
            ("triple", "lightgray", 0.4, 1),
            ("pair", "gray", 0.55, 2),
            ("total", None, 0.7, 3),
        ]
    )
    total_colors: dict[str, str] = Field(
        default_factory=lambda: {"married": "red", "other": "blue"}
    )
    special_x_offsets: dict[tuple[str, str, str], float] = Field(
        default_factory=lambda: {
            ("Anthony van Tilburg", "Phons Berkemeijer", "triple"): -0.1,
            ("Anthony van Tilburg", "Phons Berkemeijer", "pair"): -0.2,
        }
    )
    special_label_y_offsets: dict[tuple[str, str], float] = Field(
        default_factory=lambda: {("Anthony van Tilburg", "Phons Berkemeijer"): -0.5}
    )
    excluded_columns: list[str] = Field(
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
    label_bbox: dict[str, Any] = Field(
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
        self.fig: plt.Figure | None = None
        self.ax: plt.Axes | None = None

    def create_figure(self) -> tuple[plt.Figure, plt.Axes]:
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
            logger.warning("Segoe UI Emoji not available - falling back to DejaVu Sans.")
            plt.rcParams["font.family"] = "DejaVu Sans"

    # === Private helpers – feature preparation & reduction === #
    def _prepare_features(
        self,
        feature_df: pd.DataFrame,
        groupby_period: str | None = None,
        settings: DimReductionSettings | None = None,
    ) -> np.ndarray:
        """
        Prepare numerical feature matrix for dimensionality reduction.

        Args:
            feature_df: Input DataFrame with all features.
            groupby_period: Optional period column to drop.
            settings: Reduction settings (default: new instance per call).

        Returns:
            np.ndarray: Numerical feature matrix.
        """
        if settings is None:
            settings = DimReductionSettings()

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
        logger.info(f"Feature variances:\n{variances.sort_values(ascending=False).to_string()}")

        top_features = variances.nlargest(settings.n_top_features).index
        if len(top_features) < numerical_features.shape[1]:
            logger.info(f"Selected top {len(top_features)} features: {list(top_features)}")
        else:
            logger.info("Using all numeric features")

        numerical_features = numerical_features[top_features]
        logger.info(f"Prepared numerical features (shape {numerical_features.shape})")
        return numerical_features.values  # type: ignore[no-any-return]

    # === Private helpers – dimensionality reduction === #
    def _get_reducer(
        self,
        method: str,
        n_samples: int,
        settings: DimReductionSettings | None = None,
    ):
        """
        Return a fitted reducer (PCA or t-SNE) based on method.

        Args:
            method: 'pca' or 'tsne'.
            n_samples: Number of samples (for perplexity cap).
            settings: Optional reduction settings (default: new instance).

        Returns:
            sklearn reducer instance.
        """
        if settings is None:
            settings = DimReductionSettings()

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

    # === Private scatter-plot helpers (per-group / global) === #
    def _plot_per_group(
        self,
        X_reduced: np.ndarray,
        feature_df: pd.DataFrame,
        method: str,
        settings: PMNoMessageContentSettings,
    ) -> list[dict[str, Any]]:
        figs: list[dict[str, Any]] = []
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
                        logger.warning(f"Ellipse skipped for {auth} in {group}: <2 points")
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
    ) -> dict[str, Any]:
        fig, ax = plt.subplots(figsize=settings.figsize)

        for i in range(len(X_reduced)):
            group = feature_df.iloc[i][Columns.WHATSAPP_GROUP.value]
            auth = feature_df.iloc[i][Columns.AUTHOR.value]

            if auth == "Anthony van Tilburg":
                color = settings.anthony_color_map.get(group, "gray")
            else:
                color = settings.group_color_map.get(group, "black")

            ax.scatter(X_reduced[i, 0], X_reduced[i, 1], c=[color], label=None, alpha=0.6)

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

        ax.set_title("Riding the Wave of WhatsApp: Group Patterns in Messaging Behavior")
        ax.set_xlabel("Component 1", labelpad=2)
        ax.set_ylabel("Component 2", labelpad=2)
        ax.legend(handles=legend_elements, title="Group / Anthony")
        plt.tight_layout(pad=0.5)
        plt.show()

        return {"fig": fig, "filename": f"no_message_content_global_{method}"}

    # --------------------------------------------------------------------- #
    # === Public visualisation builders === #
    # --------------------------------------------------------------------- #
    def build_visual_categories(
        self,
        group_authors: pd.DataFrame,
        non_anthony_group: pd.DataFrame,
        anthony_group: pd.DataFrame,
        sorted_groups: list[str],
        settings: CategoriesPlotSettings = CategoriesPlotSettings(),
    ) -> plt.Figure | None:
        try:
            if non_anthony_group is None or anthony_group is None or not sorted_groups:
                logger.error("Invalid input for categories plot")
                return None

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
                non_anthony_group["non_anthony_avg"] * non_anthony_group["num_authors"]
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
                        arrowprops={
                            "arrowstyle": "-|>",
                            "color": settings.arrow_color,
                            "lw": settings.arrow_lw,
                            "mutation_scale": settings.arrow_mutation_scale,
                        },
                    )

            # X-axis labels with author count
            xtick_labels = [
                f"{g} ({n:.0f})"
                for g, n in zip(sorted_groups, non_anthony_group["num_authors"], strict=False)
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
        except Exception as e:
            logger.exception(f"Categories plot failed: {e}")
            return None

    def build_visual_time(
        self,
        p: pd.DataFrame,
        average_all: pd.DataFrame,
        settings: TimePlotSettings = TimePlotSettings(),
    ) -> plt.Figure | None:
        try:
            # Pre-compute period averages
            weeks_rest = average_all["isoweek"].between(1, 12) | average_all["isoweek"].between(
                35, 53
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
            for x, txt in [
                (5, settings.rest_label),
                (15, settings.prep_label),
                (26.5, settings.play_label),
                (45, settings.rest_label),
            ]:
                ax.text(
                    x,
                    y_label,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=12,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.0},
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
        except Exception as e:
            logger.exception(f"Time plot failed: {e}")
            return None

    def build_visual_distribution(
        self,
        emoji_counts_df: pd.DataFrame,
        settings: DistributionPlotSettings = DistributionPlotSettings(),
    ) -> plt.Figure | None:
        try:
            required = ["emoji", "count_once", "percent_once"]
            if not all(col in emoji_counts_df.columns for col in required):
                logger.error("emoji_counts_df missing required columns")
                return None

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
                ax.text(
                    left_mid, y_txt, f"<-- {idx_thresh + 1} emojis -->", ha="center", fontsize=12
                )
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
        except Exception as e:
            logger.exception(f"Distribution plot failed: {e}")
            return None

    def build_visual_relationships_arc(
        self,
        combined_df: pd.DataFrame,
        group: str,
        settings: ArcPlotSettings = ArcPlotSettings(),
    ) -> plt.Figure | None:
        try:
            if combined_df is None or combined_df.empty:
                logger.error("Empty DataFrame for arc diagram")
                return None

            participant_cols = [
                c for c in combined_df.columns if c not in settings.excluded_columns
            ]
            authors = sorted(set(participant_cols))
            n = len(authors)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            radius = 1.0
            pos = {
                a: (radius * np.cos(ang), radius * np.sin(ang))
                for a, ang in zip(authors, angles, strict=False)
            }

            pair_weights: dict[frozenset, float] = {}
            triple_weights: dict[frozenset, float] = {}
            total_weights: dict[frozenset, float] = {}

            # ----- Pairs -----
            pairs = combined_df[combined_df["type"] == "Pairs"]
            for _, row in pairs.iterrows():
                a1, a2 = (s.strip() for s in row[Columns.AUTHOR.value].split(" & "))
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
                        married = (
                            pair_t in settings.married_couples
                            or (a2, a1) in settings.married_couples
                        )
                        color = (
                            settings.total_colors["married"]
                            if married
                            else settings.total_colors["other"]
                        )

                    # X-offset for crowded labels
                    sorted_pair = tuple(sorted([a1, a2]))
                    x_off = settings.special_x_offsets.get((*sorted_pair, arc_type), 0)

                    # Line width scaling
                    lw = (1 + 5 * (w / max_w)) * settings.amplifier

                    t = np.linspace(0, 1, 100)
                    x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * (xm + x_off) + t**2 * x2
                    y = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * (ym + height) + t**2 * y2
                    ax.plot(x, y, color=color, linewidth=lw, zorder=z)

                    if arc_type == "total":
                        lbl_x = (x1 + x2) / 2
                        lbl_y = (y1 + y2) / 2 + height * 0.5
                        lbl_y += settings.special_label_y_offsets.get(tuple(sorted([a1, a2])), 0)
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
        except Exception as e:
            logger.exception(f"Arc diagram failed: {e}")
            return None

    def build_visual_relationships_bubble(
        self,
        feature_df: pd.DataFrame,
        settings: BubbleNewPlotSettings = BubbleNewPlotSettings(),
    ) -> plt.Figure | None:
        try:
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

            fig, ax = plt.subplots(figsize=settings.figsize)

            msg = feature_df[Columns.MESSAGE_COUNT.value]
            size_scale = (
                (msg - msg.min()) / (msg.max() - msg.min()) if msg.max() != msg.min() else 1.0
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
                settings.title or f"{Columns.AVG_WORDS.human} vs {Columns.AVG_PUNCT.human}"
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
        except Exception as e:
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

    # === Build correlation matrix === #
    def plot_month_correlations(self, correlations: pd.Series) -> plt.Figure | None:
        try:
            if correlations is None or correlations.empty:
                logger.error("No correlations to plot")
                return None

            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(correlations.index, correlations.values, color="skyblue")
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
        except Exception as e:
            logger.exception(f"Month-correlation plot failed: {e}")
            return None
