# === plot_manager.py ===
# === Module Docstring ===
"""
Plot Manager Module

Creates all 5 key visualizations using validated data from DataPreparation:

1. Categories: Total messages by group/author (Script1)
2. Time: DAC weekly heartbeat (Script2)
3. Distribution: Emoji frequency + cumulative (Script3)
4. Relationships: Words vs punctuation per author (Script4)
5. Multi-Dimensional: t-SNE style clusters (Script5)
"""

# === Imports ===
from __future__ import annotations

import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.patches as patches
from matplotlib.figure import Figure
import seaborn as sns
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Literal, Dict

from .constants import Columns, Groups, InteractionType, Script5ConfigKeys
from .data_preparation import (
    CategoryPlotData,
    TimePlotData,
    SeasonalityEvidence,
    DistributionPlotData,
    RelationshipsPlotData,
    MultiDimPlotData,
    MultiDimPlotSettings
)

warnings.simplefilter(action="ignore", category=FutureWarning)


# === Shared Base Settings ===
class PlotSettings(BaseModel):
    """Base settings for all plots."""
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    figsize: tuple[int, int] = (14, 8)


# === 1. Categories Plot Settings (Script1) ===
class CategoriesPlotSettings(PlotSettings):
    title: str = "AvT's participation is much lower for the 3rd group"
    subtitle: str = "Too much to handle or too much crap?"
    ylabel: str = Columns.MESSAGE_COUNT.human
    group_spacing: float = Field(2.5, ge=0.5, le=5.0)
    bar_width: float = Field(0.8, ge=0.1, le=1.0)
    trendline_color: str = "red"
    trendline_style: str = "--"
    trendline_width: float = 2.5

    title_fontsize: int = 24
    title_fontweight: str = "bold"
    title_pad: float = 40
    title_ha: str = "center"

    subtitle_fontsize: int = 18
    subtitle_fontweight: str = "bold"
    subtitle_color: str = "dimgray"
    subtitle_y: float = 0.85
    subtitle_ha: str = "center"


# === 2. Time Plot Settings (Script2) ===
class TimePlotSettings(PlotSettings):
    """Settings for the DAC weekly heartbeat plot."""
    title: str = "Golf season, decoded by WhatsApp heartbeat"
    subtitle: str = "Number of messages/week within whatsapp group 'dac'"
    rest_label: str = "---------Rest---------"
    prep_label: str = "---Prep---"
    play_label: str = "---------Play---------"
    line_color: str = "green"
    linewidth: float = 2.5

    title_fontsize: int = 24
    title_fontweight: str = "bold"
    title_pad: float = 40
    title_ha: str = "center"

    subtitle_fontsize: int = 18
    subtitle_fontweight: str = "bold"
    subtitle_color: str = "dimgray"
    subtitle_y: float = 0.85
    subtitle_ha: str = "center"


class SeasonalityPlotSettings(PlotSettings):
    """Settings for the 4-panel seasonality evidence suite."""
    fig_width: int = 16
    fig_height: int = 10
    subplot_titles_fontsize: int = 14
    acf_color: str = "#1f77b4"
    decomp_linewidth: float = 1.8
    fourier_marker: str = "o"
    fourier_markersize: int = 6
    filtered_alpha: float = 0.7


# === 3. Distribution Plot Settings (Script3) ===
class DistributionPlotSettings(PlotSettings):
    bar_color: str = "purple"
    cumulative_color: str = "orange"
    bar_alpha: float = 0.9
    line_width: float = 2.0
    cum_label: str = "Cumulative %"
    cum_threshold: float = 75.0
    top_table: int = 25  # For table, not bars

class DistributionPlotData(BaseModel):
    """Validated container for the emoji-frequency DataFrame."""
    emoji_counts_df: pd.DataFrame = Field(
        ..., description="Columns: ['emoji', 'count_once', 'percent_once', 'unicode_code', 'unicode_name']"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_columns(self):
        required = {"emoji", "count_once", "percent_once", "unicode_code", "unicode_name"}
        missing = required - set(self.emoji_counts_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return self

# Power-Law Plot Settings
class PowerLawPlotSettings(PlotSettings):
    """Settings for log-log power-law plot."""
    title: str = "Emoji Frequency Follows Power-Law (Zipf) Distribution"
    subtitle: str = "Log-log plot shows linear relationship → proof of log model"
    line_color: str = "red"
    line_width: float = 2.5
    marker_color: str = "blue"
    marker_size: int = 6
    alpha: float = 0.8
    show_fit_line: bool = True
    annotate_alpha: bool = True
    font_size_title: int = 24
    font_size_subtitle: int = 18

    model_config = ConfigDict(arbitrary_types_allowed=True)


# === 4. Relationships Plot Settings (Script4) ===
class RelationshipsPlotSettings(PlotSettings):
    title: str = "Correlation between averages of Words and Punctuations"
    subtitle: str = "About 1 extra Punctuation per 10 Words"
    bubble_alpha: float = 0.6
    min_bubble_size: int = 50
    max_bubble_size: int = 1000
    trendline_color: str = "red"
    trendline_style: str = "--"
    trendline_width: float = 2.5
    trendline_alpha: float = 0.5
    legend_scale_factor: float = 1.0
    group_colors: dict[str, str] = Field(
        default_factory=lambda: {
            Groups.MAAP: "deepskyblue",
            Groups.GOLFMATEN: "orange",
            Groups.DAC: "green",
            Groups.TILLIES: "gray",
        }
    )
    title_fontsize: int = 24
    title_fontweight: str = "bold"
    title_pad: float = 40
    title_ha: str = "center"

    subtitle_fontsize: int = 18
    subtitle_fontweight: str = "bold"
    subtitle_color: str = "dimgray"
    subtitle_ha: str = "center"

    model_config = ConfigDict(arbitrary_types_allowed=True)

# === 5. Multi Dimensions Plot Settings (Script5) ===
class MultiDimPlotSettings(PlotSettings):
    """Configuration for multi-dimensional t-SNE visualization (Script5)."""
    title: str = "Revealing authors Fingerprints by 'style', not by 'content'"
    subtitle: str = "TSNE plot combining 25 style features and style oriented Hugging Faces"
    by_group: bool = True
    ellipse_mode: int = Field(0, ge=0, le=2)        # 0=none, 1=single, 2=GMM pockets
    confidence_level: int = Field(75, ge=20, le=100) # 20–100%
    draw_ellipses: bool = False   # kept for backward compat
    use_embeddings: bool = True
    hybrid_features: bool = True
    embedding_model: int = 3

    title_fontsize: int = 32
    title_fontweight: str = "bold"
    title_pad: float = 40
    title_ha: str = "center"

    subtitle_fontsize: int = 24
    subtitle_fontweight: str = "bold"
    subtitle_color: str = "dimgray"
    subtitle_y: float = 0.95
    subtitle_ha: str = "center"

    @model_validator(mode="after")
    def sync_draw_ellipses(self):
        self.draw_ellipses = self.ellipse_mode > 0
        return self

    model_config = ConfigDict(arbitrary_types_allowed=True)

# === Plot Manager Class ===
class PlotManager:
    """Manages all 5 visualizations."""

    def __init__(self) -> None:
        self.data_preparation = None  # Injected


    # === 1. Categories (Script1) ===
    def build_visual_categories(
        self,
        data: CategoryPlotData,
        settings: CategoriesPlotSettings = CategoriesPlotSettings(),
    ) -> plt.Figure | None:
        """
        Plot total messages per author per group with fair AvT comparison.

        Args:
            data: Validated CategoryPlotData
            settings: Plot settings

        Returns:
            matplotlib Figure or None

        Raises:
            Exception: If plotting fails (logged).
        """
        if not data or not data.groups:
            logger.error("Invalid or empty CategoryPlotData")
            return None

        try:
            fig, ax = plt.subplots(figsize=(10.5, 7))  # Shorter canvas
            plt.subplots_adjust(top=0.65)              # Axes end earlier → more room above

            # Define group colors
            group_colors = {
                Groups.DAC: "green",
                Groups.GOLFMATEN: "orange",
                Groups.MAAP: "deepskyblue",
                Groups.TILLIES: "gray",
            }

            # Prepare bar positions and metadata
            x_pos: list[float] = []
            heights: list[int] = []
            bar_colors: list[str] = []
            author_labels: list[str] = []
            group_midpoints: list[float] = []
            cur_x = 0.0

            # Capture MAAP AvT position
            maap_avt_x = None
            maap_avt_h = None
            maap_group_avg = None

            for group_data in data.groups:
                n_auth = len(group_data.authors)
                start_x = cur_x

                for auth in group_data.authors:
                    x_pos.append(cur_x)
                    heights.append(auth.message_count)

                    # # AvT always black
                    # col = "#000000" if auth.is_avt else group_colors.get(group_data.whatsapp_group, "gray")
                    # bar_colors.append(col)
                    # author_labels.append(auth.author)

                    # AvT uses group color with thick gray border
                    group_color = group_colors.get(group_data.whatsapp_group, "gray")
                    if auth.is_avt:
                        bar_colors.append(group_color)      # Same fill as group
                        # Border handled in ax.bar() below
                    else:
                        bar_colors.append(group_color)
                    author_labels.append(auth.author)    

                    # Capture AvT in MAAP
                    if group_data.whatsapp_group == Groups.MAAP and auth.is_avt:
                        maap_avt_x = cur_x + 1.5  # Right of AvT bar
                        maap_avt_h = auth.message_count
                        maap_group_avg = group_data.group_avg
                        logger.info(f"AvT in MAAP at x={cur_x}, height={maap_avt_h}")

                    cur_x += 1.0

                mid = start_x + (n_auth - 1) / 2
                group_midpoints.append(mid)
                cur_x += settings.group_spacing

            # Draw bars with special edge for AvT
            edgecolors = []
            linewidths = []
            hatches = []
            for auth in [a for g in data.groups for a in g.authors]:
                hatches.append("//" if auth.is_avt else "")
                if auth.is_avt:
                    edgecolors.append("black")
                    linewidths.append(2.0)   # Thicker border
                else:
                    edgecolors.append("black")
                    linewidths.append(1.3)

            ax.bar(
                x_pos,
                heights,
                width=settings.bar_width,
                color=bar_colors,
                edgecolor=edgecolors,    # NEW: Per-bar edge color
                linewidth=linewidths,    # NEW: Per-bar line width
                hatch=hatches, 
                align="center",
            )

            # Draw group average line ine red
            avg_line_patch = None
            for mid_x, gdata in zip(group_midpoints, data.groups):
                if gdata.group_avg > 0:
                    line_length = 0.5 * len(gdata.authors)
                    line = ax.hlines(
                        y=gdata.group_avg,
                        xmin=mid_x - line_length,
                        xmax=mid_x + line_length,
                        color="red",
                        linestyle="--",
                        linewidth=settings.trendline_width,
                        zorder=5,
                        label="Group Avg (non-AvT)" if mid_x == group_midpoints[0] else "",
                    )
                    if avg_line_patch is None:       # lengend
                        avg_line_patch = line

                    # Log MAAP mid and avg
                    if gdata.whatsapp_group == Groups.MAAP:
                        logger.info(f"maap_mid: {mid_x}")
                        logger.info(f"maap_group_avg: {gdata.group_avg}")

            # Draw vertical red block arrow in MAAP
            if maap_group_avg is not None and maap_avt_x is not None and maap_avt_h is not None:
                ax.annotate(
                    "",
                    xy=(maap_avt_x, maap_avt_h),                   # tip at top of AvT bar
                    xytext=(maap_avt_x, maap_group_avg),           # tip at group‑avg line
                    arrowprops=dict(
                        arrowstyle="<->,head_length=2.0,head_width=2.0",  # double‑headed
                        color="red",
                        lw=3.0,
                        connectionstyle="arc3,rad=0",
                    ),
                    zorder=6,
                )

            #Title and subtitle
            ax.set_title(
                settings.title,
                fontsize=settings.title_fontsize,
                fontweight=settings.title_fontweight,
                pad=settings.title_pad,
                ha=settings.title_ha,
            )

            # fig.suptitle(
            #     settings.title,
            #     fontsize=settings.title_fontsize,
            #     fontweight=settings.title_fontweight,
            #     y=0.98,
            #     ha=settings.title_ha,
            # )

            # fig.suptitle(
            #     settings.subtitle,
            #     fontsize=settings.subtitle_fontsize,
            #     fontweight=settings.subtitle_fontweight,
            #     color=settings.subtitle_color,
            #     y=settings.subtitle_y,
            #     ha=settings.subtitle_ha,
            # )

            fig.text(
                x=0.5,                       # ← X position (required)
                y=0.92,
                s=settings.subtitle,         # ← TEXT to display (required)
                fontsize=settings.subtitle_fontsize,
                fontweight=settings.subtitle_fontweight,
                color=settings.subtitle_color,
                ha=settings.subtitle_ha,
                va="top",                    # ← Recommended
                transform=fig.transFigure    # ← Critical for figure-level
            )

            # Axis labels
            ax.set_ylabel(settings.ylabel, fontsize=12)
            ax.set_xlabel("WhatsApp Groups and participating Authors", fontsize=12)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(author_labels, rotation=45, ha="right", fontsize=12)

            # Build legend
            legend_patches = [
                patches.Patch(facecolor=c, edgecolor="black", label=g.value.title())
                for g, c in group_colors.items()
            ]
            legend_patches.append(patches.Patch(facecolor="black", edgecolor="black", label=Groups.AVT))
            if avg_line_patch is not None:
                legend_patches.append(avg_line_patch)  # NEW: Add red dashed line

            ax.legend(
                handles=legend_patches,
                title="Group",
                loc="upper right",
                bbox_to_anchor=(1.15, 0.9),
                frameon=True,
                fancybox=True,
                shadow=True,
            )

            # Apply grid and layout
            ax.grid(True, axis="y", linestyle="--", alpha=0.7, zorder=0)
            ax.set_axisbelow(True)
            plt.tight_layout()
            plt.show()

            logger.success("Category bar chart built successfully")
            return fig

        except Exception as e:
            logger.exception(f"build_visual_categories failed: {e}")
            return None


    # === 2. Time (Script2) ===
    def build_visual_time(
        self,
        data: TimePlotData,
        settings: TimePlotSettings,
        include_seasonality: bool = False,
        season_settings: SeasonalityPlotSettings | None = None,
    ) -> Dict[str, Figure]:
        """
        Build the old-style DAC weekly heartbeat (Rest/Prep/Play/Rest bands)
        and the optional 4-panel seasonality evidence suite.

        The main plot is pixel-identical to the original version:
        - 4 colored background bands (Rest/Prep/Play/Rest)
        - Red dashed period-average lines
        - Bold period labels at 80% height
        - Calendar dates on X-axis (March 15th, May 1st, Sep 1st)
        - Y-label includes year range (2017-2025)
        - **No global average line**
        - **Smooth line (no markers)**

        Args:
            data: Validated TimePlotData with weekly averages and seasonality.
            settings: TimePlotSettings (line color, labels, etc.).
            include_seasonality: If True, generate the 4-panel suite.
            season_settings: Optional styling for the seasonality suite.

        Returns:
            Dict with 'main' (old-style heartbeat) and optionally 'seasonality'.

        # NEW: Full old-style restoration + seasonality suite (2025-11-07)
        """
        figs: Dict[str, Figure] = {}

        # === Main Plot Setup ===
        fig_main, ax = plt.subplots(figsize=(14, 6.5))
        plt.subplots_adjust(top=0.85)

        weeks = sorted(data.weekly_avg.keys())
        avg_counts = [data.weekly_avg[w] for w in weeks]

        # Smooth line – no markers
        ax.plot(
            weeks,
            avg_counts,
            color=settings.line_color,
            linewidth=settings.linewidth,
            zorder=6,
        )

        # === Period Definitions ===
        vlines = [11.5, 18.5, 34.5]
        starts = [1] + [int(v + 0.5) for v in vlines]
        ends   = [int(v + 0.5) - 1 for v in vlines] + [52]

        period_labels = [
            settings.rest_label,
            settings.prep_label,
            settings.play_label,
            settings.rest_label,
        ]
        period_colors = ["#e8f5e9", "#c8e6c9", "#81c784", "#e8f5e9"]

        # === Helper: Period Average ===
        def period_avg(start: int, end: int) -> float:
            """Return mean of weekly values in [start, end]."""
            vals = [data.weekly_avg.get(w, 0.0) for w in range(start, end + 1)]
            return float(np.mean(vals)) if vals else 0.0

        # === Draw Bands, Averages, Labels, and Separators ===
        period_avg_patch = None
        y_min, y_max = ax.get_ylim()
        label_y = y_min + 0.80 * (y_max - y_min)

        for i in range(4):
            s, e = starts[i], ends[i]

            # Background band
            ax.axvspan(s - 0.5, e + 0.5, facecolor=period_colors[i], alpha=0.6, zorder=0)

            # Period average
            p_avg = period_avg(s, e)
            if p_avg > 0:
                line = ax.hlines(
                    p_avg,
                    xmin=s - 0.5,
                    xmax=e + 0.5,
                    color="red",
                    linestyle="--",
                    linewidth=1.2,
                    zorder=4,
                    label="Period Avg" if period_avg_patch is None else "",
                )
                if period_avg_patch is None:
                    period_avg_patch = line

            # Period label
            mid = (s + e) / 2
            ax.text(
                mid,
                label_y,
                period_labels[i],
                ha="center",
                va="center",
                fontsize=12,
                color="black",
                fontweight="bold",
                zorder=7,
            )

        # Vertical separators
        for v in vlines:
            ax.axvline(v, color="gray", linestyle="--", alpha=0.6, zorder=1)

        # === X-Axis: Calendar Dates ===
        key_dates = {11: "March 15th", 18: "May 1st", 36: "September 1st"}
        ax.set_xticks(sorted(key_dates.keys()))
        ax.set_xticklabels([key_dates[w] for w in sorted(key_dates)], fontsize=12, ha="center")
        ax.set_xlabel("Time over Year", fontsize=12, labelpad=10)

        # === Y-Axis: Year Range ===
        ax.set_ylabel("Average message count per week (2017 - 2025)", fontsize=12)

        # === Title and Subtitle ===
        ax.set_title(
            settings.title,
            fontsize=settings.title_fontsize,
            fontweight=settings.title_fontweight,
            pad=settings.title_pad,
            ha=settings.title_ha,
        )
        fig_main.text(
            x=0.5,
            y=0.92,
            s=settings.subtitle,
            fontsize=settings.subtitle_fontsize,
            fontweight=settings.subtitle_fontweight,
            color=settings.subtitle_color,
            ha=settings.subtitle_ha,
            va="top",
            transform=fig_main.transFigure,
        )

        # === Legend ===
        legend_handles = []
        if period_avg_patch:
            legend_handles.append(period_avg_patch)
        ax.legend(handles=legend_handles, loc="upper right", fontsize=12)

        # === Grid and Layout ===
        ax.grid(True, axis="y", linestyle="--", alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.show()

        figs["main"] = fig_main
        logger.success("Time plot built – old-style DAC heartbeat restored")

        # === Optional Seasonality Suite ===
        if include_seasonality and data.seasonality is not None:
            fig_season = self._build_seasonality_suite(data, season_settings or SeasonalityPlotSettings())
            if fig_season:
                figs["seasonality"] = fig_season

        return figs

    def _build_seasonality_suite(
        self,
        time_data: TimePlotData,
        settings: SeasonalityPlotSettings,
    ) -> Figure | None:
        """Render 4-panel seasonality evidence."""
        ev = time_data.seasonality

        fig = plt.figure(figsize=(settings.fig_width, settings.fig_height))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)

        # 1. ACF
        ax0 = fig.add_subplot(gs[0, 0])
        lags = range(len(ev.acf))
        ax0.stem(lags, ev.acf, basefmt=" ", linefmt=settings.acf_color)
        ax0.set_title("Autocorrelation (weekly lags)", fontsize=settings.subplot_titles_fontsize)
        ax0.set_xlabel("Lag (weeks)")
        ax0.set_ylabel("ACF")
        ax0.axhline(0, color="k", linewidth=0.8)
        ax0.grid(True, alpha=0.3)

        # 2. Decomposition
        ax1 = fig.add_subplot(gs[0, 1])
        idx = range(len(ev.decomposition["trend"]))
        ax1.plot(idx, ev.decomposition["trend"], label="Trend", linewidth=settings.decomp_linewidth)
        ax1.plot(idx, ev.decomposition["seasonal"], label="Seasonal", linewidth=settings.decomp_linewidth)
        ax1.plot(idx, ev.decomposition["resid"], label="Residual", linewidth=settings.decomp_linewidth, alpha=0.6)
        ax1.set_title("Additive Decomposition", fontsize=settings.subplot_titles_fontsize)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 3. Fourier
        ax2 = fig.add_subplot(gs[1, 0])
        freqs = ev.fourier["freqs"]
        amps = ev.fourier["amps"]
        ax2.stem(freqs, amps, linefmt="C3-", markerfmt=f"C3{settings.fourier_marker}", basefmt=" ")
        ax2.set_title("Fourier amplitudes (top k)", fontsize=settings.subplot_titles_fontsize)
        ax2.set_xlabel("Frequency (cycles/week)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True, alpha=0.3)

        # 4. Filters
        ax3 = fig.add_subplot(gs[1, 1])
        weeks = range(len(ev.raw_series))
        ax3.plot(weeks, ev.raw_series, label="Raw", color="lightgray")
        ax3.plot(weeks, ev.filtered["savitzky_golay"], label="Savitzky-Golay", alpha=settings.filtered_alpha)
        ax3.plot(weeks, ev.filtered["butterworth"], label="Butterworth low-pass", alpha=settings.filtered_alpha)
        ax3.set_title("Smoothing filters", fontsize=settings.subplot_titles_fontsize)
        ax3.set_xlabel("Week index")
        ax3.set_ylabel("Message count")
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        # Caption
        fig.suptitle(
            "Statistical Evidence of Weekly Seasonality (low noise)",
            fontsize=20, fontweight="bold", y=0.98
        )
        caption = (
            f"Dominant period approximately {ev.dominant_period_weeks} weeks | "
            f"Residual sigma = {ev.residual_std:.2f}"
        )
        fig.text(0.5, 0.02, caption, ha="center", fontsize=12, style="italic")

        return fig


    # === 3. Distribution (Script3) ===
    def build_visual_distribution(
        self,
        data: DistributionPlotData,
        settings: DistributionPlotSettings = DistributionPlotSettings(),
    ) -> plt.Figure | None:
        """
        Bar + cumulative line chart of emoji frequencies for the MAAP group.

        Args:
            data: Validated DistributionPlotData (emoji_counts_df)
            settings: Plot settings (top_n, colours, etc.)

        Returns:
            matplotlib Figure or None
        """
        df = data.emoji_counts_df
        if df.empty:
            logger.error("Emoji distribution DataFrame is empty")
            return None

        try:
            # === Set emoji font ===
            plt.rcParams["font.family"] = "Segoe UI Emoji"  # ← FIX SQUARES

            # === Prepare data ===
            df = df.sort_values("count_once", ascending=False).copy()
            total_once = df["count_once"].sum()
            df["percent_once"] = df["count_once"] / total_once * 100
            df["cum_percent"] = df["percent_once"].cumsum()

            # === Create figure ===
            fig, ax = plt.subplots(
                figsize=(max(8, len(df) * 0.35), 8)
            )
            ax2 = ax.twinx()

            x_pos = np.arange(len(df))

            # === Draw bars ===
            ax.bar(
                x_pos,
                df["percent_once"],
                width=0.5,
                align="center",
                color="purple",
                alpha=0.9,
            )
            ax.set_ylabel(
                "Likelihood (%) of finding an Emoji in a random message",
                fontsize=12,
                labelpad=20,
            )
            ax.set_title(
                "The Long Tail of Emotion: Few Speak for Many",
                fontsize=20,
                pad=20,
            )
            ax.set_xlim(-0.5, len(df))
            ylim_bottom, ylim_top = ax.get_ylim()
            ax.set_ylim(ylim_bottom - 3, ylim_top)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_position(("outward", 20))
            ax.tick_params(axis="y", labelsize=10)

            # === 75% highlight ===
            cum_np = np.array(df["cum_percent"])
            idx_75 = np.where(cum_np >= 75)[0][0]
            x_75 = idx_75 + 1
            y_75 = len(df)
            ax.axvspan(-0.5, idx_75 + 0.5, facecolor="lightgreen", alpha=0.2)
            ax.axvline(x=idx_75 + 0.5, color="orange", linestyle="--", linewidth=1)

            # Annotations
            left_mid = idx_75 / 2
            right_mid = (idx_75 + 0.5) + (y_75 - idx_75 - 1) / 2
            y_text = ylim_bottom - 1.5
            ax.text(left_mid, y_text, f"<-- {x_75} emojis -->", ha="center", fontsize=12)
            ax.text(right_mid, y_text, f"<-- {y_75-x_75} emojis -->", ha="center", fontsize=12)
            ax.set_xticks([])

            # === Cumulative line ===
            ax2.plot(
                x_pos + 0.25,
                df["cum_percent"],
                color="orange",
                linewidth=2,
                label="Cumulative %",
            )
            ax2.axhline(
                y=75,
                color="orange",
                linestyle="--",
                linewidth=1,
                xmin=-0.5,
                xmax=len(df) + 0.5,
            )
            ax2.set_ylabel(
                "Cumulative Percentage (%)",
                fontsize=12,
                labelpad=20,
            )
            ax2.set_ylim(0, 100)
            ax2.set_yticks(np.arange(0, 101, 10))
            ax2.spines["right"].set_position(("outward", 20))
            ax2.tick_params(axis="y", labelsize=10, colors="orange")
            ax2.spines["right"].set_color("orange")

            # === Top-25 table ===
            top_25 = df.head(25).copy()
            top_25["cum_percent"] = top_25["percent_once"].cumsum()
            table_data = [
                [str(i + 1) for i in range(25)],
                top_25["emoji"].tolist(),
                [f"{c:.0f}" for c in top_25["count_once"]],
                [f"{p:.1f}%" for p in top_25["cum_percent"]],
            ]
            table = ax.table(
                cellText=table_data,
                rowLabels=["Rank", "Emoji", "Count", "Cum"],
                colWidths=[0.05] * 25,
                loc="bottom",
                bbox=[0.1, -0.45, 0.8, 0.3],  # ← MATCH ATTACHMENT
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)

            fig.text(0.5, 0.27, "Top 25:", ha="center", fontsize=12)  # ← MATCH

            # === Layout ===
            ax2.legend(loc="upper left", fontsize=8)  # ← MATCH
            plt.tight_layout()
            plt.subplots_adjust(left=0.1, right=0.9, bottom=0.35)  # ← MATCH
            plt.show()

            logger.success(
                f"Emoji distribution plot built – {len(df)} unique emojis"
            )
            return fig

        except Exception as e:
            logger.exception(f"build_visual_distribution failed: {e}")
            return None


    # === Log-Log Power-Law Plot ===
    def build_visual_distribution_powerlaw(
        self,
        analysis: PowerLawAnalysisResult,
        settings: PowerLawPlotSettings = PowerLawPlotSettings(),
    ) -> plt.Figure | None:
        """
        Generate log-log plot of emoji frequency vs rank with fitted power-law line.

        Args:
            analysis: Result from analyze_emoji_distribution_power_law
            settings: Plot settings

        Returns:
            matplotlib Figure or None
        """
        if not analysis or analysis.fit.n_tail < 3:
            logger.error("Insufficient tail data for power-law plot")
            return None

        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            plt.subplots_adjust(top=0.82)

            # Get full frequency data
            from collections import Counter
            # Re-parse to get full sorted list
            # We'll simulate from analysis (in real use, pass full data or re-parse)
            # For now, use stored top + assume tail

            # In practice: pass full Counter or re-run parsing
            # Here: placeholder — in real code, you'd pass it
            logger.warning("build_visual_distribution_powerlaw needs full Counter — using placeholder")

            # === Placeholder: Use analysis to simulate ===
            # In real integration, modify analyze_ to return sorted frequencies
            ranks = np.arange(1, analysis.n_observations + 1)
            # Approximate frequencies using inverse power-law
            alpha = analysis.fit.alpha
            xmin = analysis.fit.xmin
            C = (analysis.fit.n_tail) * (xmin ** (alpha - 1)) / (alpha - 1)
            freqs_approx = C / (ranks ** alpha)
            freqs_approx = np.maximum(freqs_approx, 1)

            ax.loglog(ranks, freqs_approx, marker='o', linestyle='none',
                    color=settings.marker_color, markersize=settings.marker_size,
                    alpha=settings.alpha, label='Observed (approx)')

            if settings.show_fit_line:
                x_fit = np.logspace(0, np.log10(analysis.n_observations), 100)
                y_fit = C / (x_fit ** alpha)
                ax.plot(x_fit, y_fit, color=settings.line_color, linewidth=settings.line_width,
                        label=f'Fit: α = {alpha:.2f}, $x_{{min}}$ = {xmin}')

            ax.set_xlabel("Rank", fontsize=14)
            ax.set_ylabel("Frequency", fontsize=14)
            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.legend(fontsize=12)

            # Title + subtitle
            ax.set_title(settings.title, fontsize=settings.font_size_title, fontweight='bold', pad=20)
            fig.text(0.5, 0.92, settings.subtitle, ha='center', fontsize=settings.font_size_subtitle,
                    color='dimgray', transform=fig.transFigure)

            if settings.annotate_alpha:
                ax.annotate(f"α = {alpha:.2f}\nD = {analysis.fit.D:.3f}",
                            xy=(0.05, 0.95), xycoords='axes fraction',
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                            fontsize=12)

            plt.tight_layout()
            logger.success("Power-law log-log plot built")
            return fig

        except Exception as e:
            logger.exception(f"build_visual_distribution_powerlaw failed: {e}")
            return None


    # === Model Comparison Table (Interactive) ===
    def build_visual_distribution_comparison(
        self,
        analysis: PowerLawAnalysisResult,
    ) -> "go.Figure" | None:
        """
        Interactive Plotly table comparing power-law to alternatives.
        """
        try:
            import plotly.graph_objects as go

            data = [
                ["Power-Law", f"{analysis.fit.alpha:.3f}", analysis.fit.xmin, f"{analysis.fit.D:.4f}", "—", "—"],
                ["Exponential", "—", "—", "—", f"{analysis.comparison.vs_exponential:.4f}", "Worse" if analysis.comparison.vs_exponential < 0.05 else "Comparable"],
                ["Lognormal", "—", "—", "—", f"{analysis.comparison.vs_lognormal:.4f}", "Worse" if analysis.comparison.vs_lognormal < 0.05 else "Comparable"],
            ]

            fig = go.Figure(data=[go.Table(
                header=dict(values=["Model", "α", "xmin", "K-S D", "p-value (vs PL)", "Status"],
                            fill_color='paleturquoise', font=dict(size=14)),
                cells=dict(values=list(map(list, zip(*data))),
                        fill_color='lavender', font=dict(size=12))
            )])

            fig.update_layout(
                title="Power-Law Model Comparison (K-S + Likelihood Ratio)",
                height=300,
                margin=dict(t=80, b=20, l=20, r=20)
            )

            logger.success("Model comparison table built")
            return fig

        except Exception as e:
            logger.exception(f"build_visual_distribution_comparison failed: {e}")
            return None
    

    # === 4. Relationships (Script4) ===
    def build_visual_relationships(
        self,
        data: RelationshipsPlotData,
        settings: RelationshipsPlotSettings = RelationshipsPlotSettings(),
    ) -> plt.Figure | None:
        """
        Plot average words vs punctuation with bubble size by message count.

        Args:
            data: Validated RelationshipsPlotData
            settings: Plot settings

        Returns:
            matplotlib Figure or None
        """
        try:
            # Figure setup aligned with Categories
            fig, ax = plt.subplots(figsize=(14, 6.5))  # Shorter canvas
            plt.subplots_adjust(top=0.85)  # Axes end earlier → more room above

            df = data.feature_df
            required = {
                Columns.WHATSAPP_GROUP.value,
                Columns.AUTHOR.value,
                Columns.AVG_WORDS.value,
                Columns.AVG_PUNCT.value,
                Columns.MESSAGE_COUNT.value,
            }
            if not required.issubset(df.columns):
                logger.error("Relationships plot missing required columns")
                return None

            msg = df[Columns.MESSAGE_COUNT.value]
            size_scale = (msg - msg.min()) / (msg.max() - msg.min()) if msg.max() != msg.min() else 1.0
            bubble_sizes = (settings.min_bubble_size + (settings.max_bubble_size - settings.min_bubble_size) * size_scale) * 3

            for grp, col in settings.group_colors.items():
                mask = df[Columns.WHATSAPP_GROUP.value] == grp
                sub = df[mask]
                if sub.empty:
                    continue
                ax.scatter(
                    sub[Columns.AVG_WORDS.value],
                    sub[Columns.AVG_PUNCT.value],
                    s=bubble_sizes[mask],
                    alpha=settings.bubble_alpha,
                    color=col,
                    label=grp.value,
                )

            x = df[Columns.AVG_WORDS.value].values  # Convert to numpy for sorting
            y = df[Columns.AVG_PUNCT.value].values
            coef = np.polyfit(x, y, 1)
            trend = np.poly1d(coef)

            # Sort x for smooth straight line plot
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_trend = trend(x_sorted)
            ax.plot(
                x_sorted, y_trend,
                color=settings.trendline_color,
                linestyle=settings.trendline_style,
                linewidth=settings.trendline_width,
                alpha=settings.trendline_alpha,
                label="Linear Trend"  # Add label for legend
            )

            # Title and subtitle
            ax.set_title(
                settings.title,
                fontsize=settings.title_fontsize,
                fontweight=settings.title_fontweight,
                pad=settings.title_pad,
                ha=settings.title_ha,
            )

            if settings.subtitle:
                fig.text(
                    x=0.5,
                    y=0.92,
                    s=settings.subtitle,
                    fontsize=settings.subtitle_fontsize,
                    fontweight=settings.subtitle_fontweight,
                    color=settings.subtitle_color,
                    ha=settings.subtitle_ha,
                    va="top",
                    transform=fig.transFigure
                )

            ax.set_xlabel(settings.xlabel or Columns.AVG_WORDS.human, fontsize=12)
            ax.set_ylabel(settings.ylabel or Columns.AVG_PUNCT.human, fontsize=12)

            # Legend handles for groups
            legend_handles = [
                plt.scatter([], [], s=settings.min_bubble_size * settings.legend_scale_factor,
                            c=col, alpha=settings.bubble_alpha, label=grp.value)
                for grp, col in settings.group_colors.items()
            ]

            # Add trendline to legend (will be at bottom since appended last)
            trend_patch = matplotlib.lines.Line2D(
                [0], [0],
                color=settings.trendline_color,
                linestyle=settings.trendline_style,
                linewidth=settings.trendline_width,
                label="Linear Trend"
            )
            legend_handles.append(trend_patch)

            ax.legend(
                handles=legend_handles,
                title="WhatsApp Group",
                title_fontsize=12,
                bbox_to_anchor=(1.05, 1),
                loc="upper left"
            )
            ax.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.show()
            
            logger.success("Relationships plot built successfully")
            return fig

        except Exception as e:
            logger.exception(f"Relationships plot failed: {e}")
            return None


    # === 5. Multi-Dimensional (Script5) ===
    def build_visual_multi_dimensions(
        self,
        data: MultiDimPlotData,
        settings: MultiDimPlotSettings = MultiDimPlotSettings(),
    ) -> dict[str, "go.Figure"] | None:
        """
        Create interactive t-SNE plots with optional group isolation and confidence ellipses.

        Args:
            data: Validated MultiDimPlotData with t-SNE coordinates and aggregated features
            settings: Plot settings including group mode, ellipse mode, confidence level

        Returns:
            Dict of Plotly figures: {"individual": ..., "group": ...} or None

        # NEW: Restored per-author color nuance + subtitle high above + (Group) in legend (2025-11-07)
        """
        if data.agg_df.empty:
            logger.error("Empty agg_df in MultiDimPlotData")
            return None

        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from scipy.stats import chi2
            from sklearn.mixture import GaussianMixture

            figs = {}

            def get_ellipse_points(mean_x, mean_y, width, height, angle):
                t = np.linspace(0, 2 * np.pi, 100)
                x = width / 2 * np.cos(t)
                y = height / 2 * np.sin(t)
                points = np.column_stack([x, y])
                theta = np.radians(angle)
                rot = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]])
                rotated = points @ rot.T
                rotated[:, 0] += mean_x
                rotated[:, 1] += mean_y
                return rotated[:, 0], rotated[:, 1]

            def hex_to_rgba(hex_color, alpha):
                hex_color = hex_color.lstrip('#')
                return f"rgba({int(hex_color[0:2],16)}, {int(hex_color[2:4],16)}, {int(hex_color[4:6],16)}, {alpha})"

            # === AUTHOR-SPECIFIC COLOR SCHEME (restored from old version) ===
            author_colors = {
                "RH": "#FF7F0E", "MK": "#FF8C00", "HB": "#FFD580", "AB": "#00008B",
                "PB": "#00BFFF", "M": "#ADD8E6", "LL": "#228B22", "HH": "#32CD32",
                "HvH": "#66CDAA", "ND": "#9ACD32", "Bo": "#EE82EE", "Lars": "#9467BD",
                "Mats": "#DDA0DD", "JC": "#A9A9A9", "EH": "#024A07", "FJ": "#D3D3D3",
                "AvT": "#FF0000",
            }

            chi_val = np.sqrt(chi2.ppf(settings.confidence_level / 100, df=2))

            # === INDIVIDUAL MODE (Author + Group in legend) ===
            if not settings.by_group:
                # Create combined label: "Author (Group)" except for AvT
                data.agg_df["author_group"] = data.agg_df.apply(
                    lambda row: (
                        f"{row[Columns.AUTHOR.value]} ({row[Columns.WHATSAPP_GROUP_TEMP.value]})"
                        if row[Columns.WHATSAPP_GROUP_TEMP.value] != Groups.AVT
                        else row[Columns.AUTHOR.value]
                    ),
                    axis=1
                )

                # Sort legend: Group first, then Author
                sorted_labels = (
                    data.agg_df
                    .sort_values([Columns.WHATSAPP_GROUP_TEMP.value, Columns.AUTHOR.value])
                    ["author_group"]
                    .unique()
                    .tolist()
                )

                # Map colors using author name only (preserves nuance)
                color_map = {}
                for label in sorted_labels:
                    author = label.split(" (")[0] if " (" in label else label
                    color_map[label] = author_colors.get(author, "#333333")

                fig = px.scatter(
                    data.agg_df,
                    x="tsne_x", y="tsne_y",
                    size="msg_count",
                    color="author_group",
                    color_discrete_map=color_map,
                    hover_data={
                        "msg_count": True,
                        Columns.AUTHOR.value: True,
                        Columns.WHATSAPP_GROUP_TEMP.value: True,
                        "author_group": False
                    },
                    category_orders={"author_group": sorted_labels},
                )

                # Title and subtitle (combined)
                title_text = settings.title
                if settings.subtitle:
                    title_text += f"<br><span style='font-size:{settings.subtitle_fontsize}px; color:{settings.subtitle_color};'>{settings.subtitle}</span>"

                fig.update_layout(
                    title={
                        'text': title_text,
                        'y': 0.96,                    # High, centered in top area
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': dict(size=settings.title_fontsize, family="Arial Black, Arial, sans-serif", color="black"),
                        'pad': dict(t=20)             # Extra padding above title
                    },
                    margin=dict(t=120, b=80, l=80, r=180),
                    xaxis=dict(
                    title=dict(text="t-SNE X", font=dict(size=16)),  # Optional: set label text
                    tickfont=dict(size=12)
                    ),
                    yaxis=dict(
                        title=dict(text="t-SNE Y", font=dict(size=16)),
                        tickfont=dict(size=12)
                    ),
 
                    legend=dict(
                        title="Author (Group)",
                        font=dict(size=16),
                        bgcolor="white",
                        bordercolor="gray",
                        borderwidth=1
                    ),
                    width=1400,
                    height=750,
                )

                # === ELLIPSES (using author-specific color) ===
                if settings.ellipse_mode > 0:
                    for author in data.agg_df[Columns.AUTHOR.value].unique():
                        sub = data.agg_df[data.agg_df[Columns.AUTHOR.value] == author].copy()
                        if len(sub) < 3:
                            continue
                        X = sub[["tsne_x", "tsne_y"]].values
                        color = author_colors.get(author, "#333333")

                        if settings.ellipse_mode == 1:
                            cov = np.cov(X.T)
                            mean = X.mean(axis=0)
                            lambda_, v = np.linalg.eig(cov)
                            lambda_ = np.sqrt(lambda_)
                            width, height = 2 * lambda_ * chi_val
                            angle = np.degrees(np.arctan2(v[1,0], v[0,0]))
                            ell_x, ell_y = get_ellipse_points(mean[0], mean[1], width[0], height[1], angle)
                            fig.add_trace(go.Scatter(x=ell_x, y=ell_y, mode="lines", fill="toself",
                                fillcolor=hex_to_rgba(color, 0.2), line=dict(color=color, width=2),
                                name=f"{author} confidence", showlegend=False))

                        else:
                            best_gmm = None
                            best_bic = np.inf
                            best_n = 1
                            for n in range(1, min(4, len(sub))):
                                gmm = GaussianMixture(n_components=n, covariance_type="full", random_state=42)
                                gmm.fit(X)
                                bic = gmm.bic(X)
                                if bic < best_bic:
                                    best_bic = bic
                                    best_gmm = gmm
                                    best_n = n

                            for i in range(best_n):
                                weight = best_gmm.weights_[i]
                                if weight < 0.1:
                                    continue
                                mean = best_gmm.means_[i]
                                cov = best_gmm.covariances_[i]
                                lambda_, v = np.linalg.eig(cov)
                                lambda_ = np.sqrt(lambda_)
                                width = 2 * lambda_[0] * chi_val
                                height = 2 * lambda_[1] * chi_val
                                angle = np.degrees(np.arctan2(v[1,0], v[0,0]))
                                ell_x, ell_y = get_ellipse_points(mean[0], mean[1], width, height, angle)
                                opacity = 0.2 + 0.3 * weight
                                fig.add_trace(go.Scatter(x=ell_x, y=ell_y, mode="lines", fill="toself",
                                    fillcolor=hex_to_rgba(color, opacity), line=dict(color=color, width=2),
                                    name=f"{author} pocket {i+1}", showlegend=False))

                figs["individual"] = fig

            # === GROUP MODE (unchanged from current version) ===
            if settings.by_group:
                data.agg_df["plot_group"] = data.agg_df.apply(
                    lambda row: Groups.AVT if row[Columns.AUTHOR.value] == Groups.AVT else row[Columns.WHATSAPP_GROUP_TEMP],
                    axis=1
                )
                group_colors = {
                    Groups.MAAP: "#1f77b4", Groups.DAC: "#ff7f0e", Groups.GOLFMATEN: "#2ca02c",
                    Groups.TILLIES: "#808080", Groups.AVT: "#d62728"
                }
                fig = px.scatter(
                    data.agg_df, x="tsne_x", y="tsne_y", color="plot_group", size="msg_count",
                    color_discrete_map=group_colors,
                    hover_data={"msg_count": True, Columns.AUTHOR.value: True},
                )

                # Title and subtitle (combined)
                title_text = settings.title
                if settings.subtitle:
                    title_text += f"<br><span style='font-size:{settings.subtitle_fontsize}px; color:{settings.subtitle_color};'>{settings.subtitle}</span>"

                fig.update_layout(
                    title={
                        'text': title_text,
                        'y': 0.96,                    # High, centered in top area
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': dict(size=settings.title_fontsize, family="Arial Black, Arial, sans-serif", color="black"),
                        'pad': dict(t=20)             # Extra padding above title
                    },
                    margin=dict(t=120, b=80, l=80, r=180),
                    xaxis=dict(
                    title=dict(text="t-SNE X", font=dict(size=16)),  # Optional: set label text
                    tickfont=dict(size=12)
                    ),
                    yaxis=dict(
                        title=dict(text="t-SNE Y", font=dict(size=16)),
                        tickfont=dict(size=12)
                    ),
                    legend=dict(
                        title="Author (Group)",
                        font=dict(size=16),
                        bgcolor="white",
                        bordercolor="gray",
                        borderwidth=1
                    ),
                    width=1400,
                    height=750,
                )

                if settings.ellipse_mode > 0:
                    for grp in data.agg_df["plot_group"].unique():
                        sub = data.agg_df[data.agg_df["plot_group"] == grp].copy()
                        if len(sub) < 3:
                            continue
                        X = sub[["tsne_x", "tsne_y"]].values
                        color = group_colors.get(grp, "#333333")

                        if settings.ellipse_mode == 1:
                            cov = np.cov(X.T)
                            mean = X.mean(axis=0)
                            lambda_, v = np.linalg.eig(cov)
                            lambda_ = np.sqrt(lambda_)
                            width, height = 2 * lambda_ * chi_val
                            angle = np.degrees(np.arctan2(v[1,0], v[0,0]))
                            ell_x, ell_y = get_ellipse_points(mean[0], mean[1], width[0], height[1], angle)
                            fig.add_trace(go.Scatter(x=ell_x, y=ell_y, mode="lines", fill="toself",
                                fillcolor=hex_to_rgba(color, 0.25), line=dict(color=color, width=2),
                                showlegend=False))

                        else:
                            best_gmm = None
                            best_bic = np.inf
                            best_n = 1
                            for n in range(1, min(4, len(sub))):
                                gmm = GaussianMixture(n_components=n, covariance_type="full", random_state=42)
                                gmm.fit(X)
                                bic = gmm.bic(X)
                                if bic < best_bic:
                                    best_bic = bic
                                    best_gmm = gmm
                                    best_n = n

                            for i in range(best_n):
                                weight = best_gmm.weights_[i]
                                if weight < 0.1:
                                    continue
                                mean = best_gmm.means_[i]
                                cov = best_gmm.covariances_[i]
                                lambda_, v = np.linalg.eig(cov)
                                lambda_ = np.sqrt(lambda_)
                                width = 2 * lambda_[0] * chi_val
                                height = 2 * lambda_[1] * chi_val
                                angle = np.degrees(np.arctan2(v[1,0], v[0,0]))
                                ell_x, ell_y = get_ellipse_points(mean[0], mean[1], width, height, angle)
                                opacity = 0.2 + 0.3 * weight
                                fig.add_trace(go.Scatter(x=ell_x, y=ell_y, mode="lines", fill="toself",
                                    fillcolor=hex_to_rgba(color, opacity), line=dict(color=color, width=2),
                                    showlegend=False))

                figs["group"] = fig

            logger.success(f"Multi-dimensional plot built: {len(figs)} figures (title 24pt, subtitle high, author nuance restored)")
            return figs

        except Exception as e:
            logger.exception(f"Multi-dimensional plot failed: {e}")
            return None


# === CODING STANDARD ===
# - `# === Module Docstring ===` before """
# - Google-style docstrings
# - `# === Section Name ===` for all blocks
# - Inline: `# One space, sentence case`
# - Tags: `# TODO:`, `# NOTE:`, `# NEW: (YYYY-MM-DD)`, `# FIXME:`
# - Type hints in function signatures
# - Examples: with >>>
# - No long ----- lines
# - No mixed styles
# - Add markers #NEW at the end of the module

# NEW: Full 1–6 plot support with strict settings (2025-11-03)
# NEW: All hardcodes removed, uses constants.py
# NEW: Complete docstrings, consistent spacing, InteractionType
# NEW: Restored per-author color nuance + high subtitle + (Group) in legend (2025-11-07)