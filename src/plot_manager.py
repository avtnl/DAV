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
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Tuple, Dict

from .constants import Columns, Groups
from .data_preparation import (
    CategoryPlotData,
    TimePlotData,
    SeasonalityEvidence,
    DistributionPlotData,
    RelationshipsPlotData,
    MultiDimPlotData,
    MultiDimPlotSettings,
    PowerLawAnalysisResult,
    PowerLawFitResult
)

warnings.simplefilter(action="ignore", category=FutureWarning)

# === STANDARD CANVAS SIZES ===
WIDE_TALL = (14, 8)        # Distribution, PowerLaw, Seasonality
WIDE_COMPACT = (14, 6.5)    # Categories, Time, Relationships

# === Base Settings ===
class PlotSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    figsize: Tuple[float, float] = Field((11.0, 7.8), description="A4 landscape with margins")
    dpi: int = Field(300, ge=100, le=600)
    font_family: str = "Arial"
    tight_layout: bool = True
    style: str = "whitegrid"

    xlabel: str = ""
    ylabel: str = ""
    grid_alpha: float = 0.3
    grid_linestyle: str = "--"

    title: str = ""
    title_fontsize: int = 24
    title_fontweight: str = "bold"
    title_pad: float = 40.0
    title_ha: str = "center"

    subtitle: str = ""
    subtitle_fontsize: int = 18
    subtitle_fontweight: str = "bold"
    subtitle_color: str = "dimgray"
    subtitle_y: float = 0.92
    subtitle_ha: str = "center"

    legend: bool = True
    legend_loc: str = "upper right"
    legend_bbox_to_anchor: Tuple[float, float] | None = None
    legend_borderaxespad: float = 0.0
    legend_frameon: bool = True
    legend_fontsize: int = 12

# === 1. Categories ===
class CategoriesPlotSettings(PlotSettings):
    title: str = "AvT's participation is much lower for the 3rd whatsapp group!"
    subtitle: str = ""
    ylabel: str = Columns.MESSAGE_COUNT.human
    group_spacing: float = Field(2.5, ge=0.5, le=5.0)
    bar_width: float = Field(0.8, ge=0.1, le=1.0)
    trendline_color: str = "red"
    trendline_style: str = "--"
    trendline_width: float = 2.5
    tight_layout: bool = False
    figsize: Tuple[float, float] = WIDE_COMPACT
    subtitle_y: float = 0.88

# === 2. Time ===
class TimePlotSettings(PlotSettings):
    title: str = "Golf season, decoded by WhatsApp heartbeat"
    subtitle: str = "Whatsapp group = 'dac'"
    rest_label: str = "---------Rest---------"
    prep_label: str = "---Prep---"
    play_label: str = "---------Play---------"
    line_color: str = "green"
    linewidth: float = 2.5
    figsize: Tuple[float, float] = WIDE_COMPACT
    subtitle_y: float = 0.89

class SeasonalityPlotSettings(PlotSettings):
    figsize: Tuple[float, float] = (16, 10)
    subplot_titles_fontsize: int = 14
    acf_color: str = "#1f77b4"
    decomp_linewidth: float = 1.8
    fourier_marker: str = "o"
    fourier_markersize: int = 6
    filtered_alpha: float = 0.7

# === 3. Distribution ===
class DistributionPlotSettings(PlotSettings):
    figsize: Tuple[float, float] = WIDE_TALL
    bar_color: str = "purple"
    cumulative_color: str = "orange"
    bar_alpha: float = 0.9
    line_width: float = 2.0
    cum_label: str = "Cumulative %"
    cum_threshold: float = 75.0
    top_table: int = 25  # For table, not bars
    subtitle_y: float = 0.96

# Power-Law Plot Settings
class PowerLawPlotSettings(PlotSettings):
    figsize: Tuple[float, float] = WIDE_TALL
    title: str = "Emoji Frequency Follows Power-Law (Zipf) Distribution"
    subtitle: str = "Log-log plot shows linear relationship to proof of log model"
    line_color: str = "red"
    line_width: float = 2.5
    marker_color: str = "blue"
    marker_size: int = 6
    alpha: float = 0.8
    show_fit_line: bool = True
    annotate_alpha: bool = True
    tight_layout: bool = False
    subtitle_y: float = 0.96

    model_config = ConfigDict(arbitrary_types_allowed=True)


# === 4. Relationships ===
class RelationshipsPlotSettings(PlotSettings):
    title: str = "Words vs Punctuation per Author (Bubble Size = Message Count)"
    subtitle: str = "~1 extra punctuation per 10 words"
    figsize: Tuple[float, float] = WIDE_COMPACT
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
    group_colors: Dict[str, str] = Field(
        default_factory=lambda: {
            Groups.MAAP: "deepskyblue",
            Groups.GOLFMATEN: "orange",
            Groups.DAC: "green",
            Groups.TILLIES: "gray",
        }
    )
    subtitle_y: float = 0.88

# === 5. Multi Dimensions ===
class MultiDimPlotSettings(PlotSettings):
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
        if not data or not data.groups:
            logger.error("Invalid or empty CategoryPlotData")
            return None

        try:
            fig, ax = plt.subplots(figsize=settings.figsize)
            plt.subplots_adjust(top=0.85)

            group_colors = {
                Groups.DAC: "green",
                Groups.GOLFMATEN: "orange",
                Groups.MAAP: "deepskyblue",
                Groups.TILLIES: "gray",
            }

            x_pos: list[float] = []
            heights: list[int] = []
            bar_colors: list[str] = []
            author_labels: list[str] = []
            group_midpoints: list[float] = []
            cur_x = 0.0

            maap_avt_x = maap_avt_h = maap_group_avg = None

            for group_data in data.groups:
                n_auth = len(group_data.authors)
                start_x = cur_x

                for auth in group_data.authors:
                    x_pos.append(cur_x)
                    heights.append(auth.message_count)

                    group_color = group_colors.get(group_data.whatsapp_group, "gray")
                    bar_colors.append(group_color)
                    author_labels.append(auth.author)

                    if group_data.whatsapp_group == Groups.MAAP and auth.is_avt:
                        maap_avt_x = cur_x + 1.5
                        maap_avt_h = auth.message_count
                        maap_group_avg = group_data.group_avg

                    cur_x += 1.0

                mid = start_x + (n_auth - 1) / 2
                group_midpoints.append(mid)
                cur_x += settings.group_spacing

            # AvT: black border + hatch
            edgecolors = ["black"] * len(x_pos)
            linewidths = [2.0 if auth.is_avt else 1.3 for g in data.groups for auth in g.authors]
            hatches = ["//" if auth.is_avt else "" for g in data.groups for auth in g.authors]

            ax.bar(
                x_pos, heights,
                width=settings.bar_width,
                color=bar_colors,
                edgecolor=edgecolors,
                linewidth=linewidths,
                hatch=hatches,
                align="center",
            )

            # Group average lines
            avg_line_patch = None
            for mid_x, gdata in zip(group_midpoints, data.groups):
                if gdata.group_avg > 0:
                    line_length = 0.5 * len(gdata.authors)
                    line = ax.hlines(
                        gdata.group_avg,
                        mid_x - line_length, mid_x + line_length,
                        color="red", linestyle="--", linewidth=2.5,
                        zorder=5,
                        label="Group Avg (non-AvT)" if avg_line_patch is None else ""
                    )
                    avg_line_patch = avg_line_patch or line

            # MAAP AvT arrow
            if maap_group_avg and maap_avt_x and maap_avt_h:
                ax.annotate(
                    "", xy=(maap_avt_x, maap_avt_h), xytext=(maap_avt_x, maap_group_avg),
                    arrowprops=dict(
                        arrowstyle="<->,head_length=2.0,head_width=2.0",
                        color="red", lw=3.0
                    ), zorder=6
                )

            # Title and subtitle
            ax.set_title(settings.title, fontsize=settings.title_fontsize, fontweight=settings.title_fontweight,
                         pad=settings.title_pad, ha=settings.title_ha)
            if settings.subtitle:
                fig.text(0.5, settings.subtitle_y, settings.subtitle,
                         fontsize=settings.subtitle_fontsize, fontweight=settings.subtitle_fontweight,
                         color=settings.subtitle_color, ha=settings.subtitle_ha, va="center",
                         transform=fig.transFigure)

            ax.set_ylabel(settings.ylabel, fontsize=12)
            ax.set_xlabel("WhatsApp Groups and participating Authors", fontsize=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(author_labels, rotation=45, ha="right", fontsize=12)

            # Legend
            legend_patches = [
                patches.Patch(facecolor=c, edgecolor="black", label=g.value.title())
                for g, c in group_colors.items()
            ]
            legend_patches.append(patches.Patch(facecolor="black", edgecolor="black", label=Groups.AVT))
            if avg_line_patch:
                legend_patches.append(avg_line_patch)

            ax.legend(handles=legend_patches, title="Group", loc=settings.legend_loc,
                      # bbox_to_anchor=settings.legend_bbox_to_anchor or (1.15, 0.9),
                      bbox_to_anchor=None,
                      frameon=settings.legend_frameon, fontsize=settings.legend_fontsize)

            # Manually adjust margins to extend axes right border
            fig.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.85)  # right=0.95 pushes border right; adjust as needed

            ax.grid(True, axis="y", linestyle=settings.grid_linestyle, alpha=settings.grid_alpha, zorder=0)
            ax.set_axisbelow(True)
            # plt.tight_layout()
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
        include_seasonality: bool = True,
        season_settings: SeasonalityPlotSettings | None = None,
    ) -> Dict[str, Figure]:
        figs: Dict[str, Figure] = {}

        fig_category, ax = plt.subplots(figsize=settings.figsize)
        plt.subplots_adjust(top=0.85)

        weeks = sorted(data.weekly_avg.keys())
        avg_counts = [data.weekly_avg[w] for w in weeks]

        ax.plot(weeks, avg_counts, color=settings.line_color, linewidth=settings.linewidth, zorder=6)

        vlines = [11.5, 18.5, 34.5]
        starts = [1] + [int(v + 0.5) for v in vlines]
        ends = [int(v + 0.5) - 1 for v in vlines] + [52]
        period_labels = [settings.rest_label, settings.prep_label, settings.play_label, settings.rest_label]
        period_colors = ["#e8f5e9", "#c8e6c9", "#81c784", "#e8f5e9"]

        def period_avg(start: int, end: int) -> float:
            vals = [data.weekly_avg.get(w, 0.0) for w in range(start, end + 1)]
            return float(np.mean(vals)) if vals else 0.0

        period_avg_patch = None
        y_min, y_max = ax.get_ylim()
        label_y = y_min + 0.80 * (y_max - y_min)

        for i in range(4):
            s, e = starts[i], ends[i]
            ax.axvspan(s - 0.5, e + 0.5, facecolor=period_colors[i], alpha=0.6, zorder=0)
            p_avg = period_avg(s, e)
            if p_avg > 0:
                line = ax.hlines(p_avg, s - 0.5, e + 0.5, color="red", linestyle="--", linewidth=1.2, zorder=4,
                                label="Period Avg" if period_avg_patch is None else "")
                period_avg_patch = period_avg_patch or line
            mid = (s + e) / 2
            ax.text(mid, label_y, period_labels[i], ha="center", va="center", fontsize=12, fontweight="bold", zorder=7)

        for v in vlines:
            ax.axvline(v, color="gray", linestyle="--", alpha=0.6, zorder=1)

        key_dates = {11: "March 15th", 18: "May 1st", 36: "September 1st"}
        ax.set_xticks(sorted(key_dates.keys()))
        ax.set_xticklabels([key_dates[w] for w in sorted(key_dates)], fontsize=12, ha="center")
        ax.set_xlabel("Time over Year", fontsize=12, labelpad=10)
        ax.set_ylabel("Average message count per week (2017 - 2025)", fontsize=12)

        ax.set_title(settings.title, fontsize=settings.title_fontsize, fontweight=settings.title_fontweight,
                     pad=settings.title_pad, ha=settings.title_ha)
        if settings.subtitle:
            fig_category.text(0.5, settings.subtitle_y, settings.subtitle,
                              fontsize=settings.subtitle_fontsize, fontweight=settings.subtitle_fontweight,
                              color=settings.subtitle_color, ha=settings.subtitle_ha, va="top",
                              transform=fig_category.transFigure)

        legend_handles = [period_avg_patch] if period_avg_patch else []
        ax.legend(handles=legend_handles, loc="upper right", fontsize=12)

        ax.grid(True, axis="y", linestyle=settings.grid_linestyle, alpha=settings.grid_alpha, zorder=0)
        ax.set_axisbelow(True)
        plt.tight_layout() if settings.tight_layout else None
        plt.show()

        figs["category"] = fig_category
        logger.success("Time plot built")

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
        """
        Generate 4-panel seasonality evidence suite:
        1. ACF
        2. Decomposition
        3. Fourier amplitudes
        4. Smoothing filters

        Args:
            time_data: TimePlotData with seasonality evidence
            settings: SeasonalityPlotSettings for figure size and styling

        Returns:
            matplotlib Figure or None
        """
        if time_data.seasonality is None:
            logger.warning("No seasonality evidence available.")
            return None

        try:
            ev = time_data.seasonality
            fig = plt.figure(figsize=settings.figsize)
            gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

            # === 1. ACF ===
            ax0 = fig.add_subplot(gs[0, 0])
            lags = range(len(ev.acf))
            ax0.stem(lags, ev.acf, basefmt=" ", linefmt=settings.acf_color)
            ax0.set_title("Autocorrelation (weekly lags)", fontsize=settings.subplot_titles_fontsize)
            ax0.set_xlabel("Lag (weeks)")
            ax0.set_ylabel("ACF")
            ax0.axhline(0, color="k", linewidth=0.8)
            ax0.grid(True, alpha=0.3)

            # === 2. Decomposition ===
            ax1 = fig.add_subplot(gs[0, 1])
            idx = range(len(ev.decomposition["trend"]))
            ax1.plot(idx, ev.decomposition["trend"], label="Trend", linewidth=settings.decomp_linewidth)
            ax1.plot(idx, ev.decomposition["seasonal"], label="Seasonal", linewidth=settings.decomp_linewidth)
            ax1.plot(idx, ev.decomposition["resid"], label="Residual", linewidth=settings.decomp_linewidth, alpha=0.6)
            ax1.set_title("Additive Decomposition", fontsize=settings.subplot_titles_fontsize)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)

            # === 3. Fourier ===
            ax2 = fig.add_subplot(gs[1, 0])
            freqs, amps = ev.fourier["freqs"], ev.fourier["amps"]
            ax2.stem(freqs, amps, linefmt="C3-", markerfmt=f"C3{settings.fourier_marker}", basefmt=" ")
            ax2.set_title("Fourier amplitudes (top k)", fontsize=settings.subplot_titles_fontsize)
            ax2.set_xlabel("Frequency (cycles/week)")
            ax2.set_ylabel("Amplitude")
            ax2.grid(True, alpha=0.3)

            # === 4. Filters ===
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

            # === Caption ===
            caption = (
                f"Dominant period ~{ev.dominant_period_weeks} weeks | "
                f"Residual σ = {ev.residual_std:.2f} | n = {len(ev.raw_series)} weeks"
            )
            fig.suptitle(caption, fontsize=12, y=0.02, va='bottom', ha='center')

            plt.tight_layout()

            logger.success("Seasonality evidence suite built")
            return fig

        except Exception as e:
            logger.exception(f"_build_seasonality_suite failed: {e}")
            return None


    # === 3. Distribution (Script3) ===
    def build_visual_distribution(
        self,
        data: DistributionPlotData,
        settings: DistributionPlotSettings = DistributionPlotSettings(),
    ) -> plt.Figure | None:
        """
        Generate emoji frequency bar chart with cumulative percentage line.
        Handles 0, 1, or many emojis with dynamic width and robust plotting.
        """
        df = data.emoji_counts_df
        if df.empty:
            logger.error("Emoji distribution DataFrame is empty")
            return None

        try:
            # === DYNAMIC WIDTH: Original behavior restored ===
            n_emojis = len(df)
            figsize = (max(8, n_emojis * 0.35), 8)
            fig, ax = plt.subplots(figsize=figsize)
            ax2 = ax.twinx()

            plt.rcParams["font.family"] = "Segoe UI Emoji"

            # === Sort and compute percentages ===
            df = df.sort_values("count_once", ascending=False).copy()
            total_once = df["count_once"].sum()
            df["percent_once"] = df["count_once"] / total_once * 100
            df["cum_percent"] = df["percent_once"].cumsum()

            x_pos = np.arange(len(df))

            # === BARS: Always visible ===
            ax.bar(
                x_pos, df["percent_once"],
                width=0.5, align="center",
                color=settings.bar_color,
                alpha=settings.bar_alpha
            )
            ax.set_ylabel("Likelihood (%) of finding an Emoji in a random message", fontsize=12, labelpad=20)
            ax.set_title("The Long Tail of Emotion: Few Speak for Many", fontsize=20, pad=20)
            ax.set_xlim(-0.5, len(df) - 0.5)
            ylim_bottom, ylim_top = ax.get_ylim()
            ax.set_ylim(ylim_bottom - 3, ylim_top)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_position(("outward", 20))
            ax.tick_params(axis="y", labelsize=10)

            # === 75% THRESHOLD MARKERS ===
            cum_np = np.array(df["cum_percent"])
            idx_75 = np.where(cum_np >= 75)[0]
            if len(idx_75) > 0:
                idx_75 = idx_75[0]
                x_75 = idx_75 + 1
                ax.axvspan(-0.5, idx_75 + 0.5, facecolor="lightgreen", alpha=0.2)
                ax.axvline(x=idx_75 + 0.5, color="orange", linestyle="--", linewidth=1)

                left_mid = idx_75 / 2
                right_mid = (idx_75 + 0.5) + (len(df) - idx_75 - 1) / 2
                y_text = ylim_bottom - 1.5
                ax.text(left_mid, y_text, f"<-- {x_75} emojis -->", ha="center", fontsize=12)
                ax.text(right_mid, y_text, f"--> {len(df) - x_75} emojis -->", ha="center", fontsize=12)

            # === CUMULATIVE LINE: 1 vs 2+ points ===
            if len(df) == 1:
                # Single emoji → show bold dot + "100%" label
                ax2.scatter(x_pos, df["cum_percent"], color=settings.cumulative_color,
                            s=120, zorder=5, label=settings.cum_label)
                ax2.text(x_pos[0], df["cum_percent"].iloc[0], "100%", 
                        ha='center', va='bottom', fontsize=11, color=settings.cumulative_color,
                        fontweight='bold')
            else:
                # 2+ emojis → full line
                ax2.plot(x_pos, df["cum_percent"], color=settings.cumulative_color,
                        linewidth=settings.line_width, label=settings.cum_label, zorder=4)

            # === Right axis (cumulative %) ===
            ax2.set_ylim(0, 100)
            ax2.set_yticks(np.arange(0, 101, 20))
            ax2.spines["right"].set_position(("outward", 20))
            ax2.tick_params(axis="y", labelsize=10, colors="orange")
            ax2.spines["right"].set_color("orange")

            # === TOP N TABLE (DYNAMIC: fix for small len(df)) ===
            n_top = min(settings.top_table, n_emojis)
            top_n = df.head(n_top).copy()
            top_n["cum_percent"] = top_n["percent_once"].cumsum()
            table_data = [
                [str(i + 1) for i in range(n_top)],
                top_n["emoji"].tolist(),
                [f"{c:.0f}" for c in top_n["count_once"]],
                [f"{p:.1f}%" for p in top_n["cum_percent"]],
            ]
            # Adjust bbox width to fit small tables (add padding)
            table_width = min(0.8, n_top * 0.05 + 0.1)
            table = ax.table(
                cellText=table_data,
                rowLabels=["Rank", "Emoji", "Count", "Cum"],
                colWidths=[0.05] * n_top,
                loc="bottom",
                bbox=[0.1, -0.45, table_width, 0.3],
            )

            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.5)

            # Dynamic label
            fig.text(0.5, 0.27, f"Top {n_top}:", ha="center", fontsize=12)

            # === Final layout ===
            ax2.legend(loc="upper right", fontsize=8)
            plt.tight_layout()
            plt.subplots_adjust(left=0.1, right=0.9, bottom=0.35)
            plt.show()

            logger.success(f"Emoji distribution plot built – {len(df)} unique emojis")
            return fig

        except Exception as e:
            logger.exception(f"build_visual_distribution failed: {e}")
            return None


    # === Power-Law (Script3) ===
    def build_visual_distribution_powerlaw(
        self,
        analysis: PowerLawAnalysisResult,
        settings: PowerLawPlotSettings = PowerLawPlotSettings(),
    ) -> plt.Figure | None:
        """Log-log power-law fit visualisation."""
        if not analysis or analysis.fit.n_tail < 3:
            logger.error("Insufficient tail data for power-law plot")
            return None

        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            plt.subplots_adjust(top=0.85)

            ranks = np.arange(1, analysis.n_observations + 1)
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
            settings: RelationshipsPlotSettings with styling

        Returns:
            matplotlib Figure or None
        """
        try:
            # === Canvas & Layout ===
            fig, ax = plt.subplots(figsize=settings.figsize)
            plt.subplots_adjust(top=0.85)

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

            # === Bubble sizing ===
            msg = df[Columns.MESSAGE_COUNT.value]
            size_scale = (msg - msg.min()) / (msg.max() - msg.min()) if msg.max() != msg.min() else 1.0
            bubble_sizes = (settings.min_bubble_size + (settings.max_bubble_size - settings.min_bubble_size) * size_scale) * 3

            # === Plot each group ===
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
                    label=grp.value
                )

            # === Trendline ===
            x = df[Columns.AVG_WORDS.value].values
            y = df[Columns.AVG_PUNCT.value].values
            coef = np.polyfit(x, y, 1)
            trend = np.poly1d(coef)
            sort_idx = np.argsort(x)
            ax.plot(x[sort_idx], trend(x[sort_idx]), color=settings.trendline_color,
                    linestyle=settings.trendline_style, linewidth=settings.trendline_width,
                    alpha=settings.trendline_alpha, label="Linear Trend")

            # === Title & Subtitle ===
            ax.set_title(
                settings.title,
                fontsize=settings.title_fontsize,
                fontweight=settings.title_fontweight,
                pad=settings.title_pad,
                ha=settings.title_ha
            )
            if settings.subtitle:
                fig.text(
                    0.5, settings.subtitle_y, settings.subtitle,
                    fontsize=settings.subtitle_fontsize,
                    fontweight=settings.subtitle_fontweight,
                    color=settings.subtitle_color,
                    ha=settings.subtitle_ha, va="center",
                    transform=fig.transFigure
                )

            # === Axis labels ===
            ax.set_xlabel(settings.xlabel or Columns.AVG_WORDS.human, fontsize=12)
            ax.set_ylabel(settings.ylabel or Columns.AVG_PUNCT.human, fontsize=12)

            # === Legend with Bubble Size Explanation ===
            legend_handles = [
                plt.scatter([], [], s=settings.min_bubble_size * settings.legend_scale_factor,
                            c=col, alpha=settings.bubble_alpha, label=grp.value)
                for grp, col in settings.group_colors.items()
            ]
            trend_patch = matplotlib.lines.Line2D([0], [0], color=settings.trendline_color,
                                                 linestyle=settings.trendline_style, linewidth=settings.trendline_width,
                                                 label="Linear Trend")
            legend_handles.append(trend_patch)

            # Add bubble size explanation (3 sizes)
            min_size = settings.min_bubble_size
            max_size = settings.max_bubble_size
            mid_size = (min_size + max_size) // 2

            legend_handles.append(
                plt.scatter([], [], s=min_size, c="gray", alpha=0.7, label="Few messages")
            )
            legend_handles.append(
                plt.scatter([], [], s=mid_size, c="gray", alpha=0.7, label="Medium")
            )
            legend_handles.append(
                plt.scatter([], [], s=max_size, c="gray", alpha=0.7, label="Many messages")
            )

            ax.legend(
                handles=legend_handles,
                title="WhatsApp Group & Bubble Size",
                title_fontsize=12,
                loc="lower right",
                bbox_to_anchor=(1.0, 0.0),
                bbox_transform=ax.transAxes,
                fontsize=settings.legend_fontsize,
                frameon=True,
                borderaxespad=0.5,
                ncol=1  # Keep single column
            )

            # === Layout ===
            fig.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.85)

            # === Grid & Finalize ===
            ax.grid(True, linestyle=settings.grid_linestyle, alpha=settings.grid_alpha)
            # plt.tight_layout()
            plt.show()

            logger.success("Relationships plot built successfully")
            return fig

        except Exception as e:
            logger.exception(f"Relationships plot failed: {e}")
            return None


    # === 5. Multi Dimensions (Script5) ===
    def build_visual_multi_dimensions(
        self,
        data: MultiDimPlotData,
        settings: MultiDimPlotSettings,
    ) -> Dict[str, Figure] | None:
        """
        Build interactive PCA or t-SNE scatter plots with optional confidence ellipses.

        Args:
            data: Validated MultiDimPlotData (must contain agg_df + plot_type).
            settings: Plot configuration.

        Returns:
            Dict of Plotly figures keyed by mode (individual / group).
            For PLOT_TYPE="both" returns prefixed keys (pca_…, tsne_…).
        """
        # === 1. Strict type guard ===
        if not isinstance(data, MultiDimPlotData):
            logger.error("Invalid input: expected MultiDimPlotData, received %s", type(data).__name__)
            return None
        if data.agg_df.empty:
            logger.error("agg_df is empty")
            return None

        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from scipy.stats import chi2
            from sklearn.mixture import GaussianMixture

            # === 2. Resolve plot_type ===
            plot_type = data.plot_type.lower()
            if plot_type not in {"pca", "tsne", "both"}:
                logger.warning("Invalid plot_type '%s' – falling back to 'tsne'", plot_type)
                plot_type = "tsne"

            # === 3. "both" → recursive generation ===
            if plot_type == "both":
                pca_data = data.model_copy(update={"plot_type": "pca"})
                tsne_data = data.model_copy(update={"plot_type": "tsne"})
                pca_figs = self.build_visual_multi_dimensions(pca_data, settings)
                tsne_figs = self.build_visual_multi_dimensions(tsne_data, settings)
                if not pca_figs or not tsne_figs:
                    logger.error("Failed to generate one half of 'both' mode")
                    return None
                return {f"pca_{k}": v for k, v in pca_figs.items()} | \
                    {f"tsne_{k}": v for k, v in tsne_figs.items()}

            # === 4. Choose columns / titles ===
            if plot_type == "pca":
                prefix = "PCA"
                x_col, y_col = "pca_1", "pca_2"
                x_title, y_title = "PCA Component 1", "PCA Component 2"
                required = {"pca_1", "pca_2"}
            else:  # tsne
                prefix = "TSNE"
                x_col, y_col = "tsne_x", "tsne_y"
                x_title, y_title = "t-SNE X", "t-SNE Y"
                required = {"tsne_x", "tsne_y"}

            missing = required - set(data.agg_df.columns)
            if missing:
                logger.error("Missing %s coordinates in agg_df", missing)
                return None

            # === 5. Dynamic subtitle ===
            base_subtitle = "plot combining 25 style features and style oriented Hugging Faces"
            settings.subtitle = f"{prefix} {base_subtitle}"

            # === 6. Confidence scaling ===
            chi_val = np.sqrt(chi2.ppf(settings.confidence_level / 100, df=2))

            # === 7. Helper utilities ===
            def hex_to_rgba(hex_color: str, alpha: float) -> str:
                h = hex_color.lstrip("#")
                rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
                return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"

            def get_ellipse_points(cx, cy, w, h, angle_deg):
                t = np.linspace(0, 2*np.pi, 100)
                ex = cx + w/2*np.cos(t)
                ey = cy + h/2*np.sin(t)
                rot = np.array([[np.cos(np.radians(angle_deg)), -np.sin(np.radians(angle_deg))],
                                [np.sin(np.radians(angle_deg)),  np.cos(np.radians(angle_deg))]])
                pts = np.vstack([ex, ey])
                return (rot @ pts).T[:, 0], (rot @ pts).T[:, 1]

            figs: Dict[str, Figure] = {}

            # === 8. Individual-author mode ===
            if not settings.by_group:
                df = data.agg_df.copy()
                df["author_group"] = df.apply(
                    lambda r: (
                        f"{r[Columns.AUTHOR.value]} ({r[Columns.WHATSAPP_GROUP_TEMP.value]})"
                        if r[Columns.WHATSAPP_GROUP_TEMP.value] != Groups.AVT
                        else r[Columns.AUTHOR.value]
                    ),
                    axis=1,
                )
                sorted_labels = (
                    df.sort_values([Columns.WHATSAPP_GROUP_TEMP.value, Columns.AUTHOR.value])
                    ["author_group"].unique().tolist()
                )
                author_colors = {
                    "RH": "#FF7F0E","MK":"#FF8C00","HB":"#FFD580","AB":"#00008B",
                    "PB":"#00BFFF","M":"#ADD8E6","LL":"#228B22","HH":"#32CD32",
                    "HvH":"#66CDAA","ND":"#9ACD32","Bo":"#EE82EE","Lars":"#9467BD",
                    "Mats":"#DDA0DD","JC":"#A9A9A9","EH":"#024A07","FJ":"#D3D3D3",
                    "AvT":"#FF0000",
                }
                color_map = {lbl: author_colors.get(lbl.split(" (")[0], "#333333") for lbl in sorted_labels}

                fig = px.scatter(
                    df, x=x_col, y=y_col, size="msg_count", color="author_group",
                    color_discrete_map=color_map,
                    hover_data={"msg_count":True, Columns.AUTHOR.value:True,
                                Columns.WHATSAPP_GROUP_TEMP.value:True,
                                "author_group":False, x_col:False, y_col:False},
                    category_orders={"author_group": sorted_labels},
                )

                # === Ellipses Per Author ===
                if settings.ellipse_mode > 0:
                    for author in df[Columns.AUTHOR.value].unique():
                        sub = df[df[Columns.AUTHOR.value] == author]
                        if len(sub) < 3: continue
                        X = sub[[x_col, y_col]].values
                        col = author_colors.get(author, "#333333")

                        if settings.ellipse_mode == 1:          # single ellipse
                            cov = np.cov(X.T)
                            mu = X.mean(axis=0)
                            lam, v = np.linalg.eig(cov)
                            lam = np.sqrt(lam)
                            w, h = 2*lam*chi_val
                            ang = np.degrees(np.arctan2(v[1,0], v[0,0]))
                            ex, ey = get_ellipse_points(mu[0], mu[1], w[0], h[1], ang)
                            fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", fill="toself",
                                                    fillcolor=hex_to_rgba(col, 0.2),
                                                    line=dict(color=col, width=2),
                                                    showlegend=False))

                        else:                                   # GMM pockets
                            best_gmm, best_bic, best_n = None, float('inf'), 1
                            for n in range(1, min(4, len(sub))):
                                g = GaussianMixture(n_components=n, covariance_type="full", random_state=42)
                                g.fit(X)
                                bic = g.bic(X)
                                if bic < best_bic:
                                    best_bic, best_gmm, best_n = bic, g, n
                            for i in range(best_n):
                                wgt = best_gmm.weights_[i]
                                if wgt < 0.1: continue
                                mu = best_gmm.means_[i]
                                cov = best_gmm.covariances_[i]
                                lam, v = np.linalg.eig(cov)
                                lam = np.sqrt(lam)
                                w, h = 2*lam*chi_val
                                ang = np.degrees(np.arctan2(v[1,0], v[0,0]))
                                ex, ey = get_ellipse_points(mu[0], mu[1], w, h, ang)
                                op = 0.2 + 0.3*wgt
                                fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", fill="toself",
                                                        fillcolor=hex_to_rgba(col, op),
                                                        line=dict(color=col, width=2),
                                                        showlegend=False))

                figs["individual"] = fig

            # === 9. Group-level Mode ===
            if settings.by_group:
                df = data.agg_df.copy()
                df["plot_group"] = df.apply(
                    lambda r: Groups.AVT if r[Columns.AUTHOR.value] == Groups.AVT else r[Columns.WHATSAPP_GROUP_TEMP],
                    axis=1,
                )
                group_colors = {
                    Groups.MAAP: "#1f77b4", Groups.DAC: "#ff7f0e",
                    Groups.GOLFMATEN: "#2ca02c", Groups.TILLIES: "#808080",
                    Groups.AVT: "#d62728",
                }
                fig = px.scatter(
                    df, x=x_col, y=y_col, size="msg_count", color="plot_group",
                    color_discrete_map=group_colors,
                    hover_data={"msg_count":True, Columns.AUTHOR.value:True,
                                x_col:False, y_col:False},
                )

                # === Ellipses Per Group ===
                if settings.ellipse_mode > 0:
                    for grp in df["plot_group"].unique():
                        sub = df[df["plot_group"] == grp]
                        if len(sub) < 3: continue
                        X = sub[[x_col, y_col]].values
                        col = group_colors.get(grp, "#333333")

                        if settings.ellipse_mode == 1:
                            cov = np.cov(X.T)
                            mu = X.mean(axis=0)
                            lam, v = np.linalg.eig(cov)
                            lam = np.sqrt(lam)
                            w, h = 2*lam*chi_val
                            ang = np.degrees(np.arctan2(v[1,0], v[0,0]))
                            ex, ey = get_ellipse_points(mu[0], mu[1], w[0], h[1], ang)
                            fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", fill="toself",
                                                    fillcolor=hex_to_rgba(col, 0.25),
                                                    line=dict(color=col, width=2),
                                                    showlegend=False))

                        else:  # GMM (Max 3 ellipses to identify one and the asme author/ group)
                            best_gmm, best_bic, best_n = None, float('inf'), 1
                            for n in range(1, min(4, len(sub))):
                                g = GaussianMixture(n_components=n, covariance_type="full", random_state=42)
                                g.fit(X)
                                bic = g.bic(X)
                                if bic < best_bic:
                                    best_bic, best_gmm, best_n = bic, g, n
                            for i in range(best_n):
                                wgt = best_gmm.weights_[i]
                                if wgt < 0.1: continue
                                mu = best_gmm.means_[i]
                                cov = best_gmm.covariances_[i]
                                lam, v = np.linalg.eig(cov)
                                lam = np.sqrt(lam)
                                w, h = 2*lam*chi_val
                                ang = np.degrees(np.arctan2(v[1,0], v[0,0]))
                                ex, ey = get_ellipse_points(mu[0], mu[1], w, h, ang)
                                op = 0.2 + 0.3*wgt
                                fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", fill="toself",
                                                        fillcolor=hex_to_rgba(col, op),
                                                        line=dict(color=col, width=2),
                                                        showlegend=False))

                figs["group"] = fig

            # === 10. Shared Layout ===
            for name, fig in figs.items():
                title_text = settings.title
                if settings.subtitle:
                    title_text += (
                        f"<br><span style='font-size:{settings.subtitle_fontsize}px; "
                        f"color:{settings.subtitle_color};'>{settings.subtitle}</span>"
                    )
                fig.update_layout(
                    title=dict(text=title_text, y=0.96, x=0.5, xanchor='center',
                            yanchor='top',
                            font=dict(size=settings.title_fontsize,
                                        family="Arial Black, Arial, sans-serif",
                                        color="black"),
                            pad=dict(t=20)),
                    margin=dict(t=120, b=80, l=80, r=180),
                    xaxis=dict(title=x_title, title_font=dict(size=16), tickfont=dict(size=12)),
                    yaxis=dict(title=y_title, title_font=dict(size=16), tickfont=dict(size=12)),
                    legend=dict(
                        title="Author (Group)" if "individual" in name else "Group",
                        font=dict(size=16), bgcolor="white",
                        bordercolor="gray", borderwidth=1),
                    width=1400, height=750,
                )

            logger.success(f"Multi-dimensional plot built – {len(figs)} figure(s) [{plot_type}]")
            return figs

        except Exception as e:
            logger.exception(f"build_visual_multi_dimensions failed: {e}")
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

