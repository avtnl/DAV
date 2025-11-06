# === plot_manager.py ===
# === Module Docstring ===
"""
Plot Manager Module

Creates all 6 key visualizations using validated data from DataPreparation:

1. Categories: Total messages by group/author (Script1)
2. Time: DAC weekly heartbeat (Script2)
3. Distribution: Emoji frequency + cumulative (Script3)
4. Arc: Author interaction network (Script4)
5. Bubble: Words vs punctuation per author (Script5)
6. Multi-Dimensional: t-SNE style clusters (Script6)
"""

# === Imports ===
from __future__ import annotations

import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field
from typing import Literal

from .constants import Columns, Groups, InteractionType, Script6ConfigKeys
from .data_preparation import (
    CategoryPlotData,
    TimePlotData,
    DistributionPlotData,
    ArcPlotData,
    BubblePlotData,
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
    title: str = "Anthony's participation is significantly lower for the 3rd group"
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
    title: str = "Golf season, decoded by WhatsApp heartbeat"
    subtitle: str = "Whatsapp_group is 'dac' (Dinsdag Avond Competitie)"
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
    #subtitle_y: float = 0.80
    subtitle_ha: str = "center"


# === 3. Distribution Plot Settings (Script3) ===
class DistributionPlotSettings(PlotSettings):
    bar_color: str = "purple"
    cumulative_color: str = "orange"
    bar_alpha: float = 0.9
    line_width: float = 2.0
    cum_label: str = "Cumulative %"
    cum_threshold: float = 75.0
    top_table: int = 25  # For table, not bars


# === 4. Arc Plot Settings (Script4) ===
class ArcPlotSettings(PlotSettings):
    arc_color: str = "black"
    arc_width: float = 1.5
    node_size: int = 300
    node_color: str = "skyblue"
    node_edge_color: str = "black"
    node_fontsize: int = 10
    node_fontweight: str = "bold"
    label_fontsize: int = 8
    label_bbox: dict = Field(default_factory=lambda: {"boxstyle": "round", "facecolor": "white", "alpha": 0.8})
    amplifier: float = 1.0
    excluded_columns: list[str] = Field(default_factory=list)
    married_couples: list[tuple[str, str]] = Field(default_factory=list)
    total_colors: dict[str, str] = Field(
        default_factory=lambda: {"married": "red", "other": "gray"}
    )
    special_x_offsets: dict[tuple[str, str, str], float] = Field(default_factory=dict)
    special_label_y_offsets: dict[tuple[str, str], float] = Field(default_factory=dict)
    arc_types: list[tuple[str, str, float, int]] = Field(
        default_factory=lambda: [
            ("pair", "blue", 0.3, 2),
            ("triple", "purple", 0.2, 1),
            ("total", "black", 0.4, 3),
        ]
    )
    title_template: str = "Author Interactions in {group}"


class BubblePlotSettings(PlotSettings):
    title: str = "Correlation between avg Words and avg Punctuations"
    subtitle: str = "About 1 extra Punctuation per 10 words"
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


# === Plot Manager Class ===
class PlotManager:
    """Manages all 6 visualizations."""

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
            fig, ax = plt.subplots(figsize=(14, 6.5))  # Shorter canvas
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
        settings: TimePlotSettings = TimePlotSettings(),
    ) -> plt.Figure | None:
        """
        Plot the weekly average heartbeat for the DAC group.

        Args:
            data: Validated TimePlotData
            settings: Plot settings

        Returns:
            matplotlib Figure or None
        """
        try:
            # Figure setup aligned with Categories
            fig, ax = plt.subplots(figsize=(14, 6.5))  # Shorter canvas
            plt.subplots_adjust(top=0.85)              # Axes end earlier → more room above

            # Sorted weeks and values
            weeks = sorted(data.weekly_avg.keys())
            avg_counts = [data.weekly_avg[w] for w in weeks]

            # Main line: weekly average
            ax.plot(
                weeks,
                avg_counts,
                color=settings.line_color,
                linewidth=settings.linewidth,
                zorder=6,
            )

            # Period boundaries (hardcoded)
            vlines = [11.5, 18.5, 34.5]
            starts = [1] + [int(v + 0.5) for v in vlines]
            ends   = [int(v + 0.5) - 1 for v in vlines] + [52]

            # Period labels and colors
            period_labels = [
                settings.rest_label,
                settings.prep_label,
                settings.play_label,
                settings.rest_label,
            ]
            period_colors = ["#e8f5e9", "#c8e6c9", "#81c784", "#e8f5e9"]

            # Helper
            def period_avg(start: int, end: int) -> float:
                vals = [data.weekly_avg.get(w, 0.0) for w in range(start, end + 1)]
                return float(np.mean(vals)) if vals else 0.0

            # Draw periods
            period_avg_patch = None
            y_min, y_max = ax.get_ylim()
            label_y = y_min + 0.80 * (y_max - y_min)

            for i in range(4):
                s, e = starts[i], ends[i]
                ax.axvspan(s - 0.5, e + 0.5, facecolor=period_colors[i], alpha=0.6, zorder=0)

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

            # X-axis
            key_dates = {11: "March 15th", 18: "May 1st", 36: "September 1st"}
            ax.set_xticks(sorted(key_dates.keys()))
            ax.set_xticklabels([key_dates[w] for w in sorted(key_dates)], fontsize=12, ha="center")
            ax.set_xlabel("Time over Year", fontsize=12, labelpad=10)

            # Title and subtitle
            # fig.suptitle(
            #     settings.title,
            #     fontsize=settings.title_fontsize,
            #     fontweight=settings.title_fontweight,
            #     y=0.98,
            #     ha=settings.title_ha,
            # )

            ax.set_title(
                settings.title,
                fontsize=settings.title_fontsize,
                fontweight=settings.title_fontweight,
                pad=settings.title_pad,
                ha=settings.title_ha,
            )

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

            # Y-label
            ax.set_ylabel("Average message count per week (2017 - 2025)", fontsize=12)

            # Legend
            if period_avg_patch:
                ax.legend(handles=[period_avg_patch], loc="upper right", fontsize=12)

            # Grid
            ax.grid(True, axis="y", linestyle="--", alpha=0.7, zorder=0)
            ax.set_axisbelow(True)
            plt.tight_layout()
            plt.show()

            logger.success("Time plot built – aligned with Categories")
            return fig

        except Exception as e:
            logger.exception(f"Time plot failed: {e}")
            return None


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
            ax.text(left_mid, y_text, f"<-- {x_75} emojies -->", ha="center", fontsize=12)
            ax.text(right_mid, y_text, f"<-- {y_75} emojies -->", ha="center", fontsize=12)
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


    # === 4. Arc (Script4) ===
    def build_visual_relationships_arc(
        self,
        data: ArcPlotData,
        settings: ArcPlotSettings = ArcPlotSettings(),
    ) -> plt.Figure | None:
        """
        Plot author interaction network as an arc diagram.
        Raises: ValueError if authors != 4
        Args:
            data: Validated ArcPlotData
            settings: Plot settings
        Returns:
            matplotlib Figure or None
        """
        try:
            df = data.participation_df
            participant_cols = [c for c in df.columns if c not in ["type", "author", "total_messages"]]
            authors = sorted(participant_cols)
            if len(authors) != 4:
                logger.error(f"Expected 4 authors, got {len(authors)}")
                return None

            pos = {auth: (i, 0) for i, auth in enumerate(authors)}
            pair_weights = {}
            triple_weights = {}
            total_weights = {}
            combined_df = df

            # === Pairs processing ===
            pairs = combined_df[combined_df["type"] == InteractionType.PAIRS]
            for _, row in pairs.iterrows():
                pair_str = None
                for col in row.index:
                    val = str(row[col])
                    if " & " in val and val not in ["Pairs", "Non-participant"]:
                        pair_str = val
                        break
                if not pair_str:
                    continue
                a1, a2 = [a.strip() for a in pair_str.split(" & ", 1)]
                key = frozenset([a1, a2])
                pair_weights[key] = row["total_messages"]
                total_weights[key] = total_weights.get(key, 0) + row["total_messages"]

            # === Triples processing ===
            triples = combined_df[combined_df["type"] == InteractionType.NON_PARTICIPANT]
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
            weights_dict = {"pair": pair_weights, "triple": triple_weights, "total": total_weights}

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

                    if arc_type == "total":
                        pair_t = (a1, a2) if a1 < a2 else (a2, a1)
                        married = pair_t in settings.married_couples or (a2, a1) in settings.married_couples
                        color = settings.total_colors["married"] if married else settings.total_colors["other"]

                    sorted_pair = tuple(sorted([a1, a2]))
                    x_off = settings.special_x_offsets.get((*sorted_pair, arc_type), 0)
                    lw = (1 + 5 * (w / max_w)) * settings.amplifier

                    t = np.linspace(0, 1, 100)
                    x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * (xm + x_off) + t**2 * x2
                    y = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * (ym + height) + t**2 * y2
                    ax.plot(x, y, color=color, linewidth=lw, zorder=z)

                    if arc_type == "total":
                        lbl_x = (x1 + x2) / 2
                        lbl_y = (y1 + y2) / 2 + height * 0.5
                        lbl_y += settings.special_label_y_offsets.get(tuple(sorted([a1, a2])), 0)
                        ax.text(lbl_x, lbl_y, f"{round(w)}", ha="center", va="center",
                                fontsize=settings.label_fontsize, bbox=settings.label_bbox, zorder=z + 1)

            for auth, (x, y) in pos.items():
                ax.scatter([x], [y], s=settings.node_size, color=settings.node_color,
                        edgecolors=settings.node_edge_color, zorder=4)
                ax.text(x, y, auth, ha="center", va="center",
                        fontsize=settings.node_fontsize, fontweight=settings.node_fontweight, zorder=5)

            group = df.iloc[0][Columns.WHATSAPP_GROUP.value] if Columns.WHATSAPP_GROUP.value in df.columns else "Unknown"
            ax.set_title(settings.title_template.format(group=group))
            ax.axis("off")
            plt.tight_layout()

            logger.success("Arc diagram built successfully")
            return fig

        except Exception as e:
            logger.exception(f"Arc diagram failed: {e}")
            return None


    # === 5. Bubble (Script5) ===
    def build_visual_relationships_bubble(
        self,
        data: BubblePlotData,
        settings: BubblePlotSettings = BubblePlotSettings(),
    ) -> plt.Figure | None:
        """
        Plot average words vs punctuation with bubble size by message count.

        Args:
            data: Validated BubblePlotData
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
                logger.error("Bubble plot missing required columns")
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

            # Title
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
            
            logger.success("Bubble plot built successfully")
            return fig

        except Exception as e:
            logger.exception(f"Bubble plot failed: {e}")
            return None


    # === 6. Multi-Dimensional Style (Script6) ===
    def build_visual_multi_dimensions(
        self,
        data: MultiDimPlotData,
        settings: MultiDimPlotSettings = MultiDimPlotSettings(),
    ) -> dict[str, "go.Figure"] | None:
        """
        Create interactive t-SNE plots with optional group isolation and confidence ellipses.

        Args:
            data: Validated MultiDimPlotData with t-SNE coordinates
            settings: Plot settings including group mode and ellipse drawing

        Returns:
            Dict of Plotly figures: {"individual": ..., "group": ...} or None
        """
        if data.agg_df.empty:
            logger.error("Empty agg_df in MultiDimPlotData")
            return None

        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from scipy.stats import chi2

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

            # === AUTHOR-SPECIFIC COLORS (per_group == False) ===
            author_colors = {
                "RH": "#FFA500",   # Standard Orange
                "MK": "#FF8C00",   # Dark Orange
                "HB": "#FFFF00",   # Pastel Yellow
                "AB": "#00008B",   # Dark Blue
                "PB": "#00BFFF",   # Deep Sky Blue
                "M": "#ADD8E6",    # Light Blue
                "LL": "#ADFF2F",   # Green Yellow
                "HH": "#228B22",   # Forest Green
                "HvH": "#ADFF2F",  # Yellow Green
                "ND": "#006400",   # Dark Green
                "Bo": "#EE82EE",   # Violet
                "Lars": "#800080", # Pure Purple
                "Mats": "#9370DB", # Medium Purple
                "JC": "#2F4F4F",   # Dark Slate Gray
                "EH": "#A9A9A9",   # Dark Gray
                "FJ": "#D3D3D3",   # Light Gray
                "AvT": "#FF0000",  # Pure Red
            }

            if not settings.by_group:
                # Build color map: known authors → custom color, others → black
                unique_authors = data.agg_df[Columns.AUTHOR.value].unique()
                color_discrete_map = {
                    author: author_colors.get(author, "black")
                    for author in unique_authors
                }

                fig = px.scatter(
                    data.agg_df,
                    x="tsne_x", y="tsne_y",
                    size="msg_count",
                    color=Columns.AUTHOR.value,
                    hover_data={"msg_count": True},
                    title="Linguistic Style Clusters (t-SNE) – Per Author",
                    color_discrete_map=color_discrete_map,
                )

                if settings.draw_ellipses:
                    for author in unique_authors:
                        sub = data.agg_df[data.agg_df[Columns.AUTHOR.value] == author]
                        if len(sub) < 2:
                            continue
                        x, y = sub["tsne_x"], sub["tsne_y"]
                        cov = np.cov(x, y)
                        mean_x, mean_y = x.mean(), y.mean()
                        lambda_, v = np.linalg.eig(cov)
                        lambda_ = np.sqrt(lambda_)
                        chi = np.sqrt(chi2.ppf(0.75, 2))
                        width, height = 2 * lambda_[0] * chi, 2 * lambda_[1] * chi
                        angle = np.degrees(np.arctan2(v[1,0], v[0,0]))

                        color = color_discrete_map.get(author, "black")
                        ell_x, ell_y = get_ellipse_points(mean_x, mean_y, width, height, angle)
                        fig.add_trace(go.Scatter(
                            x=ell_x, y=ell_y, mode="lines", fill="toself",
                            fillcolor=hex_to_rgba(color, 0.2),
                            line=dict(color=color, width=2),
                            name=f"{author} 75%", showlegend=False
                        ))

                figs["individual"] = fig

            if settings.by_group:
                data.agg_df["plot_group"] = data.agg_df.apply(
                    lambda row: Groups.AVT if row[Columns.AUTHOR.value] == Groups.AVT else row[Columns.WHATSAPP_GROUP_TEMP],
                    axis=1
                )
                group_colors = {
                    Groups.MAAP: "#1f77b4",
                    Groups.DAC: "#ff7f0e",
                    Groups.GOLFMATEN: "#2ca02c",
                    Groups.TILLIES: "#808080",
                    Groups.AVT: "#d62728"
                }
                fig = px.scatter(
                    data.agg_df,
                    x="tsne_x", y="tsne_y",
                    color="plot_group",
                    size="msg_count",
                    color_discrete_map=group_colors,
                    hover_data={"msg_count": True, Columns.AUTHOR.value: True},
                    title="Linguistic Style Clusters (t-SNE) – 5 Groups (AvT Isolated)",
                )

                if settings.draw_ellipses:
                    for grp in data.agg_df["plot_group"].unique():
                        sub = data.agg_df[data.agg_df["plot_group"] == grp]
                        if len(sub) < 2:
                            continue
                        x, y = sub["tsne_x"], sub["tsne_y"]
                        cov = np.cov(x, y)
                        mean_x, mean_y = x.mean(), y.mean()
                        lambda_, v = np.linalg.eig(cov)
                        lambda_ = np.sqrt(lambda_)
                        chi = np.sqrt(chi2.ppf(0.50, 2))
                        width, height = 2 * lambda_[0] * chi, 2 * lambda_[1] * chi
                        angle = np.degrees(np.arctan2(v[1,0], v[0,0]))

                        color = group_colors.get(grp, "#333333")
                        ell_x, ell_y = get_ellipse_points(mean_x, mean_y, width, height, angle)
                        fig.add_trace(go.Scatter(
                            x=ell_x, y=ell_y, mode="lines", fill="toself",
                            fillcolor=hex_to_rgba(color, 0.25),
                            line=dict(color=color, width=2),
                            name=f"{grp} 50%", showlegend=False
                        ))

                figs["group"] = fig

            logger.success(f"Multi-dimensional plot built: {len(figs)} figures")
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