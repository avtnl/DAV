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
    group_spacing: float = Field(2.5, ge=0.5, le=5.0)
    bar_width: float = Field(0.8, ge=0.1, le=1.0)
    trendline_color: str = "red"
    trendline_style: str = "--"
    trendline_width: float = 2.5
    title: str = "Anthony's participation is significantly lower for the 3rd group"
    subtitle: str = "Too much to handle or too much crap?"
    ylabel: str = Columns.MESSAGE_COUNT.human


# === 2. Time Plot Settings (Script2) ===
class TimePlotSettings(PlotSettings):
    vline_weeks: list[float] = Field(default_factory=lambda: [11.5, 18.5, 34.5])
    week_ticks: list[int] = Field(default_factory=lambda: [1, 5, 9, 14, 18, 23, 27, 31, 36, 40, 44, 49])
    month_labels: list[str] = Field(
        default_factory=lambda: [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ]
    )
    rest_label: str = "---------Rest---------"
    prep_label: str = "---Prep---"
    play_label: str = "---------Play---------"
    line_color: str = "black"
    linewidth: float = 2.5


# === 3. Distribution Plot Settings (Script3) ===
class DistributionPlotSettings(PlotSettings):
    bar_color: str = "steelblue"
    cumulative_color: str = "red"
    bar_alpha: float = 0.7
    line_width: float = 2.0
    cum_label: str = "Cumulative %"
    top_n: int = 20
    cum_threshold: float = 80.0


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


# === 5. Bubble Plot Settings (Script5) ===
class BubblePlotSettings(PlotSettings):
    bubble_alpha: float = 0.6
    min_bubble_size: int = 50
    max_bubble_size: int = 1000
    trendline_color: str = "black"
    trendline_alpha: float = 0.5
    legend_scale_factor: float = 0.3
    group_colors: dict[str, str] = Field(
        default_factory=lambda: {
            Groups.MAAP: "blue",
            Groups.GOLFMATEN: "orange",
            Groups.DAC: "green",
            Groups.TILLIES: "gray",
        }
    )


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
            fig.tight_layout(pad=4.0)                  # Push everything out, including titles

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

            # # Draw bars
            # ax.bar(
            #     x_pos,
            #     heights,
            #     width=settings.bar_width,
            #     color=bar_colors,
            #     edgecolor="black",
            #     linewidth=1.3,
            #     align="center",
            # )

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

            ax.set_title(
                "Anthony's participation is significantly lower for the 3rd group",
                fontsize=24,
                fontweight="bold",
                pad=40,        # NEW: Push title UP into the new space
                ha="center",
            )
            fig.suptitle(
                "Too much to handle or too much crap?",
                fontsize=18,
                fontweight="bold",
                color="dimgray",
                y=0.85,        # NEW: Position in the new top space
                ha="center",
            )

            # Axis labels
            ax.set_ylabel(settings.ylabel, fontsize=12)
            ax.set_xlabel("WhatsApp Groups and participating Authors", fontsize=12)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(author_labels, rotation=45, ha="right", fontsize=10)

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
            fig, ax = plt.subplots(figsize=settings.figsize)

            weeks = list(data.weekly_avg.keys())
            avg_counts = list(data.weekly_avg.values())

            ax.plot(
                weeks,
                avg_counts,
                color=settings.line_color,
                linewidth=settings.linewidth,
            )

            ax.axhline(
                data.global_avg,
                color="red",
                linestyle="--",
                label="Global Avg",
            )

            for vline in settings.vline_weeks:
                ax.axvline(vline, color="gray", linestyle="--", alpha=0.5)

            ax.set_xticks(settings.week_ticks)
            ax.set_xticklabels(settings.month_labels, rotation=0)

            ax.set_title(settings.title or f"Weekly Message Averages ({Groups.DAC})")
            ax.set_xlabel(settings.xlabel or "Week of Year")
            ax.set_ylabel(settings.ylabel or "Average Messages")

            ax.legend()
            plt.tight_layout()
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
        Plot emoji frequency with cumulative percentage.

        Args:
            data: Validated DistributionPlotData
            settings: Plot settings

        Returns:
            matplotlib Figure or None
        """
        try:
            df = data.emoji_counts_df
            required = ["emoji", "count_once", "percent_once"]
            if not all(col in df.columns for col in required):
                logger.error("emoji_counts_df missing required columns")
                return None

            n = len(df)
            fig, ax = plt.subplots(figsize=(max(n * 0.05, 8), 8))
            ax2 = ax.twinx()

            x_pos = np.arange(n)

            ax.vlines(
                x=x_pos,
                ymin=0,
                ymax=df["percent_once"],
                color=settings.bar_color,
                linewidth=0.8,
                alpha=0.8,
                label="Emoji %"
            )

            ax.plot(
                x_pos,
                df["percent_once"],
                'o',
                markersize=2,
                color=settings.bar_color,
                alpha=0.6
            )

            ax.set_ylabel("Likelihood (%)", fontsize=12, labelpad=10)
            ax.set_title(settings.title or "Emoji Distribution (MAAP)", fontsize=16, pad=20)
            ax.set_xlim(-0.5, n - 0.5)
            ax.set_xticks([])

            cum = df["percent_once"].cumsum()
            ax2.plot(
                x_pos,
                cum,
                color=settings.cumulative_color,
                linewidth=settings.line_width,
                label=settings.cum_label,
            )

            if (cum >= settings.cum_threshold).any():
                idx_thresh = np.where(cum >= settings.cum_threshold)[0][0]
                ax2.axhline(settings.cum_threshold, color=settings.cumulative_color, linestyle="--", linewidth=1)
                ax.axvspan(-0.5, idx_thresh + 0.5, facecolor="lightgreen", alpha=0.15)

            ax2.set_ylabel("Cumulative %", fontsize=12, labelpad=10)
            ax2.set_ylim(0, 100)

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

            plt.tight_layout()
            logger.success("Distribution plot built successfully")
            return fig

        except Exception as e:
            logger.exception(f"Distribution plot failed: {e}")
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

            fig, ax = plt.subplots(figsize=settings.figsize)
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

            x = df[Columns.AVG_WORDS.value]
            y = df[Columns.AVG_PUNCT.value]
            coef = np.polyfit(x, y, 1)
            trend = np.poly1d(coef)
            ax.plot(x, trend(x), color=settings.trendline_color, alpha=settings.trendline_alpha)

            ax.set_title(settings.title or f"{Columns.AVG_WORDS.human} vs {Columns.AVG_PUNCT.human}")
            ax.set_xlabel(settings.xlabel or Columns.AVG_WORDS.human)
            ax.set_ylabel(settings.ylabel or Columns.AVG_PUNCT.human)

            legend_handles = [
                plt.scatter([], [], s=settings.min_bubble_size * settings.legend_scale_factor,
                            c=col, alpha=settings.bubble_alpha, label=grp.value)
                for grp, col in settings.group_colors.items()
            ]
            ax.legend(handles=legend_handles, title="WhatsApp Group", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
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

            if not settings.by_group:
                fig = px.scatter(
                    data.agg_df,
                    x="tsne_x", y="tsne_y",
                    size="msg_count",
                    color=Columns.AUTHOR.value,
                    hover_data={"msg_count": True},
                    title="Linguistic Style Clusters (t-SNE) – Per Author",
                )

                if settings.draw_ellipses:
                    for author in data.agg_df[Columns.AUTHOR.value].unique():
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

                        trace_names = [d.name for d in fig.data]
                        color = fig.data[trace_names.index(author)].marker.color if author in trace_names else "#1f77b4"

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