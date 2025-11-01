# === plot_manager.py ===
# === Module Docstring ===
"""
Plot Manager Module

Creates all 5 key visualizations:
1. Categories: Total messages by group/author (Script1)
2. Time: DAC weekly heartbeat (Script2)
3. Distribution: Emoji frequency + cumulative (Script3)
4. Arc: Author interaction network (Script4)
5. Bubble: Words vs punctuation per author (Script5)
"""

# === Imports ===
from __future__ import annotations

import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches  # ADD THIS LINE
import seaborn as sns
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from .constants import Columns
from .data_preparation import CategoryPlotData, TimePlotData


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
class BubbleNewPlotSettings(PlotSettings):
    bubble_alpha: float = 0.6
    min_bubble_size: int = 50
    max_bubble_size: int = 1000
    trendline_color: str = "black"
    trendline_alpha: float = 0.5
    legend_scale_factor: float = 0.3
    group_colors: dict[str, str] = Field(
        default_factory=lambda: {
            "maap": "blue",
            "golfmaten": "orange",
            "dac": "green",
            "tillies": "gray",
        }
    )


# === Plot Manager Class ===
class PlotManager:
    """Manages all 5 visualizations."""

    def __init__(self) -> None:
        self.data_preparation = None  # Injected

    # === 1. Categories (Script1) ===
    def build_visual_categories(
        self,
        data: CategoryPlotData,
        settings: CategoriesPlotSettings,
    ) -> plt.Figure | None:
        """
        Plot total messages per author per group.

        Args:
            data: Validated CategoryPlotData
            settings: Plot settings

        Returns:
            matplotlib Figure or None
        """
        if not data or not data.groups:
            logger.error("Invalid or empty CategoryPlotData")
            return None

        try:
            fig, ax = plt.subplots(figsize=settings.figsize)
            plt.subplots_adjust(top=0.88, bottom=0.2)

            group_colors = {
                "dac": "green",
                "golfmaten": "orange",
                "maap": "blue",
                "tillies": "gray",
            }

            x_pos = []
            heights = []
            bar_colors = []
            author_labels = []
            group_midpoints = []
            current_x = 0.0

            for group_data in data.groups:
                n_authors = len(group_data.authors)
                start_x = current_x

                for author in group_data.authors:
                    x_pos.append(current_x)
                    heights.append(author.message_count)
                    color = "#000000" if author.is_avt else group_colors.get(group_data.whatsapp_group, "gray")
                    bar_colors.append(color)
                    author_labels.append(author.author)
                    current_x += 1.0

                mid_x = start_x + (n_authors - 1) / 2
                group_midpoints.append(mid_x)
                current_x += settings.group_spacing

            bars = ax.bar(
                x_pos,
                heights,
                width=settings.bar_width,
                color=bar_colors,
                edgecolor="black",
                linewidth=1.3,
                align="center",
            )

            for mid_x, group_data in zip(group_midpoints, data.groups):
                ax.hlines(
                    y=group_data.group_avg,
                    xmin=mid_x - 0.5,
                    xmax=mid_x + 0.5,
                    color=settings.trendline_color,
                    linestyle=settings.trendline_style,
                    linewidth=settings.trendline_width,
                    zorder=5,
                    label="Group Avg" if mid_x == group_midpoints[0] else "",
                )

            ax.set_title(settings.title, fontsize=16, fontweight="bold", pad=20)
            fig.suptitle(settings.subtitle, fontsize=11, color="dimgray", y=0.93)
            ax.set_ylabel(settings.ylabel, fontsize=12)
            ax.set_xlabel("WhatsApp Group → Author", fontsize=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(author_labels, rotation=45, ha="right", fontsize=10)

            for mid_x, group_data in zip(group_midpoints, data.groups):
                ax.text(
                    mid_x,
                    ax.get_ylim()[1] * 1.03,
                    group_data.whatsapp_group.upper(),
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=12,
                    color=group_colors.get(group_data.whatsapp_group, "black"),
                )

            legend_patches = [
                patches.Patch(facecolor=color, edgecolor="black", label=group.title())
                for group, color in group_colors.items()
            ]
            legend_patches.append(patches.Patch(facecolor="black", edgecolor="black", label="AvT"))
            ax.legend(
                handles=legend_patches,
                title="Group",
                loc="upper right",
                bbox_to_anchor=(1.15, 1),
                frameon=True,
                fancybox=True,
                shadow=True,
            )

            ax.grid(True, axis="y", linestyle="--", alpha=0.7, zorder=0)
            ax.set_axisbelow(True)
            plt.tight_layout()
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
        """
        try:
            fig, ax = plt.subplots(figsize=settings.figsize)

            # data.weekly_avg is a dict {week_number: avg_messages}
            weeks = list(data.weekly_avg.keys())
            avg_counts = list(data.weekly_avg.values())

            # ----- line plot -----
            ax.plot(
                weeks,
                avg_counts,
                color=settings.line_color,
                linewidth=settings.linewidth,
            )

            # ----- global average line -----
            ax.axhline(
                data.global_avg,
                color="red",
                linestyle="--",
                label="Global Avg",
            )

            # ----- vertical separation lines -----
            for vline in settings.vline_weeks:
                ax.axvline(vline, color="gray", linestyle="--", alpha=0.5)

            # ----- X-axis ticks & month labels -----
            ax.set_xticks(settings.week_ticks)
            ax.set_xticklabels(settings.month_labels, rotation=0)

            # ----- titles / labels -----
            ax.set_title(settings.title or "Weekly Message Averages (DAC)")
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
        emoji_counts_df: pd.DataFrame,
        settings: DistributionPlotSettings = DistributionPlotSettings(),
    ) -> plt.Figure | None:
        try:
            required = ["emoji", "count_once", "percent_once"]
            if not all(col in emoji_counts_df.columns for col in required):
                logger.error("emoji_counts_df missing required columns")
                return None

            n = len(emoji_counts_df)
            fig, ax = plt.subplots(figsize=(max(n * 0.05, 8), 8))  # Narrower per emoji
            ax2 = ax.twinx()

            x_pos = np.arange(n)

            # === THIN LINES INSTEAD OF BARS ===
            ax.vlines(
                x=x_pos,
                ymin=0,
                ymax=emoji_counts_df["percent_once"],
                color=settings.bar_color,
                linewidth=0.8,        # Thin line
                alpha=0.8,
                label="Emoji %"
            )

            # Optional: tiny markers at top
            ax.plot(
                x_pos,
                emoji_counts_df["percent_once"],
                'o',
                markersize=2,
                color=settings.bar_color,
                alpha=0.6
            )

            ax.set_ylabel("Likelihood (%)", fontsize=12, labelpad=10)
            ax.set_title(settings.title or "Emoji Distribution (MAAP)", fontsize=16, pad=20)
            ax.set_xlim(-0.5, n - 0.5)
            ax.set_xticks([])  # No x-ticks — too many

            # === Cumulative Line ===
            cum = emoji_counts_df["percent_once"].cumsum()
            ax2.plot(
                x_pos,
                cum,
                color=settings.cumulative_color,
                linewidth=settings.line_width,
                label=settings.cum_label,
            )

            # Threshold line
            idx_thresh = None
            if (cum >= settings.cum_threshold).any():
                idx_thresh = np.where(cum >= settings.cum_threshold)[0][0]
                ax2.axhline(settings.cum_threshold, color=settings.cumulative_color, linestyle="--", linewidth=1)
                ax.axvspan(-0.5, idx_thresh + 0.5, facecolor="lightgreen", alpha=0.15)

            ax2.set_ylabel("Cumulative %", fontsize=12, labelpad=10, color=settings.cumulative_color)
            ax2.set_ylim(0, 100)
            ax2.set_yticks(np.arange(0, 101, 10))
            ax2.tick_params(axis='y', colors=settings.cumulative_color)

            # === Top N Table (Bottom) ===
            top = emoji_counts_df.head(settings.top_n)
            cum_top = top["percent_once"].cumsum().round(1)
            table_data = [
                [f"{i+1}" for i in range(len(top))],
                [row["emoji"] for _, row in top.iterrows()],
                [f"{c:,.0f}" for c in top["count_once"]],
                [f"{c}%" for c in cum_top],
            ]
            col_width = 0.8 / len(top)
            table = ax.table(
                cellText=table_data,
                rowLabels=["Rank", "Emoji", "Count", "Cum"],
                colWidths=[col_width] * len(top),
                loc="bottom",
                bbox=[0.1, -0.5, 0.8, 0.35],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)

            fig.text(0.5, 0.28, f"Top {settings.top_n} Emojis", ha="center", fontsize=11)

            # === Legends ===
            ax2.legend(loc="upper left", fontsize=9)
            # ax.legend(loc="upper right", fontsize=9)  # Optional

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.35)
            return fig

        except Exception as e:
            logger.exception(f"Distribution plot failed: {e}")
            return None

    # === 4. Arc (Script4) ===
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
            participant_cols = [c for c in combined_df.columns if c not in settings.excluded_columns]
            authors = sorted(set(participant_cols))
            n = len(authors)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            radius = 1.0
            pos = {a: (radius * np.cos(ang), radius * np.sin(ang)) for a, ang in zip(authors, angles, strict=False)}
            pair_weights: dict[frozenset, float] = {}
            triple_weights: dict[frozenset, float] = {}
            total_weights: dict[frozenset, float] = {}
            pairs = combined_df[combined_df["type"] == "Pairs"]
            for _, row in pairs.iterrows():
                a1, a2 = (s.strip() for s in row[Columns.AUTHOR.value].split(" & "))
                key = frozenset([a1, a2])
                pair_weights[key] = row["total_messages"]
                total_weights[key] = total_weights.get(key, 0) + row["total_messages"]
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
                        ax.text(lbl_x, lbl_y, f"{round(w)}", ha="center", va="center", fontsize=settings.label_fontsize, bbox=settings.label_bbox, zorder=z + 1)
            for auth, (x, y) in pos.items():
                ax.scatter([x], [y], s=settings.node_size, color=settings.node_color, edgecolors=settings.node_edge_color, zorder=4)
                ax.text(x, y, auth, ha="center", va="center", fontsize=settings.node_fontsize, fontweight=settings.node_fontweight, zorder=5)
            ax.set_title(settings.title_template.format(group=group))
            ax.axis("off")
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.exception(f"Arc diagram failed: {e}")
            return None

    # === 5. Bubble (Script5) ===
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
            size_scale = (msg - msg.min()) / (msg.max() - msg.min()) if msg.max() != msg.min() else 1.0
            bubble_sizes = (settings.min_bubble_size + (settings.max_bubble_size - settings.min_bubble_size) * size_scale) * 3
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
            x = feature_df[Columns.AVG_WORDS.value]
            y = feature_df[Columns.AVG_PUNCT.value]
            coef = np.polyfit(x, y, 1)
            trend = np.poly1d(coef)
            ax.plot(x, trend(x), color=settings.trendline_color, alpha=settings.trendline_alpha)
            ax.set_title(settings.title or f"{Columns.AVG_WORDS.human} vs {Columns.AVG_PUNCT.human}")
            ax.set_xlabel(settings.xlabel or Columns.AVG_WORDS.human)
            ax.set_ylabel(settings.ylabel or Columns.AVG_PUNCT.human)
            legend_handles = [
                plt.scatter([], [], s=settings.min_bubble_size * settings.legend_scale_factor, c=col, alpha=settings.bubble_alpha, label=grp)
                for grp, col in settings.group_colors.items()
            ]
            ax.legend(handles=legend_handles, title="WhatsApp Group", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.exception(f"Bubble plot failed: {e}")
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

# NEW: Full 1–5 plot support (2025-11-01)
# NEW: BaseModel contracts for Script1 & Script2
# NEW: Clean, modular, production-ready
# NEW: All unused code removed