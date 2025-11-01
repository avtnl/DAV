# === script2.py ===
# === Module Docstring ===
"""
Time plot: DAC weekly heartbeat (Script 2).

Uses enriched DataFrame, builds TimePlotData,
and generates weekly average line chart.
"""

# === Imports ===
from pathlib import Path

from loguru import logger
from src.constants import Columns, Groups
from src.plot_manager import TimePlotSettings

from .base import BaseScript


# === Script 2 ===
class Script2(BaseScript):
    """DAC weekly message heartbeat."""

    def __init__(
        self,
        file_manager,
        data_preparation,
        plot_manager,
        image_dir: Path,
        df,
    ) -> None:
        super().__init__(
            file_manager=file_manager,
            data_preparation=data_preparation,
            plot_manager=plot_manager,
        )
        self.image_dir = image_dir
        self.df = df

    def run(self) -> Path | None:
        if self.df is None or self.df.empty:
            self.log_error("No DataFrame provided to Script2.")
            return None

        df_dac = self.df[self.df[Columns.WHATSAPP_GROUP.value] == Groups.DAC.value]
        logger.info(f"DAC group rows: {len(df_dac)}")  # ← ADD THIS
        if df_dac.empty:
            self.log_error(f"No data for group '{Groups.DAC.value}'. Skipping.")
            return None

        data = self.data_preparation.build_visual_time(df_dac)
        if data is None:
            self.log_error("build_visual_time returned None.")
            return None

        logger.info(f"Time plot: {len(data.weekly_avg)} weeks, avg {data.global_avg:.1f}")

        fig = self.plot_manager.build_visual_time(data, TimePlotSettings())
        if fig is None:
            self.log_error("Failed to create time plot.")
            return None

        return self.save_figure(fig, self.image_dir, "time_plot_dac")


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

# NEW: Fixed df injection – stored locally, not passed to BaseScript (2025-11-02)
# NEW: Uses keyword args in super().__init__() to avoid DataFrame truth error