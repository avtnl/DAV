# === script2.py ===
# === Module Docstring ===
"""
Time plot: DAC weekly heartbeat (Script 2).

Uses enriched DataFrame, builds TimePlotData,
and generates weekly average line chart + optional seasonality suite.

Examples
--------
>>> script = Script2(file_manager, data_preparation, plot_manager, image_dir, df)
>>> main_path, season_path = script.run()
"""

# === Imports ===
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
from loguru import logger

from src.constants import Columns, Groups
from src.plot_manager import TimePlotSettings, SeasonalityPlotSettings

from .base import BaseScript


# === Script 2 ===
class Script2(BaseScript):
    """DAC weekly message heartbeat with optional seasonality proof."""

    def __init__(
        self,
        file_manager,
        data_preparation,
        plot_manager,
        image_dir: Path,
        df: pd.DataFrame,
        settings: TimePlotSettings | None = None,
        include_seasonality: bool = True,  # ← Re-added for flexibility
    ) -> None:
        super().__init__(
            file_manager=file_manager,
            data_preparation=data_preparation,
            plot_manager=plot_manager,
            df=df,
        )
        self.image_dir = image_dir
        self.settings = settings or TimePlotSettings()
        self.include_seasonality = include_seasonality

    def run(self) -> Tuple[Path, Optional[Path]]:
        """
        Generate and save DAC weekly heartbeat + optional seasonality evidence.

        Returns:
            Tuple[Path, Optional[Path]]:
                - Path to main time plot
                - Path to seasonality evidence plot (or None)
        """
        if self.df is None or self.df.empty:
            self.log_error("No DataFrame provided to Script2.")
            return self._error_path(), None

        df_dac = self.df[self.df[Columns.WHATSAPP_GROUP.value] == Groups.DAC.value]
        logger.info(f"DAC group rows: {len(df_dac)}")
        if df_dac.empty:
            self.log_error(f"No data for group '{Groups.DAC.value}'. Skipping.")
            return self._error_path(), None

        # Build data (only once!)
        try:
            data = self.data_preparation.build_visual_time(
                df_dac, compute_seasonality=self.include_seasonality
            )
        except Exception as e:
            self.log_error(f"Failed to build TimePlotData: {e}")
            return self._error_path(), None

        if data is None:
            self.log_error("build_visual_time returned None.")
            return self._error_path(), None

        logger.info(
            f"Time plot: {len(data.weekly_avg)} weeks, "
            f"global avg {data.global_avg:.1f}, "
            f"seasonality={'included' if data.seasonality else 'skipped'}"
        )

        # Generate plots
        try:
            figs = self.plot_manager.build_visual_time(
                data=data,
                settings=self.settings,
                include_seasonality=self.include_seasonality,
                season_settings=SeasonalityPlotSettings(),
            )
        except Exception as e:
            self.log_error(f"Failed to build plots: {e}")
            return self._error_path(), None

        main_path = self.save_figure(figs["category"], self.image_dir, "time_plot_dac")
        season_path = None
        if "seasonality" in figs and figs["seasonality"]:
            season_path = self.save_figure(
                figs["seasonality"], self.image_dir, "seasonality_evidence_time_plot"
            )

        logger.success(f"Script2 completed: main={main_path}, season={season_path}")
        return main_path, season_path

    def _error_path(self) -> Path:
        """Return a dummy path on error."""
        return self.image_dir / "error_time_plot_dac.png"


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

# NEW: Final Script2 with dual output, include_seasonality, no duplicate calls (2025-11-07)