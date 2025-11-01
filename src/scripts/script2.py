# === Module Docstring ===
"""
Script2 - Average Weekly Total Messages for Group DAC

Builds and saves the time base graph using validated data.
DAC = Dinsdag Avond Competitie, which is a weekly golf tournament starting May till end of August.
"""

# === Imports ===
from __future__ import annotations

import pandas as pd  # ADD THIS LINE
from pathlib import Path

from src.constants import Columns, Groups
from src.plot_manager import TimePlotSettings
from .base import BaseScript


class Script2(BaseScript):
    def __init__(self, file_manager, data_preparation, plot_manager, image_dir: Path, df, settings: TimePlotSettings | None = None):
        super().__init__(file_manager, data_preparation=data_preparation, plot_manager=plot_manager, settings=settings or TimePlotSettings())
        self.image_dir = image_dir
        self.df = df

    def run(self):
        df_dac = self.df[self.df[Columns.WHATSAPP_GROUP] == Groups.DAC]
        if df_dac.empty:
            return self.log_error("No data for DAC group.")

        result = self.data_preparation.build_visual_time(df_dac)
        if not result:
            return self.log_error("build_visual_time failed.")

        # Convert dict to DataFrame for plot_manager
        p_df = pd.DataFrame(
            list(result.weekly_avg.items()),
            columns=["week", "avg_count_all"]
        ).set_index("week")

        average_all_df = pd.DataFrame({
            "isoweek": range(1, 53),
            "avg_count_all": [result.weekly_avg.get(w, 0.0) for w in range(1, 53)]
        })

        fig = self.plot_manager.build_visual_time(p_df, average_all_df, self.settings)
        if not fig:
            return self.log_error("Failed to create time plot.")

        return self.save_figure(fig, self.image_dir, "golf_decode_by_wa_heartbeat")


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

# NEW: Added pandas import for DataFrame conversion (2025-11-01)