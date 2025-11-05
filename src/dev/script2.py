from pathlib import Path

from src.constants import Columns, Groups
from src.plot_manager import TimePlotSettings
src.dev.plot_manager
from .base import BaseScript


class Script2(BaseScript):
    """Time-based visualization for the DAC group."""

    def __init__(self, file_manager, data_preparation, plot_manager,
                 image_dir: Path, df, settings: TimePlotSettings | None = None) -> None:
        super().__init__(file_manager, data_preparation=data_preparation,
                         plot_manager=plot_manager, settings=settings or TimePlotSettings())
        self.image_dir = image_dir
        self.df = df

    def run(self):
        df_dac = self.df[self.df[Columns.WHATSAPP_GROUP.value] == Groups.DAC.value].copy()
        if df_dac.empty:
            return self.log_error(f"No data for group '{Groups.DAC.value}'. Skipping.")

        df_dac, p, average_all = self.data_preparation.build_visual_time(df_dac)
        if df_dac is None or p is None or average_all is None:
            return self.log_error("build_visual_time failed.")

        fig = self.plot_manager.build_visual_time(p, average_all, self.settings)
        if fig is None:
            return self.log_error("Failed to create time plot.")
        return self.save_figure(fig, self.image_dir, "golf_decode_by_wa_heartbeat")
