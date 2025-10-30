from pathlib import Path
from typing import Optional
from .base import BaseScript
from src.constants import Groups, Columns
from src.plot_manager import BubbleNewPlotSettings


class Script5(BaseScript):
    """Bubble plot across multiple groups."""

    def __init__(self, file_manager, data_preparation, plot_manager,
                 image_dir: Path, df,
                 settings: Optional[BubbleNewPlotSettings] = None):
        super().__init__(file_manager, data_preparation=data_preparation,
                         plot_manager=plot_manager,
                         settings=settings or BubbleNewPlotSettings())
        self.image_dir = image_dir
        self.df = df

    def run(self):
        groups = [Groups.MAAP.value, Groups.GOLFMATEN.value,
                  Groups.DAC.value, Groups.TILLIES.value]
        df_groups = self.df[self.df[Columns.WHATSAPP_GROUP.value].isin(groups)].copy()
        if df_groups.empty:
            return self.log_error(f"No data for groups {groups}. Skipping bubble plot.")

        bubble_df = self.data_preparation.build_visual_relationships_bubble(df_groups, groups)
        if bubble_df is None or bubble_df.empty:
            return self.log_error("build_visual_relationships_bubble returned nothing.")

        fig = self.plot_manager.build_visual_relationships_bubble(bubble_df, self.settings)
        if fig is None:
            return self.log_error("Bubble plot creation failed.")
        return self.save_figure(fig, self.image_dir, "bubble_plot_words_vs_punct")