# scripts/script1.py
from pathlib import Path
from typing import Optional
from .base import BaseScript
from src.plot_manager import CategoriesPlotSettings


class Script1(BaseScript):
    """Build the combined yearly bar chart (categories)."""

    def __init__(self, file_manager, plot_manager, image_dir: Path,
                 group_authors, non_anthony_group, anthony_group, sorted_groups,
                 settings: Optional[CategoriesPlotSettings] = None):
        super().__init__(file_manager, plot_manager=plot_manager,
                         settings=settings or CategoriesPlotSettings())
        self.image_dir = image_dir
        self.group_authors = group_authors
        self.non_anthony_group = non_anthony_group
        self.anthony_group = anthony_group
        self.sorted_groups = sorted_groups

    def run(self):
        fig = self.plot_manager.build_visual_categories(
            self.group_authors, self.non_anthony_group,
            self.anthony_group, self.sorted_groups, self.settings
        )
        if fig is None:
            return self.log_error("Failed to create yearly bar chart.")
        return self.save_figure(fig, self.image_dir, "yearly_bar_chart_combined")