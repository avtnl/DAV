# === Module Docstring ===
"""
Build the combined yearly bar chart comparing Anthony vs non-Anthony messages.

Uses :meth:`src.plot_manager.PlotManager.build_visual_categories`.
"""

# === Imports ===
from pathlib import Path

from src.dev.plot_manager import CategoriesPlotSettings
from src.file_manager import FileManager  # Adjust import based on actual module
from src.plot_manager import PlotManager  # Adjust import based on actual module

from .base import BaseScript


# === Script 1 ===
class Script1(BaseScript):
    """Build the combined yearly bar chart (categories)."""

    def __init__(
        self,
        file_manager: FileManager,
        plot_manager: PlotManager,
        image_dir: Path,
        group_authors: dict[str, list[str]],
        non_anthony_group: list[str],
        anthony_group: list[str],
        sorted_groups: list[str],
        settings: CategoriesPlotSettings | None = None,
    ) -> None:
        super().__init__(
            file_manager,
            plot_manager=plot_manager,
            settings=settings or CategoriesPlotSettings(),
        )
        self.image_dir = image_dir
        self.group_authors = group_authors
        self.non_anthony_group = non_anthony_group
        self.anthony_group = anthony_group
        self.sorted_groups = sorted_groups

    def run(self) -> Path | None:
        """Generate and save the categories bar chart.

        Returns:
            Path to the saved image file, or None if generation failed.
        """
        fig = self.plot_manager.build_visual_categories(
            self.group_authors,
            self.non_anthony_group,
            self.anthony_group,
            self.sorted_groups,
            self.settings,
        )
        if fig is None:
            self.log_error("Failed to create yearly bar chart.")
            return None
        return self.save_figure(fig, self.image_dir, "yearly_bar_chart_combined")


# === CODING STANDARD (APPLY TO ALL CODE) ===
# - `# === Module Docstring ===` before """
# - Google-style docstrings
# - `# === Section Name ===` for all blocks
# - Inline: `# One space, sentence case`
# - Tags: `# TODO:`, `# NOTE:`, `# NEW: (YYYY-MM-DD)`, `# FIXME:`
# - Type hints in function signatures
# - Examples: with >>>
# - No long ----- lines
# - No mixed styles
# - Add markers #NEW at the end of the module capturing the latest changes. There can be a list of more #NEW lines.

# NEW: Fixed PLR0913, ANN001, ANN401, mypy override, and typing issues (2025-11-01)
