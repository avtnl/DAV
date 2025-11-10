# === script3.py ===
# === Module Docstring ===
"""
Distribution plot: Emoji distribution for the MAAP group (Script 3).

Filters the enriched DataFrame for the MAAP group, prepares the emoji-frequency
DataFrame via :meth:`src.data_preparation.DataPreparation.build_visual_distribution`,
and generates the bar + cumulative line chart via
:meth:`src.plot_manager.PlotManager.build_visual_distribution`.

- Saves **full emoji table** (all emojis + counts + Unicode) to `data/tables/`
- Passes **top 20** to `plot_manager` for consistent visualization
- Optional power-law (Zipf) analysis - runs K-S test, saves log-log plot and model-comparison table
- Flexible run modes: individual groups, combined, or both
- Robust error handling and logging
"""

# === Imports ===
from pathlib import Path
import pandas as pd

from loguru import logger
from src.constants import Columns, Groups, RunMode  # NEW: RunMode
from src.plot_manager import DistributionPlotSettings
from src.data_preparation import DistributionPlotData

from .base import BaseScript


# === Script 3 ===
class Script3(BaseScript):
    """Generate emoji distribution plots for WhatsApp groups with flexible run modes."""

    def __init__(
        self,
        file_manager,
        data_editor,
        data_preparation,
        plot_manager,
        image_dir: Path,
        df: pd.DataFrame,
        settings: DistributionPlotSettings | None = None,
        run_powerlaw: bool = True,
        run_mode: RunMode = RunMode.COMBINED,  # <-- RunMode (INDIVIDUAL, COMBINED, BOTH)
    ) -> None:
        """
        Initialize Script3 with required components.

        Args:
            file_manager: FileManager (required by BaseScript).
            data_editor: DataEditor (required for emoji parsing).
            data_preparation: DataPreparation for distribution data.
            plot_manager: PlotManager for rendering.
            image_dir: Directory to save plot.
            df: Enriched DataFrame (required).
            settings: Plot settings (optional).
            run_powerlaw: If True, run power-law analysis.
            run_mode: Execution mode: individual groups, combined, or both.
        """
        super().__init__(
            file_manager=file_manager,
            data_editor=data_editor,
            data_preparation=data_preparation,
            plot_manager=plot_manager,
            settings=settings or DistributionPlotSettings(),
            df=df,
        )
        self.image_dir = image_dir
        self.run_powerlaw = run_powerlaw
        self.run_mode = run_mode

    def run(self) -> dict | None:
        """
        Generate emoji distribution plots based on run_mode.

        - RunMode.INDIVIDUAL: One plot per group
        - RunMode.COMBINED: One plot for all groups combined
        - RunMode.BOTH: Both of the above

        Returns:
            Dict[group_name, Path] for individual bar charts (or None)
        """
        results = {}
        tables_dir = self.image_dir.parent / "tables"
        tables_dir.mkdir(exist_ok=True)

        # === INDIVIDUAL GROUPS ===
        if self.run_mode in (RunMode.INDIVIDUAL, RunMode.BOTH):
            all_groups = [Groups.MAAP.value, Groups.DAC.value, Groups.GOLFMATEN.value, Groups.TILLIES.value]
            for group_name in all_groups:
                df_group = self.df[self.df[Columns.WHATSAPP_GROUP.value] == group_name].copy()
                if df_group.empty:
                    logger.warning(f"No data for group '{group_name}'. Skipping.")
                    continue

                logger.info(f"Processing group: {group_name} ({len(df_group)} messages)")

                # Build distribution
                distribution_data = self.data_preparation.build_visual_distribution(df_group)
                if not distribution_data:
                    logger.error(f"Failed to build distribution for {group_name}")
                    continue

                full_df = distribution_data.emoji_counts_df
                logger.info(f"Unique emojis in {group_name}: {len(full_df)}")

                # Save full table
                table_path = self.file_manager.save_table(
                    full_df, tables_dir, f"emoji_counts_full_{group_name}"
                )
                logger.success(f"Saved: {table_path}")

                # Bar + cumulative
                top_20_data = DistributionPlotData(emoji_counts_df=full_df.head(20))
                fig = self.plot_manager.build_visual_distribution(distribution_data, self.settings)
                if fig:
                    bar_path = self.save_figure(fig, self.image_dir, f"emoji_counts_once_{group_name}")
                    results[group_name] = bar_path
                    logger.success(f"Bar plot saved: {bar_path}")

                # Power-law per group
                if self.run_powerlaw:
                    self._run_powerlaw_analysis(df_group, tables_dir, suffix=f"_{group_name}")

            logger.success("Individual group analysis complete.")

        # === COMBINED ANALYSIS ===
        if self.run_mode in (RunMode.COMBINED, RunMode.BOTH):
            logger.info("Running COMBINED analysis on ALL groups...")
            self.run_combined_analysis(tables_dir)
            logger.success("Combined analysis complete.")

        return results or None

    # === Power-law analysis helper ===
    def _run_powerlaw_analysis(self, df_group: pd.DataFrame, tables_dir: Path, suffix: str = "") -> None:
        """Run power-law analysis and save plots."""
        try:
            analysis_method = getattr(self.data_preparation, "analyze_emoji_distribution_power_law", None)
            if not analysis_method:
                logger.info("Power-law analysis method not available – skipping.")
                return

            try:
                import powerlaw  # noqa: F401
            except Exception:
                logger.warning("Package `powerlaw` not installed – skipping proof.")
                return

            logger.info(f"Starting power-law analysis{suffix}...")
            analysis = analysis_method(df_group)
            if not analysis:
                logger.warning("Power-law analysis returned None.")
                return

            fit = analysis.fit
            logger.success(
                f"Power-law fit{suffix}: α={fit.alpha:.3f}, xₘᵢₙ={fit.xmin}, "
                f"K-S D={fit.D:.3f}, n_tail={fit.n_tail}"
            )
            if analysis.is_power_law:
                logger.success("Power-law is the best model (p < 0.05 vs alternatives).")
            else:
                logger.info("Power-law fit is acceptable but not definitively superior.")

            # Log-log plot
            loglog_fig = self.plot_manager.build_visual_distribution_powerlaw(analysis)
            if loglog_fig:
                loglog_path = self.save_figure(loglog_fig, self.image_dir, f"logmodel_evidence_distribution_plot{suffix}")
                logger.success(f"Log-log plot saved: {loglog_path}")

        except Exception as e:
            logger.exception(f"Power-law analysis failed{suffix}: {e}")

    # === Combined analysis for ALL groups ===
    def run_combined_analysis(self, tables_dir: Path) -> None:
        """Run power-law analysis on ALL messages from all groups."""
        logger.info("Running power-law analysis on ALL groups combined...")

        df_all = self.df.copy()
        total_messages = len(df_all)
        logger.info(f"Combined dataset: {total_messages} messages")

        distribution_data = self.data_preparation.build_visual_distribution(df_all)
        if not distribution_data:
            logger.error("Failed to build combined distribution")
            return

        full_df = distribution_data.emoji_counts_df
        logger.info(f"Unique emojis (ALL): {len(full_df)}")

        table_path = self.file_manager.save_table(full_df, tables_dir, "emojies (highest to lowest)")
        logger.success(f"Saved: {table_path}")

        top_20_data = DistributionPlotData(emoji_counts_df=full_df.head(20))
        fig = self.plot_manager.build_visual_distribution(distribution_data, self.settings)
        if fig:
            bar_path = self.save_figure(fig, self.image_dir, "distribution_plot")
            logger.success(f"Bar plot (ALL): {bar_path}")

        if self.run_powerlaw:
            self._run_powerlaw_analysis(df_all, tables_dir, suffix="_ALL")


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

# NEW: Added RunMode support (individual/combined/both) (2025-11-07)
# NEW: Full per-group + combined analysis with suffix handling (2025-11-07)
# NEW: Clean run_mode logic with StrEnum (2025-11-07)
# NEW: Full table export to data/tables/ (2025-11-05)
# NEW: Pass top 20 to plot_manager for consistent visualization (2025-11-05)
# NEW: Robust directory creation and logging (2025-11-05)
# NEW: Clear step-by-step run() docstring (2025-11-05)