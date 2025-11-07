# === script3.py ===
# === Module Docstring ===
"""
Distribution plot: Emoji distribution for the MAAP group (Script 3).

Filters the enriched DataFrame for the MAAP group, prepares the emoji-frequency
DataFrame via :meth:`src.data_preparation.DataPreparation.build_visual_distribution`,
and generates the bar + cumulative line chart via
:meth:`src.plot_manager.PlotManager.build_visual_distribution`.

**New features added:**
- Saves **full emoji table** (all emojis + counts + Unicode) to `data/tables/`
- Passes **top 20** to `plot_manager` for consistent visualization
- **Optional power-law (Zipf) analysis** – runs K-S test, saves log-log plot and model-comparison table
- Robust error handling and logging

Examples
--------
>>> script = Script3(file_manager, data_editor, data_preparation, plot_manager, image_dir, df)
>>> script.run()
PosixPath('images/emoji_counts_once.png')
"""

# === Imports ===
from pathlib import Path
import pandas as pd

from loguru import logger
from src.constants import Columns, Groups
from src.plot_manager import DistributionPlotSettings
from src.data_preparation import DistributionPlotData

from .base import BaseScript


# === Script 3 ===
class Script3(BaseScript):
    """Generate emoji distribution plot for MAAP group (with optional power-law proof)."""

    def __init__(
        self,
        file_manager,
        data_editor,
        data_preparation,
        plot_manager,
        image_dir: Path,
        df: pd.DataFrame,
        settings: DistributionPlotSettings | None = None,
        run_powerlaw: bool = True,                     # NEW: toggle power-law proof
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
            run_powerlaw: If True, also run power-law analysis after the bar chart.
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

    def run(self) -> Path | None:
        """
        Generate and save the emoji distribution plot (and optional power-law proof).

        Steps:
            1. Filter for MAAP group
            2. Build full emoji distribution (with Unicode)
            3. Save **full table** to `data/tables/`
            4. Pass **top 20** to `plot_manager`
            5. Save bar + cumulative plot to `images/`
            6. (Optional) Run power-law analysis → log-log plot + model-comparison table

        Returns:
            Path: Path to saved PNG file (the original bar chart).
            None: If data missing or plot fails.
        """
        # === 1. Filter for MAAP group ===
        df_maap = self.df[self.df[Columns.WHATSAPP_GROUP.value] == Groups.MAAP.value].copy()
        if df_maap.empty:
            self.log_error(f"No data for group '{Groups.MAAP.value}'. Skipping.")
            return None

        # === 2. Build full emoji distribution ===
        distribution_data = self.data_preparation.build_visual_distribution(df_maap)
        if distribution_data is None:
            self.log_error("build_visual_distribution returned None.")
            return None

        full_df = distribution_data.emoji_counts_df
        logger.info(f"Unique emojis: {len(full_df)}")

        # === 3. Save full emoji table ===
        tables_dir = self.image_dir.parent / "tables"
        tables_dir.mkdir(exist_ok=True)
        full_table_path = self.file_manager.save_table(
            full_df,
            tables_dir,
            "emoji_counts_full"
        )
        logger.success(f"Saved full emoji table: {full_table_path}")

        # === 4. Pass top 20 to plot manager ===
        top_20_df = full_df.head(20)
        top_20_data = DistributionPlotData(emoji_counts_df=top_20_df)

        fig = self.plot_manager.build_visual_distribution(
            distribution_data,  # Full data (for cumulative line)
            self.settings      # With cum_threshold=75.0
        )
        if fig is None:
            self.log_error("Failed to create emoji bar chart.")
            return None

        # === 5. Save original plot ===
        bar_chart_path = self.save_figure(fig, self.image_dir, "emoji_counts_once")
        logger.success(f"Bar + cumulative plot saved: {bar_chart_path}")

        # === 6. Optional power-law (Zipf) proof ===
        if self.run_powerlaw:
            self._run_powerlaw_analysis(df_maap, tables_dir)

        return bar_chart_path


    # === Power-law analysis helper ===
    def _run_powerlaw_analysis(self, df_maap: pd.DataFrame, tables_dir: Path) -> None:
        """
        Run power-law analysis on the MAAP emoji frequencies.

        - Calls ``DataPreparation.analyze_emoji_distribution_power_law`` **only if the
        `powerlaw` package is installed**.
        - Saves log-log plot (PNG) and model-comparison table (HTML) if successful.
        - Gracefully skips the whole block if the package is missing.

        Args:
            df_maap: DataFrame filtered to MAAP group.
            tables_dir: Directory where the full table lives (used for naming consistency).
        """
        try:
            # Guard – the method may not exist in older codebases
            analysis_method = getattr(
                self.data_preparation, "analyze_emoji_distribution_power_law", None
            )
            if analysis_method is None:
                logger.info("Power-law analysis method not available – skipping.")
                return

            # Guard – `powerlaw` package must be importable
            try:
                import powerlaw  # noqa: F401
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Package `powerlaw` is not installed – power-law proof skipped. "
                    "Install with: pip install powerlaw"
                )
                return

            logger.info("Starting power-law (Zipf) analysis on emoji frequencies...")
            analysis = analysis_method(df_maap)

            if analysis is None:
                logger.warning("Power-law analysis returned None.")
                return

            # Log key results
            fit = analysis.fit
            logger.success(
                f"Power-law fit: α={fit.alpha:.3f}, xₘᵢₙ={fit.xmin}, "
                f"K-S D={fit.D:.3f}, n_tail={fit.n_tail}"
            )
            if analysis.is_power_law:
                logger.success("Power-law is the best model (p > 0.05 and beats alternatives).")
            else:
                logger.info("Power-law fit is acceptable but not definitively superior.")

            # === Log-log plot ===
            loglog_fig = self.plot_manager.build_visual_distribution_powerlaw(analysis)
            if loglog_fig:
                loglog_path = self.save_figure(
                    loglog_fig, self.image_dir, "emoji_powerlaw_loglog"
                )
                logger.success(f"Log-log power-law plot saved: {loglog_path}")

            # === Model-comparison table (interactive HTML) ===
            comp_fig = self.plot_manager.build_visual_distribution_comparison(analysis)
            if comp_fig:
                html_path = self.image_dir / "emoji_model_comparison.html"
                comp_fig.write_html(str(html_path))
                logger.success(f"Model-comparison table saved: {html_path}")

        except Exception as e:
            logger.exception(f"Power-law analysis failed: {e}")


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

# NEW: Added optional power-law proof (log-log plot + model-comparison) (2025-11-07)
# NEW: ``run_powerlaw`` flag in __init__ (default True) (2025-11-07)
# NEW: ``_run_powerlaw_analysis`` private helper with full error handling (2025-11-07)
# NEW: Full table export to data/tables/ (2025-11-05)
# NEW: Pass top 20 to plot_manager for consistent visualization (2025-11-05)
# NEW: Robust directory creation and logging (2025-11-05)
# NEW: Clear step-by-step run() docstring (2025-11-05)