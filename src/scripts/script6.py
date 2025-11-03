# === script6.py ===
# === Module Docstring ===
"""
Multi-dimensional plot: PCA/t-SNE clustering with embeddings (Script 6).

Aggregates yearly author style, applies embeddings, reduces dimensions,
and renders interactive t-SNE plots (individual or group-level).

=== SCRIPT_6_DETAILS README ===
Configure Script6 via: SCRIPT_6_DETAILS = [PLOT_TYPE, BY_GROUP, DRAW_ELLIPSES, USE_EMBEDDINGS, HYBRID_FEATURES, EMBEDDING_MODEL]

1. PLOT_TYPE (str): "pca" | "tsne" | "both"
2. BY_GROUP (bool): True → group-level, False → per-author
3. DRAW_ELLIPSES (bool): True → draw confidence ellipses
4. USE_EMBEDDINGS (bool): True → load Hugging-Face embeddings
5. HYBRID_FEATURES (bool): True → combine hand-crafted + embeddings
6. EMBEDDING_MODEL (int): 1=Style-Embedding, 2=all-MiniLM, 3=all-mpnet (default)

Example:
>>> SCRIPT_6_DETAILS = ["tsne", True, True, False, True, 1]
"""

# === Imports ===
from pathlib import Path
from typing import Any
import pandas as pd

import pandas as pd
from loguru import logger
import warnings

warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores for the following reason"
)

from src.constants import Script6ConfigKeys
from src.plot_manager import MultiDimPlotSettings

from .base import BaseScript


# === Validation ===
def _validate_script6_details(details: list) -> None:
    """
    Validate SCRIPT_6_DETAILS configuration.

    Args:
        details: List of 6 configuration values.

    Raises:
        ValueError: If length or value invalid.
        TypeError: If type mismatch.
    """
    if len(details) != 6:
        raise ValueError("SCRIPT_6_DETAILS must have exactly 6 values")

    plot_type, by_group, draw_ellipses, use_emb, hybrid, model_id = details

    if plot_type not in {"pca", "tsne", "both"}:
        raise ValueError("PLOT_TYPE must be 'pca', 'tsne', or 'both'")

    if not all(isinstance(x, bool) for x in [by_group, draw_ellipses, use_emb]):
        raise TypeError("BY_GROUP, DRAW_ELLIPSES, USE_EMBEDDINGS must be bool")

    if use_emb:
        if not isinstance(hybrid, bool):
            raise TypeError("HYBRID_FEATURES must be bool when USE_EMBEDDINGS=True")
        if model_id not in {1, 2, 3}:
            raise ValueError("EMBEDDING_MODEL must be 1, 2, or 3 when USE_EMBEDDINGS=True")
    else:
        logger.info("USE_EMBEDDINGS=False → HYBRID_FEATURES and EMBEDDING_MODEL are ignored")


# === Script 6 ===
class Script6(BaseScript):
    """Multi-dimensional linguistic style analysis."""

    def __init__(
        self,
        file_manager,
        data_preparation,
        plot_manager,
        image_dir: Path,
        df: pd.DataFrame,
        settings: MultiDimPlotSettings | None = None,
        script_details: list | None = None,
    ) -> None:
        """
        Initialize Script6 with configuration.

        Args:
            file_manager: FileManager (required).
            data_preparation: DataPreparation for style aggregation.
            plot_manager: PlotManager for rendering.
            image_dir: Directory to save plots.
            df: Enriched DataFrame (required).
            settings: Multi-dimensional plot settings.
            script_details: List of 6 config values (see README).
        """
        super().__init__(
            file_manager=file_manager,
            data_preparation=data_preparation,
            plot_manager=plot_manager,
            settings=settings or MultiDimPlotSettings(),
            df=df,
        )
        self.image_dir = image_dir

        if script_details is not None:
            _validate_script6_details(script_details)
            self.script_details = script_details
        else:
            self.script_details = ["tsne", True, False, True, True, 3]
            logger.info("No SCRIPT_6_DETAILS provided. Using defaults.")

    def _get_config_code(self) -> str:
        """Convert SCRIPT_6_DETAILS to compact code: TTFT1_tsne"""
        details = self.script_details
        mapping = {True: "T", False: "F"}
        bool_code = "".join(mapping.get(details[i], "X") for i in range(1, 5))
        return "_".join([bool_code, str(details[5]), details[0]])

    def run(self) -> dict[str, Path] | None:
        """
        Generate and save t-SNE/PCA style plots with config-coded filenames.

        Returns:
            dict: Mapping of plot name to PNG path.
            None: If data or plot fails.
        """
        if self.df is None or self.df.empty:
            self.log_error("No enriched DataFrame. Run Script0 first.")
            return None

        details = self.script_details
        settings_dict = {
            Script6ConfigKeys.PLOT_TYPE: details[0],
            Script6ConfigKeys.BY_GROUP: details[1],
            Script6ConfigKeys.DRAW_ELLIPSES: details[2],
            Script6ConfigKeys.USE_EMBEDDINGS: details[3],
            Script6ConfigKeys.HYBRID_FEATURES: details[4],
            Script6ConfigKeys.EMBEDDING_MODEL: details[5],
        }

        data = self.data_preparation.build_visual_multi_dimensions(self.df, settings_dict)
        if not data:
            self.log_error("Data preparation failed.")
            return None

        figs = self.plot_manager.build_visual_multi_dimensions(data, self.settings)
        if not figs:
            self.log_error("Plot creation failed.")
            return None

        config_code = self._get_config_code()
        logger.info(f"Script6 config: {details} → {config_code}")

        results = {}
        style_dir = self.image_dir / "style_output"
        style_dir.mkdir(parents=True, exist_ok=True)

        for name, fig in figs.items():
            suffix = f"_{config_code}"
            html_path = style_dir / f"style_{name}{suffix}.html"
            fig.write_html(str(html_path))
            logger.success(f"Saved HTML: {html_path}")

            png_path = style_dir / f"style_{name}{suffix}.png"
            try:
                fig.write_image(str(png_path), width=1200, height=800)
                logger.success(f"Saved PNG: {png_path}")
            except Exception as e:
                logger.warning(f"PNG export failed (kaleido missing?): {e}")
                logger.info("Install with: uv add kaleido")

            results[name] = png_path

        csv_path = style_dir / f"style_summary{suffix}.csv"
        data.agg_df.to_csv(csv_path, index=False)
        logger.success(f"Saved summary: {csv_path}")

        return results


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
# - Add markers #NEW at the end of the module capturing the latest changes.

# NEW: df passed to super(); no *args, **kwargs (2025-11-03)