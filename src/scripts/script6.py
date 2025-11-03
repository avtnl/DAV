# === script6.py ===
# === Module Docstring ===
"""
Multi-dimensional plot: PCA/t-SNE clustering with embeddings (Script 6).

Aggregates yearly author style, applies embeddings, reduces dimensions,
and renders interactive t-SNE plots (individual or group-level).

=== SCRIPT_6_DETAILS README ===
Configure Script6 via: SCRIPT_6_DETAILS = [PLOT_TYPE, BY_GROUP, DRAW_ELLIPSES, USE_EMBEDDINGS, HYBRID_FEATURES, EMBEDDING_MODEL]

1. PLOT_TYPE (str):
   "pca"  -> 2-D PCA projection only
   "tsne" -> t-SNE projection (PCA pre-step)
   "both" -> separate PCA and t-SNE plots

2. BY_GROUP (bool):
   True  -> group-level plot (AvT isolated)
   False -> per-author plot

3. DRAW_ELLIPSES (bool):
   True  -> draw confidence ellipses (75% individual, 50% group)
   False -> no ellipses

4. USE_EMBEDDINGS (bool):
   True  -> load Hugging-Face sentence embeddings
   False -> hand-crafted features only

5. HYBRID_FEATURES (bool):
   True  -> combine hand-crafted + embeddings (best results)
   False -> embeddings only
   (ignored when USE_EMBEDDINGS=False)

6. EMBEDDING_MODEL (int):
   1 -> "AnnaWegmann/Style-Embedding"        -> style-focused, ignores content
   2 -> "sentence-transformers/all-MiniLM-L6-v2" -> fast, general
   3 -> "sentence-transformers/all-mpnet-base-v2" -> high-quality (default)
   (ignored when USE_EMBEDDINGS=False)

Example:
SCRIPT_6_DETAILS = ["tsne", True, True, False, True, 1]   # -> TTFT1_tsne
SCRIPT_6_DETAILS = ["pca",   False, False, False, True, 3] # -> FFFF3_pca

Note: by_year is always True (yearly aggregation for stability)
"""

# === Imports ===
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
import warnings

# === Suppress joblib warning on Windows ===
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
        details: List of 6 configuration values

    Raises:
        ValueError, TypeError: If invalid
    """
    if len(details) != 6:
        raise ValueError("SCRIPT_6_DETAILS must have exactly 6 values")

    plot_type, by_group, draw_ellipses, use_emb, hybrid, model_id = details

    # Validate plot_type
    if plot_type not in {"pca", "tsne", "both"}:
        raise ValueError("PLOT_TYPE must be 'pca', 'tsne', or 'both'")

    # Validate bools
    if not all(isinstance(x, bool) for x in [by_group, draw_ellipses, use_emb]):
        raise TypeError("BY_GROUP, DRAW_ELLIPSES, USE_EMBEDDINGS must be bool")

    # Validate hybrid & model only if embeddings are used
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
        df: pd.DataFrame | None = None,
        settings: MultiDimPlotSettings | None = None,
        script_details: list | None = None,
    ) -> None:
        super().__init__(
            file_manager,
            data_preparation=data_preparation,
            plot_manager=plot_manager,
            settings=settings or MultiDimPlotSettings(),
            df=df,
        )
        self.image_dir = image_dir

        # === Validate configuration ===
        if script_details is not None:
            _validate_script6_details(script_details)
            self.script_details = script_details
        else:
            self.script_details = ["tsne", True, False, True, True, 3]  # default
            logger.info("No SCRIPT_6_DETAILS provided. Using defaults.")

    def _get_config_code(self) -> str:
        """Convert SCRIPT_6_DETAILS to compact code: TTFT1_tsne"""
        details = self.script_details
        mapping = {True: "T", False: "F"}
        
        # Bool code (indices 1-4) - compact, no separator
        bool_code = "".join(mapping.get(details[i], "X") for i in range(1, 5))
        
        # model_id (index 5)
        model_code = str(details[5])
        
        # plot_type (index 0)
        plot_type = details[0]
        
        # Join with _
        return "_".join([bool_code, model_code, plot_type])

    def run(self) -> dict[str, Path] | None:
        """Generate and save t-SNE style plots with config-coded filenames."""
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

        # === Generate config code: TTFT1_tsne ===
        config_code = self._get_config_code()
        logger.info(f"Script6 config: {details} → {config_code}")

        results = {}
        style_dir = self.image_dir / "style_output"
        style_dir.mkdir(parents=True, exist_ok=True)

        for name, fig in figs.items():
            suffix = f"_{config_code}"

            # Save HTML
            html_path = style_dir / f"style_{name}{suffix}.html"
            fig.write_html(str(html_path))
            logger.success(f"Saved HTML: {html_path}")

            # Save PNG (requires kaleido)
            png_path = style_dir / f"style_{name}{suffix}.png"
            try:
                fig.write_image(str(png_path), width=1200, height=800)
                logger.success(f"Saved PNG: {png_path}")
            except Exception as e:
                logger.warning(f"PNG export failed (kaleido missing?): {e}")
                logger.info("Install with: uv add kaleido")

            results[name] = png_path

        # Save CSV
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

# NEW: Initial Script6 with full integration (2025-11-03)
# NEW: Validation moved into script6.py (2025-11-03)
# NEW: Direct script_6_details injection from pipeline (2025-11-03)
# NEW: Embedded README in docstring (2025-11-03)
# NEW: Conditional validation for USE_EMBEDDINGS (2025-11-03)
# NEW: Added PLOT_TYPE toggle, updated validation and config code (2025-11-03)