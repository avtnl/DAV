# === Module Docstring ===
"""
Multi-dimensional plot: PCA/t-SNE clustering with embeddings (Script 5).

Aggregates yearly author style, applies embeddings, reduces dimensions,
and renders interactive t-SNE/PCA plots (individual or group-level).

=== SCRIPT_5_DETAILS README ===
Configure Script6 via: SCRIPT_5_DETAILS = [PLOT_TYPE, BY_GROUP, ELLIPSE_MODE, CONFIDENCE_LEVEL, USE_EMBEDDINGS, HYBRID_FEATURES, EMBEDDING_MODEL]

1. PLOT_TYPE (str): "pca" | "tsne" | "both"
2. BY_GROUP (bool): True → group-level, False → per-author
3. ELLIPSE_MODE (int): 0=no ellipses, 1=single ellipse, 2=more elipses for max 3 pockets (GMM pockets)
4. CONFIDENCE_LEVEL (int): 20-100 (%) → only used if ELLIPSE_MODE > 0
5. USE_EMBEDDINGS (bool): True → load Hugging-Face embeddings
6. HYBRID_FEATURES (bool): True → combine hand-crafted + embeddings
7. EMBEDDING_MODEL (int): 1=Style-Embedding, 2=all-MiniLM, 3=all-mpnet

Example (GMM pockets @ 75% confidence):
>>> SCRIPT_5_DETAILS = ["tsne", False, 2, 75, True, True, 1]
"""

# === Imports ===
from pathlib import Path
from typing import Any
import pandas as pd

from loguru import logger
import warnings

warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

from src.constants import Script5ConfigKeys
from src.plot_manager import MultiDimPlotSettings

from .base import BaseScript


# === Validation ===
def _validate_script5_details(details: list) -> None:
    if len(details) != 7:
        raise ValueError("SCRIPT_5_DETAILS must have exactly 7 values")

    plot_type, by_group, ellipse_mode, conf_level, use_emb, hybrid, model_id = details

    if plot_type not in {"pca", "tsne", "both"}:
        raise ValueError("PLOT_TYPE must be 'pca', 'tsne', or 'both'")

    if not isinstance(by_group, bool):
        raise TypeError("BY_GROUP must be bool")

    if ellipse_mode not in {0, 1, 2}:
        raise ValueError("ELLIPSE_MODE must be 0 (none), 1 (single), or 2 (GMM pockets)")

    if not (20 <= conf_level <= 100):
        raise ValueError("CONFIDENCE_LEVEL must be 20–100 (%)")
    if ellipse_mode == 0 and conf_level != 75:
        logger.warning("CONFIDENCE_LEVEL ignored when ELLIPSE_MODE=0")

    if not isinstance(use_emb, bool):
        raise TypeError("USE_EMBEDDINGS must be bool")

    if use_emb:
        if not isinstance(hybrid, bool):
            raise TypeError("HYBRID_FEATURES must be bool")
        if model_id not in {1, 2, 3}:
            raise ValueError("EMBEDDING_MODEL must be 1, 2, or 3")
    else:
        logger.info("USE_EMBEDDINGS=False → HYBRID_FEATURES and EMBEDDING_MODEL ignored")


# === Script 5 ===
class Script5(BaseScript):
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
        super().__init__(
            file_manager=file_manager,
            data_preparation=data_preparation,
            plot_manager=plot_manager,
            settings=settings or MultiDimPlotSettings(),
            df=df,
        )
        self.image_dir = image_dir  # e.g. Path("img")

        if script_details is not None:
            _validate_script5_details(script_details)
            self.script_details = script_details
        else:
            self.script_details = ["tsne", False, 0, 75, True, True, 3]
            logger.info("No SCRIPT_5_DETAILS → using safe defaults (per-author)")

    def _get_config_code(self) -> str:
        """Convert to: TT75T1_tsne"""
        d = self.script_details
        mapping = {True: "T", False: "F"}
        bool_code = mapping[d[1]] + str(d[2]) + str(d[3])
        emb_code = mapping[d[4]] + mapping[d[5]]
        return f"_{bool_code}{emb_code}{d[6]}_{d[0]}"

    def run(self) -> dict[str, Path] | None:
        if self.df is None or self.df.empty:
            self.log_error("No enriched DataFrame. Run Script0 first.")
            return None

        details = self.script_details
        settings_dict = {
            Script5ConfigKeys.PLOT_TYPE: details[0],
            Script5ConfigKeys.BY_GROUP: details[1],
            Script5ConfigKeys.ELLIPSE_MODE: details[2],
            Script5ConfigKeys.CONFIDENCE_LEVEL: details[3],
            Script5ConfigKeys.USE_EMBEDDINGS: details[4],
            Script5ConfigKeys.HYBRID_FEATURES: details[5],
            Script5ConfigKeys.EMBEDDING_MODEL: details[6],
        }

        data = self.data_preparation.build_visual_multi_dimensions(self.df, settings_dict)
        if not data:
            self.log_error("Data preparation failed.")
            return None

        # Override settings regarding ellipse logic
        self.settings.ellipse_mode = details[2]
        self.settings.confidence_level = details[3]
        self.settings.by_group = details[1]

        figs = self.plot_manager.build_visual_multi_dimensions(data, self.settings)
        if not figs:
            self.log_error("Plot creation failed.")
            return None

        config_code = self._get_config_code()  # e.g. _T025FT1_both
        logger.info(f"Script5 config: {details} → {config_code}")

        results = {}

        # PNG & HTML → img/ (top-level)
        plot_dir = self.image_dir
        plot_dir.mkdir(parents=True, exist_ok=True)

        # CSV summary → src/style_output/
        csv_dir = Path("src") / "style_output"
        csv_dir.mkdir(parents=True, exist_ok=True)

        # === Pre-extract values to avoid repeated indexing ===
        d = self.script_details
        mapping = {True: "T", False: "F"}
        bool_code = mapping[d[1]] + str(d[2]) + str(d[3])
        emb_code = mapping[d[4]] + mapping[d[5]]
        model_id = d[6]

        for name, fig in figs.items():
            # === Determine inner plot_type from figure key ===
            if name.startswith("pca_"):
                inner_plot_type = "pca"
            elif name.startswith("tsne_"):
                inner_plot_type = "tsne"
            else:
                inner_plot_type = d[0]  # fallback

            #  Rebuild suffix with correct plot_type 
            suffix = f"_{bool_code}{emb_code}{model_id}_{inner_plot_type}"

            # Clean name (remove pca_/tsne_ prefix)
            clean_name = name.replace("pca_", "").replace("tsne_", "")

            # File paths
            png_path = plot_dir / f"style_{clean_name}{suffix}.png"
            html_path = plot_dir / f"style_{clean_name}{suffix}.html"

            # Save HTML
            fig.write_html(str(html_path))
            logger.success(f"Saved HTML: {html_path}")

            # Save PNG
            try:
                fig.write_image(str(png_path), width=1200, height=800)
                logger.success(f"Saved PNG: {png_path}")
            except Exception as e:
                logger.warning(f"PNG export failed: {e}")

            results[name] = png_path

        # Save CSV summary (uses original config_code or last inner type)
        final_suffix = f"_{bool_code}{emb_code}{model_id}_{d[0]}"
        csv_path = csv_dir / f"style_summary{final_suffix}.csv"
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

