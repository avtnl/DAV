# === base.py ===
from pathlib import Path
from typing import Any

from loguru import logger

# === Base Script ===
class BaseScript:
    """Common functionality for all scripts."""

    def __init__(
        self,
        file_manager,
        data_editor=None,
        data_preparation=None,
        plot_manager=None,
        settings=None,
        *args,
        **kwargs,
    ) -> None:
        self.file_manager = file_manager
        self.data_editor = data_editor
        self.data_preparation = data_preparation
        self.plot_manager = plot_manager
        self.settings = settings

    def log_error(self, msg: str) -> None:
        logger.error(msg)

    def save_figure(self, fig, image_dir: Path, name: str) -> Path:
        image_dir.mkdir(parents=True, exist_ok=True)
        path = image_dir / f"{name}.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        logger.success(f"Plot saved: {path}")
        return path