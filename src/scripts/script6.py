# === Module Docstring ===
"""
Launch Streamlit dashboard (Script6).

Starts the interactive WhatsApp dashboard using subprocess.
Runs `streamlit run src/dashboard/streamlit_app.py` with safe auto-open.

Examples
--------
>>> script = Script6(file_manager, image_dir=Path("images"))
>>> result = script.run()
>>> print(result["url"])
http://localhost:8501
"""

# === Imports ===
from __future__ import annotations

import sys
import time
import webbrowser
from pathlib import Path
from subprocess import Popen
from typing import Any

from loguru import logger

from .base import BaseScript


# === Script 6 ===
class Script6(BaseScript):
    """Launch the Streamlit dashboard interactively."""

    def __init__(
        self,
        file_manager,
        image_dir: Path,
    ) -> None:
        """
        Initialize Script6 with required file_manager and image directory.

        Args:
            file_manager: FileManager instance (required for BaseScript).
            image_dir: Directory containing generated plots.
        """
        super().__init__(file_manager=file_manager)
        self.image_dir = image_dir
        self.dashboard_path = Path("src/dashboard/streamlit_app.py").resolve()
        self.port = 8501
        self.url = f"http://localhost:{self.port}"

    def run(self) -> dict[str, Any] | None:
        """
        Start Streamlit dashboard in a subprocess with smart browser auto-open.

        Returns:
            dict: Contains 'process', 'url', 'port'.
            None: If dashboard file missing or launch fails.
        """
        if not self.dashboard_path.exists():
            self.log_error(f"Dashboard file not found: {self.dashboard_path}")
            return None

        if not self.image_dir.exists():
            logger.warning(f"Image directory not found: {self.image_dir}")
            logger.info("Dashboard will run but plots may not appear.")

        cmd = [
            "streamlit", "run", str(self.dashboard_path),
            "--server.port", str(self.port),
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
        ]

        logger.info(f"Launching Streamlit dashboard: {' '.join(cmd)}")

        try:
            process = Popen(cmd)
            logger.info(f"Streamlit process started (PID: {process.pid})")

            if sys.stdin and sys.stdin.isatty():
                logger.info("Interactive terminal detected – opening browser in 2s...")
                time.sleep(2)
                success = webbrowser.open(self.url)
                if success:
                    logger.success(f"Browser opened: {self.url}")
                else:
                    logger.warning(f"Failed to open browser. Visit manually: {self.url}")
            else:
                logger.info(f"Non-interactive environment – visit manually: {self.url}")

            logger.success(f"Streamlit dashboard running at: {self.url}")
            return {"process": process, "url": self.url, "port": self.port}

        except FileNotFoundError:
            self.log_error("Streamlit CLI not found. Install with: pip install streamlit")
            return None
        except Exception as e:
            self.log_error(f"Failed to start Streamlit dashboard: {e}")
            return None


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

# NEW: Removed *args, **kwargs; use keyword args in super() (2025-11-03)
# NEW: No df needed — not passed to BaseScript (2025-11-03)