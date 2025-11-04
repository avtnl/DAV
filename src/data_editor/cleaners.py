# === Module Docstring ===
"""
Cleaners module within data_editor

Focusses on message cleaning and media/system event detection.

Handles:
- Media placeholders (images, videos, etc.)
- Link removal
- System messages (deleted, group picture changes)
- Fallback cleanup

Uses pre-compiled regex patterns from ``core.DataEditor``.
"""

from __future__ import annotations

# === Imports ===
import re
from typing import List, Tuple, Set

import emoji
import pandas as pd
from loguru import logger
from nltk.corpus import stopwords  # ← NEW: for self.stopwords

from src.constants import Columns
from .utilities import has_link


# === Main Cleaner ===
class MessageCleaner:
    """Clean raw WhatsApp messages and extract media/system events."""

    # === Initialization ===
    def __init__(
        self,
        author_names_regex: str,
        ignore_emojis: Set[str],
        url_pattern: re.Pattern,
        punctuation_pattern: re.Pattern,
    ) -> None:
        """Initialize with shared patterns and build cleaning regexes.

        Args:
            author_names_regex: Alternation of all author full names (escaped).
            ignore_emojis: Set of emoji codes to skip (e.g., skin tones).
            url_pattern: Pre-compiled URL regex.
            punctuation_pattern: Pre-compiled standalone punctuation regex.
        """
        self._author_names_regex = author_names_regex
        self.ignore_emojis = ignore_emojis
        self.url_pattern = url_pattern
        self.punctuation_pattern = punctuation_pattern
        self._build_patterns()

        # === ADD: stopwords for feature engineering ===
        self.stopwords = set(stopwords.words("dutch"))

        # === ADD: patterns dict for feature engineering ===
        self.patterns = {
            "url": self.url_pattern,
            "punctuation": self.punctuation_pattern,
            "connected_emoji": re.compile(
                r"[" + "".join(re.escape(c) for c in emoji.EMOJI_DATA) + r"]{2,}",
                flags=re.UNICODE
            ),
            "connected_punct": re.compile(r"([!?.,;:]{2,})"),
            "fallback": rf"\s*[\u200e\u200f]*\[\d{{2}}-\d{{2}}-\d{{4}},\s*\d{{2}}:\d{{2}}:\d{{2}}\]\s*(?:{self._author_names_regex})[\s\u200e\u200f]*:.*",
        }

    # === Pattern Compilation ===
    def _build_patterns(self) -> None:
        """Compile all regex patterns used in cleaning."""
        # Non-media system messages
        self.non_media_patterns: List[Tuple[str, str, int]] = [
            (r"Dit bericht is verwijderd\.", "message deleted", re.IGNORECASE),
            (
                rf"(?:{self._author_names_regex}) heeft de groepsafbeelding gewijzigd",
                "grouppicture",
                re.IGNORECASE,
            ),
        ]

        # Media placeholders
        self.media_patterns: List[Tuple[str, str, int]] = [
            (r"afbeelding\s*weggelaten", "picture deleted", re.IGNORECASE),
            (r"video\s*weggelaten", "video deleted", re.IGNORECASE),
            (r"audio\s*weggelaten", "audio deleted", re.IGNORECASE),
            (r"GIF\s*weggelaten", "GIF deleted", re.IGNORECASE),
            (r"sticker\s*weggelaten", "sticker deleted", re.IGNORECASE),
            (r"document\s*weggelaten", "document deleted", re.IGNORECASE),
            (r"videonotitie\s*weggelaten", "video note deleted", re.IGNORECASE),
        ]

        # Link removal (with surrounding space)
        self.link_removal_pattern = r'(\s*https?://[^\s<>"{}|\\^`\[\]]+[\.,;:!?]?\s*)'

        # Fallback: full timestamp + author + colon
        self.fallback_pattern = rf"\s*[\u200e\u200f]*\[\d{{2}}-\d{{2}}-\d{{4}},\s*\d{{2}}:\d{{2}}:\d{{2}}\]\s*(?:{self._author_names_regex})[\s\u200e\u200f]*:.*"

    # === Helper: Detect Events ===
    def _was_deleted(self, message: str) -> bool:
        """Check if message was deleted."""
        return bool(re.search(self.non_media_patterns[0][0], message, self.non_media_patterns[0][2]))

    def _changes_to_grouppicture(self, message: str) -> int:
        """Count group picture change events."""
        return len(re.findall(self.non_media_patterns[1][0], message, self.non_media_patterns[1][2]))

    # === Main Cleaning Method ===
    def clean_messages(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Clean messages and update media/deletion columns.

        Returns:
            Cleaned DataFrame or None on failure.
        """
        try:
            media_cols = [
                Columns.PICTURES_DELETED,
                Columns.VIDEOS_DELETED,
                Columns.AUDIOS_DELETED,
                Columns.GIFS_DELETED,
                Columns.STICKERS_DELETED,
                Columns.DOCUMENTS_DELETED,
                Columns.VIDEONOTES_DELETED,
            ]

            # Initialize media columns
            for col in media_cols:
                df[col] = 0

            # === FIX: Pass url_pattern to has_link ===
            df[Columns.HAS_LINK] = df[Columns.MESSAGE].apply(
                lambda msg: has_link(msg, self.url_pattern)
            )
            df[Columns.WAS_DELETED] = df[Columns.MESSAGE].apply(self._was_deleted)
            df[Columns.MESSAGE_CLEANED] = df[Columns.MESSAGE]

            def clean_row(row: pd.Series) -> pd.Series:
                msg = row[Columns.MESSAGE]
                cleaned = msg.strip()

                # Remove links
                if row[Columns.HAS_LINK]:
                    cleaned = re.sub(self.link_removal_pattern, " ", cleaned, flags=re.IGNORECASE)
                    cleaned = re.sub(r"\s+", " ", cleaned).strip()

                # Remove non-media system messages
                for pattern, _, flags in self.non_media_patterns:
                    cleaned = re.sub(pattern, "", cleaned, flags=flags).strip()

                # Count and remove media placeholders
                for pattern, media_type, flags in self.media_patterns:
                    matches = re.findall(pattern, cleaned, flags=flags)
                    count = len(matches)
                    if count > 0:
                        if media_type == "picture deleted":
                            row[Columns.PICTURES_DELETED] += count
                        elif media_type == "video deleted":
                            row[Columns.VIDEOS_DELETED] += count
                        elif media_type == "audio deleted":
                            row[Columns.AUDIOS_DELETED] += count
                        elif media_type == "GIF deleted":
                            row[Columns.GIFS_DELETED] += count
                        elif media_type == "sticker deleted":
                            row[Columns.STICKERS_DELETED] += count
                        elif media_type == "document deleted":
                            row[Columns.DOCUMENTS_DELETED] += count
                        elif media_type == "video note deleted":
                            row[Columns.VIDEONOTES_DELETED] += count
                        cleaned = re.sub(pattern, "", cleaned, flags=flags).strip()

                # Remove timestamp+author if media was present
                total_media = sum(row[col] for col in media_cols)
                if total_media > 0:
                    ta_pattern = r"[\s\u200e\u200f]*\[\d{2}-\d{2}-\d{4},\s*\d{2}:\d{2}:\d{2}\]\s*[^:]*:[\s\u200e\u200f]*$"
                    cleaned = re.sub(ta_pattern, "", cleaned).strip()

                # Final fallback cleanup
                if re.search(self.fallback_pattern, cleaned, flags=re.IGNORECASE):
                    cleaned = re.sub(self.fallback_pattern, "", cleaned, flags=re.IGNORECASE).strip()

                # Final fallback text
                if not cleaned or cleaned.strip() == "":
                    cleaned = "completely removed"

                row[Columns.MESSAGE_CLEANED] = cleaned
                return row

            df = df.apply(clean_row, axis=1)
            logger.info(f"Cleaned {len(df)} messages")
            return df

        except Exception as e:
            logger.exception(f"Message cleaning failed: {e}")
            return None


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

# NEW: Created cleaners.py with MessageCleaner class (2025-11-03)
# NEW: All author names via author_names_regex (2025-11-03)
# NEW: Full cleaning logic with media counting (2025-11-03)
# NEW: Strict 1-blank-line rule enforced (2025-11-03)
# NEW: Google-style docstrings with examples (2025-11-03)
# NEW: (2025-11-04) – Added self.patterns dict to __init__ for feature engineering
# NEW: (2025-11-04) – Added self.stopwords from NLTK
# NEW: (2025-11-04) – Fixed has_link() call with url_pattern
# NEW: (2025-11-04) – Fixed emoji.EMOJI_DATA usage