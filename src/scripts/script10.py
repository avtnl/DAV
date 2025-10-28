# src/scripts/script10.py
from .base import BaseScript
from loguru import logger
import pandas as pd

class Script10(BaseScript):
    def __init__(self, file_manager, data_editor, data_preparation, processed_dir, tables_dir):
        super().__init__(file_manager, data_editor=data_editor, data_preparation=data_preparation)
        self.processed_dir = processed_dir
        self.tables_dir = tables_dir

    def run(self):
        # -------------------------------------------------------------
        # 1. Load preprocessed data (from Script0)
        # -------------------------------------------------------------
        latest_parquet = max(self.processed_dir.glob("combined_*.parq"), key=lambda p: p.stat().st_mtime)
        if not latest_parquet.exists():
            return self.log_error("No preprocessed parquet found. Run Script0 first.")

        df = pd.read_parquet(latest_parquet)
        logger.info(f"Loaded preprocessed data: {latest_parquet.name} → {df.shape}")

        # -------------------------------------------------------------
        # 2. ADD ONLY NEW COLUMNS (if not already present)
        # -------------------------------------------------------------
        # Example: Add a simple derived column
        if 'day_of_week' not in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['day_of_week'] = df['timestamp'].dt.day_name()

        if 'is_weekend' not in df.columns:
            df['is_weekend'] = df['timestamp'].dt.weekday >= 5

        # Add more columns here as needed...

        # -------------------------------------------------------------
        # 3. REORDER COLUMNS (your desired order)
        # -------------------------------------------------------------
        desired_order = [
            'whatsapp_group', 'timestamp', 'year', 'month', 'week', 'day_of_week', 'is_weekend',
            'author', 'active_years', 'early_leaver',
            'number_of_chats_that_day', 'length_chat', 'response_time',
            'has_emoji', 'number_of_emojis', 'pct_emojis',
            'has_punctuation', 'number_of_punctuations', 'pct_punctuations',
            'has_link', 'was_deleted',
            'pictures_deleted', 'videos_deleted', 'audios_deleted',
            'gifs_deleted', 'stickers_deleted', 'documents_deleted', 'videonotes_deleted',
            'day_pct_length_chat', 'day_pct_length_emojis', 'day_pct_length_punctuations',
            'number_of_unique_participants_that_day', 'day_pct_authors',
            'previous_author', 'next_author', 'sequence_authors', 'sequence_response_times'
        ]

        # Keep only existing columns
        final_cols = [col for col in desired_order if col in df.columns]
        missing = [col for col in desired_order if col not in df.columns]
        if missing:
            logger.warning(f"Columns not found (will be skipped): {missing}")

        df_ordered = df[final_cols].copy()

        # -------------------------------------------------------------
        # 4. SAVE TO TABLES DIR
        # -------------------------------------------------------------
        output_path = self.tables_dir / "script10_enhanced_table.csv"
        df_ordered.to_csv(output_path, index=False)
        logger.info(f"Script10 complete: Saved enhanced table → {output_path}")
        logger.info(f"Final shape: {df_ordered.shape}, Columns: {len(final_cols)}")

        return df_ordered  # optional: return for pipeline