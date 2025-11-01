from src.file_manager import FileManager
from src.data_editor import DataEditor
from pathlib import Path

fm = FileManager()
de = DataEditor()

processed_dir = Path("data") / "processed"
enriched_csv = fm.enrich_all_groups(de, processed_dir)

if enriched_csv:
    print(f"\nREADY! Use this file in style_analyzer:\n{enriched_csv}")